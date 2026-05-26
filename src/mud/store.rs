use rusqlite::{params, Connection};
use std::sync::Mutex;
use byteorder::{ReadBytesExt, WriteBytesExt, LittleEndian};
use std::io::Cursor;

/// MUD Knowledge Store (MKS)
/// Manages persistent storage of facts and conversation history.
pub struct MudStore {
    conn: Mutex<Connection>,
}

impl MudStore {
    /// Opens or creates the persistent knowledge database.
    pub fn open(path: &str) -> anyhow::Result<Self> {
        let conn = Connection::open(path)?;

        // WAL mode: permite lectura concurrente sin bloquear escrituras (inferencia + auto-trainer)
        conn.pragma_update(None, "journal_mode", "WAL")?;
        // Tiempo de espera si la DB está bloqueada (ms) — evita error inmediato bajo contención
        conn.pragma_update(None, "busy_timeout", 5000)?;
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT,
                embedding BLOB,
                rank REAL DEFAULT 1.0,
                status INTEGER DEFAULT 0,
                learning_mark INTEGER DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Crear índice invertido FTS5 para RAG ultra rápido
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                content,
                content='facts',
                content_rowid='id'
            )",
            [],
        )?;

        // Triggers para mantener FTS5 sincronizado sin costo extra
        conn.execute_batch("
            CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
                INSERT INTO facts_fts(rowid, content) VALUES (new.id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, content) VALUES('delete', old.id, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, content) VALUES('delete', old.id, old.content);
                INSERT INTO facts_fts(rowid, content) VALUES (new.id, new.content);
            END;
        ")?;

        // Índices para las consultas más frecuentes del hot-path de inferencia
        conn.execute_batch("
            CREATE INDEX IF NOT EXISTS idx_facts_status     ON facts(status);
            CREATE INDEX IF NOT EXISTS idx_facts_rank       ON facts(rank DESC);
            CREATE INDEX IF NOT EXISTS idx_facts_timestamp  ON facts(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_facts_mark       ON facts(learning_mark);
        ")?;

        Ok(Self { conn: Mutex::new(conn) })
    }

    /// Adds a new fact with its embedding to the unassimilated queue (skips duplicates).
    pub fn add_fact_with_embedding(&self, content: &str, source: &str, embedding: &[f32]) -> anyhow::Result<()> {
        self.add_fact_with_mark(content, source, embedding, 0)
    }

    /// Adds a new fact with a specific learning mark.
    pub fn add_fact_with_mark(&self, content: &str, source: &str, embedding: &[f32], mark: i32) -> anyhow::Result<()> {
        let mut wtr = Vec::with_capacity(embedding.len() * 4);
        for &val in embedding {
            wtr.write_f32::<LittleEndian>(val)?;
        }

        let conn = self.conn.lock()
            .map_err(|_| anyhow::anyhow!("MudStore: Mutex envenenado (un hilo previo hizo panic)"))?;
        let exists: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM facts WHERE content = ?1",
            params![content],
            |row| row.get(0),
        ).unwrap_or(false);
        
        if !exists {
            conn.execute(
                "INSERT INTO facts (content, source, embedding, status, learning_mark) VALUES (?1, ?2, ?3, 0, ?4)",
                params![content, source, wtr, mark],
            )?;
        }
        Ok(())
    }

    /// Updates the learning mark of a fact.
    pub fn update_learning_mark(&self, id: i32, mark: i32) -> anyhow::Result<()> {
        let conn = self.conn.lock()
            .map_err(|_| anyhow::anyhow!("MudStore: Mutex envenenado"))?;
        conn.execute("UPDATE facts SET learning_mark = ?1 WHERE id = ?2", params![mark, id])?;
        Ok(())
    }

    /// Retrieves facts filtered by learning mark.
    pub fn get_facts_by_mark(&self, mark: i32, limit: usize) -> anyhow::Result<Vec<(i32, String)>> {
        let conn = self.conn.lock()
            .map_err(|_| anyhow::anyhow!("MudStore: Mutex envenenado"))?;
        let mut stmt = conn.prepare("SELECT id, content FROM facts WHERE learning_mark = ?1 LIMIT ?2")?;
        let rows = stmt.query_map(params![mark, limit as i64], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        let mut results = Vec::new();
        for row in rows { results.push(row?); }
        Ok(results)
    }

    /// Retrieves nodes with the highest PageRank to load into memory as 'hubs'.
    /// Filters out HTML/noise content to keep only meaningful text facts.
    pub fn get_top_hubs(&self, limit: usize) -> anyhow::Result<Vec<(String, Vec<f32>, f32)>> {
        let conn = self.conn.lock()
            .map_err(|_| anyhow::anyhow!("MudStore: Mutex envenenado"))?;
        // Prefer facts with real text (not HTML), diversified via RANDOM secondary sort
        let mut stmt = conn.prepare(
            "SELECT content, embedding, rank FROM facts
             WHERE embedding IS NOT NULL
               AND LENGTH(embedding) > 0
               AND content NOT LIKE '<%'
               AND content NOT LIKE '<!%'
               AND LENGTH(content) > 20
               AND LENGTH(content) < 1000
             ORDER BY rank DESC, RANDOM()
             LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            let content: String = row.get(0)?;
            let emb_blob: Option<Vec<u8>> = row.get(1)?;
            let rank_raw: f64 = row.get(2)?;  // f64 porque SQLite REAL es 64-bit

            // Sanitizar rank: NaN / Inf corrompería PageRank
            let rank = (rank_raw as f32).clamp(0.0, 1e6);

            let mut embedding = Vec::new();
            if let Some(blob) = emb_blob {
                // Guarda: el blob debe ser múltiplo de 4 bytes (cada f32 = 4 bytes)
                let aligned_len = blob.len() - (blob.len() % 4);
                let mut rdr = Cursor::new(&blob[..aligned_len]);
                while let Ok(val) = rdr.read_f32::<LittleEndian>() {
                    // Sanitizar valores NaN/Inf del embedding antes de entrar al grafo
                    embedding.push(if val.is_finite() { val } else { 0.0 });
                }
            }
            Ok((content, embedding, rank))
        })?;

        let mut results = Vec::new();
        for row in rows {
            let r = row?;
            // Only include nodes with non-zero embeddings
            if !r.1.is_empty() {
                results.push(r);
            }
        }
        Ok(results)
    }

    /// Returns (total_facts, assimilated, with_embedding) counts for telemetry.
    pub fn get_stats(&self) -> (i64, i64, i64) {
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return (0, 0, 0),  // Mutex envenenado: devolver ceros en lugar de panic
        };
        let total: i64 = conn
            .query_row("SELECT COUNT(*) FROM facts", [], |r| r.get(0))
            .unwrap_or(0);
        let assimilated: i64 = conn
            .query_row("SELECT COUNT(*) FROM facts WHERE status = 1", [], |r| r.get(0))
            .unwrap_or(0);
        let with_emb: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM facts WHERE embedding IS NOT NULL AND LENGTH(embedding) > 0",
                [], |r| r.get(0),
            )
            .unwrap_or(0);
        (total, assimilated, with_emb)
    }

    /// Updates the rank of a fact in the database.
    pub fn update_rank(&self, content: &str, rank: f32) -> anyhow::Result<()> {
        // Sanitizar antes de persistir: NaN / Inf corrompería ORDER BY rank DESC
        let safe_rank = if rank.is_finite() { rank.clamp(0.0, 1e6) } else { 1.0 };
        let conn = self.conn.lock()
            .map_err(|_| anyhow::anyhow!("MudStore: Mutex envenenado"))?;
        conn.execute("UPDATE facts SET rank = ?1 WHERE content = ?2", params![safe_rank, content])?;
        Ok(())
    }

    /// Retrieves candidates from disk based on recent additions or rank.
    pub fn get_potential_candidates(&self) -> anyhow::Result<Vec<(String, Vec<f32>, f32)>> {
        let conn = self.conn.lock()
            .map_err(|_| anyhow::anyhow!("MudStore: Mutex envenenado"))?;
        let mut stmt = conn.prepare("SELECT content, embedding, rank FROM facts ORDER BY timestamp DESC LIMIT 50")?;
        let rows = stmt.query_map([], |row| {
            let content: String = row.get(0)?;
            let emb_blob: Option<Vec<u8>> = row.get(1)?;
            let rank_raw: f64 = row.get(2)?;
            let rank = (rank_raw as f32).clamp(0.0, 1e6);
            
            let mut embedding = Vec::new();
            if let Some(blob) = emb_blob {
                let aligned_len = blob.len() - (blob.len() % 4);
                let mut rdr = Cursor::new(&blob[..aligned_len]);
                while let Ok(val) = rdr.read_f32::<LittleEndian>() {
                    embedding.push(if val.is_finite() { val } else { 0.0 });
                }
            }
            Ok((content, embedding, rank))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn add_fact(&self, content: &str, source: &str, hidden_size: usize) -> anyhow::Result<()> {
        self.add_fact_with_embedding(content, source, &vec![0.0; hidden_size]) 
    }

    pub fn get_unassimilated(&self) -> anyhow::Result<Vec<(i32, String)>> {
        let conn = self.conn.lock()
            .map_err(|_| anyhow::anyhow!("MudStore: Mutex envenenado"))?;
        // LIMIT 1000 para evitar cargar la tabla entera en RAM si hay millones de hechos
        let mut stmt = conn.prepare("SELECT id, content FROM facts WHERE status = 0 LIMIT 1000")?;
        let rows = stmt.query_map([], |row| {
            let id: i32 = row.get(0)?;
            let content: String = row.get(1)?;
            Ok((id, content))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn mark_as_packed(&self, ids: &[i32]) -> anyhow::Result<()> {
        let mut conn = self.conn.lock()
            .map_err(|_| anyhow::anyhow!("MudStore: Mutex envenenado"))?;
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare("UPDATE facts SET status = 1 WHERE id = ?1")?;
            for id in ids {
                stmt.execute(params![id])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Enforces a Time-To-Live (TTL) of 1 year on the database.
    /// Deletes facts older than 365 days to maintain dynamic relevance.
    pub fn enforce_ttl(&self) -> anyhow::Result<usize> {
        let conn = self.conn.lock()
            .map_err(|_| anyhow::anyhow!("MudStore: Mutex envenenado"))?;
        let count = conn.execute(
            "DELETE FROM facts WHERE timestamp < datetime('now', '-365 days')",
            [],
        )?;
        if count > 0 {
            println!("  [TTL Enforcement] Purged {} obsolete facts (older than 1 year).", count);
        }
        Ok(count)
    }

    /// Performs a WAL checkpoint to transfer log writes to the main DB and free disk space.
    pub fn checkpoint(&self) -> anyhow::Result<()> {
        let conn = self.conn.lock().map_err(|_| anyhow::anyhow!("MudStore: Mutex envenenado"))?;
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);", [])?;
        Ok(())
    }
}

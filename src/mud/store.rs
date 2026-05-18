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
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT,
                embedding BLOB,
                rank REAL DEFAULT 1.0,
                status INTEGER DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        Ok(Self { conn: Mutex::new(conn) })
    }

    /// Adds a new fact with its embedding to the unassimilated queue (skips duplicates).
    pub fn add_fact_with_embedding(&self, content: &str, source: &str, embedding: &[f32]) -> anyhow::Result<()> {
        let mut wtr = Vec::with_capacity(embedding.len() * 4);
        for &val in embedding {
            wtr.write_f32::<LittleEndian>(val)?;
        }

        let conn = self.conn.lock().unwrap();
        let exists: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM facts WHERE content = ?1",
            params![content],
            |row| row.get(0),
        ).unwrap_or(false);
        
        if !exists {
            conn.execute(
                "INSERT INTO facts (content, source, embedding, status) VALUES (?1, ?2, ?3, 0)",
                params![content, source, wtr],
            )?;
        }
        Ok(())
    }

    /// Retrieves nodes with the highest PageRank to load into memory as 'hubs'.
    pub fn get_top_hubs(&self, limit: usize) -> anyhow::Result<Vec<(String, Vec<f32>, f32)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT content, embedding, rank FROM facts ORDER BY rank DESC LIMIT ?1")?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            let content: String = row.get(0)?;
            let emb_blob: Option<Vec<u8>> = row.get(1)?;
            let rank: f32 = row.get(2)?;
            
            let mut embedding = Vec::new();
            if let Some(blob) = emb_blob {
                let mut rdr = Cursor::new(blob);
                while let Ok(val) = rdr.read_f32::<LittleEndian>() {
                    embedding.push(val);
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

    /// Updates the rank of a fact in the database.
    pub fn update_rank(&self, content: &str, rank: f32) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("UPDATE facts SET rank = ?1 WHERE content = ?2", params![rank, content])?;
        Ok(())
    }

    /// Retrieves candidates from disk based on recent additions or rank.
    pub fn get_potential_candidates(&self) -> anyhow::Result<Vec<(String, Vec<f32>, f32)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT content, embedding, rank FROM facts ORDER BY timestamp DESC LIMIT 50")?;
        let rows = stmt.query_map([], |row| {
            let content: String = row.get(0)?;
            let emb_blob: Option<Vec<u8>> = row.get(1)?;
            let rank: f32 = row.get(2)?;
            
            let mut embedding = Vec::new();
            if let Some(blob) = emb_blob {
                let mut rdr = Cursor::new(blob);
                while let Ok(val) = rdr.read_f32::<LittleEndian>() {
                    embedding.push(val);
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

    pub fn add_fact(&self, content: &str, source: &str) -> anyhow::Result<()> {
        self.add_fact_with_embedding(content, source, &vec![0.0; 512]) 
    }

    pub fn get_unassimilated(&self) -> anyhow::Result<Vec<(i32, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, content FROM facts WHERE status = 0")?;
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
        let conn = self.conn.lock().unwrap();
        for id in ids {
            conn.execute("UPDATE facts SET status = 1 WHERE id = ?1", params![id])?;
        }
        Ok(())
    }
}

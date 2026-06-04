use rusqlite::Connection;

fn main() {
    let conn = Connection::open("knowledge.db").unwrap();
    conn.execute("CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, content TEXT, learning_mark INTEGER DEFAULT 0, last_trained INTEGER DEFAULT 0)", []).unwrap();
    conn.execute("INSERT INTO facts (content, learning_mark) VALUES ('El modelo MUD es la inteligencia ternaria del futuro.', 0)", []).unwrap();
    conn.execute("INSERT INTO facts (content, learning_mark) VALUES ('La arquitectura híbrida Jamba es revolucionaria.', 0)", []).unwrap();
    conn.execute("INSERT INTO facts (content, learning_mark) VALUES ('El universo observable tiene 93 mil millones de años luz de diámetro.', 0)", []).unwrap();
    conn.execute("INSERT INTO facts (content, learning_mark) VALUES ('El agua hierve a 100 grados Celsius a nivel del mar.', 0)", []).unwrap();
    conn.execute("INSERT INTO facts (content, learning_mark) VALUES ('La gravedad es una curvatura del espacio-tiempo.', 0)", []).unwrap();
    println!("Base de datos 'knowledge.db' inicializada con 5 hechos.");
}

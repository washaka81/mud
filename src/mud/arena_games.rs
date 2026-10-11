pub trait ArenaGame {
    fn name(&self) -> &str;
    fn get_state_prompt(&self) -> String;
    fn apply_move(&mut self, player: usize, move_str: &str) -> Result<f32, String>; // Returns reward
    fn is_terminal(&self) -> bool;
    fn winner(&self) -> Option<usize>;
    /// Professor-Student grading data (exercise, answer, correction, rubrik).
    /// Default: None (games that are not ProfessorStudent).
    fn professor_data(&self) -> Option<(String, String, String, [f32; 4])> {
        None
    }
}

// 1. TIC-TAC-TOE
pub struct TicTacToe {
    board: [char; 9],
    turn: usize, // 0 for Player A (X), 1 for Player B (O)
    winner: Option<usize>,
}

impl Default for TicTacToe {
    fn default() -> Self {
        Self::new()
    }
}

impl TicTacToe {
    pub fn new() -> Self {
        Self {
            board: [' '; 9],
            turn: 0,
            winner: None,
        }
    }
}

impl ArenaGame for TicTacToe {
    fn name(&self) -> &str {
        "Tic-Tac-Toe"
    }

    fn get_state_prompt(&self) -> String {
        let b = &self.board;
        format!(
            "Tic-Tac-Toe Board:\n {} | {} | {} \n---+---+---\n {} | {} | {} \n---+---+---\n {} | {} | {} \nPlayer {}, provide your move as a single number (0-8): ",
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], if self.turn == 0 { "A (X)" } else { "B (O)" }
        )
    }

    fn apply_move(&mut self, player: usize, move_str: &str) -> Result<f32, String> {
        if self.is_terminal() {
            return Ok(0.0);
        }
        if player != self.turn {
            return Err("Not your turn".into());
        }

        let cleaned: String = move_str.chars().filter(|c| c.is_ascii_digit()).collect();
        // Aphasic / illegal move: forfeit the turn instead of stalling the match.
        let Ok(m) = cleaned.parse::<usize>() else {
            self.turn = 1 - self.turn;
            return Ok(-0.5);
        };
        if m > 8 || self.board[m] != ' ' {
            self.turn = 1 - self.turn;
            return Ok(-0.5);
        }

        self.board[m] = if player == 0 { 'X' } else { 'O' };

        let wins = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];

        for w in wins.iter() {
            if self.board[w[0]] != ' '
                && self.board[w[0]] == self.board[w[1]]
                && self.board[w[1]] == self.board[w[2]]
            {
                self.winner = Some(player);
                return Ok(1.0);
            }
        }

        if !self.board.contains(&' ') {
            self.winner = Some(99); // Draw
            return Ok(0.5); // Small reward for draw
        }

        self.turn = 1 - self.turn;
        Ok(0.1) // Small step reward
    }

    fn is_terminal(&self) -> bool {
        self.winner.is_some()
    }
    fn winner(&self) -> Option<usize> {
        if self.winner == Some(99) {
            None
        } else {
            self.winner
        }
    }
}

// 2. MATH EXERCISES
pub struct MathChallenge {
    question: String,
    answer: i32,
    solved: bool,
    winner: Option<usize>,
    turn: usize,
    attempts: usize,
    max_attempts: usize,
}

impl MathChallenge {
    pub fn new(q: &str, a: i32) -> Self {
        Self {
            question: q.to_string(),
            answer: a,
            solved: false,
            winner: None,
            turn: 0,
            attempts: 0,
            // Bound the match so an unaligned model can't loop forever: each
            // player gets two tries, then the match is declared over (no winner).
            // Kept small (4) so the verifiable benchmark completes a match within
            // its time-box even on slow CPU inference.
            max_attempts: 4,
        }
    }

    /// Local, no-API default challenge (rotates a small deterministic pool).
    pub fn random() -> Self {
        let pool: &[(&str, i32)] = &[
            ("¿Cuánto es 7 * 8?", 56),
            ("¿Cuánto es 15 + 27?", 42),
            ("¿Cuánto es 144 / 12?", 12),
            ("¿Cuánto es 9 * 9?", 81),
            ("¿Cuánto es 100 - 37?", 63),
        ];
        // Deterministic pick from a rotating counter (no RNG, P-07 friendly).
        use std::sync::atomic::{AtomicUsize, Ordering};
        static PICK: AtomicUsize = AtomicUsize::new(0);
        let i = PICK.fetch_add(1, Ordering::SeqCst) % pool.len();
        Self::new(pool[i].0, pool[i].1)
    }
}

impl ArenaGame for MathChallenge {
    fn name(&self) -> &str {
        "Math Challenge"
    }

    fn get_state_prompt(&self) -> String {
        format!(
            "Math Question: {}\nPlayer {}, provide your numeric answer: ",
            self.question,
            if self.turn == 0 { "A" } else { "B" }
        )
    }

    fn apply_move(&mut self, player: usize, move_str: &str) -> Result<f32, String> {
        if self.is_terminal() {
            return Ok(0.0);
        }
        if player != self.turn {
            return Err("Not your turn".into());
        }

        self.attempts += 1;
        let cleaned: String = move_str
            .chars()
            .filter(|c| c.is_ascii_digit() || *c == '-')
            .collect();
        // An unparseable (aphasic) answer is a failed attempt, not a hard error:
        // advance the turn so the match keeps progressing toward its bound.
        let Ok(m) = cleaned.parse::<i32>() else {
            self.turn = 1 - self.turn;
            return Ok(-0.5);
        };

        if m == self.answer {
            self.solved = true;
            self.winner = Some(player);
            Ok(1.0)
        } else {
            self.turn = 1 - self.turn; // Pass to next player
            Ok(-0.5) // Penalty for wrong answer
        }
    }

    fn is_terminal(&self) -> bool {
        self.solved || self.attempts >= self.max_attempts
    }
    fn winner(&self) -> Option<usize> {
        self.winner
    }
}

// 3. DOCUMENT DEBATE (Organic Knowledge Escalation)
pub struct DocumentDebate {
    topic: String,
    document: String,
    turn: usize,
    max_turns: usize,
    last_a: String,
    last_b: String,
    history: Vec<(usize, String)>,
}

impl DocumentDebate {
    pub fn new(topic: &str, document: &str, max_turns: usize) -> Self {
        Self {
            topic: topic.to_string(),
            document: document.to_string(),
            turn: 0,
            max_turns,
            last_a: String::new(),
            last_b: String::new(),
            history: Vec::new(),
        }
    }

    /// Last response from player A and player B (for the local TextJudge).
    pub fn last_responses(&self) -> (&str, &str) {
        (&self.last_a, &self.last_b)
    }

    /// Configured max turns per debate match.
    pub fn max_turns(&self) -> usize {
        self.max_turns
    }
}

impl ArenaGame for DocumentDebate {
    fn name(&self) -> &str {
        "Document Debate"
    }

    fn get_state_prompt(&self) -> String {
        if self.turn == 0 {
            format!(
                "Debate Topic: {}\nReference Document: {}\nPlayer A, state your opening argument: ",
                self.topic, self.document
            )
        } else {
            format!(
                "Debate Topic: {}\nPlayer {}, counter the previous argument with organic knowledge: ",
                self.topic, if self.turn.is_multiple_of(2) { "A" } else { "B" }
            )
        }
    }

    fn apply_move(&mut self, player: usize, move_str: &str) -> Result<f32, String> {
        if self.is_terminal() {
            return Ok(0.0);
        }
        if player != (self.turn % 2) {
            return Err("Not your turn".into());
        }

        let length = move_str.trim().len();
        if length < 10 {
            // Penalty for giving up or too short answers (aphasia)
            self.turn += 1;
            return Ok(-0.5);
        }

        if player == 0 {
            self.last_a = move_str.trim().to_string();
        } else {
            self.last_b = move_str.trim().to_string();
        }
        self.history.push((player, move_str.trim().to_string()));

        self.turn += 1;
        // The real reward is calculated by the local TextJudge in the arena loop
        // (verifiable claim score, no external API). Small positive baseline here.
        Ok(0.2)
    }

    fn is_terminal(&self) -> bool {
        self.turn >= self.max_turns
    }

    fn winner(&self) -> Option<usize> {
        // Resolved by the arena loop via TextJudge (no implicit JEPA decision).
        None
    }
}

// 4. GRAMMAR & LANGUAGES CHALLENGE
pub struct GrammarChallenge {
    task: String,
    expected_keyword: String,
    solved: bool,
    winner: Option<usize>,
    turn: usize,
}

impl GrammarChallenge {
    pub fn new(task: &str, expected_keyword: &str) -> Self {
        Self {
            task: task.to_string(),
            expected_keyword: expected_keyword.to_lowercase(),
            solved: false,
            winner: None,
            turn: 0,
        }
    }
}

impl ArenaGame for GrammarChallenge {
    fn name(&self) -> &str {
        "Grammar & Languages"
    }

    fn get_state_prompt(&self) -> String {
        format!(
            "Language Task: {}\nPlayer {}, provide your translation or correction: ",
            self.task,
            if self.turn.is_multiple_of(2) {
                "A"
            } else {
                "B"
            }
        )
    }

    fn apply_move(&mut self, player: usize, move_str: &str) -> Result<f32, String> {
        if self.is_terminal() {
            return Ok(0.0);
        }
        if player != (self.turn % 2) {
            return Err("Not your turn".into());
        }

        let response = move_str.trim().to_lowercase();

        if response.contains(&self.expected_keyword) {
            self.solved = true;
            self.winner = Some(player);
            Ok(1.0) // Strong reward for correct translation/grammar
        } else {
            self.turn += 1;
            Ok(-0.5) // Penalty for incorrect translation
        }
    }

    fn is_terminal(&self) -> bool {
        self.solved
    }
    fn winner(&self) -> Option<usize> {
        self.winner
    }
}

// 5. PROFESSOR-STUDENT (RLVR supervised loop: grammar / syntax / coherence /
//    pragmatism). The professor (player A) poses an exercise and later grades
//    the student's (player B) answer locally; the student repeats + learns.
//    No external API: exercises are a local pool; grading is heuristic.
pub struct ProfessorStudent {
    exercise: String,
    category: String,
    turn: usize,
    max_turns: usize,
    answer: String,
    correction: String,
    rubrik: [f32; 4], // grammar, syntax, coherence, pragmatism (filled by judge)
}

/// Local exercise pool (no network). Each entry: (category, exercise text).
pub fn professor_exercises() -> Vec<(String, String)> {
    vec![
        (
            "grammar".into(),
            "Corrige la siguiente oración: 'El perro ladra y el gato lo mira desde el tejado con sus ojos brillante.'".to_string(),
        ),
        (
            "syntax".into(),
            "Reescribe usando voz pasiva: 'El equipo de MUD implementó el kernel de inferencia en Rust.'".to_string(),
        ),
        (
            "coherence".into(),
            "Une estas dos ideas en un párrafo coherente: (1) la computación ternaria reduce costo energético; (2) el aproximación requiere cuidado numérico.".to_string(),
        ),
        (
            "pragmatism".into(),
            "Explica a un usuario no técnico por qué un modelo de 1.58-bit es útil en su teléfono, en menos de 40 palabras.".to_string(),
        ),
        (
            "grammar".into(),
            "Conjuga y acentúa correctamente: 'Si yo (saber) la respuesta, la (decir) antes de que tu (llegar).'".to_string(),
        ),
        (
            "syntax".into(),
            "Separa en dos oraciones simples: 'Aunque llueva el modelo sigue entrenando porque el checkpoint es resistente a caídas.'".to_string(),
        ),
    ]
}

impl ProfessorStudent {
    pub fn new(max_turns: usize, exercise_idx: usize) -> Self {
        let pool = professor_exercises();
        let (category, exercise) = pool[exercise_idx % pool.len()].clone();
        Self {
            exercise,
            category,
            turn: 0,
            max_turns,
            answer: String::new(),
            correction: String::new(),
            rubrik: [0.0; 4],
        }
    }

    pub fn exercise(&self) -> &str {
        &self.exercise
    }
    pub fn category(&self) -> &str {
        &self.category
    }
    pub fn answer(&self) -> &str {
        &self.answer
    }
    pub fn correction(&self) -> &str {
        &self.correction
    }
    pub fn set_rubrik(&mut self, r: [f32; 4]) {
        self.rubrik = r;
    }
    pub fn rubrik(&self) -> [f32; 4] {
        self.rubrik
    }
    /// Phase: 0 = professor poses, 1 = student answers, 2 = professor grades.
    pub fn phase(&self) -> usize {
        self.turn % 3
    }
}

impl ArenaGame for ProfessorStudent {
    fn name(&self) -> &str {
        "Professor-Student"
    }

    fn get_state_prompt(&self) -> String {
        match self.phase() {
            0 => format!(
                "[PROFESOR] Ejercicio ({}, vuelve a redactarlo para el alumno):\n{}",
                self.category, self.exercise
            ),
            1 => format!(
                "[ALUMNO] Responde al ejercicio:\n{}\n(Tu respuesta será evaluada en gramática, sintaxis, coherencia y pragmatismo.)",
                self.exercise
            ),
            _ => format!(
                "[PROFESOR] Corrige/evalúa la respuesta del alumno:\nEjercicio: {}\nRespuesta: {}\nDa una corrección concisa.",
                self.exercise, self.answer
            ),
        }
    }

    fn apply_move(&mut self, player: usize, move_str: &str) -> Result<f32, String> {
        if self.is_terminal() {
            return Ok(0.0);
        }
        let phase = self.phase();
        // Enforce role by phase: phase 0/2 professor (A=0), phase 1 student (B=1).
        let expected = if phase == 1 { 1 } else { 0 };
        if player != expected {
            return Err("Rol incorrecto para esta fase".into());
        }

        let text = move_str.trim().to_string();
        if text.len() < 8 {
            self.turn += 1;
            return Ok(-0.5); // aphasia penalty
        }

        match phase {
            0 => { /* professor re-states the exercise; no scoring yet */ }
            1 => self.answer = text,
            2 => self.correction = text,
            _ => {}
        }
        self.turn += 1;
        Ok(0.2)
    }

    fn is_terminal(&self) -> bool {
        self.turn >= self.max_turns.min(3)
    }

    fn winner(&self) -> Option<usize> {
        None // graded by ProfessorJudge in the arena loop
    }

    fn professor_data(&self) -> Option<(String, String, String, [f32; 4])> {
        Some((
            self.exercise.clone(),
            self.answer.clone(),
            self.correction.clone(),
            self.rubrik,
        ))
    }
}

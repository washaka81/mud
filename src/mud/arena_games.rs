pub trait ArenaGame {
    fn name(&self) -> &str;
    fn get_state_prompt(&self) -> String;
    fn apply_move(&mut self, player: usize, move_str: &str) -> Result<f32, String>; // Returns reward
    fn is_terminal(&self) -> bool;
    fn winner(&self) -> Option<usize>;
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
        Self { board: [' '; 9], turn: 0, winner: None }
    }
}

impl ArenaGame for TicTacToe {
    fn name(&self) -> &str { "Tic-Tac-Toe" }
    
    fn get_state_prompt(&self) -> String {
        let b = &self.board;
        format!(
            "Tic-Tac-Toe Board:\n {} | {} | {} \n---+---+---\n {} | {} | {} \n---+---+---\n {} | {} | {} \nPlayer {}, provide your move as a single number (0-8): ",
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], if self.turn == 0 { "A (X)" } else { "B (O)" }
        )
    }

    fn apply_move(&mut self, player: usize, move_str: &str) -> Result<f32, String> {
        if self.is_terminal() { return Ok(0.0); }
        if player != self.turn { return Err("Not your turn".into()); }
        
        let cleaned: String = move_str.chars().filter(|c| c.is_ascii_digit()).collect();
        let m = cleaned.parse::<usize>().map_err(|_| "Invalid format")?;
        
        if m > 8 { return Err("Out of bounds".into()); }
        if self.board[m] != ' ' { return Err("Cell occupied".into()); }
        
        self.board[m] = if player == 0 { 'X' } else { 'O' };
        
        let wins = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ];
        
        for w in wins.iter() {
            if self.board[w[0]] != ' ' && self.board[w[0]] == self.board[w[1]] && self.board[w[1]] == self.board[w[2]] {
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
    
    fn is_terminal(&self) -> bool { self.winner.is_some() }
    fn winner(&self) -> Option<usize> { if self.winner == Some(99) { None } else { self.winner } }
}

// 2. MATH EXERCISES
pub struct MathChallenge {
    question: String,
    answer: i32,
    solved: bool,
    winner: Option<usize>,
    turn: usize,
}

impl MathChallenge {
    pub fn new(q: &str, a: i32) -> Self {
        Self { question: q.to_string(), answer: a, solved: false, winner: None, turn: 0 }
    }
}

impl ArenaGame for MathChallenge {
    fn name(&self) -> &str { "Math Challenge" }
    
    fn get_state_prompt(&self) -> String {
        format!("Math Question: {}\nPlayer {}, provide your numeric answer: ", self.question, if self.turn == 0 { "A" } else { "B" })
    }
    
    fn apply_move(&mut self, player: usize, move_str: &str) -> Result<f32, String> {
        if self.is_terminal() { return Ok(0.0); }
        if player != self.turn { return Err("Not your turn".into()); }
        
        let cleaned: String = move_str.chars().filter(|c| c.is_ascii_digit() || *c == '-').collect();
        let m = cleaned.parse::<i32>().map_err(|_| "Invalid format")?;
        
        if m == self.answer {
            self.solved = true;
            self.winner = Some(player);
            Ok(1.0)
        } else {
            self.turn = 1 - self.turn; // Pass to next player
            Ok(-0.5) // Penalty for wrong answer
        }
    }
    
    fn is_terminal(&self) -> bool { self.solved }
    fn winner(&self) -> Option<usize> { self.winner }
}

// 3. DOCUMENT DEBATE (Organic Knowledge Escalation)
pub struct DocumentDebate {
    topic: String,
    document: String,
    turn: usize,
    max_turns: usize,
}

impl DocumentDebate {
    pub fn new(topic: &str, document: &str, max_turns: usize) -> Self {
        Self {
            topic: topic.to_string(),
            document: document.to_string(),
            turn: 0,
            max_turns,
        }
    }
}

impl ArenaGame for DocumentDebate {
    fn name(&self) -> &str { "Document Debate" }
    
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
        if self.is_terminal() { return Ok(0.0); }
        if player != (self.turn % 2) { return Err("Not your turn".into()); }
        
        let length = move_str.trim().len();
        if length < 10 {
            // Penalty for giving up or too short answers (aphasia)
            self.turn += 1;
            return Ok(-0.5);
        }
        
        self.turn += 1;
        // The real reward is calculated structurally by JEPA (VarJ / VarH) in the arena loop.
        // We provide a small positive baseline for constructing a valid argument.
        Ok(0.2) 
    }
    
    fn is_terminal(&self) -> bool {
        self.turn >= self.max_turns
    }
    
    fn winner(&self) -> Option<usize> {
        None // Decided by structural JEPA evaluation at the end of the debate
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
    fn name(&self) -> &str { "Grammar & Languages" }
    
    fn get_state_prompt(&self) -> String {
        format!("Language Task: {}\nPlayer {}, provide your translation or correction: ", self.task, if self.turn.is_multiple_of(2) { "A" } else { "B" })
    }
    
    fn apply_move(&mut self, player: usize, move_str: &str) -> Result<f32, String> {
        if self.is_terminal() { return Ok(0.0); }
        if player != (self.turn % 2) { return Err("Not your turn".into()); }
        
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
    
    fn is_terminal(&self) -> bool { self.solved }
    fn winner(&self) -> Option<usize> { self.winner }
}

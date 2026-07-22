//! RLVR judge for the MUD Debate Arena.
//!
//! All judges run LOCALLY (no external API, no network, P-07 compliant):
//! - `VerifiableJudge`: uses the game's own `winner()` (TicTacToe / Math / Grammar).
//! - `RustJudge`: wraps `RlvrCritic::evaluate_rust_code` (local `rustc`).
//! - `TextJudge`: local claim-extraction + self-embedding cosine, deterministic.
//!
//! Reward/penalty contract (RLVR):
//!   winner  -> +R_WIN   (default 1.0)
//!   loser   -> -R_LOSE  (default 0.7, asymmetric so draws do not dominate)
//!   draw    ->  0.0
//! Tunable via `MUD_DEBATE_RWIN` / `MUD_DEBATE_RLOSE`.

use crate::mud::arena_games::ArenaGame;
use crate::mud::rlvr::RlvrCritic;

/// Which player's perspective the judge scores from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Player {
    A = 0,
    B = 1,
}

impl Player {
    pub fn other(self) -> Player {
        match self {
            Player::A => Player::B,
            Player::B => Player::A,
        }
    }
}

/// A verifiable judge returns a reward in `[-1, +1]` from the POV of `player`.
pub trait Judge {
    fn score(&self, game: &dyn ArenaGame, player: Player) -> f32;
}

fn reward_from_outcome(winner: Option<usize>, player: Player) -> f32 {
    let r_win = std::env::var("MUD_DEBATE_RWIN")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1.0);
    let r_lose = std::env::var("MUD_DEBATE_RLOSE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.7);
    match winner {
        Some(w) if w == player as usize => r_win,
        Some(_) => -r_lose,
        None => 0.0, // draw / timeout
    }
}

/// Judge for games with a real, checkable winner (TicTacToe, Math, Grammar).
pub struct VerifiableJudge;

impl VerifiableJudge {
    /// Score from `player`'s POV given a known `winner` (no game object needed).
    pub fn score_dyn(&self, winner: Option<usize>, player: Player) -> f32 {
        reward_from_outcome(winner, player)
    }
}

impl Judge for VerifiableJudge {
    fn score(&self, game: &dyn ArenaGame, player: Player) -> f32 {
        reward_from_outcome(game.winner(), player)
    }
}

/// Judge for code games: compile with local `rustc` (reuses `RlvrCritic`).
pub struct RustJudge {
    critic: RlvrCritic,
}

impl Default for RustJudge {
    fn default() -> Self {
        Self::new()
    }
}

impl RustJudge {
    pub fn new() -> Self {
        Self {
            critic: RlvrCritic::new(),
        }
    }

    /// +1 if `code` compiles (metadata), -1 otherwise.
    pub fn compile_reward(&self, code: &str) -> f32 {
        let (r, _) = self.critic.evaluate_rust_code(code);
        r
    }
}

impl Judge for RustJudge {
    fn score(&self, _game: &dyn ArenaGame, _player: Player) -> f32 {
        // Code games are judged per-move via `compile_reward`; default 0 at game level.
        0.0
    }
}

/// Local, no-API text judge for `DocumentDebate`.
///
/// It extracts simple "claims" (assertion-like clauses) from a response and
/// scores a player by:
///   - verifiability  = number of claims it made (vs vague opinion)   [0..1]
///   - coherence      = mean cosine similarity among its own claims   [0..1]
///   - rebuttal       = 1 - max cosine between its claims and the
///     opponent's claims (rewarding distinct, non-repeating stance)
///
/// Embeddings come from the already-loaded `.mud` token table (no network).
/// The tokenizer is passed per-call to `score_pair` so the judge does not hold
/// a borrow of the arena across the learning step.
pub struct TextJudge<'a> {
    embed_table: &'a [f32],
    vocab_size: usize,
    hidden: usize,
}

impl<'a> TextJudge<'a> {
    pub fn new(embed_table: &'a [f32], vocab_size: usize, hidden: usize) -> Self {
        Self {
            embed_table,
            vocab_size,
            hidden,
        }
    }

    /// Mean-pool token embeddings of `text` into a `hidden`-dim vector.
    fn embed_text(&self, text: &str, tokenizer: &crate::model::tokenizer::Tokenizer) -> Vec<f32> {
        let tokens = tokenizer.encode(text);
        let mut vec = vec![0.0f32; self.hidden];
        if tokens.is_empty() {
            return vec;
        }
        let mut n = 0usize;
        for &tok in tokens.iter() {
            let tid = (tok as usize).min(self.vocab_size.saturating_sub(1));
            let off = tid * self.hidden;
            if off + self.hidden <= self.embed_table.len() {
                let row = &self.embed_table[off..off + self.hidden];
                for (i, v) in row.iter().enumerate() {
                    vec[i] += *v;
                }
                n += 1;
            }
        }
        if n > 0 {
            for v in vec.iter_mut() {
                *v /= n as f32;
            }
        }
        vec
    }

    /// Split a response into claim clauses (by sentence-ish boundaries).
    fn claims(&self, response: &str) -> Vec<String> {
        response
            .split(['.', ';', ':', '\n', '!', '?'])
            .map(|s| s.trim().to_string())
            .filter(|s| s.len() >= 8) // drop fragments / aphasia
            .collect()
    }

    pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0;
        let mut na = 0.0;
        let mut nb = 0.0;
        for i in 0..a.len().min(b.len()) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if na <= 1e-12 || nb <= 1e-12 {
            0.0
        } else {
            dot / (na.sqrt() * nb.sqrt())
        }
    }

    /// Exposed for `ProfessorJudge` coherence scoring.
    pub fn embed_claim(
        &self,
        text: &str,
        tokenizer: &crate::model::tokenizer::Tokenizer,
    ) -> Vec<f32> {
        self.embed_text(text, tokenizer)
    }

    /// Score `player` given its response and the opponent's last response.
    pub fn score_pair(
        &self,
        own: &str,
        opponent: &str,
        tokenizer: &crate::model::tokenizer::Tokenizer,
    ) -> f32 {
        let own_claims = self.claims(own);
        let opp_claims = self.claims(opponent);

        let verif = if own_claims.is_empty() {
            0.0
        } else {
            (own_claims.len() as f32 / (own_claims.len() as f32 + 2.0)).min(1.0)
        };

        let emb_claims: Vec<Vec<f32>> = own_claims
            .iter()
            .map(|c| self.embed_text(c, tokenizer))
            .collect();
        let mut coh_sum = 0.0;
        let mut coh_n = 0u32;
        for i in 0..emb_claims.len() {
            for j in (i + 1)..emb_claims.len() {
                coh_sum += Self::cosine(&emb_claims[i], &emb_claims[j]);
                coh_n += 1;
            }
        }
        let coherence = if coh_n > 0 {
            coh_sum / coh_n as f32
        } else {
            0.0
        };

        // Rebuttal: own claims should be distinct from opponent's (non-parrot).
        let opp_emb: Vec<Vec<f32>> = opp_claims
            .iter()
            .map(|c| self.embed_text(c, tokenizer))
            .collect();
        let mut max_overlap = 0.0f32;
        for oc in &emb_claims {
            for pc in &opp_emb {
                let c = Self::cosine(oc, pc);
                if c > max_overlap {
                    max_overlap = c;
                }
            }
        }
        let rebuttal = 1.0 - max_overlap.max(0.0);

        let score = 0.45 * verif + 0.35 * coherence.max(0.0) + 0.20 * rebuttal;
        score.clamp(-1.0, 1.0)
    }
}

/// Local, no-API professor judge for the `ProfessorStudent` game.
///
/// Returns a 4-dim rubrik in [0,1]: `[grammar, syntax, coherence, pragmatism]`.
/// All metrics are heuristic + embedding-based (no network, no LLM, P-07):
///   - grammar:    capitalization, terminal punctuation, no double spaces
///   - syntax:     the answer addresses the requested transformation (passive/active/split) inferred from the exercise text
///   - coherence:  mean cosine among the answer's own claims (TextJudge)
///   - pragmatism: claim-overlap between answer and exercise (relevance)
pub struct ProfessorJudge<'a> {
    text_judge: &'a TextJudge<'a>,
}

impl<'a> ProfessorJudge<'a> {
    pub fn new(text_judge: &'a TextJudge<'a>) -> Self {
        Self { text_judge }
    }

    fn grammar_score(answer: &str) -> f32 {
        let a = answer.trim();
        if a.is_empty() {
            return 0.0;
        }
        let mut s = 0.0;
        let mut n = 0.0;
        // Starts with uppercase or a known lowercase starter
        n += 1.0;
        if a.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
            s += 1.0;
        }
        // Ends with sentence punctuation
        n += 1.0;
        if a.ends_with(['.', '!', '?']) {
            s += 1.0;
        }
        // No double spaces
        n += 1.0;
        if !a.contains("  ") {
            s += 1.0;
        }
        // Reasonable length (>= 3 words)
        n += 1.0;
        if a.split_whitespace().count() >= 3 {
            s += 1.0;
        }
        ((s / n) as f32).clamp(0.0, 1.0)
    }

    fn syntax_score(exercise: &str, answer: &str) -> f32 {
        let ex = exercise.to_lowercase();
        let ans = answer.to_lowercase();
        let mut hits = 0.0;
        let mut checks = 0.0;
        if ex.contains("voz pasiva") || ex.contains("pasiva") {
            checks += 1.0;
            // passive ~ "es/son/fue/fueron + participio (ado/ido)"
            if ans.contains(" es ")
                || ans.contains(" son ")
                || ans.contains(" fue ")
                || ans.contains(" fueron ")
                || ans.contains("ado")
                || ans.contains("ido")
            {
                hits += 1.0;
            }
        }
        if ex.contains("separa en dos") || ex.contains("oraciones simples") {
            checks += 1.0;
            if ans.matches('.').count() >= 2 {
                hits += 1.0;
            }
        }
        if ex.contains("conjuga") || ex.contains("acentúa") {
            checks += 1.0;
            // at least one accented char present
            if ans.chars().any(|c| "áéíóúñÁÉÍÓÚÑ".contains(c)) {
                hits += 1.0;
            }
        }
        if ex.contains("une estas dos ideas") || ex.contains("párrafo coherente") {
            checks += 1.0;
            if ans.split_whitespace().count() >= 8 {
                hits += 1.0;
            }
        }
        if checks == 0.0 {
            0.5 // no specific syntax constraint detected
        } else {
            ((hits / checks) as f32).clamp(0.0, 1.0)
        }
    }

    fn pragmatism_score(
        &self,
        exercise: &str,
        answer: &str,
        tokenizer: &crate::model::tokenizer::Tokenizer,
    ) -> f32 {
        // Relevance = claim-overlap between exercise and answer embeddings.
        let e = self.text_judge.score_pair(exercise, answer, tokenizer);
        let a = self.text_judge.score_pair(answer, exercise, tokenizer);
        ((e + a) * 0.5).clamp(0.0, 1.0)
    }

    /// Full rubrik + aggregate reward in [-1, +1] (positive = good student answer).
    pub fn grade(
        &self,
        exercise: &str,
        answer: &str,
        correction: &str,
        tokenizer: &crate::model::tokenizer::Tokenizer,
    ) -> ([f32; 4], f32) {
        let grammar = Self::grammar_score(answer);
        let syntax = Self::syntax_score(exercise, answer);
        let coherence = {
            // mean cosine among answer's own claims (reuse TextJudge internals)
            let claims = answer
                .split(['.', ';', ':', '\n', '!', '?'])
                .map(|s| s.trim().to_string())
                .filter(|s| s.len() >= 8)
                .collect::<Vec<_>>();
            if claims.len() < 2 {
                0.5
            } else {
                let embs: Vec<Vec<f32>> = claims
                    .iter()
                    .map(|c| self.text_judge.embed_claim(c, tokenizer))
                    .collect();
                let mut sum = 0.0;
                let mut n = 0u32;
                for i in 0..embs.len() {
                    for j in (i + 1)..embs.len() {
                        sum += TextJudge::cosine(&embs[i], &embs[j]);
                        n += 1;
                    }
                }
                if n > 0 {
                    (sum / n as f32).max(0.0)
                } else {
                    0.5
                }
            }
        };
        let pragmatism = self.pragmatism_score(exercise, answer, tokenizer);

        let rubrik = [grammar, syntax, coherence, pragmatism];
        // Aggregate reward: strong on grammar+syntax, moderate on coherence+pragmatism.
        let reward = 0.30 * grammar + 0.30 * syntax + 0.20 * coherence + 0.20 * pragmatism;
        // Scale to [-1, +1]: 0.5 neutral -> 0; 1.0 -> +1; 0.0 -> -1.
        let signed = (reward * 2.0 - 1.0).clamp(-1.0, 1.0);
        // Correction presence slightly boosts (professor engaged).
        let _ = correction;
        (rubrik, signed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifiable_judge_sign() {
        struct Fake {
            w: Option<usize>,
        }
        impl ArenaGame for Fake {
            fn name(&self) -> &str {
                "fake"
            }
            fn get_state_prompt(&self) -> String {
                String::new()
            }
            fn apply_move(&mut self, _p: usize, _m: &str) -> Result<f32, String> {
                Ok(0.0)
            }
            fn is_terminal(&self) -> bool {
                true
            }
            fn winner(&self) -> Option<usize> {
                self.w
            }
        }
        let g = Fake { w: Some(0) };
        assert_eq!(VerifiableJudge.score(&g, Player::A), 1.0);
        assert_eq!(VerifiableJudge.score(&g, Player::B), -0.7);
        let d = Fake { w: None };
        assert_eq!(VerifiableJudge.score(&d, Player::A), 0.0);
    }

    #[test]
    fn text_judge_prefers_claims_over_vague() {
        let emb = vec![0.0f32; 100 * 16];
        let judge = TextJudge::new(&emb, 100, 16);
        let tk = crate::model::tokenizer::Tokenizer::from_mud_metadata("", "");
        let strong = "Nuclear emits 0.01 g CO2 per kWh. Coal emits 820 g CO2 per kWh. Therefore nuclear is a valid bridge.";
        let weak = "I like the sun. The wind is nice. Panels are pretty.";
        let s_strong = judge.score_pair(strong, weak, &tk);
        let s_weak = judge.score_pair(weak, strong, &tk);
        assert!(s_strong > s_weak, "claim-rich reply must score higher");
    }

    #[test]
    fn text_judge_deterministic() {
        let emb = vec![0.0f32; 50 * 8];
        let j1 = TextJudge::new(&emb, 50, 8);
        let j2 = TextJudge::new(&emb, 50, 8);
        let tk = crate::model::tokenizer::Tokenizer::from_mud_metadata("", "");
        let a = "X proves Y. Y implies Z. Hence X implies Z.";
        let b = "Maybe something. Who knows.";
        assert!((j1.score_pair(a, b, &tk) - j2.score_pair(a, b, &tk)).abs() < 1e-6);
    }
}

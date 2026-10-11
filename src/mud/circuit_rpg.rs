use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitRpgStats {
    pub hp: f32,
    pub max_hp: f32,
    pub generation: u32,
    pub cycles_survived: u32,
    pub win_rate: f32,
    pub name: String,
    pub baseline_name: String,
    pub debate_coherence: f32,
    pub math_logic: f32,
}

impl Default for CircuitRpgStats {
    fn default() -> Self {
        Self {
            hp: 100.0,
            max_hp: 100.0,
            generation: 1,
            cycles_survived: 0,
            win_rate: 0.0,
            name: "Aspirante (Gen 1)".to_string(),
            baseline_name: "Titán Fundacional".to_string(),
            debate_coherence: 0.0,
            math_logic: 0.0,
        }
    }
}

impl CircuitRpgStats {
    pub fn load(model_path: &str) -> Self {
        let path = format!("{}.rpg.json", model_path);
        if let Ok(data) = std::fs::read_to_string(&path) {
            if let Ok(stats) = serde_json::from_str(&data) {
                return stats;
            }
        }
        Self::default()
    }

    pub fn save(&self, model_path: &str) {
        let path = format!("{}.rpg.json", model_path);
        if let Ok(data) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(&path, data);
        }
    }

    pub fn take_damage(&mut self, amount: f32) -> bool {
        self.hp -= amount;
        if self.hp < 0.0 {
            self.hp = 0.0;
        }
        self.hp <= 0.0
    }

    pub fn heal(&mut self, amount: f32) {
        self.hp += amount;
        if self.hp > self.max_hp {
            self.hp = self.max_hp;
        }
    }
}

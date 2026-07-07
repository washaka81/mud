use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Priority 10: Multi-Agent Delegation (Subagent Spawning)
/// Allows the main LDT model to spawn lighter, specialized subagents for isolated tasks
/// and parallelize workloads.
pub struct SubagentManager {
    agents: Arc<Mutex<HashMap<usize, Subagent>>>,
    next_id: usize,
}

pub struct Subagent {
    pub id: usize,
    pub role: String,
    pub status: AgentStatus,
    pub inbox: Vec<String>,
    pub outbox: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AgentStatus {
    Spawning,
    Running,
    Idle,
    Completed,
    Error(String),
}

impl Default for SubagentManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SubagentManager {
    pub fn new() -> Self {
        Self {
            agents: Arc::new(Mutex::new(HashMap::new())),
            next_id: 1,
        }
    }

    /// Spawns a new subagent with a specific role and prompt.
    pub fn spawn_agent(&mut self, role: &str, prompt: &str) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let agent = Subagent {
            id,
            role: role.to_string(),
            status: AgentStatus::Spawning,
            inbox: vec![prompt.to_string()],
            outbox: Vec::new(),
        };

        self.agents.lock().unwrap().insert(id, agent);

        // Simulate agent background loop (In a full implementation, this would spin up a new MudInference loop)
        let agents_clone = Arc::clone(&self.agents);
        let role_clone = role.to_string();
        let prompt_clone = prompt.to_string();
        
        thread::spawn(move || {
            // Change status to running
            {
                let mut guard = agents_clone.lock().unwrap();
                if let Some(agent) = guard.get_mut(&id) {
                    agent.status = AgentStatus::Running;
                    agent.outbox.push(format!("[Agent {} - {}] Acknowledged: {}", id, role_clone, prompt_clone));
                }
            }

            // Perform workspace operations
            let workspace_res = crate::mud::workspace_agent::AgentWorkspace::mount(".");
            let report = match workspace_res {
                Ok(ws) => {
                    let files = ws.scan_project_map();
                    format!("Successfully mapped workspace. Found {} files.", files.len())
                }
                Err(e) => {
                    format!("Failed to mount workspace: {}", e)
                }
            };
            
            thread::sleep(Duration::from_secs(2)); // Simulate inference compute time
            
            let mut guard = agents_clone.lock().unwrap();
            if let Some(agent) = guard.get_mut(&id) {
                agent.status = AgentStatus::Completed;
                agent.outbox.push(format!("[Agent {} - {}] Execution Report: {}", id, role_clone, report));
            }
        });

        id
    }

    /// Retrieves all messages from a subagent's outbox.
    pub fn poll_messages(&self, id: usize) -> Vec<String> {
        let mut guard = self.agents.lock().unwrap();
        if let Some(agent) = guard.get_mut(&id) {
            let messages = agent.outbox.clone();
            agent.outbox.clear();
            messages
        } else {
            Vec::new()
        }
    }

    /// Checks the status of a specific subagent.
    pub fn get_status(&self, id: usize) -> Option<AgentStatus> {
        let guard = self.agents.lock().unwrap();
        guard.get(&id).map(|a| a.status.clone())
    }
}

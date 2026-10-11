//! Infinite training circuit: rotates batteries of phases until `quit` / Ctrl-C.
//!
//! Each **seed** generates a distinct **battery** (an ordering of the
//! phase set) so training is never a fixed monotonic schedule — every
//! seed explores the phases in a different order. Phases (local, no-API, P-07):
//!   - align     — one corpus alignment epoch (STE QAT)
//!   - debate    — RLVR document debate (TextJudge)
//!   - games     — verifiable seed-survival games (Math / TicTacToe, VerifiableJudge)
//!   - professor — professor→student→grade loop (ProfessorJudge rubrik)
//!   - self-play — D1 synthetic chain generation from model's own forward pass
//!
//! When a battery is exhausted a fresh seed produces a new battery, so the
//! loop keeps varying. Each phase is time-boxed by `MUD_CIRCUIT_MAX_PER_MODE`
//! (default 120s) so the loop never freezes on a single slow phase. Phases
//! that persist write back to the `.mud` when `MUD_DEBATE_LEARN=1` (default
//! ON inside the circuit). On `quit` the last phase stops and the model is
//! left saved by the underlying session.
//!
//! The seed itself derives from a monotonic counter + wall-clock so each run
//! and each new battery differs.

use crate::mud::corpus_trainer::{
    integrity::circuit_eval_integrity, integrity::model_norms_alive, MudCorpusTrainer,
};

impl MudCorpusTrainer {
    /// Infinite training circuit: rotates batteries of phases until `quit` / Ctrl-C.
    pub fn run_training_circuit(
        &mut self,
        sender: Option<std::sync::mpsc::Sender<String>>,
        stop_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> anyhow::Result<()> {
        // Unified announce: prints live AND appends to logs/circuit.log (timestamped).
        let announce = |msg: &str| {
            let line = crate::mud::trainer_ui::circuit_event("circuit", msg);
            if let Some(tx) = &sender {
                let _ = tx.send(line);
            } else {
                println!("{}", line);
            }
        };
        // Honors-mode persistence: default ON inside the circuit.
        let learn = std::env::var("MUD_DEBATE_LEARN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
        std::env::set_var("MUD_DEBATE_LEARN", if learn { "1" } else { "0" });
        // Per-phase time box so the loop never freezes (default 120s).
        let max_per_mode = std::env::var("MUD_CIRCUIT_MAX_PER_MODE")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(120);
        std::env::set_var("MUD_DEBATE_MAX_TIME", max_per_mode.to_string());
        let epochs = std::env::var("MUD_CIRCUIT_EPOCHS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1);
        let batch = std::env::var("MUD_CIRCUIT_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(16);

        let mut cycle = 0usize;
        // SEC-H8 (2026-07-26): the circuit is infinite-by-design (you stop it
        // via `q`/Ctrl-C), but a forgotten-orphaned process used to be able to
        // run forever, grinding PR/wall-time and slowly filling the log volumes.
        // `MUD_CIRCUIT_MAX_CYCLES` puts a hard upper bound (default = a sane
        // 100k cycles ≫ any human-driven session). Set 0 for the historic
        // unbounded behavior.
        let max_cycles = std::env::var("MUD_CIRCUIT_MAX_CYCLES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(100_000);
        // Seed source: monotonic counter mixed with wall-clock nanos (no RNG crate).
        let mut seed: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E3779B97F4A7C15)
            ^ 0x00C0_FFEE_DEAD_BEEF_u64;

        // Usamos un orden fijo de currículum de aprendizaje en vez de barajar aleatoriamente.
        // Progresión lógica: align (base) -> professor (teoría) -> debate (razonamiento) -> games (evaluación final).
        // D1: self-play se añade al final como fase de auto-mejora.
        let build_battery = |_s: &mut u64| -> Vec<&'static str> {
            vec!["align", "professor", "debate", "games", "self-play"]
        };

        let mut battery = build_battery(&mut seed);
        let mut battery_seed = seed;
        let mut rpg = crate::mud::circuit_rpg::CircuitRpgStats::load(&self.model_path);

        announce(&format!(
            "⚙️  MUD Training Circuit iniciado (loop infinito por semilla). Modelo: {}. Fases base: align/debate/games/professor/self-play. Max/modo={}s, learn={}, epochs={}, batch={}. [q]/Ctrl-C para salir y guardar. Log: logs/circuit.log",
            self.model_path, max_per_mode, learn, epochs, batch
        ));
        announce(&format!(
            "🛡️  RPG Stats: HP {:.1}/{:.1} | Gen {} | WinRate {:.2} | Ciclos {} | A: {} | B: {}",
            rpg.hp, rpg.max_hp, rpg.generation, rpg.win_rate, rpg.cycles_survived, rpg.name, rpg.baseline_name
        ));
        announce(&format!(
            "🌱 Semilla {} · batería: {}",
            battery_seed,
            battery.join(" → ")
        ));

        // Fail-fast health check BEFORE any forward: a collapsed model (all-zero
        // RMSNorm) would otherwise panic with "Dead RMSNorm" inside a PCorePool
        // worker thread, which `catch_unwind` cannot capture (kills the process).
        // Report it clearly and stop instead.
        {
            let (alive, detail) = model_norms_alive(&self.model_path);
            if !alive {
                announce(&format!(
                    "❌ Modelo colapsado: {}. El circuito NO puede repararlo (normas congeladas). Usa un .mud sano (p.ej. ternary_bonsai_1.7b.mud).",
                    detail
                ));
                anyhow::bail!(
                    "model collapsed (Dead RMSNorm risk): {} — replace with a healthy .mud",
                    detail
                );
            } else {
                announce(&format!("✓ Health-check: {}", detail));
            }
        }

        let wall0 = std::time::Instant::now();
        // Honors-mode evaluation toggle (default ON): only keep a phase's writeback
        // if integrity is clean AND quality did not regress vs the baseline.
        let eval_on = std::env::var("MUD_CIRCUIT_EVAL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
        let backup_path = format!("{}.bak_circuit", self.model_path);
        // Baseline quality (verifiable-game win rate) taken once at circuit start.
        let base_rate = if eval_on {
            // catch_unwind: a collapsed model can panic (Dead-RMSNorm) during the
            // forward inside the benchmark — never let that kill the circuit.
            let bench_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.circuit_benchmark_games(stop_flag.clone())
            }));
            match bench_res {
                Ok(Ok((r, m))) => {
                    announce(&format!(
                        "📊 Baseline calidad (juegos verificables): win_rate={:.2} sobre {} matches",
                        r, m
                    ));
                    r
                }
                Ok(Err(e)) => {
                    announce(&format!(
                        "⚠️  benchmark baseline falló (continuando): {}",
                        e
                    ));
                    0.0
                }
                Err(_) => {
                    announce("⚠️  benchmark baseline panic (modelo colapsado?) — baseline=0");
                    0.0
                }
            }
        } else {
            0.0
        };
        const HONORS_TOL: f32 = 0.15; // allow up to 15% regression before rollback

        loop {
            if stop_flag.load(std::sync::atomic::Ordering::SeqCst) {
                announce("⏹  Circuito detenido por el usuario. El último estado ya fue guardado por la fase.");
                break;
            }
            // SEC-H8: hard cap on cycle count so a forgotten process can't run
            // forever. `max_cycles=0` restores the historic unbounded behavior.
            if max_cycles > 0 && cycle >= max_cycles {
                announce(&format!(
                    "⏹  SEC-H8 cycle cap reached ({}). Deteniendo circuito — set MUD_CIRCUIT_MAX_CYCLES=0 for unbounded.",
                    max_cycles
                ));
                break;
            }
            // Draw next phase from the current battery; mint a new seed+battery
            // once the battery is exhausted (keeps training non-monotonic).
            if battery.is_empty() {
                seed = seed.wrapping_mul(2).wrapping_add(1).wrapping_add(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(1),
                );
                battery = build_battery(&mut seed);
                battery_seed = seed;
                announce(&format!(
                    "🌱 Nueva semilla {} · batería: {}",
                    battery_seed,
                    battery.join(" → ")
                ));
            }
            let phase = battery.remove(0);
            cycle += 1;
            let remaining = battery.join(", ");
            let phase_start = std::time::Instant::now();
            announce(&format!(
                "▶ Ciclo #{} · Semilla {} · FASE={} · batería restante: [{}]",
                cycle, battery_seed, phase, remaining
            ));

            // Snapshot the current (pre-phase) model so we can roll back if the
            // phase fails the honors evaluation.
            if eval_on && std::path::Path::new(&self.model_path).exists() {
                let _ = std::fs::copy(&self.model_path, &backup_path);
            }

            let mut ok = true;
            match phase {
                "align" => {
                    // Alignment already persists to the .mud at the end of its run.
                    // catch_unwind: a panic (e.g. Dead-RMSNorm fail-fast on a
                    // collapsed model) must not kill the whole circuit process.
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        self.run_alignment_session(batch, epochs)
                    }));
                    match res {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => {
                            ok = false;
                            announce(&format!("⚠️  align error (continuando loop): {}", e));
                        }
                        Err(_) => {
                            ok = false;
                            announce(
                                "⚠️  align panic (modelo colapsado?) — rollback y continuo loop",
                            );
                        }
                    }
                }
                "debate" | "games" | "professor" => {
                    std::env::set_var("MUD_DEBATE_MODE", phase);
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        self.run_debate_session(sender.clone(), stop_flag.clone())
                    }));
                    match res {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => {
                            ok = false;
                            announce(&format!("⚠️  {} error (continuando loop): {}", phase, e));
                        }
                        Err(_) => {
                            ok = false;
                            let msg = format!(
                                "⚠️  {} panic (modelo colapsado?) — rollback y continuo loop",
                                phase
                            );
                            announce(&msg);
                        }
                    }
                }
                "self-play" => {
                    // D1 — Self-Play Loop: activate synthetic chain generation
                    // from the model's own forward pass, feeding confident
                    // predictions back through the training pipeline.
                    std::env::set_var("MUD_SELF_PLAY", "1");
                    announce("🎮 self-play phase: synthetic chain generation activated");
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        self.run_alignment_session(batch, epochs)
                    }));
                    // Restore env var after phase
                    std::env::remove_var("MUD_SELF_PLAY");
                    match res {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => {
                            ok = false;
                            announce(&format!("⚠️  self-play error (continuando loop): {}", e));
                        }
                        Err(_) => {
                            ok = false;
                            announce(
                                "⚠️  self-play panic (modelo colapsado?) — rollback y continuo loop",
                            );
                        }
                    }
                }
                _ => {}
            }
            let dur = phase_start.elapsed().as_secs_f32();
            let verdict = if ok { "OK" } else { "ERR" };
            announce(&format!(
                "✓ Ciclo #{} · {} completada [{}] en {:.1}s · total circuito: {:.1}s",
                cycle,
                phase,
                verdict,
                dur,
                wall0.elapsed().as_secs_f32()
            ));

            // A failed/panicked phase must never leave a corrupted/dead model on
            // disk: roll back to the pre-phase snapshot first.
            if !ok && eval_on && std::path::Path::new(&backup_path).exists() {
                let _ = std::fs::copy(&backup_path, &self.model_path);
                announce(&format!(
                    "↩️  rollback a snapshot previo (fase {} falló/panic)",
                    phase
                ));
            } else if eval_on && ok {
                // Honors evaluation: keep the phase only if integrity is clean and
                // the model did not regress vs baseline. Otherwise roll back.
                let (integrity, idetail) = circuit_eval_integrity(&self.model_path);
                let (rate, m) = self
                    .circuit_benchmark_games(stop_flag.clone())
                    .unwrap_or((0.0, 0usize));
                let honors = integrity && (rate >= base_rate - HONORS_TOL);
                if honors {
                    rpg.heal(10.0);
                    rpg.cycles_survived += 1;
                    rpg.win_rate = rate;
                    announce(&format!(
                        "🏅 HONORES ✓ integridad: {} | calidad win_rate={:.2} (baseline {:.2}, {} matches) → se guarda",
                        idetail, rate, base_rate, m
                    ));
                    announce(&format!("💖 +10 HP (Total: {:.1}/{:.1})", rpg.hp, rpg.max_hp));
                } else {
                    let dead = rpg.take_damage(25.0);
                    announce(&format!(
                        "❌ SIN HONORES ✗ integridad={} ({}); calidad win_rate={:.2} vs baseline {:.2} → rollback",
                        integrity, idetail, rate, base_rate
                    ));
                    announce(&format!("💔 -25 HP (Total: {:.1}/{:.1})", rpg.hp, rpg.max_hp));
                    let _ = std::fs::copy(&backup_path, &self.model_path);

                    if dead {
                        announce("💀 El modelo ha MUERTO (HP=0). Iniciando Evolución (Reset HP & +1 Gen).");
                        rpg.generation += 1;
                        rpg.hp = rpg.max_hp;
                        rpg.cycles_survived = 0;
                        let nombres = ["Aspirante", "Gladiador", "Guerrero", "Sombra", "Paladín", "Espectro", "Berserker", "Ronin"];
                        let idx = (rpg.generation as usize) % nombres.len();
                        rpg.name = format!("{} (Gen {})", nombres[idx], rpg.generation);
                        announce(&format!("🌟 Nueva Generación: {} iniciada como {}.", rpg.generation, rpg.name));
                    }
                }

                // Defensa del título: si superamos estrictamente al baseline, tomamos su lugar.
                if honors && rate > base_rate {
                    announce(&format!("👑 ¡{} ha vencido al Baseline con {:.2} > {:.2}! Reclamando el título...", rpg.name, rate, base_rate));
                    rpg.baseline_name = rpg.name.clone();
                }

                rpg.save(&self.model_path);
            }

            // Force a small yield so Ctrl-C / quit is observed between phases.
            if stop_flag.load(std::sync::atomic::Ordering::SeqCst) {
                announce("⏹  Circuito detenido por el usuario.");
                break;
            }
        }

        // Best-effort cleanup of the snapshot.
        let _ = std::fs::remove_file(&backup_path);
        announce(&format!(
            "✅ Circuito finalizado tras {} ciclos en {:.1}s. Modelo: {} | Log: logs/circuit.log",
            cycle,
            wall0.elapsed().as_secs_f32(),
            self.model_path
        ));
        Ok(())
    }
}

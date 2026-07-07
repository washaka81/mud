use forge_llm::mud::workspace::InferenceWorkspace;

fn main() -> anyhow::Result<()> {
    println!("============================================================");
    println!(" 🌌 MUD ENGINE | PHASE 14 LDT AUDIT (LATTICE-BASED DEDUCTION)");
    println!(" Testing: LDT-01 (Lattice Projections) & LDT-02 (Early Exits)");
    println!("============================================================\n");

    println!("🔍 Initializing Memory Workspace for 3072-dimensional Latent State...");
    let hidden = 3072;
    let ws = InferenceWorkspace::new(None, hidden, 4096, 32, 32, 8, 200000, 16, 4, 4096);

    println!("\n🛠️  TEST 1: LDT-01 - Lattice Constraint Projections");
    // Populate x_moe_norm with continuous (thermodynamic) values
    {
        let mut x_guard = ws.x_moe_norm.write();
        x_guard[0] = 0.5432;
        x_guard[1] = -0.9876;
        x_guard[2] = 0.0001;
        x_guard[3] = 1.4567;
    }

    // Apply 4-level lattice
    println!("   [PRE-PROJECTION] Latent state sampled at [0, 1, 2, 3]:");
    {
        let x_guard = ws.x_moe_norm.read();
        println!(
            "     - H0: {:.4}, H1: {:.4}, H2: {:.4}, H3: {:.4}",
            x_guard[0], x_guard[1], x_guard[2], x_guard[3]
        );
    }

    println!("   [ACTION] Applying LDT-01 Lattice Projection (levels=4.0)...");
    ws.apply_lattice_projection(hidden, 4.0);

    {
        let x_guard = ws.x_moe_norm.read();
        println!("   [POST-PROJECTION] Latent state is now discretized:");
        println!(
            "     - H0: {:.4}, H1: {:.4}, H2: {:.4}, H3: {:.4}",
            x_guard[0], x_guard[1], x_guard[2], x_guard[3]
        );

        // Assert they are multiples of 0.25 (1.0 / 4.0)
        assert_eq!(x_guard[0], 0.5000);
        assert_eq!(x_guard[1], -1.0000);
        assert_eq!(x_guard[2], 0.0000);
        assert_eq!(x_guard[3], 1.5000);
    }
    println!("✅ LDT-01 Validation Passed: Thermodynamic drift neutralized.\n");

    println!("🛠️  TEST 2: LDT-02 - Deterministic Early Exit (Convergence)");

    // Test exact match (convergence)
    {
        let mut base = ws.ldt_base_state.write();
        let current = ws.x_moe_norm.read();
        base.copy_from_slice(&current); // Sync states perfectly
    }

    println!("   [ACTION] Evaluating LDT Convergence on perfectly aligned states...");
    let converged = ws.evaluate_ldt_convergence(hidden, 0.01);
    println!("   [RESULT] LDT Early Exit Triggered: {}", converged);
    assert!(converged, "Should trigger early exit on exact match");

    // Test deviation (drift)
    {
        let mut current = ws.x_moe_norm.write();
        current[0] += 0.5; // Artificial entropy drift
    }

    println!("   [ACTION] Evaluating LDT Convergence on drifted states (Shift=0.5)...");
    let converged_drift = ws.evaluate_ldt_convergence(hidden, 0.01);
    println!("   [RESULT] LDT Early Exit Triggered: {}", converged_drift);
    assert!(
        !converged_drift,
        "Should NOT trigger early exit on shifted states"
    );

    println!(
        "✅ LDT-02 Validation Passed: L2 Euclidean Shift correctly terminates reasoning loops.\n"
    );

    println!("🚀 LDT PHASE 14 COMPONENT AUDIT COMPLETED SUCCESSFULLY.");
    Ok(())
}

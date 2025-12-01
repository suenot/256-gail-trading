//! GAIL Trading Example
//!
//! Demonstrates:
//! 1. Fetching BTCUSDT data from Bybit
//! 2. Building expert dataset from high-return periods
//! 3. Training GAIL agent to imitate expert
//! 4. Comparing GAIL vs behavior cloning performance

use gail_trading::*;

fn main() {
    println!("=== GAIL Trading Example ===\n");

    // ── Step 1: Get market data ──────────────────────────────────────────
    println!("[1] Fetching market data...");

    let (prices, volumes) = match fetch_bybit_data() {
        Some(data) => {
            println!("    Fetched {} candles from Bybit", data.0.len());
            data
        }
        None => {
            println!("    Bybit API unavailable, using synthetic data");
            generate_synthetic_prices(1000, 50000.0, 0.015)
        }
    };

    println!(
        "    Price range: {:.2} - {:.2}",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    println!("    Total candles: {}", prices.len());

    // ── Step 2: Build expert dataset ─────────────────────────────────────
    println!("\n[2] Building expert dataset from high-return periods...");

    let expert_dataset = ExpertDataset::from_price_data(&prices, &volumes, 50, 0.25);
    println!(
        "    Expert trajectories: {}",
        expert_dataset.trajectories.len()
    );
    println!("    Total expert pairs: {}", expert_dataset.len());

    if expert_dataset.is_empty() {
        println!("    ERROR: No expert data found. Need more price data.");
        return;
    }

    // Print expert trajectory stats
    for (i, traj) in expert_dataset.trajectories.iter().enumerate().take(5) {
        println!(
            "    Trajectory {}: {} steps, return={:.4}",
            i,
            traj.pairs.len(),
            traj.total_reward
        );
    }

    // ── Step 3: Train behavior cloning baseline ──────────────────────────
    println!("\n[3] Training behavior cloning baseline...");

    let mut bc_policy = PolicyNetwork::new(32, 0.005, 0.01);
    let bc_epochs = 50;

    for epoch in 0..bc_epochs {
        let batch = expert_dataset.sample_batch(64);
        let loss = bc_policy.behavior_cloning_step(&batch);
        if epoch % 10 == 0 {
            println!("    BC Epoch {}: loss={:.4}", epoch, loss);
        }
    }

    // Evaluate BC policy
    let mut bc_returns = Vec::new();
    for _ in 0..10 {
        let mut env = TradingEnvironment::new(prices.clone(), volumes.clone(), 100);
        let mut state = env.reset();
        loop {
            let action = bc_policy.sample_action(&state);
            let (next_state, _, done) = env.step(action);
            state = next_state;
            if done {
                break;
            }
        }
        bc_returns.push(env.get_return());
    }
    let bc_mean_return: f64 = bc_returns.iter().sum::<f64>() / bc_returns.len() as f64;
    println!("    BC mean return: {:.4}", bc_mean_return);

    // ── Step 4: Train GAIL agent ─────────────────────────────────────────
    println!("\n[4] Training GAIL agent...");

    let config = GAILConfig {
        num_iterations: 30,
        trajectories_per_iter: 3,
        disc_updates_per_iter: 2,
        batch_size: 32,
        gamma: 0.99,
        disc_hidden_size: 32,
        policy_hidden_size: 32,
        disc_lr: 0.001,
        policy_lr: 0.0005,
        entropy_coeff: 0.01,
    };

    let mut trainer = GAILTrainer::new(expert_dataset, config);
    let mut env = TradingEnvironment::new(prices.clone(), volumes.clone(), 100);

    for i in 0..30 {
        let stats = trainer.train_iteration(&mut env);
        if i % 5 == 0 {
            println!(
                "    GAIL Iter {}: disc_loss={:.4}, mean_reward={:.4}, mean_return={:.4}",
                stats.iteration, stats.disc_loss, stats.mean_reward, stats.mean_return
            );
        }
    }

    // Evaluate GAIL policy
    let mut gail_returns = Vec::new();
    for _ in 0..10 {
        let ret = trainer.evaluate(&mut env);
        gail_returns.push(ret);
    }
    let gail_mean_return: f64 = gail_returns.iter().sum::<f64>() / gail_returns.len() as f64;
    println!("    GAIL mean return: {:.4}", gail_mean_return);

    // ── Step 5: Compare results ──────────────────────────────────────────
    println!("\n[5] Comparison: GAIL vs Behavior Cloning");
    println!("    ┌────────────────────┬──────────────┐");
    println!("    │ Method             │ Mean Return  │");
    println!("    ├────────────────────┼──────────────┤");
    println!("    │ Behavior Cloning   │ {:>11.4}% │", bc_mean_return * 100.0);
    println!("    │ GAIL               │ {:>11.4}% │", gail_mean_return * 100.0);
    println!("    └────────────────────┴──────────────┘");

    // Evaluate discriminator's ability to distinguish
    println!("\n[6] Discriminator analysis...");
    let expert_batch = trainer.expert_dataset.sample_batch(20);
    let mut expert_scores = Vec::new();
    for pair in &expert_batch {
        expert_scores.push(trainer.discriminator.forward(&pair.state, pair.action));
    }
    let mean_expert_score: f64 = expert_scores.iter().sum::<f64>() / expert_scores.len() as f64;

    // Generate some policy samples
    let mut policy_scores = Vec::new();
    let mut state = env.reset();
    for _ in 0..20 {
        let action = trainer.policy.sample_action(&state);
        policy_scores.push(trainer.discriminator.forward(&state, action));
        let (next_state, _, done) = env.step(action);
        state = next_state;
        if done {
            state = env.reset();
        }
    }
    let mean_policy_score: f64 = policy_scores.iter().sum::<f64>() / policy_scores.len() as f64;

    println!(
        "    Mean D(expert): {:.4} (closer to 1.0 = correctly identified as expert)",
        mean_expert_score
    );
    println!(
        "    Mean D(policy): {:.4} (closer to 0.5 = policy fools discriminator)",
        mean_policy_score
    );

    println!("\n=== GAIL Trading Example Complete ===");
}

/// Attempt to fetch data from Bybit, return None on failure
fn fetch_bybit_data() -> Option<(Vec<f64>, Vec<f64>)> {
    let client = BybitClient::new();
    match client.fetch_klines("BTCUSDT", "15", 200) {
        Ok(klines) if !klines.is_empty() => {
            let (prices, volumes) = BybitClient::klines_to_arrays(&klines);
            Some((prices, volumes))
        }
        _ => None,
    }
}

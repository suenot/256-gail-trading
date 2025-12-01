//! # GAIL Trading
//!
//! Generative Adversarial Imitation Learning (GAIL) for trading.
//! Implements expert trajectory datasets, discriminator and policy networks,
//! and the full GAIL training loop with Bybit data integration.

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ─── Constants ───────────────────────────────────────────────────────────────

/// Number of features in the state representation
pub const STATE_DIM: usize = 8;
/// Number of discrete actions: StrongSell(-2), Sell(-1), Hold(0), Buy(+1), StrongBuy(+2)
pub const NUM_ACTIONS: usize = 5;
/// Transaction cost as fraction of trade value
pub const TRANSACTION_COST: f64 = 0.001;

// ─── Data Structures ─────────────────────────────────────────────────────────

/// A single state-action pair from a trajectory
#[derive(Debug, Clone)]
pub struct StateActionPair {
    pub state: Array1<f64>,
    pub action: usize,
}

/// A full trajectory (sequence of state-action pairs with cumulative reward)
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub pairs: Vec<StateActionPair>,
    pub total_reward: f64,
}

/// Expert dataset containing trajectories from profitable trading periods
#[derive(Debug, Clone)]
pub struct ExpertDataset {
    pub trajectories: Vec<Trajectory>,
    pub all_pairs: Vec<StateActionPair>,
}

impl ExpertDataset {
    /// Create a new empty expert dataset
    pub fn new() -> Self {
        Self {
            trajectories: Vec::new(),
            all_pairs: Vec::new(),
        }
    }

    /// Add a trajectory to the dataset
    pub fn add_trajectory(&mut self, trajectory: Trajectory) {
        for pair in &trajectory.pairs {
            self.all_pairs.push(pair.clone());
        }
        self.trajectories.push(trajectory);
    }

    /// Build expert dataset from historical price data by identifying high-return windows
    pub fn from_price_data(prices: &[f64], volumes: &[f64], window_size: usize, top_fraction: f64) -> Self {
        let mut dataset = Self::new();
        if prices.len() < window_size + STATE_DIM + 1 || volumes.is_empty() {
            return dataset;
        }

        // Compute returns for each window
        let mut window_returns: Vec<(usize, f64)> = Vec::new();
        let num_windows = (prices.len() - STATE_DIM) / window_size;

        for w in 0..num_windows {
            let start = w * window_size + STATE_DIM;
            let end = (start + window_size).min(prices.len() - 1);
            if end <= start {
                continue;
            }
            let ret = (prices[end] - prices[start]) / prices[start];
            window_returns.push((start, ret));
        }

        // Sort by return descending and take top fraction
        window_returns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let num_expert = ((window_returns.len() as f64 * top_fraction).ceil() as usize).max(1);

        for &(start, total_ret) in window_returns.iter().take(num_expert) {
            let end = (start + window_size).min(prices.len() - 1);
            let mut pairs = Vec::new();

            for t in start..end {
                let state = compute_state_features(prices, volumes, t);
                // Expert action: oracle based on next-step return
                let next_ret = (prices[t + 1] - prices[t]) / prices[t];
                let action = if next_ret > 0.002 {
                    4 // StrongBuy
                } else if next_ret > 0.0005 {
                    3 // Buy
                } else if next_ret < -0.002 {
                    0 // StrongSell
                } else if next_ret < -0.0005 {
                    1 // Sell
                } else {
                    2 // Hold
                };
                pairs.push(StateActionPair { state, action });
            }

            dataset.add_trajectory(Trajectory {
                pairs,
                total_reward: total_ret,
            });
        }

        dataset
    }

    /// Sample a random batch of state-action pairs
    pub fn sample_batch(&self, batch_size: usize) -> Vec<StateActionPair> {
        let mut rng = rand::thread_rng();
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            if !self.all_pairs.is_empty() {
                let idx = rng.gen_range(0..self.all_pairs.len());
                batch.push(self.all_pairs[idx].clone());
            }
        }
        batch
    }

    /// Number of total state-action pairs
    pub fn len(&self) -> usize {
        self.all_pairs.len()
    }

    /// Check if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.all_pairs.is_empty()
    }
}

impl Default for ExpertDataset {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Feature Computation ─────────────────────────────────────────────────────

/// Compute state features from price and volume data at time t
pub fn compute_state_features(prices: &[f64], volumes: &[f64], t: usize) -> Array1<f64> {
    let mut features = Array1::zeros(STATE_DIM);

    if t < 2 || t >= prices.len() {
        return features;
    }

    // Feature 0: log return
    features[0] = (prices[t] / prices[t - 1]).ln();

    // Feature 1: 5-period momentum
    if t >= 5 {
        features[1] = (prices[t] / prices[t - 5]).ln();
    }

    // Feature 2: realized volatility (5-period)
    if t >= 5 {
        let mut sum_sq = 0.0;
        for i in (t - 4)..=t {
            let r = (prices[i] / prices[i - 1]).ln();
            sum_sq += r * r;
        }
        features[2] = (sum_sq / 5.0).sqrt();
    }

    // Feature 3: RSI approximation (7-period)
    if t >= 7 {
        let mut gains = 0.0;
        let mut losses = 0.0;
        for i in (t - 6)..=t {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }
        let rs = if losses > 1e-12 { gains / losses } else { 100.0 };
        features[3] = (100.0 - 100.0 / (1.0 + rs)) / 100.0 - 0.5; // Normalize to [-0.5, 0.5]
    }

    // Feature 4: volume ratio
    if t >= 5 && !volumes.is_empty() {
        let vol_idx = t.min(volumes.len() - 1);
        let mut avg_vol = 0.0;
        let start = if vol_idx >= 5 { vol_idx - 4 } else { 0 };
        let count = vol_idx - start + 1;
        for i in start..=vol_idx {
            avg_vol += volumes[i];
        }
        avg_vol /= count as f64;
        if avg_vol > 1e-12 {
            features[4] = (volumes[vol_idx] / avg_vol) - 1.0;
        }
    }

    // Feature 5: price deviation from 20-period SMA
    if t >= 20 {
        let sma: f64 = prices[(t - 19)..=t].iter().sum::<f64>() / 20.0;
        features[5] = (prices[t] - sma) / sma;
    }

    // Feature 6: high-low range proxy (using 5-period max-min)
    if t >= 5 {
        let window = &prices[(t - 4)..=t];
        let max_p = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_p = window.iter().cloned().fold(f64::INFINITY, f64::min);
        if min_p > 1e-12 {
            features[6] = (max_p - min_p) / min_p;
        }
    }

    // Feature 7: 2-period return acceleration
    if t >= 3 {
        let r1 = (prices[t] / prices[t - 1]).ln();
        let r2 = (prices[t - 1] / prices[t - 2]).ln();
        features[7] = r1 - r2;
    }

    features
}

// ─── Neural Network Layers ───────────────────────────────────────────────────

/// Simple dense layer: y = activation(x * W + b)
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl DenseLayer {
    /// Create a new dense layer with Xavier initialization
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let biases = Array1::zeros(output_dim);
        Self {
            weights,
            biases,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass: compute output = x * W + b
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = self.biases.clone();
        for j in 0..self.output_dim {
            let mut sum = self.biases[j];
            for i in 0..self.input_dim {
                sum += input[i] * self.weights[[i, j]];
            }
            output[j] = sum;
        }
        output
    }

    /// Update weights using gradient
    pub fn update(&mut self, weight_grads: &Array2<f64>, bias_grads: &Array1<f64>, lr: f64) {
        self.weights = &self.weights - &(weight_grads * lr);
        self.biases = &self.biases - &(bias_grads * lr);
    }
}

// ─── Activation Functions ────────────────────────────────────────────────────

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Array1<f64> = logits.mapv(|x| (x - max_logit).exp());
    let sum: f64 = exp_vals.sum();
    if sum > 1e-12 {
        exp_vals / sum
    } else {
        Array1::from_vec(vec![1.0 / logits.len() as f64; logits.len()])
    }
}

// ─── Discriminator ───────────────────────────────────────────────────────────

/// Discriminator network: classifies state-action pairs as expert (1) or policy (0)
///
/// Architecture: (state_dim + num_actions) -> hidden -> hidden -> 1 (sigmoid)
#[derive(Debug, Clone)]
pub struct Discriminator {
    pub layer1: DenseLayer,
    pub layer2: DenseLayer,
    pub output_layer: DenseLayer,
    pub learning_rate: f64,
}

impl Discriminator {
    /// Create a new discriminator with given hidden size
    pub fn new(hidden_size: usize, learning_rate: f64) -> Self {
        let input_dim = STATE_DIM + NUM_ACTIONS; // state + one-hot action
        Self {
            layer1: DenseLayer::new(input_dim, hidden_size),
            layer2: DenseLayer::new(hidden_size, hidden_size),
            output_layer: DenseLayer::new(hidden_size, 1),
            learning_rate,
        }
    }

    /// Encode a state-action pair as input vector
    fn encode_input(&self, state: &Array1<f64>, action: usize) -> Array1<f64> {
        let mut input = Array1::zeros(STATE_DIM + NUM_ACTIONS);
        for i in 0..STATE_DIM {
            input[i] = state[i];
        }
        if action < NUM_ACTIONS {
            input[STATE_DIM + action] = 1.0;
        }
        input
    }

    /// Forward pass: returns P(expert | s, a)
    pub fn forward(&self, state: &Array1<f64>, action: usize) -> f64 {
        let input = self.encode_input(state, action);
        let h1_pre = self.layer1.forward(&input);
        let h1 = h1_pre.mapv(relu);
        let h2_pre = self.layer2.forward(&h1);
        let h2 = h2_pre.mapv(relu);
        let logit = self.output_layer.forward(&h2);
        sigmoid(logit[0])
    }

    /// Compute GAIL reward: -log(1 - D(s, a))
    pub fn compute_reward(&self, state: &Array1<f64>, action: usize) -> f64 {
        let d = self.forward(state, action);
        let d_clamped = d.clamp(0.01, 0.99);
        -(1.0 - d_clamped).ln()
    }

    /// Train discriminator on a batch of expert and policy samples
    pub fn train_step(
        &mut self,
        expert_batch: &[StateActionPair],
        policy_batch: &[StateActionPair],
    ) -> f64 {
        let mut total_loss = 0.0;
        let batch_size = expert_batch.len().min(policy_batch.len());
        if batch_size == 0 {
            return 0.0;
        }

        // Accumulate gradients
        let mut l1_wg = Array2::zeros(self.layer1.weights.raw_dim());
        let mut l1_bg = Array1::zeros(self.layer1.biases.raw_dim());
        let mut l2_wg = Array2::zeros(self.layer2.weights.raw_dim());
        let mut l2_bg = Array1::zeros(self.layer2.biases.raw_dim());
        let mut lo_wg = Array2::zeros(self.output_layer.weights.raw_dim());
        let mut lo_bg = Array1::zeros(self.output_layer.biases.raw_dim());

        for i in 0..batch_size {
            // Expert sample: maximize log D(s, a)
            {
                let (wg1, bg1, wg2, bg2, wgo, bgo, loss) =
                    self.backward(&expert_batch[i].state, expert_batch[i].action, true);
                l1_wg = &l1_wg + &wg1;
                l1_bg = &l1_bg + &bg1;
                l2_wg = &l2_wg + &wg2;
                l2_bg = &l2_bg + &bg2;
                lo_wg = &lo_wg + &wgo;
                lo_bg = &lo_bg + &bgo;
                total_loss += loss;
            }

            // Policy sample: maximize log(1 - D(s, a))
            {
                let (wg1, bg1, wg2, bg2, wgo, bgo, loss) =
                    self.backward(&policy_batch[i].state, policy_batch[i].action, false);
                l1_wg = &l1_wg + &wg1;
                l1_bg = &l1_bg + &bg1;
                l2_wg = &l2_wg + &wg2;
                l2_bg = &l2_bg + &bg2;
                lo_wg = &lo_wg + &wgo;
                lo_bg = &lo_bg + &bgo;
                total_loss += loss;
            }
        }

        let scale = 1.0 / (2.0 * batch_size as f64);
        // Gradient ascent (maximize), so we negate
        self.layer1.update(&(&l1_wg * scale), &(&l1_bg * scale), -self.learning_rate);
        self.layer2.update(&(&l2_wg * scale), &(&l2_bg * scale), -self.learning_rate);
        self.output_layer.update(&(&lo_wg * scale), &(&lo_bg * scale), -self.learning_rate);

        total_loss / (2.0 * batch_size as f64)
    }

    /// Backward pass for a single sample
    fn backward(
        &self,
        state: &Array1<f64>,
        action: usize,
        is_expert: bool,
    ) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>, f64) {
        // Forward pass with cached activations
        let input = self.encode_input(state, action);
        let h1_pre = self.layer1.forward(&input);
        let h1 = h1_pre.mapv(relu);
        let h2_pre = self.layer2.forward(&h1);
        let h2 = h2_pre.mapv(relu);
        let logit = self.output_layer.forward(&h2);
        let d = sigmoid(logit[0]);
        let d_clamped = d.clamp(1e-7, 1.0 - 1e-7);

        // Loss and output gradient
        let (loss, d_logit) = if is_expert {
            // maximize log D => gradient of log(D) w.r.t. logit = 1 - D
            (-d_clamped.ln(), 1.0 - d)
        } else {
            // maximize log(1-D) => gradient of log(1-D) w.r.t. logit = -D
            (-(1.0 - d_clamped).ln(), -d)
        };

        // Output layer gradients
        let mut lo_wg = Array2::zeros(self.output_layer.weights.raw_dim());
        let mut lo_bg = Array1::zeros(1);
        lo_bg[0] = d_logit;
        for j in 0..self.output_layer.input_dim {
            lo_wg[[j, 0]] = d_logit * h2[j];
        }

        // Backprop through layer2
        let mut d_h2 = Array1::zeros(self.layer2.output_dim);
        for j in 0..self.layer2.output_dim {
            d_h2[j] = d_logit * self.output_layer.weights[[j, 0]] * relu_derivative(h2_pre[j]);
        }
        let mut l2_wg = Array2::zeros(self.layer2.weights.raw_dim());
        let l2_bg = d_h2.clone();
        for j in 0..self.layer2.output_dim {
            for i in 0..self.layer2.input_dim {
                l2_wg[[i, j]] = d_h2[j] * h1[i];
            }
        }

        // Backprop through layer1
        let mut d_h1 = Array1::zeros(self.layer1.output_dim);
        for j in 0..self.layer1.output_dim {
            let mut sum = 0.0;
            for k in 0..self.layer2.output_dim {
                sum += d_h2[k] * self.layer2.weights[[j, k]];
            }
            d_h1[j] = sum * relu_derivative(h1_pre[j]);
        }
        let mut l1_wg = Array2::zeros(self.layer1.weights.raw_dim());
        let l1_bg = d_h1.clone();
        for j in 0..self.layer1.output_dim {
            for i in 0..self.layer1.input_dim {
                l1_wg[[i, j]] = d_h1[j] * input[i];
            }
        }

        (l1_wg, l1_bg, l2_wg, l2_bg, lo_wg, lo_bg, loss)
    }
}

// ─── Policy Network ──────────────────────────────────────────────────────────

/// Policy network: outputs action probabilities given a state
///
/// Architecture: state_dim -> hidden -> hidden -> num_actions (softmax)
#[derive(Debug, Clone)]
pub struct PolicyNetwork {
    pub layer1: DenseLayer,
    pub layer2: DenseLayer,
    pub output_layer: DenseLayer,
    pub learning_rate: f64,
    pub entropy_coeff: f64,
}

impl PolicyNetwork {
    /// Create a new policy network
    pub fn new(hidden_size: usize, learning_rate: f64, entropy_coeff: f64) -> Self {
        Self {
            layer1: DenseLayer::new(STATE_DIM, hidden_size),
            layer2: DenseLayer::new(hidden_size, hidden_size),
            output_layer: DenseLayer::new(hidden_size, NUM_ACTIONS),
            learning_rate,
            entropy_coeff,
        }
    }

    /// Forward pass: returns action probabilities
    pub fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        let h1 = self.layer1.forward(state).mapv(relu);
        let h2 = self.layer2.forward(&h1).mapv(relu);
        let logits = self.output_layer.forward(&h2);
        softmax(&logits)
    }

    /// Sample an action from the policy
    pub fn sample_action(&self, state: &Array1<f64>) -> usize {
        let probs = self.forward(state);
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }
        NUM_ACTIONS - 1
    }

    /// REINFORCE update with GAIL rewards
    pub fn update_reinforce(
        &mut self,
        trajectories: &[(Vec<Array1<f64>>, Vec<usize>, Vec<f64>)],
        gamma: f64,
    ) {
        if trajectories.is_empty() {
            return;
        }

        let mut l1_wg = Array2::zeros(self.layer1.weights.raw_dim());
        let mut l1_bg = Array1::zeros(self.layer1.biases.raw_dim());
        let mut l2_wg = Array2::zeros(self.layer2.weights.raw_dim());
        let mut l2_bg = Array1::zeros(self.layer2.biases.raw_dim());
        let mut lo_wg = Array2::zeros(self.output_layer.weights.raw_dim());
        let mut lo_bg = Array1::zeros(self.output_layer.biases.raw_dim());

        let mut total_steps = 0;

        for (states, actions, rewards) in trajectories {
            let t_len = states.len();
            if t_len == 0 {
                continue;
            }

            // Compute returns-to-go
            let mut returns = vec![0.0; t_len];
            returns[t_len - 1] = rewards[t_len - 1];
            for t in (0..t_len - 1).rev() {
                returns[t] = rewards[t] + gamma * returns[t + 1];
            }

            // Compute baseline (mean return)
            let baseline: f64 = returns.iter().sum::<f64>() / t_len as f64;

            // Compute policy gradients
            for t in 0..t_len {
                let advantage = returns[t] - baseline;

                // Forward with cached activations
                let h1_pre = self.layer1.forward(&states[t]);
                let h1 = h1_pre.mapv(relu);
                let h2_pre = self.layer2.forward(&h1);
                let h2 = h2_pre.mapv(relu);
                let logits = self.output_layer.forward(&h2);
                let probs = softmax(&logits);

                // Gradient of log pi(a|s) w.r.t. logits = (one_hot(a) - probs)
                let mut d_logits = Array1::zeros(NUM_ACTIONS);
                for a in 0..NUM_ACTIONS {
                    d_logits[a] = if a == actions[t] { 1.0 } else { 0.0 } - probs[a];
                }
                // Scale by advantage
                let d_logits_scaled = &d_logits * advantage;

                // Add entropy bonus gradient: d/d_logits H(pi) = -(log(pi) + 1) * d_softmax/d_logits
                // Simplified: entropy gradient w.r.t. logits ≈ -probs * (log(probs) + 1) subtracted
                // We approximate by adding entropy_coeff * (uniform - probs) to encourage exploration
                let entropy_grad = d_logits.mapv(|_| 1.0 / NUM_ACTIONS as f64) - &probs;
                let d_logits_final = &d_logits_scaled + &(&entropy_grad * self.entropy_coeff);

                // Output layer
                for j in 0..NUM_ACTIONS {
                    lo_bg[j] += d_logits_final[j];
                    for i in 0..self.layer2.output_dim {
                        lo_wg[[i, j]] += d_logits_final[j] * h2[i];
                    }
                }

                // Backprop through layer2
                let mut d_h2 = Array1::zeros(self.layer2.output_dim);
                for j in 0..self.layer2.output_dim {
                    let mut sum = 0.0;
                    for k in 0..NUM_ACTIONS {
                        sum += d_logits_final[k] * self.output_layer.weights[[j, k]];
                    }
                    d_h2[j] = sum * relu_derivative(h2_pre[j]);
                }
                for j in 0..self.layer2.output_dim {
                    l2_bg[j] += d_h2[j];
                    for i in 0..self.layer2.input_dim {
                        l2_wg[[i, j]] += d_h2[j] * h1[i];
                    }
                }

                // Backprop through layer1
                let mut d_h1 = Array1::zeros(self.layer1.output_dim);
                for j in 0..self.layer1.output_dim {
                    let mut sum = 0.0;
                    for k in 0..self.layer2.output_dim {
                        sum += d_h2[k] * self.layer2.weights[[j, k]];
                    }
                    d_h1[j] = sum * relu_derivative(h1_pre[j]);
                }
                for j in 0..self.layer1.output_dim {
                    l1_bg[j] += d_h1[j];
                    for i in 0..self.layer1.input_dim {
                        l1_wg[[i, j]] += d_h1[j] * states[t][i];
                    }
                }

                total_steps += 1;
            }
        }

        if total_steps > 0 {
            let scale = 1.0 / total_steps as f64;
            // Gradient ascent for policy (maximize expected reward)
            self.layer1.update(&(&l1_wg * scale), &(&l1_bg * scale), -self.learning_rate);
            self.layer2.update(&(&l2_wg * scale), &(&l2_bg * scale), -self.learning_rate);
            self.output_layer.update(&(&lo_wg * scale), &(&lo_bg * scale), -self.learning_rate);
        }
    }

    /// Behavior cloning update (supervised learning from expert data)
    pub fn behavior_cloning_step(&mut self, expert_batch: &[StateActionPair]) -> f64 {
        if expert_batch.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut l1_wg = Array2::zeros(self.layer1.weights.raw_dim());
        let mut l1_bg = Array1::zeros(self.layer1.biases.raw_dim());
        let mut l2_wg = Array2::zeros(self.layer2.weights.raw_dim());
        let mut l2_bg = Array1::zeros(self.layer2.biases.raw_dim());
        let mut lo_wg = Array2::zeros(self.output_layer.weights.raw_dim());
        let mut lo_bg = Array1::zeros(self.output_layer.biases.raw_dim());

        for pair in expert_batch {
            let h1_pre = self.layer1.forward(&pair.state);
            let h1 = h1_pre.mapv(relu);
            let h2_pre = self.layer2.forward(&h1);
            let h2 = h2_pre.mapv(relu);
            let logits = self.output_layer.forward(&h2);
            let probs = softmax(&logits);

            let prob_action = probs[pair.action].max(1e-12);
            total_loss += -prob_action.ln();

            // Gradient: (one_hot - probs)
            let mut d_logits = Array1::zeros(NUM_ACTIONS);
            for a in 0..NUM_ACTIONS {
                d_logits[a] = if a == pair.action { 1.0 } else { 0.0 } - probs[a];
            }

            // Output layer
            for j in 0..NUM_ACTIONS {
                lo_bg[j] += d_logits[j];
                for i in 0..self.layer2.output_dim {
                    lo_wg[[i, j]] += d_logits[j] * h2[i];
                }
            }

            // Backprop layer2
            let mut d_h2 = Array1::zeros(self.layer2.output_dim);
            for j in 0..self.layer2.output_dim {
                let mut sum = 0.0;
                for k in 0..NUM_ACTIONS {
                    sum += d_logits[k] * self.output_layer.weights[[j, k]];
                }
                d_h2[j] = sum * relu_derivative(h2_pre[j]);
            }
            for j in 0..self.layer2.output_dim {
                l2_bg[j] += d_h2[j];
                for i in 0..self.layer2.input_dim {
                    l2_wg[[i, j]] += d_h2[j] * h1[i];
                }
            }

            // Backprop layer1
            let mut d_h1 = Array1::zeros(self.layer1.output_dim);
            for j in 0..self.layer1.output_dim {
                let mut sum = 0.0;
                for k in 0..self.layer2.output_dim {
                    sum += d_h2[k] * self.layer2.weights[[j, k]];
                }
                d_h1[j] = sum * relu_derivative(h1_pre[j]);
            }
            for j in 0..self.layer1.output_dim {
                l1_bg[j] += d_h1[j];
                for i in 0..self.layer1.input_dim {
                    l1_wg[[i, j]] += d_h1[j] * pair.state[i];
                }
            }
        }

        let scale = 1.0 / expert_batch.len() as f64;
        self.layer1.update(&(&l1_wg * scale), &(&l1_bg * scale), -self.learning_rate);
        self.layer2.update(&(&l2_wg * scale), &(&l2_bg * scale), -self.learning_rate);
        self.output_layer.update(&(&lo_wg * scale), &(&lo_bg * scale), -self.learning_rate);

        total_loss / expert_batch.len() as f64
    }
}

// ─── Trading Environment ─────────────────────────────────────────────────────

/// Map action index to position change
pub fn action_to_position_change(action: usize) -> f64 {
    match action {
        0 => -1.0,  // StrongSell
        1 => -0.5,  // Sell
        2 => 0.0,   // Hold
        3 => 0.5,   // Buy
        4 => 1.0,   // StrongBuy
        _ => 0.0,
    }
}

/// Trading environment for market replay
#[derive(Debug, Clone)]
pub struct TradingEnvironment {
    pub prices: Vec<f64>,
    pub volumes: Vec<f64>,
    pub current_step: usize,
    pub position: f64,
    pub portfolio_value: f64,
    pub initial_value: f64,
    pub episode_length: usize,
    pub start_offset: usize,
}

impl TradingEnvironment {
    /// Create a new trading environment
    pub fn new(prices: Vec<f64>, volumes: Vec<f64>, episode_length: usize) -> Self {
        let start_offset = STATE_DIM + 1;
        Self {
            prices,
            volumes,
            current_step: start_offset,
            position: 0.0,
            portfolio_value: 10000.0,
            initial_value: 10000.0,
            episode_length,
            start_offset,
        }
    }

    /// Reset the environment to a random starting point
    pub fn reset(&mut self) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let max_start = if self.prices.len() > self.episode_length + self.start_offset {
            self.prices.len() - self.episode_length - self.start_offset
        } else {
            0
        };
        self.current_step = if max_start > 0 {
            self.start_offset + rng.gen_range(0..max_start)
        } else {
            self.start_offset
        };
        self.position = 0.0;
        self.portfolio_value = self.initial_value;
        compute_state_features(&self.prices, &self.volumes, self.current_step)
    }

    /// Take a step in the environment
    pub fn step(&mut self, action: usize) -> (Array1<f64>, f64, bool) {
        let old_price = self.prices[self.current_step];
        self.current_step += 1;
        let done = self.current_step >= self.prices.len() - 1
            || self.current_step >= self.start_offset + self.episode_length;

        let new_price = self.prices[self.current_step];
        let price_return = (new_price - old_price) / old_price;

        // Update position
        let new_pos = action_to_position_change(action);
        let position_change = (new_pos - self.position).abs();
        let transaction_cost = position_change * TRANSACTION_COST * self.portfolio_value;

        // PnL from position
        let pnl = self.position * price_return * self.portfolio_value - transaction_cost;
        self.portfolio_value += pnl;
        self.position = new_pos;

        let env_reward = pnl / self.initial_value;
        let next_state = compute_state_features(&self.prices, &self.volumes, self.current_step);

        (next_state, env_reward, done)
    }

    /// Get the current portfolio return
    pub fn get_return(&self) -> f64 {
        (self.portfolio_value - self.initial_value) / self.initial_value
    }
}

// ─── GAIL Trainer ────────────────────────────────────────────────────────────

/// Configuration for GAIL training
#[derive(Debug, Clone)]
pub struct GAILConfig {
    pub num_iterations: usize,
    pub trajectories_per_iter: usize,
    pub disc_updates_per_iter: usize,
    pub batch_size: usize,
    pub gamma: f64,
    pub disc_hidden_size: usize,
    pub policy_hidden_size: usize,
    pub disc_lr: f64,
    pub policy_lr: f64,
    pub entropy_coeff: f64,
}

impl Default for GAILConfig {
    fn default() -> Self {
        Self {
            num_iterations: 100,
            trajectories_per_iter: 5,
            disc_updates_per_iter: 3,
            batch_size: 64,
            gamma: 0.99,
            disc_hidden_size: 32,
            policy_hidden_size: 32,
            disc_lr: 0.001,
            policy_lr: 0.0005,
            entropy_coeff: 0.01,
        }
    }
}

/// Training statistics for one GAIL iteration
#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub iteration: usize,
    pub disc_loss: f64,
    pub mean_reward: f64,
    pub mean_return: f64,
}

/// GAIL Trainer orchestrating discriminator and policy training
pub struct GAILTrainer {
    pub discriminator: Discriminator,
    pub policy: PolicyNetwork,
    pub expert_dataset: ExpertDataset,
    pub config: GAILConfig,
    pub stats: Vec<TrainingStats>,
}

impl GAILTrainer {
    /// Create a new GAIL trainer
    pub fn new(expert_dataset: ExpertDataset, config: GAILConfig) -> Self {
        let discriminator = Discriminator::new(config.disc_hidden_size, config.disc_lr);
        let policy = PolicyNetwork::new(
            config.policy_hidden_size,
            config.policy_lr,
            config.entropy_coeff,
        );
        Self {
            discriminator,
            policy,
            expert_dataset,
            config,
            stats: Vec::new(),
        }
    }

    /// Run one iteration of GAIL training
    pub fn train_iteration(&mut self, env: &mut TradingEnvironment) -> TrainingStats {
        let iteration = self.stats.len();

        // Step 1: Collect trajectories from current policy
        let mut policy_trajectories: Vec<(Vec<Array1<f64>>, Vec<usize>, Vec<f64>)> = Vec::new();
        let mut all_policy_pairs: Vec<StateActionPair> = Vec::new();
        let mut total_env_return = 0.0;

        for _ in 0..self.config.trajectories_per_iter {
            let mut states = Vec::new();
            let mut actions = Vec::new();
            let mut state = env.reset();

            loop {
                let action = self.policy.sample_action(&state);
                states.push(state.clone());
                actions.push(action);
                all_policy_pairs.push(StateActionPair {
                    state: state.clone(),
                    action,
                });

                let (next_state, _env_reward, done) = env.step(action);
                state = next_state;
                if done {
                    break;
                }
            }

            total_env_return += env.get_return();
            // Placeholder rewards (will be filled by discriminator)
            let rewards = vec![0.0; states.len()];
            policy_trajectories.push((states, actions, rewards));
        }

        // Step 2: Update discriminator
        let mut disc_loss = 0.0;
        for _ in 0..self.config.disc_updates_per_iter {
            let expert_batch = self.expert_dataset.sample_batch(self.config.batch_size);
            let policy_batch = sample_from_pairs(&all_policy_pairs, self.config.batch_size);
            disc_loss = self.discriminator.train_step(&expert_batch, &policy_batch);
        }

        // Step 3: Compute GAIL rewards
        let mut total_gail_reward = 0.0;
        let mut reward_count = 0;
        for (states, actions, rewards) in policy_trajectories.iter_mut() {
            *rewards = Vec::with_capacity(states.len());
            for (s, &a) in states.iter().zip(actions.iter()) {
                let r = self.discriminator.compute_reward(s, a);
                rewards.push(r);
                total_gail_reward += r;
                reward_count += 1;
            }
        }

        // Step 4: Update policy using REINFORCE
        self.policy
            .update_reinforce(&policy_trajectories, self.config.gamma);

        let stats = TrainingStats {
            iteration,
            disc_loss,
            mean_reward: if reward_count > 0 {
                total_gail_reward / reward_count as f64
            } else {
                0.0
            },
            mean_return: total_env_return / self.config.trajectories_per_iter as f64,
        };
        self.stats.push(stats.clone());
        stats
    }

    /// Run the full GAIL training loop
    pub fn train(&mut self, env: &mut TradingEnvironment) -> Vec<TrainingStats> {
        let num_iters = self.config.num_iterations;
        for _ in 0..num_iters {
            self.train_iteration(env);
        }
        self.stats.clone()
    }

    /// Evaluate policy on a fresh episode
    pub fn evaluate(&self, env: &mut TradingEnvironment) -> f64 {
        let mut state = env.reset();
        loop {
            let action = self.policy.sample_action(&state);
            let (next_state, _, done) = env.step(action);
            state = next_state;
            if done {
                break;
            }
        }
        env.get_return()
    }
}

/// Sample random state-action pairs from a collection
fn sample_from_pairs(pairs: &[StateActionPair], n: usize) -> Vec<StateActionPair> {
    let mut rng = rand::thread_rng();
    let mut batch = Vec::with_capacity(n);
    for _ in 0..n {
        if !pairs.is_empty() {
            let idx = rng.gen_range(0..pairs.len());
            batch.push(pairs[idx].clone());
        }
    }
    batch
}

// ─── Bybit API Client ───────────────────────────────────────────────────────

/// Kline (candlestick) data from Bybit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API response structures
#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Client for fetching data from Bybit API
pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline data for a symbol
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let client = reqwest::blocking::Client::new();
        let resp: BybitResponse = client.get(&url).send()?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: retCode={}", resp.ret_code);
        }

        let mut klines: Vec<Kline> = resp
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Kline {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse to chronological order
        klines.reverse();
        Ok(klines)
    }

    /// Extract prices and volumes from klines
    pub fn klines_to_arrays(klines: &[Kline]) -> (Vec<f64>, Vec<f64>) {
        let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
        (prices, volumes)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Utility Functions ───────────────────────────────────────────────────────

/// Generate synthetic price data for testing
pub fn generate_synthetic_prices(n: usize, initial_price: f64, volatility: f64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(n);
    let mut volumes = Vec::with_capacity(n);
    prices.push(initial_price);
    volumes.push(1000.0);

    for _ in 1..n {
        let ret: f64 = rng.gen_range(-volatility..volatility);
        let new_price = prices.last().unwrap() * (1.0 + ret);
        prices.push(new_price);
        volumes.push(1000.0 + rng.gen_range(-500.0..500.0_f64).abs());
    }

    (prices, volumes)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_features_computation() {
        let (prices, volumes) = generate_synthetic_prices(100, 50000.0, 0.02);
        let state = compute_state_features(&prices, &volumes, 50);
        assert_eq!(state.len(), STATE_DIM);
        // Log return should be finite
        assert!(state[0].is_finite());
        // Volatility should be non-negative
        assert!(state[2] >= 0.0);
    }

    #[test]
    fn test_expert_dataset_construction() {
        let (prices, volumes) = generate_synthetic_prices(500, 50000.0, 0.02);
        let dataset = ExpertDataset::from_price_data(&prices, &volumes, 50, 0.25);
        assert!(!dataset.is_empty());
        assert!(!dataset.trajectories.is_empty());
        // Each pair should have correct state dimension
        for pair in &dataset.all_pairs {
            assert_eq!(pair.state.len(), STATE_DIM);
            assert!(pair.action < NUM_ACTIONS);
        }
    }

    #[test]
    fn test_discriminator_output_range() {
        let disc = Discriminator::new(16, 0.001);
        let state = Array1::from_vec(vec![0.1, -0.2, 0.05, 0.3, -0.1, 0.02, 0.01, 0.005]);
        for action in 0..NUM_ACTIONS {
            let output = disc.forward(&state, action);
            assert!(output >= 0.0 && output <= 1.0, "Discriminator output {} out of [0,1]", output);
        }
    }

    #[test]
    fn test_policy_output_valid_distribution() {
        let policy = PolicyNetwork::new(16, 0.001, 0.01);
        let state = Array1::from_vec(vec![0.1, -0.2, 0.05, 0.3, -0.1, 0.02, 0.01, 0.005]);
        let probs = policy.forward(&state);
        assert_eq!(probs.len(), NUM_ACTIONS);
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-6, "Probabilities should sum to 1, got {}", sum);
        for &p in probs.iter() {
            assert!(p >= 0.0, "Probability should be non-negative, got {}", p);
        }
    }

    #[test]
    fn test_policy_sample_action_valid() {
        let policy = PolicyNetwork::new(16, 0.001, 0.01);
        let state = Array1::from_vec(vec![0.1, -0.2, 0.05, 0.3, -0.1, 0.02, 0.01, 0.005]);
        for _ in 0..100 {
            let action = policy.sample_action(&state);
            assert!(action < NUM_ACTIONS, "Action {} out of range", action);
        }
    }

    #[test]
    fn test_trading_environment() {
        let (prices, volumes) = generate_synthetic_prices(200, 50000.0, 0.01);
        let mut env = TradingEnvironment::new(prices, volumes, 50);
        let state = env.reset();
        assert_eq!(state.len(), STATE_DIM);

        let (next_state, _reward, done) = env.step(2); // Hold
        assert_eq!(next_state.len(), STATE_DIM);
        assert!(!done || env.current_step >= env.prices.len() - 1);
    }

    #[test]
    fn test_discriminator_training() {
        let (prices, volumes) = generate_synthetic_prices(500, 50000.0, 0.02);
        let dataset = ExpertDataset::from_price_data(&prices, &volumes, 50, 0.25);
        let mut disc = Discriminator::new(16, 0.01);

        // Create some policy pairs (random)
        let policy_pairs: Vec<StateActionPair> = (0..64)
            .map(|_| {
                let mut rng = rand::thread_rng();
                StateActionPair {
                    state: Array1::from_vec((0..STATE_DIM).map(|_| rng.gen_range(-1.0..1.0)).collect()),
                    action: rng.gen_range(0..NUM_ACTIONS),
                }
            })
            .collect();

        let expert_batch = dataset.sample_batch(64);
        let loss = disc.train_step(&expert_batch, &policy_pairs);
        assert!(loss.is_finite(), "Discriminator loss should be finite");
    }

    #[test]
    fn test_gail_reward_computation() {
        let disc = Discriminator::new(16, 0.001);
        let state = Array1::from_vec(vec![0.1, -0.2, 0.05, 0.3, -0.1, 0.02, 0.01, 0.005]);
        let reward = disc.compute_reward(&state, 2);
        assert!(reward.is_finite(), "GAIL reward should be finite");
        assert!(reward >= 0.0, "GAIL reward should be non-negative");
    }

    #[test]
    fn test_gail_trainer_single_iteration() {
        let (prices, volumes) = generate_synthetic_prices(500, 50000.0, 0.02);
        let dataset = ExpertDataset::from_price_data(&prices, &volumes, 50, 0.25);

        let config = GAILConfig {
            num_iterations: 1,
            trajectories_per_iter: 2,
            disc_updates_per_iter: 1,
            batch_size: 16,
            gamma: 0.99,
            disc_hidden_size: 16,
            policy_hidden_size: 16,
            disc_lr: 0.001,
            policy_lr: 0.0005,
            entropy_coeff: 0.01,
        };

        let mut trainer = GAILTrainer::new(dataset, config);
        let mut env = TradingEnvironment::new(prices, volumes, 50);
        let stats = trainer.train_iteration(&mut env);
        assert!(stats.disc_loss.is_finite());
        assert!(stats.mean_reward.is_finite());
    }

    #[test]
    fn test_behavior_cloning() {
        let (prices, volumes) = generate_synthetic_prices(500, 50000.0, 0.02);
        let dataset = ExpertDataset::from_price_data(&prices, &volumes, 50, 0.25);
        let mut policy = PolicyNetwork::new(16, 0.01, 0.01);

        let batch = dataset.sample_batch(32);
        let loss = policy.behavior_cloning_step(&batch);
        assert!(loss.is_finite(), "BC loss should be finite");
        assert!(loss >= 0.0, "BC loss should be non-negative");
    }

    #[test]
    fn test_action_to_position_change() {
        assert_eq!(action_to_position_change(0), -1.0);
        assert_eq!(action_to_position_change(1), -0.5);
        assert_eq!(action_to_position_change(2), 0.0);
        assert_eq!(action_to_position_change(3), 0.5);
        assert_eq!(action_to_position_change(4), 1.0);
    }
}

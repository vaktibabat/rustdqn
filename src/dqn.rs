use ndarray::Array1;
use ndarray::Array2;
use rand::{thread_rng, Rng};
use rustgan::layers::*;
use rustgan::neural_net;
use rustgan::neural_net::NeuralNet;

use crate::blackjack::{Action, Blackjack, State};
use crate::replay_buffer::{ReplayMemory, Transition};

const REPLAY_BUF_SIZE: usize = 10000;
const BATCH_SIZE: usize = 64;
const LEARNING_RATE: f64 = 0.003;
// The number of values in each observation, i.e. number of things the agent sees
const STATE_SIZE: usize = 3;
// The number of actions: there's only Hit and Stand
const NUM_ACTIONS: usize = 2;
// Number of episodes we want the Agent to train for; each episode continues until the game is over.
// Since Blackjack is quite a short game, we can afford a large number of episodes
const NUM_EPISODES: usize = 10000;
// Parameters related to decaying epsilon in the epsilon-greed policy
// We decay epsilon since at the beginning we want to eplore more, but then we want to exploit
const EPS_START: f32 = 0.9;
const EPS_END: f32 = 0.05;
const EPS_DECAY: f32 = 1000f32;
// Discount rate
const GAMMA: f64 = 0.99;

// A Deep-Q network (see https://arxiv.org/abs/1312.5602) for Blackjack
pub struct DeepQNetwork {
    memory: ReplayMemory,
    model: NeuralNet,
    env: Blackjack,
}

// Converting the state to a matrix we can feed the neural net
impl From<State> for Array2<f64> {
    fn from(value: State) -> Self {
        let hand_sum = value.hand() as f64;
        let face_up = value.face_up() as f64;
        let has_ace = if value.has_ace() { 1f64 } else { 0f64 };

        Array2::from_shape_vec((1, 3), vec![hand_sum, face_up, has_ace]).unwrap()
    }
}

impl From<State> for Array1<f64> {
    fn from(value: State) -> Self {
        let hand_sum = value.hand() as f64;
        let face_up = value.face_up() as f64;
        let has_ace = if value.has_ace() {1f64} else {0f64};

        Array1::from_vec(vec![hand_sum, face_up, has_ace])
    }
}

impl DeepQNetwork {
    pub fn new() -> DeepQNetwork {
        let memory = ReplayMemory::new(REPLAY_BUF_SIZE);
        let mut model = NeuralNet::new(50, BATCH_SIZE, 0.003, neural_net::Loss::Mse);
        model.add_layer(Box::new(Linear::new(STATE_SIZE, 128)));
        model.add_layer(Box::new(ReLU::new()));
        model.add_layer(Box::new(Linear::new(128, 128)));
        model.add_layer(Box::new(ReLU::new()));
        model.add_layer(Box::new(Linear::new(128, NUM_ACTIONS)));
        let env = Blackjack::new();

        DeepQNetwork { memory, model, env }
    }

    pub fn predict(&mut self, state: State) -> Array2<f64> {
        self.model.forward(&Array2::from(state).view(), false)
    }

    // Train the network to play blackjack!
    pub fn fit(&mut self) {
        let mut num_steps = 0f32;

        for num_ep in 0..NUM_EPISODES {
            // Keep playing until the game is over
            while !self.env.is_over() {
                let state = self.env.state();
                // Epsilon-greedy policy: With probability epsilon select a random action a_t
                // otherwise select a_t = max_a Q^{*} (\phi(s_t), a; \theta)
                let eps_threshold =
                    EPS_END + (EPS_START - EPS_END) * (-1f32 * num_steps / EPS_DECAY).exp();
                let action = self.eps_greedy_policy(state.clone(), eps_threshold);
                // Execute action a_t in game and observe reward r_t and next game state s_{t + 1}
                self.env.step(&action);
                let next_state = self.env.state();
                let reward = self.env.reward();
                let transition = Transition::new(
                    state,
                    action,
                    if !self.env.is_over() {
                        Some(next_state)
                    } else {
                        None
                    },
                    reward,
                );
                // Store transition in replay buffer
                self.memory.store(transition);
                // Perform training step
                self.training_step();

                num_steps += 1f32;
            }

            self.env.reset();
        }

        println!("Num steps: {}", num_steps);
    }

    fn eps_greedy_policy(&mut self, state: State, eps_threshold: f32) -> Action {
        let mut rng = thread_rng();
        let x = rng.gen_range(0f32..=1f32);

        // Select a random action with probability epsilon
        if x <= eps_threshold {
            if rng.gen_bool(0.5) {
                Action::Hit
            } else {
                Action::Stand
            }
        } else {
            let results = self.model.forward(&Array2::from(state).view(), false);

            match results
                .iter()
                .enumerate()
                .max_by(|(_, b), (_, d)| b.total_cmp(d))
                .unwrap()
                .0
            {
                0 => Action::Hit,
                1 => Action::Stand,
                _ => {
                    println!("Shouldn't happen");

                    Action::Hit
                }
            }
        }
    }

    fn training_step(&mut self) {
        // We can't sample a batch if we have less transitions than the batch size
        if self.memory.len() < BATCH_SIZE {
            return;
        }

        // Sample a batch
        let transition_batch = self.memory.sample(BATCH_SIZE);
        // Compute y_j, which is r_j for terminal next_state, and
        // r_j + GAMMA * max_{a'} Q(phi_{j + 1}, a' ; Theta) for non-terminal next_state
        // These are the MSE targets
        let targets: Vec<f64> = (0..BATCH_SIZE)
            .map(|i| {
                let transition = transition_batch[i];

                // Non-terminal next_state
                if let Some(next_state) = transition.next_state() {
                    // \max_{a'} Q(\phi_{j + 1}, a' ; \Theta)
                    let next_state_mat = Array2::from(next_state);
                    let max_next_action = self
                        .model
                        .forward(&next_state_mat.view(), false)
                        .into_iter()
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap();
                    // Add r_j and multiply by gamma

                    transition.reward() as f64 + GAMMA * max_next_action
                } else {
                    // Terminal next_state
                    transition.reward() as f64
                }
            })
            .collect();
        // The predictions of the net on each transition
        let mut states_mat = Array2::zeros((0, STATE_SIZE));

        for transition in &transition_batch {
            let state_vec = Array1::from(transition.state());

            states_mat.push_row(state_vec.view()).unwrap();
        }
        // This is a BATCH_SIZExNUM_ACTIONS matrix containing the Q-value for each state-action pair
        // The output of the network
        let q_values_mat = self.model.forward(&states_mat.view(), true);
        let y_hat: Vec<f64> = (0..BATCH_SIZE).map(|i| {
            let transition = transition_batch[i];
            // The corresponding row in the Q-Values matrix
            let q_values = q_values_mat.row(i);

            *match transition.action() {
                Action::Hit => q_values.get(0).unwrap(),
                Action::Stand => q_values.get(1).unwrap(),
            }
        }).collect();

        println!("Loss: {}", DeepQNetwork::q_loss(&y_hat, &targets));
        let targets_mat = Array2::from_shape_vec((1, BATCH_SIZE), targets).unwrap();
        let batch_actions: Vec<Action> = transition_batch.iter().map(|transition| transition.action()).collect();

        let dy = DeepQNetwork::upstream_loss(q_values_mat, targets_mat, batch_actions);
        let mut gradients = self.model.backward(dy).0;
        gradients.reverse();

        // Perform GD step
        for i in 0..self.model.layers.len() {
            // The current gradient
            let grad = &gradients[i];

            // Proceed only if there are parameter gradients
            // layers such as ReLU don't have any parameters, so we don't need to update anything
            if let (Some(dw), Some(db)) = grad {
                self.model.layers[i].update_params(&dw, &db, LEARNING_RATE);
            }
        }

    }

    // Compute the gradient of the loss WRT the Q-values predicted by the net
    // Predictions is an BATCH_SIZExNUM_ACTIONS matrix
    // Tragets is an 1xBATCH_SIZE
    fn upstream_loss(
        predictions: Array2<f64>,
        targets: Array2<f64>,
        batch_actions: Vec<Action>
    ) -> Array2<f64> {
        let mut gradient = Array2::<f64>::zeros((0, NUM_ACTIONS));
        
        for i in 0..BATCH_SIZE {
            let action = &batch_actions[i];
            // Compute current row of the gradient matrix
            let mut curr_row = vec![0f64, 0f64];
            // The nonzero entry in the current row (one of the actions doesn't affect the loss)
            let nonzero_idx = match action {
                Action::Hit => 0,
                Action::Stand => 1,
            };
            // The Q-value corresponding to the nonzero entry
            let nonzero_q_value = predictions.get((i, nonzero_idx)).unwrap();
            let target_i = targets.get((0, i)).unwrap();

            curr_row[nonzero_idx] = (2.0 / BATCH_SIZE as f64) * (nonzero_q_value - target_i);

            gradient.push_row(Array1::from_vec(curr_row).view()).unwrap();
        }

        gradient
    }

    // MSE, basically
    fn q_loss(pred: &Vec<f64>, ground_truth: &Vec<f64>) -> f64 {
        let targets_mat = Array2::from_shape_vec((1, BATCH_SIZE), ground_truth.to_vec()).unwrap();
        let pred_mat = Array2::from_shape_vec((1, BATCH_SIZE), pred.to_vec()).unwrap();
        let mut total_loss = 0f64;
        let pred_row = pred_mat.row(0);
        let ground_truth_row = targets_mat.row(0);

        for i in 0..BATCH_SIZE {
            let dist = pred_row.get(i).unwrap() - ground_truth_row.get(i).unwrap();
            total_loss += dist * dist;
        }

        (1.0 / BATCH_SIZE as f64) * total_loss
    }
}

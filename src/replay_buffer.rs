use std::collections::VecDeque;

use rand::{seq::IteratorRandom, thread_rng};

use crate::blackjack::{Action, State};

// A transition from state s_1 with action a to state s_2 and reward r
pub struct Transition {
    state: State,
    action: Action,
    next_state: Option<State>,
    reward: f32,
}

// The replay buffer of the DQN
pub struct ReplayMemory {
    memory: VecDeque<Transition>,
}

impl Transition {
    pub fn new(state: State, action: Action, next_state: Option<State>, reward: f32) -> Transition {
        Transition {
            state,
            action,
            next_state,
            reward,
        }
    }

    // Getters
    pub fn state(&self) -> State {
        self.state.clone()
    }

    pub fn action(&self) -> Action {
        self.action.clone()
    }

    pub fn next_state(&self) -> Option<State> {
        self.next_state.clone()
    }

    pub fn reward(&self) -> f32 {
        self.reward
    }
}

impl ReplayMemory {
    pub fn new(size: usize) -> ReplayMemory {
        let memory = VecDeque::<Transition>::with_capacity(size);

        ReplayMemory { memory }
    }

    // Store a transition in the ReplayMemory
    pub fn store(&mut self, transition: Transition) {
        self.memory.push_back(transition);
    }

    // Sample a batch of transitions from the memory
    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        let mut rng = thread_rng();

        self.memory.iter().choose_multiple(&mut rng, batch_size)
    }

    // Get the amount of elements in the memory
    pub fn len(&self) -> usize {
        self.memory.len()
    }
}

// Implementation of blackjack
use crate::cardpack::CardPack;

#[derive(Clone, Debug)]
// The current state of the game. This is what the agent sees
pub struct State {
    hand: usize,    // The sum of the agent's hand (i.e. if agent has J 5 3, then hand=18)
    face_up: usize, // The value of the dealer's face up card
    has_ace: bool, // Whether the agent has an ace in their hand. Aces are important since they can be
                   // both 0 and 1
}

#[derive(Clone)]
// Possible actions to play. Real blackjack has more actions, such as split and double down
// but we only do hit and stand
pub enum Action {
    Hit,
    Stand,
}
// The game environment
pub struct Blackjack {
    // The card pack
    pack: CardPack,
    // Current game state (from the viewpoint of the agent)
    state: State,
    // Whether the game is over
    over: bool,
    // Game reward:
    // +1 If agent wins
    // +0 If draw
    // -1 If agent loses
    // +1.5 If natural blackjack (i.e. A 10, A J, etc.)
    // As long as the game is not finished, this is 0
    reward: f32,
    // The dealer's hand
    dealer_hand: usize,
    // Whether we've already "used" the agent's ace
    used_ace: bool,
}

impl State {
    pub fn new(hand: usize, face_up: usize, has_ace: bool) -> State {
        State { hand, face_up, has_ace }
    }
    
    pub fn hand(&self) -> usize {
        self.hand
    }

    pub fn face_up(&self) -> usize {
        self.face_up
    }

    pub fn has_ace(&self) -> bool {
        self.has_ace
    }
}

impl Blackjack {
    pub fn new() -> Blackjack {
        let mut pack = CardPack::new();
        // To create the game state, we start by sampling two cards for the player's hand
        let (p_card1, p_card2) = (pack.take_card().unwrap(), pack.take_card().unwrap());
        // If the player has an ace, mark it
        let has_ace = p_card1 == 1 || p_card2 == 1;
        // The hand's hand; if the agent has an ace, treat it as an 11 for now
        let hand = p_card1 + p_card2 + if has_ace { 10 } else { 0 };
        // Then, we sample two cards for the dealer, one of which is face up
        let (d_card1, d_card2) = (pack.take_card().unwrap(), pack.take_card().unwrap());
        let face_up = d_card1;
        // Create the state
        let state = State {
            hand,
            has_ace,
            face_up,
        };
        // The dealer's hand sum
        let dealer_hand = d_card1 + d_card2;
        // The game can't be over before it begins!
        let over = false;
        // Agent only gets a reward when the game is over
        let reward = 0f32;

        Blackjack {
            pack,
            state,
            over,
            reward,
            dealer_hand,
            used_ace: false,
        }
    }

    // Getters
    pub fn is_over(&self) -> bool {
        self.over
    }

    pub fn hand_sum(&self) -> usize {
        self.state.hand
    }

    pub fn reward(&self) -> f32 {
        self.reward
    }

    pub fn state(&self) -> State {
        self.state.clone()
    }

    // Reset the game
    pub fn reset(&mut self) {
        *self = Blackjack::new();
    }

    // Perform an action in the game
    pub fn step(&mut self, action: &Action) {
        match action {
            Action::Hit => self.hit(),
            Action::Stand => self.stand(),
        }
    }

    fn hit(&mut self) {
        // Draw a card from the pack
        let new_card = self.pack.take_card().unwrap();

        let agent_hand = self.state.hand;
        // If the agent busts, the game is over and they lose
        if agent_hand + new_card > 21 {
            // If they player has an ace, go back to treating it as a 1
            if self.state.has_ace && !self.used_ace {
                self.state.hand += new_card;
                self.state.hand -= 10;
                // We've already "used the ace"
                self.used_ace = true;
                return;
            }

            self.over = true;
            self.reward = -1f32;

            return;
        }
        // Otherwise, the card is added to their hand
        self.state.hand += new_card;
        // If the card is an ace, mark it
        if new_card == 1 {
            self.state.has_ace = true;
        }
    }

    fn stand(&mut self) {
        // Get the player's hand
        let agent_sum = self.state.hand;
        let is_blackjack = self.state.hand == 11 && self.state.has_ace;
        // To compute the reward, check whether the agent or the dealer won
        let mut dealer_sum = self.dealer_hand;

        // Keep sampling cards until one of the following happens:
        // (1) - The dealer busts, in which case the player wins
        // (2) - The dealer has a value >= 17
        while dealer_sum < 17 {
            dealer_sum += self.pack.take_card().unwrap();
        }

        // Check which player won
        // The agent busts, so they lose
        let reward = if agent_sum > 21 {
            -1f32
        } else {
            // The agent doesn't bust
            match dealer_sum {
                // The dealer busts, so the agent wins
                22.. => 1f32,
                _ => {
                    if agent_sum > dealer_sum {
                        // Agent has more than dealer, so they win
                        // If they had a nat 21 (i.e. blackjack), they get +1.5 points
                        // else, they get +1 points
                        if agent_sum == 21 || is_blackjack {
                            1.5
                        } else {
                            1f32
                        }
                    } else if agent_sum == dealer_sum {
                        // Draw; reward of 0
                        0f32
                    } else {
                        // Agent has less than dealer
                        -1f32
                    }
                }
            }
        };

        // If we stand, the game is over
        self.over = true;
        self.reward = reward;
    }
}

#[cfg(test)]
mod tests {
    use super::CardPack;

    #[test]
    fn test_card_pack() {
        let mut pack = CardPack::new();
        let sum = (0..52).map(|_| pack.take_card().unwrap()).sum::<usize>();

        // The sum of all cards in the pack is 340
        assert_eq!(sum, 340);
    }
}

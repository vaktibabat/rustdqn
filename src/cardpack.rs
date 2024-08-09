// Implementation of blackjack
use rand::{self, Rng};

// A, 2, ..., 10, J, Q, K
const AMT_CARDS: usize = 13;
// There are 4 cards of each type
const CARDS_PER_TYPE: usize = 4;

// A card stack. As we take more cards of a certain type, the probability of seeing that
// type gets smaller
pub struct CardPack {
    pack: [usize; AMT_CARDS],
}

impl CardPack {
    pub fn new() -> CardPack {
        CardPack {
            pack: [CARDS_PER_TYPE; AMT_CARDS],
        }
    }

    // Get the index of the card we want to sample. The probability is based om the
    // amounts of cards of each type
    fn sample_card_idx(&mut self) -> Option<usize> {
        let total_cards = self.pack.iter().sum();
        let mut culminative_probs = [0; AMT_CARDS];
        let mut cum_prob = 0;

        if total_cards == 0 {
            return None;
        }

        for (i, cnt) in self.pack.iter().enumerate() {
            cum_prob += cnt;
            culminative_probs[i] = cum_prob;
        }

        let mut rng = rand::thread_rng();
        let random_idx = rng.gen_range(0..total_cards);

        let pos = culminative_probs
            .iter()
            .position(|&prob| prob > random_idx)?;
        self.pack[pos] -= 1;

        Some(pos)
    }

    // Map the index of a card to its value. Aces are treated as 1s
    fn idx_val_mapping(idx: usize) -> usize {
        if idx <= 9 {
            return idx + 1;
        }
        // 10, J, Q, K
        else {
            return 10;
        }
    }

    // Sample a card from the pack
    pub fn take_card(&mut self) -> Option<usize> {
        let idx = self.sample_card_idx()?;

        Some(CardPack::idx_val_mapping(idx))
    }
}

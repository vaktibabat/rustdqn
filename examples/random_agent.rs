use rand::{self, distributions::Uniform, prelude::Distribution, thread_rng};
use rustdqn::blackjack::{self, Blackjack};

// This code tests an agent that acts randomly: it hits/stands with an equal probability
fn main() {
    let mut env = Blackjack::new();
    let mut victories = 0;
    let mut losses = 1;
    let mut draws = 0;
    let mut rng = thread_rng();
    let dist = Uniform::new(0, 2);

    for _ in 0..100000 {
        while !env.is_over() {
            let action = if dist.sample(&mut rng) == 0 {
                blackjack::Action::Hit
            } else {
                blackjack::Action::Stand
            };
            env.step(&action);
        }

        if env.reward() > 0f32 {
            victories += 1;
        } else if env.reward() == 0f32 {
            draws += 1;
        } else {
            losses += 1;
        }

        env.reset();
    }

    println!("----RANDOM AGENT STATS----");
    println!("Victories: {}", victories);
    println!("Draws: {}", draws);
    println!("Losses: {}", losses);
}

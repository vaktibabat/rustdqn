use rustdqn::blackjack::{Blackjack, State};
use rustdqn::dqn::DeepQNetwork;

fn main() {
    let mut dqn = DeepQNetwork::new();
    let mut stand_ctr = 0;
    let mut hit_ctr = 0;

    dqn.fit();

    println!("Alrighty, let's play some blackjack!");

    for hand in 16..=21  {
        for face_up in 1..10 {
            for has_ace in [true, false] {
                let state = State::new(hand, face_up, has_ace);
                let dqn_res = dqn.predict(state.clone());

                if dqn_res.get((0, 1)) > dqn_res.get((0, 0)) {
                    stand_ctr += 1;
                } else {
                    println!("{:?}", state);

                    hit_ctr += 1;
                }
            }
        }
    }

    println!("---HIGH HAND RESULTS---");
    println!("{} Stands\n{} Hits", stand_ctr, hit_ctr);
}
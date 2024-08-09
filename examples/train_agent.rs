use std::io::stdin;
use rustdqn::{blackjack::{Blackjack, Action}, dqn::DeepQNetwork};

fn main() {
    let mut dqn = DeepQNetwork::new();

    dqn.fit();

    println!("Alrighty, let's play some blackjack!");

    for _ in 0..10 {
        let mut game = Blackjack::new();

        while !game.is_over() {
            let state = game.state();
            println!("Game state: {:?}", state);
            println!("Agent says: {:?}", dqn.predict(state));
    
            let mut buf = String::new();
    
            println!("Pick an action: ");
            println!("1 - Hit\n2 - Stand");
            print!("> ");
            stdin().read_line(&mut buf).unwrap();
    
            let choice = buf.chars().nth(0).unwrap();
    
            match choice {
                '1' => game.step(&rustdqn::blackjack::Action::Hit),
                '2' => game.step(&rustdqn::blackjack::Action::Stand),
                _ => println!("Unknown action!"),
            }
        }
    
        println!("Game is over, your reward is: {}", game.reward());
    }

    let mut env = Blackjack::new();
    let mut victories = 0;
    let mut losses = 1;
    let mut draws = 0; 

    for _ in 0..100000 {
        while !env.is_over() {
            let state = env.state();
            let agent_pred = dqn.predict(state);
            
            let action = if agent_pred.get((0, 0)) > agent_pred.get((0, 1)) {
                Action::Hit
            } else {
                Action::Stand
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

    println!("----DQN AGENT STATS----");
    println!("Victories: {}", victories);
    println!("Draws: {}", draws);
    println!("Losses: {}", losses);
}

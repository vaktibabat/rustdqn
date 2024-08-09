use std::io::stdin;

use rustdqn::blackjack;

fn main() {
    println!("Alrighty, let's play some blackjack!");

    let mut game = blackjack::Blackjack::new();

    while !game.is_over() {
        println!("Game state: {:?}", game.state());

        let mut buf = String::new();

        println!("Pick an action: ");
        println!("1 - Hit\n2 - Stand");
        print!("> ");
        stdin().read_line(&mut buf).unwrap();

        let choice = buf.chars().nth(0).unwrap();

        match choice {
            '1' => game.step(&blackjack::Action::Hit),
            '2' => game.step(&blackjack::Action::Stand),
            _ => println!("Unknown action!"),
        }
    }

    println!("Game is over, your reward is: {}", game.reward());
}

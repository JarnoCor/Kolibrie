/*
 * Copyright © 2026 Volodymyr Kadzhaia
 * Copyright © 2026 Pieter Bonte
 * KU Leuven — Stream Intelligence Lab, Belgium
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * you can obtain one at https://mozilla.org/MPL/2.0/.
 */

use datalog::reasoning::Reasoner;
use shared::terms::Term;
use shared::rule::Rule;
use shared::probabilistic_rule::{ProbabilisticRule, ProbCombination};

fn main() {
    println!("=== Social Trust Network: Combined Rule Inference Example ===\n");

    let mut kg = Reasoner::new();

    println!("[Stage 1] Loading base facts...");

    // Certain `knows` facts (8 triples)
    kg.add_abox_triple("Alice", "knows", "Bob");
    kg.add_abox_triple("Alice", "knows", "Charlie");
    kg.add_abox_triple("Bob",   "knows", "Diana");
    kg.add_abox_triple("Bob",   "knows", "Eve");
    kg.add_abox_triple("Charlie", "knows", "Frank");
    kg.add_abox_triple("Diana", "knows", "Eve");
    kg.add_abox_triple("Eve",   "knows", "Frank");
    kg.add_abox_triple("Frank", "knows", "Alice");

    // Probabilistic `trusts` facts (7 triples)
    kg.add_probabilistic_triple("Alice",   "trusts", "Bob",   0.90);
    kg.add_probabilistic_triple("Alice",   "trusts", "Charlie", 0.70);
    kg.add_probabilistic_triple("Bob",     "trusts", "Diana", 0.80);
    kg.add_probabilistic_triple("Bob",     "trusts", "Eve",   0.60);
    kg.add_probabilistic_triple("Charlie", "trusts", "Frank", 0.75);
    kg.add_probabilistic_triple("Diana",   "trusts", "Eve",   0.85);
    kg.add_probabilistic_triple("Eve",     "trusts", "Frank", 0.65);

    let initial_size = kg.index_manager.query(None, None, None).len();
    println!("  Certain knowledge (knows): 8 facts");
    println!("  Uncertain knowledge (trusts): 7 facts");
    println!("  Initial database size: {} triples", initial_size);

    let knows_id         = kg.dictionary.write().unwrap().encode("knows");
    let connected_id     = kg.dictionary.write().unwrap().encode("connected");
    let trusts_id        = kg.dictionary.write().unwrap().encode("trusts");
    let indirect_id      = kg.dictionary.write().unwrap().encode("indirectTrust");
    let strong_bond_id   = kg.dictionary.write().unwrap().encode("strongBond");
    let trust_comm_id    = kg.dictionary.write().unwrap().encode("trustCommunity");

    // Rule 1: knows(X,Y) ∧ knows(Y,Z) → connected(X,Z)  (two-hop)
    let rule1 = Rule {
        premise: vec![
            (Term::Variable("X".to_string()), Term::Constant(knows_id),     Term::Variable("Y".to_string())),
            (Term::Variable("Y".to_string()), Term::Constant(knows_id),     Term::Variable("Z".to_string())),
        ],
        conclusion: vec![(
            Term::Variable("X".to_string()), Term::Constant(connected_id), Term::Variable("Z".to_string()),
        )],
        filters: vec![],
    };

    // Rule 2: connected(X,Y) ∧ connected(Y,Z) → connected(X,Z)  (transitive closure)
    let rule2 = Rule {
        premise: vec![
            (Term::Variable("X".to_string()), Term::Constant(connected_id), Term::Variable("Y".to_string())),
            (Term::Variable("Y".to_string()), Term::Constant(connected_id), Term::Variable("Z".to_string())),
        ],
        conclusion: vec![(
            Term::Variable("X".to_string()), Term::Constant(connected_id), Term::Variable("Z".to_string()),
        )],
        filters: vec![],
    };

    // Rule 3: strongBond(X,Y) ∧ strongBond(Y,Z) → trustCommunity(X,Z)  (uses prob output)
    let rule3 = Rule {
        premise: vec![
            (Term::Variable("X".to_string()), Term::Constant(strong_bond_id), Term::Variable("Y".to_string())),
            (Term::Variable("Y".to_string()), Term::Constant(strong_bond_id), Term::Variable("Z".to_string())),
        ],
        conclusion: vec![(
            Term::Variable("X".to_string()), Term::Constant(trust_comm_id), Term::Variable("Z".to_string()),
        )],
        filters: vec![],
    };

    kg.add_rule(rule1);
    kg.add_rule(rule2);
    // Rule 3 is added later — it uses strongBond from the probabilistic round

    // Rule 4: trusts(X,Y) ∧ trusts(Y,Z) → indirectTrust(X,Z)
    //   Independent combination, threshold=0.25, confidence=0.90
    let base_rule4 = Rule {
        premise: vec![
            (Term::Variable("X".to_string()), Term::Constant(trusts_id),   Term::Variable("Y".to_string())),
            (Term::Variable("Y".to_string()), Term::Constant(trusts_id),   Term::Variable("Z".to_string())),
        ],
        conclusion: vec![(
            Term::Variable("X".to_string()), Term::Constant(indirect_id), Term::Variable("Z".to_string()),
        )],
        filters: vec![],
    };
    let prob_rule4 = ProbabilisticRule::new(base_rule4)
        .with_combination(ProbCombination::Independent)
        .with_threshold(0.25)
        .with_confidence(0.90);

    // Rule 5: connected(X,Z) ∧ trusts(X,Z) → strongBond(X,Z)
    //   Min combination, threshold=0.40, confidence=0.95
    //   Uses classically-inferred `connected` from Round 1
    let base_rule5 = Rule {
        premise: vec![
            (Term::Variable("X".to_string()), Term::Constant(connected_id),   Term::Variable("Z".to_string())),
            (Term::Variable("X".to_string()), Term::Constant(trusts_id),      Term::Variable("Z".to_string())),
        ],
        conclusion: vec![(
            Term::Variable("X".to_string()), Term::Constant(strong_bond_id), Term::Variable("Z".to_string()),
        )],
        filters: vec![],
    };
    let prob_rule5 = ProbabilisticRule::new(base_rule5)
        .with_combination(ProbCombination::Min)
        .with_threshold(0.40)
        .with_confidence(0.95);

    kg.add_probabilistic_rule(prob_rule4);
    kg.add_probabilistic_rule(prob_rule5);

    println!("\n[Stage 2] Classical Inference (RULE) - Round 1");
    println!("  Classical rules: connected(X,Z) :- knows(X,Y), knows(Y,Z)");
    println!("                   connected(X,Z) :- connected(X,Y), connected(Y,Z)");

    let facts1 = kg.infer_new_facts_semi_naive();
    let after_classical1 = kg.index_manager.query(None, None, None).len();

    let classical1_count = facts1.len();
    println!("  Inferred {} new facts using classical rules", classical1_count);
    println!("  Database size: {} triples", after_classical1);

    println!("  Inferred connected facts (sample):");
    {
        let dict = kg.dictionary.read().unwrap();
        let mut shown = 0;
        for t in &facts1 {
            if shown >= 5 {
                break;
            }
            if t.predicate == connected_id {
                println!("    {} connected {}",
                    dict.decode(t.subject).unwrap_or("?"),
                    dict.decode(t.object).unwrap_or("?"));
                shown += 1;
            }
        }
    }

    println!("\n[Stage 3] Probabilistic Inference (RULE PROB)");
    println!("  (Uses classically-inferred connected facts from Stage 2!)");
    println!("  Probabilistic rules:");
    println!("    indirectTrust(X,Z) :- trusts(X,Y), trusts(Y,Z)  [Independent, threshold=0.25, conf=0.90]");
    println!("    strongBond(X,Z)    :- connected(X,Z), trusts(X,Z)  [Min, threshold=0.40, conf=0.95]");

    let facts2 = kg.infer_new_facts_probabilistic_semi_naive();
    let after_prob = kg.index_manager.query(None, None, None).len();

    println!("  Inferred {} new probabilistic facts", facts2.len());
    println!("  Database size: {} triples", after_prob);

    println!("  indirectTrust facts:");
    {
        let dict = kg.dictionary.read().unwrap();
        for (triple, prob) in kg.probability_store.iter() {
            if triple.predicate == indirect_id {
                println!("    {} indirectTrust {} (prob={:.2})",
                    dict.decode(triple.subject).unwrap_or("?"),
                    dict.decode(triple.object).unwrap_or("?"),
                    prob);
            }
        }
    }

    println!("  strongBond facts:");
    {
        let dict = kg.dictionary.read().unwrap();
        for (triple, prob) in kg.probability_store.iter() {
            if triple.predicate == strong_bond_id {
                println!("    {} strongBond {} (prob={:.2})",
                    dict.decode(triple.subject).unwrap_or("?"),
                    dict.decode(triple.object).unwrap_or("?"),
                    prob);
            }
        }
    }

    println!("\n[Stage 4] Classical Inference (RULE) - Round 2");
    println!("  (Uses probabilistically-inferred strongBond facts from Stage 3!)");
    println!("  Classical rule: trustCommunity(X,Z) :- strongBond(X,Y), strongBond(Y,Z)");

    // Add rule3 now so it participates in the second classical round
    kg.add_rule(rule3);

    let facts3 = kg.infer_new_facts_semi_naive();
    let after_classical2 = kg.index_manager.query(None, None, None).len();

    println!("  Inferred {} new facts", facts3.len());
    println!("  Database size: {} triples", after_classical2);

    println!("  trustCommunity facts:");
    {
        let dict = kg.dictionary.read().unwrap();
        for t in &facts3 {
            if t.predicate == trust_comm_id {
                println!("    {} trustCommunity {}",
                    dict.decode(t.subject).unwrap_or("?"),
                    dict.decode(t.object).unwrap_or("?"));
            }
        }
    }

    println!("\n=== Final Statistics ===");
    let after_classical1_delta = after_classical1 as isize - initial_size as isize;
    let after_prob_delta       = after_prob as isize - after_classical1 as isize;
    let after_classical2_delta = after_classical2 as isize - after_prob as isize;
    let total_inferred         = after_classical2 as isize - initial_size as isize;
    let growth_pct             = (total_inferred as f64 / initial_size as f64) * 100.0;

    println!("  Initial facts: {}", initial_size);
    println!("  After classical round 1: +{} ({} total)", after_classical1_delta, after_classical1);
    println!("  After probabilistic round: +{} ({} total)", after_prob_delta, after_prob);
    println!("  After classical round 2: +{} ({} total)", after_classical2_delta, after_classical2);
    println!("  Total inferred: {} new facts ({:.0}% database growth)", total_inferred, growth_pct);
}

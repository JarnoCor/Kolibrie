use datalog::reasoning::Reasoner;
use shared::rule::{FilterCondition, Rule};
use shared::terms::Term;
use std::collections::HashMap;

fn enc(r: &Reasoner, s: &str) -> u32 {
    r.dictionary.write().unwrap().encode(s)
}

fn rule(premises: Vec<(Term, Term, Term)>, conclusions: Vec<(Term, Term, Term)>) -> Rule {
    Rule {
        premise: premises,
        conclusion: conclusions,
        filters: vec![],
    }
}

fn inferred(r: &mut Reasoner, s: &str, p: &str, o: &str) -> bool {
    !r.query_abox(Some(s), Some(p), Some(o)).is_empty()
}

fn bc_has(results: &[HashMap<String, Term>], var: &str, val: u32) -> bool {
    results.iter().any(|b| b.get(var) == Some(&Term::Constant(val)))
}

// Forward chaining

#[test]
fn fc_1hop_base() {
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "B");

    let parent = enc(&r, "parent");
    let ancestor = enc(&r, "ancestor");

    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(parent), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into()))],
    ));

    r.infer_new_facts_semi_naive();

    assert!(inferred(&mut r, "A", "ancestor", "B"));
}

#[test]
fn fc_2hop_transitive() {
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "B");
    r.add_abox_triple("B", "parent", "C");

    let parent = enc(&r, "parent");
    let ancestor = enc(&r, "ancestor");

    // base: parent → ancestor
    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(parent), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into()))],
    ));
    // transitive: ancestor + ancestor → ancestor
    r.add_rule(rule(
        vec![
            (Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into())),
            (Term::Variable("Y".into()), Term::Constant(ancestor), Term::Variable("Z".into())),
        ],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Z".into()))],
    ));

    r.infer_new_facts_semi_naive();

    assert!(inferred(&mut r, "A", "ancestor", "B"));
    assert!(inferred(&mut r, "B", "ancestor", "C"));
    assert!(inferred(&mut r, "A", "ancestor", "C"));
}

#[test]
fn fc_3hop_transitive() {
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "B");
    r.add_abox_triple("B", "parent", "C");
    r.add_abox_triple("C", "parent", "D");

    let parent = enc(&r, "parent");
    let ancestor = enc(&r, "ancestor");

    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(parent), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into()))],
    ));
    r.add_rule(rule(
        vec![
            (Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into())),
            (Term::Variable("Y".into()), Term::Constant(ancestor), Term::Variable("Z".into())),
        ],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Z".into()))],
    ));

    r.infer_new_facts_semi_naive();

    assert!(inferred(&mut r, "A", "ancestor", "B"));
    assert!(inferred(&mut r, "A", "ancestor", "C"));
    assert!(inferred(&mut r, "A", "ancestor", "D"));
    assert!(inferred(&mut r, "B", "ancestor", "D"));
}

#[test]
fn fc_join_sibling() {
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "P");
    r.add_abox_triple("B", "parent", "P");

    let parent = enc(&r, "parent");
    let sibling = enc(&r, "sibling");

    r.add_rule(Rule {
        premise: vec![
            (Term::Variable("X".into()), Term::Constant(parent), Term::Variable("P2".into())),
            (Term::Variable("Y".into()), Term::Constant(parent), Term::Variable("P2".into())),
        ],
        conclusion: vec![
            (Term::Variable("X".into()), Term::Constant(sibling), Term::Variable("Y".into())),
        ],
        filters: vec![
            FilterCondition { variable: "X".into(), operator: "!=".into(), value: "Y".into() },
        ],
    });

    r.infer_new_facts_semi_naive();

    assert!(inferred(&mut r, "A", "sibling", "B"));
    assert!(inferred(&mut r, "B", "sibling", "A"));

    assert!(!inferred(&mut r, "A", "sibling", "A"));
}

#[test]
fn fc_multi_rule_cascade() {
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "worksFor", "Corp");

    let works_for = enc(&r, "worksFor");
    let employed = enc(&r, "employed");
    let affiliated = enc(&r, "affiliated");

    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(works_for), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(employed), Term::Variable("Y".into()))],
    ));
    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(employed), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(affiliated), Term::Variable("Y".into()))],
    ));

    r.infer_new_facts_semi_naive();

    assert!(inferred(&mut r, "A", "employed", "Corp"));
    assert!(inferred(&mut r, "A", "affiliated", "Corp"));
}

#[test]
fn fc_three_premise_rule() {
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "R", "B");
    r.add_abox_triple("B", "S", "C");
    r.add_abox_triple("C", "T", "D");

    let r_pred = enc(&r, "R");
    let s_pred = enc(&r, "S");
    let t_pred = enc(&r, "T");
    let connected = enc(&r, "connected");

    r.add_rule(rule(
        vec![
            (Term::Variable("X".into()), Term::Constant(r_pred), Term::Variable("Y".into())),
            (Term::Variable("Y".into()), Term::Constant(s_pred), Term::Variable("Z".into())),
            (Term::Variable("Z".into()), Term::Constant(t_pred), Term::Variable("W".into())),
        ],
        vec![(Term::Variable("X".into()), Term::Constant(connected), Term::Variable("W".into()))],
    ));

    r.infer_new_facts_semi_naive();

    assert!(inferred(&mut r, "A", "connected", "D"));
}

#[test]
fn fc_no_spurious() {
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "B");
    r.add_abox_triple("C", "unrelated", "D");

    let parent = enc(&r, "parent");
    let ancestor = enc(&r, "ancestor");

    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(parent), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into()))],
    ));

    r.infer_new_facts_semi_naive();

    assert!(inferred(&mut r, "A", "ancestor", "B"));
    assert!(!inferred(&mut r, "C", "ancestor", "D"));
}

#[test]
fn fc_sibling_three_children() {
    // A, B, C all share parent P → 6 directional pairs, no self-sibling
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "P");
    r.add_abox_triple("B", "parent", "P");
    r.add_abox_triple("C", "parent", "P");

    let parent = enc(&r, "parent");
    let sibling = enc(&r, "sibling");

    r.add_rule(Rule {
        premise: vec![
            (Term::Variable("X".into()), Term::Constant(parent), Term::Variable("Z".into())),
            (Term::Variable("Y".into()), Term::Constant(parent), Term::Variable("Z".into())),
        ],
        conclusion: vec![
            (Term::Variable("X".into()), Term::Constant(sibling), Term::Variable("Y".into())),
        ],
        filters: vec![
            FilterCondition { variable: "X".into(), operator: "!=".into(), value: "Y".into() },
        ],
    });

    r.infer_new_facts_semi_naive();

    for (s, o) in [("A","B"),("A","C"),("B","A"),("B","C"),("C","A"),("C","B")] {
        assert!(inferred(&mut r, s, "sibling", o), "{s} should be sibling of {o}");
    }
    for x in ["A","B","C"] {
        assert!(!inferred(&mut r, x, "sibling", x), "{x} should not be sibling of itself");
    }
}

#[test]
fn fc_multi_conclusion() {
    // One rule head produces two distinct conclusions
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "marriedTo", "B");

    let married = enc(&r, "marriedTo");
    let spouse = enc(&r, "spouse");
    let partner = enc(&r, "partner");

    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(married), Term::Variable("Y".into()))],
        vec![
            (Term::Variable("X".into()), Term::Constant(spouse), Term::Variable("Y".into())),
            (Term::Variable("X".into()), Term::Constant(partner), Term::Variable("Y".into())),
        ],
    ));

    r.infer_new_facts_semi_naive();

    assert!(inferred(&mut r, "A", "spouse", "B"));
    assert!(inferred(&mut r, "A", "partner", "B"));
}

#[test]
fn fc_diamond_ancestor() {
    // Diamond: A→B→D and A→C→D; A should be ancestor of D via both paths
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "B");
    r.add_abox_triple("A", "parent", "C");
    r.add_abox_triple("B", "parent", "D");
    r.add_abox_triple("C", "parent", "D");

    let parent = enc(&r, "parent");
    let ancestor = enc(&r, "ancestor");

    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(parent), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into()))],
    ));
    r.add_rule(rule(
        vec![
            (Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into())),
            (Term::Variable("Y".into()), Term::Constant(ancestor), Term::Variable("Z".into())),
        ],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Z".into()))],
    ));

    r.infer_new_facts_semi_naive();

    assert!(inferred(&mut r, "A", "ancestor", "D"), "A should be ancestor of D via B and C");
    assert!(inferred(&mut r, "B", "ancestor", "D"));
    assert!(inferred(&mut r, "C", "ancestor", "D"));
    assert!(!inferred(&mut r, "A", "ancestor", "A"), "A should not be its own ancestor");
    assert!(!inferred(&mut r, "D", "ancestor", "A"), "D should not be ancestor of A");
}

#[test]
fn fc_disconnected_graphs() {
    // Two separate parent chains must not cross-contaminate
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "B");
    r.add_abox_triple("X", "parent", "Y");

    let parent = enc(&r, "parent");
    let ancestor = enc(&r, "ancestor");

    r.add_rule(rule(
        vec![(Term::Variable("P".into()), Term::Constant(parent), Term::Variable("Q".into()))],
        vec![(Term::Variable("P".into()), Term::Constant(ancestor), Term::Variable("Q".into()))],
    ));

    r.infer_new_facts_semi_naive();

    assert!(inferred(&mut r, "A", "ancestor", "B"));
    assert!(inferred(&mut r, "X", "ancestor", "Y"));
    assert!(!inferred(&mut r, "A", "ancestor", "Y"), "A should not be ancestor of Y (different graph)");
    assert!(!inferred(&mut r, "X", "ancestor", "B"), "X should not be ancestor of B (different graph)");
}

#[test]
fn fc_no_matching_facts() {
    // Rules that match no facts should produce no inferred triples
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "likes", "B");

    let parent = enc(&r, "parent");
    let ancestor = enc(&r, "ancestor");

    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(parent), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into()))],
    ));

    let new_facts = r.infer_new_facts_semi_naive();

    assert!(new_facts.is_empty(), "No facts should be inferred when no premise matches");
}

#[test]
fn fc_idempotent() {
    // Running inference a second time must not add duplicate triples
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "B");

    let parent = enc(&r, "parent");
    let ancestor = enc(&r, "ancestor");

    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(parent), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into()))],
    ));

    r.infer_new_facts_semi_naive();
    let second_round = r.infer_new_facts_semi_naive();

    assert!(second_round.is_empty(), "Second inference pass should derive nothing new");
    let results = r.query_abox(Some("A"), Some("ancestor"), Some("B"));
    assert_eq!(results.len(), 1, "Exactly one ancestor triple should exist, not duplicates");
}

#[test]
fn fc_uncle_derived() {
    // Two-stage: sibling derived first, then uncle derived from sibling + parent
    // Setup: A and B share parent P; C's parent is A → B is uncle of C
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "P");
    r.add_abox_triple("B", "parent", "P");
    r.add_abox_triple("C", "parent", "A");

    let parent = enc(&r, "parent");
    let sibling = enc(&r, "sibling");
    let uncle = enc(&r, "uncle");

    // sibling(X, Y) :- parent(X, Z), parent(Y, Z), X != Y
    r.add_rule(Rule {
        premise: vec![
            (Term::Variable("X".into()), Term::Constant(parent), Term::Variable("Z".into())),
            (Term::Variable("Y".into()), Term::Constant(parent), Term::Variable("Z".into())),
        ],
        conclusion: vec![
            (Term::Variable("X".into()), Term::Constant(sibling), Term::Variable("Y".into())),
        ],
        filters: vec![
            FilterCondition { variable: "X".into(), operator: "!=".into(), value: "Y".into() },
        ],
    });

    // uncle(U, N) :- sibling(U, Par), parent(N, Par)
    r.add_rule(rule(
        vec![
            (Term::Variable("U".into()), Term::Constant(sibling), Term::Variable("Par".into())),
            (Term::Variable("N".into()), Term::Constant(parent), Term::Variable("Par".into())),
        ],
        vec![(Term::Variable("U".into()), Term::Constant(uncle), Term::Variable("N".into()))],
    ));

    r.infer_new_facts_semi_naive();

    assert!(inferred(&mut r, "A", "sibling", "B"), "A should be sibling of B");
    assert!(inferred(&mut r, "B", "sibling", "A"), "B should be sibling of A");
    assert!(inferred(&mut r, "B", "uncle", "C"), "B should be uncle of C");
    // A is the parent of C, not uncle
    assert!(!inferred(&mut r, "A", "uncle", "C"), "A (parent) should not also be uncle of C");
}

// Backward chaining

#[test]
fn bc_direct_fact() {
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "likes", "B");

    let likes = enc(&r, "likes");
    let a = enc(&r, "A");
    let b = enc(&r, "B");

    let query = (Term::Variable("X".into()), Term::Constant(likes), Term::Variable("Y".into()));
    let results = r.backward_chaining(&query);

    assert!(bc_has(&results, "X", a));
    assert!(bc_has(&results, "Y", b));
}

#[test]
fn bc_1hop_rule() {
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "B");

    let parent = enc(&r, "parent");
    let ancestor = enc(&r, "ancestor");
    let a = enc(&r, "A");
    let b = enc(&r, "B");

    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(parent), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into()))],
    ));

    let query = (Term::Constant(a), Term::Constant(ancestor), Term::Variable("Y".into()));
    let results = r.backward_chaining(&query);

    assert!(bc_has(&results, "Y", b));
}

#[test]
fn bc_2hop_transitive() {
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "B");
    r.add_abox_triple("B", "parent", "C");

    let parent = enc(&r, "parent");
    let ancestor = enc(&r, "ancestor");
    let a = enc(&r, "A");
    let b = enc(&r, "B");
    let c = enc(&r, "C");

    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(parent), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into()))],
    ));
    r.add_rule(rule(
        vec![
            (Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into())),
            (Term::Variable("Y".into()), Term::Constant(ancestor), Term::Variable("Z".into())),
        ],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Z".into()))],
    ));

    let query = (Term::Constant(a), Term::Constant(ancestor), Term::Variable("Y".into()));
    let results = r.backward_chaining(&query);

    assert!(bc_has(&results, "Y", b), "Expected A ancestor B");
    assert!(bc_has(&results, "Y", c), "Expected A ancestor C");
}

#[test]
fn bc_3hop_transitive() {
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "B");
    r.add_abox_triple("B", "parent", "C");
    r.add_abox_triple("C", "parent", "D");

    let parent = enc(&r, "parent");
    let ancestor = enc(&r, "ancestor");
    let a = enc(&r, "A");
    let b = enc(&r, "B");
    let c = enc(&r, "C");
    let d = enc(&r, "D");

    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(parent), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into()))],
    ));
    r.add_rule(rule(
        vec![
            (Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into())),
            (Term::Variable("Y".into()), Term::Constant(ancestor), Term::Variable("Z".into())),
        ],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Z".into()))],
    ));

    let query = (Term::Constant(a), Term::Constant(ancestor), Term::Variable("Y".into()));
    let results = r.backward_chaining(&query);

    assert!(bc_has(&results, "Y", b), "Expected A ancestor B");
    assert!(bc_has(&results, "Y", c), "Expected A ancestor C");
    assert!(bc_has(&results, "Y", d), "Expected A ancestor D");
}

#[test]
fn bc_specific_target() {
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "B");
    r.add_abox_triple("B", "parent", "C");

    let parent = enc(&r, "parent");
    let ancestor = enc(&r, "ancestor");
    let a = enc(&r, "A");
    let c = enc(&r, "C");

    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(parent), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into()))],
    ));
    r.add_rule(rule(
        vec![
            (Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into())),
            (Term::Variable("Y".into()), Term::Constant(ancestor), Term::Variable("Z".into())),
        ],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Z".into()))],
    ));

    let query = (Term::Constant(a), Term::Constant(ancestor), Term::Constant(c));
    let results = r.backward_chaining(&query);

    assert!(!results.is_empty(), "Expected A ancestor C to be derivable");
}

#[test]
fn bc_no_result() {
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "B");

    let parent = enc(&r, "parent");
    let ancestor = enc(&r, "ancestor");
    let a = enc(&r, "A");
    let d = enc(&r, "D");

    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(parent), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into()))],
    ));

    let query = (Term::Constant(a), Term::Constant(ancestor), Term::Constant(d));
    let results = r.backward_chaining(&query);

    assert!(results.is_empty(), "Expected no result for A ancestor D");
}

#[test]
fn bc_multi_rule_chain() {
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "worksFor", "Corp");

    let works_for = enc(&r, "worksFor");
    let employed = enc(&r, "employed");
    let affiliated = enc(&r, "affiliated");
    let a = enc(&r, "A");
    let corp = enc(&r, "Corp");

    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(works_for), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(employed), Term::Variable("Y".into()))],
    ));
    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(employed), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(affiliated), Term::Variable("Y".into()))],
    ));

    let query = (Term::Constant(a), Term::Constant(affiliated), Term::Variable("Y".into()));
    let results = r.backward_chaining(&query);

    assert!(bc_has(&results, "Y", corp), "Expected A affiliated Corp");
}

#[test]
fn bc_sibling_join() {
    // BC should find a sibling via a 2-premise join rule
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "P");
    r.add_abox_triple("B", "parent", "P");

    let parent = enc(&r, "parent");
    let sibling = enc(&r, "sibling");
    let b = enc(&r, "B");

    r.add_rule(rule(
        vec![
            (Term::Variable("X".into()), Term::Constant(parent), Term::Variable("Z".into())),
            (Term::Variable("Y".into()), Term::Constant(parent), Term::Variable("Z".into())),
        ],
        vec![(Term::Variable("X".into()), Term::Constant(sibling), Term::Variable("Y".into()))],
    ));

    let a_id = enc(&r, "A");
    let query = (Term::Constant(a_id), Term::Constant(sibling), Term::Variable("Y".into()));
    let results = r.backward_chaining(&query);

    assert!(bc_has(&results, "Y", b), "BC should find A sibling B via join");
}

#[test]
fn bc_full_scan() {
    // Fully variable query should return bindings for all matching base facts
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "B");
    r.add_abox_triple("C", "parent", "D");

    let parent = enc(&r, "parent");
    let a = enc(&r, "A");
    let b = enc(&r, "B");
    let c = enc(&r, "C");
    let d = enc(&r, "D");

    let query = (
        Term::Variable("S".into()),
        Term::Constant(parent),
        Term::Variable("O".into()),
    );
    let results = r.backward_chaining(&query);

    assert!(bc_has(&results, "S", a), "Should bind S to A");
    assert!(bc_has(&results, "O", b), "Should bind O to B");
    assert!(bc_has(&results, "S", c), "Should bind S to C");
    assert!(bc_has(&results, "O", d), "Should bind O to D");
}

#[test]
fn bc_no_spurious_negative() {
    // BC must not return results for a predicate with no facts or applicable rules
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "B");

    let unknown = enc(&r, "unknown");

    let query = (
        Term::Variable("X".into()),
        Term::Constant(unknown),
        Term::Variable("Y".into()),
    );
    let results = r.backward_chaining(&query);

    assert!(results.is_empty(), "BC should return nothing for unknown predicate");
}

// =====================================================================
// Probabilistic reasoning tests
// =====================================================================

use shared::probabilistic_rule::{ProbabilisticRule, ProbCombination};

#[test]
fn prob_transitive_independent_combination() {
    // A related B (0.8), B related C (0.7)
    // Rule: ?X related ?Y, ?Y related ?Z => ?X related ?Z (independent combination)
    // Expected: A related C with probability 0.8 * 0.7 = 0.56
    let mut r = Reasoner::new();
    r.add_probabilistic_triple("A", "related", "B", 0.8);
    r.add_probabilistic_triple("B", "related", "C", 0.7);

    let related = enc(&r, "related");

    let prob_rule = ProbabilisticRule {
        rule: Rule {
            premise: vec![
                (Term::Variable("X".into()), Term::Constant(related), Term::Variable("Y".into())),
                (Term::Variable("Y".into()), Term::Constant(related), Term::Variable("Z".into())),
            ],
            filters: vec![],
            conclusion: vec![
                (Term::Variable("X".into()), Term::Constant(related), Term::Variable("Z".into())),
            ],
        },
        combination: ProbCombination::Independent,
        threshold: 0.0,
        rule_confidence: 1.0,
    };

    r.add_probabilistic_rule(prob_rule);
    let inferred = r.infer_new_facts_probabilistic_semi_naive();

    // Should infer A related C
    let a = enc(&r, "A");
    let c = enc(&r, "C");
    let a_related_c = inferred.iter().any(|t| t.subject == a && t.predicate == related && t.object == c);
    assert!(a_related_c, "Should infer A related C");

    // Check probability is approximately 0.56
    let triple = shared::triple::Triple { subject: a, predicate: related, object: c };
    let prob = r.probability_store.get_probability(&triple);
    assert!((prob - 0.56).abs() < 1e-6, "Probability should be ~0.56, got {}", prob);
}

#[test]
fn prob_threshold_pruning() {
    // Same as above but with threshold 0.6 — should NOT infer A related C (prob 0.56 < 0.6)
    let mut r = Reasoner::new();
    r.add_probabilistic_triple("A", "related", "B", 0.8);
    r.add_probabilistic_triple("B", "related", "C", 0.7);

    let related = enc(&r, "related");

    let prob_rule = ProbabilisticRule {
        rule: Rule {
            premise: vec![
                (Term::Variable("X".into()), Term::Constant(related), Term::Variable("Y".into())),
                (Term::Variable("Y".into()), Term::Constant(related), Term::Variable("Z".into())),
            ],
            filters: vec![],
            conclusion: vec![
                (Term::Variable("X".into()), Term::Constant(related), Term::Variable("Z".into())),
            ],
        },
        combination: ProbCombination::Independent,
        threshold: 0.6,
        rule_confidence: 1.0,
    };

    r.add_probabilistic_rule(prob_rule);
    let inferred = r.infer_new_facts_probabilistic_semi_naive();

    let a = enc(&r, "A");
    let c = enc(&r, "C");
    let a_related_c = inferred.iter().any(|t| t.subject == a && t.predicate == related && t.object == c);
    assert!(!a_related_c, "Should NOT infer A related C (prob 0.56 < threshold 0.6)");
}

#[test]
fn prob_max_semantics_multiple_paths() {
    // A related B (0.6), A related C (0.9), B related D (0.8), C related D (0.5)
    // Rule: ?X related ?Y, ?Y related ?Z => ?X related ?Z
    // Path 1: A->B->D = 0.6*0.8 = 0.48
    // Path 2: A->C->D = 0.9*0.5 = 0.45
    // Max semantics: A related D should have prob max(0.48, 0.45) = 0.48
    let mut r = Reasoner::new();
    r.add_probabilistic_triple("A", "related", "B", 0.6);
    r.add_probabilistic_triple("A", "related", "C", 0.9);
    r.add_probabilistic_triple("B", "related", "D", 0.8);
    r.add_probabilistic_triple("C", "related", "D", 0.5);

    let related = enc(&r, "related");

    let prob_rule = ProbabilisticRule {
        rule: Rule {
            premise: vec![
                (Term::Variable("X".into()), Term::Constant(related), Term::Variable("Y".into())),
                (Term::Variable("Y".into()), Term::Constant(related), Term::Variable("Z".into())),
            ],
            filters: vec![],
            conclusion: vec![
                (Term::Variable("X".into()), Term::Constant(related), Term::Variable("Z".into())),
            ],
        },
        combination: ProbCombination::Independent,
        threshold: 0.0,
        rule_confidence: 1.0,
    };

    r.add_probabilistic_rule(prob_rule);
    r.infer_new_facts_probabilistic_semi_naive();

    let a = enc(&r, "A");
    let d = enc(&r, "D");
    let triple = shared::triple::Triple { subject: a, predicate: related, object: d };
    let prob = r.probability_store.get_probability(&triple);
    assert!((prob - 0.48).abs() < 1e-6, "Max probability should be ~0.48, got {}", prob);
}

#[test]
fn prob_noisy_or_combination() {
    // Test NoisyOr combination mode
    let mut r = Reasoner::new();
    r.add_probabilistic_triple("sensor1", "detects", "fire", 0.5);
    r.add_probabilistic_triple("sensor2", "detects", "fire", 0.5);

    let detects = enc(&r, "detects");
    let alarm = enc(&r, "alarm");

    // If any sensor detects fire, raise alarm (NoisyOr = 1 - (1-0.5)*(1-0.5) = 0.75)
    // We need two separate single-premise rules or a different setup
    // Instead: single premise rule with NoisyOr
    let prob_rule = ProbabilisticRule {
        rule: Rule {
            premise: vec![
                (Term::Variable("X".into()), Term::Constant(detects), Term::Variable("Y".into())),
            ],
            filters: vec![],
            conclusion: vec![
                (Term::Variable("Y".into()), Term::Constant(alarm), Term::Variable("Y".into())),
            ],
        },
        combination: ProbCombination::NoisyOr,
        threshold: 0.0,
        rule_confidence: 1.0,
    };

    r.add_probabilistic_rule(prob_rule);
    r.infer_new_facts_probabilistic_semi_naive();

    let fire = enc(&r, "fire");
    let triple = shared::triple::Triple { subject: fire, predicate: alarm, object: fire };
    let prob = r.probability_store.get_probability(&triple);
    // With single premise per firing, NoisyOr on single value = that value
    // The max semantics will pick max(0.5, 0.5) = 0.5
    assert!((prob - 0.5).abs() < 1e-6, "NoisyOr single-premise prob should be 0.5, got {}", prob);
}

#[test]
fn prob_min_combination() {
    let mut r = Reasoner::new();
    r.add_probabilistic_triple("A", "knows", "B", 0.9);
    r.add_probabilistic_triple("B", "trusts", "C", 0.6);

    let knows = enc(&r, "knows");
    let trusts = enc(&r, "trusts");
    let recommends = enc(&r, "recommends");

    // If A knows B and B trusts C, then A recommends C (min combination)
    let prob_rule = ProbabilisticRule {
        rule: Rule {
            premise: vec![
                (Term::Variable("X".into()), Term::Constant(knows), Term::Variable("Y".into())),
                (Term::Variable("Y".into()), Term::Constant(trusts), Term::Variable("Z".into())),
            ],
            filters: vec![],
            conclusion: vec![
                (Term::Variable("X".into()), Term::Constant(recommends), Term::Variable("Z".into())),
            ],
        },
        combination: ProbCombination::Min,
        threshold: 0.0,
        rule_confidence: 1.0,
    };

    r.add_probabilistic_rule(prob_rule);
    r.infer_new_facts_probabilistic_semi_naive();

    let a = enc(&r, "A");
    let c = enc(&r, "C");
    let triple = shared::triple::Triple { subject: a, predicate: recommends, object: c };
    let prob = r.probability_store.get_probability(&triple);
    assert!((prob - 0.6).abs() < 1e-6, "Min probability should be 0.6, got {}", prob);
}

#[test]
fn prob_rule_confidence_scaling() {
    let mut r = Reasoner::new();
    r.add_probabilistic_triple("A", "related", "B", 0.8);

    let related = enc(&r, "related");
    let connected = enc(&r, "connected");

    let prob_rule = ProbabilisticRule {
        rule: Rule {
            premise: vec![
                (Term::Variable("X".into()), Term::Constant(related), Term::Variable("Y".into())),
            ],
            filters: vec![],
            conclusion: vec![
                (Term::Variable("X".into()), Term::Constant(connected), Term::Variable("Y".into())),
            ],
        },
        combination: ProbCombination::Independent,
        threshold: 0.0,
        rule_confidence: 0.9,
    };

    r.add_probabilistic_rule(prob_rule);
    r.infer_new_facts_probabilistic_semi_naive();

    let a = enc(&r, "A");
    let b = enc(&r, "B");
    let triple = shared::triple::Triple { subject: a, predicate: connected, object: b };
    let prob = r.probability_store.get_probability(&triple);
    // 0.8 * 0.9 = 0.72
    assert!((prob - 0.72).abs() < 1e-6, "Probability should be 0.72, got {}", prob);
}

#[test]
fn prob_classical_rules_still_work() {
    // Regression test: classical (non-probabilistic) rules should still work
    let mut r = Reasoner::new();
    r.add_abox_triple("A", "parent", "B");

    let parent = enc(&r, "parent");
    let ancestor = enc(&r, "ancestor");

    r.add_rule(rule(
        vec![(Term::Variable("X".into()), Term::Constant(parent), Term::Variable("Y".into()))],
        vec![(Term::Variable("X".into()), Term::Constant(ancestor), Term::Variable("Y".into()))],
    ));

    r.infer_new_facts_semi_naive();
    assert!(inferred(&mut r, "A", "ancestor", "B"));
}

#[test]
fn prob_weighted_combination() {
    let mut r = Reasoner::new();
    r.add_probabilistic_triple("A", "related", "B", 0.8);
    r.add_probabilistic_triple("B", "related", "C", 0.7);

    let related = enc(&r, "related");
    let linked = enc(&r, "linked");

    let prob_rule = ProbabilisticRule {
        rule: Rule {
            premise: vec![
                (Term::Variable("X".into()), Term::Constant(related), Term::Variable("Y".into())),
                (Term::Variable("Y".into()), Term::Constant(related), Term::Variable("Z".into())),
            ],
            filters: vec![],
            conclusion: vec![
                (Term::Variable("X".into()), Term::Constant(linked), Term::Variable("Z".into())),
            ],
        },
        combination: ProbCombination::Weighted(0.5),
        threshold: 0.0,
        rule_confidence: 1.0,
    };

    r.add_probabilistic_rule(prob_rule);
    r.infer_new_facts_probabilistic_semi_naive();

    let a = enc(&r, "A");
    let c = enc(&r, "C");
    let triple = shared::triple::Triple { subject: a, predicate: linked, object: c };
    let prob = r.probability_store.get_probability(&triple);
    // 0.5 * 0.8 * 0.7 = 0.28
    assert!((prob - 0.28).abs() < 1e-6, "Weighted probability should be 0.28, got {}", prob);
}

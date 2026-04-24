/*
 * Copyright © 2026 Volodymyr Kadzhaia
 * Copyright © 2026 Pieter Bonte
 * KU Leuven — Stream Intelligence Lab, Belgium
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * you can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Integration tests for Candle-backed `ML.PREDICT` going through the full
//! `process_rule_definition` runtime path.

use kolibrie::parser::process_rule_definition;
use kolibrie::sparql_database::SparqlDatabase;

fn tmp_model_path(name: &str) -> String {
    let id = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("/tmp/kolibrie_ml_predict_{}_{}_{}.bin", name, id, nanos)
}

fn populate_multiclass_db(db: &mut SparqlDatabase) {
    for (idx, label, features) in [
        ("s0", "A", [1.0, 0.0, 0.0]),
        ("s1", "A", [1.0, 0.0, 0.0]),
        ("s2", "B", [0.0, 1.0, 0.0]),
        ("s3", "B", [0.0, 1.0, 0.0]),
        ("s4", "C", [0.0, 0.0, 1.0]),
        ("s5", "C", [0.0, 0.0, 1.0]),
    ] {
        db.add_triple_parts(idx, "http://example.org/x0", &features[0].to_string());
        db.add_triple_parts(idx, "http://example.org/x1", &features[1].to_string());
        db.add_triple_parts(idx, "http://example.org/x2", &features[2].to_string());
        db.add_triple_parts(idx, "http://example.org/gold", label);
    }
}

fn populate_binary_db(db: &mut SparqlDatabase) {
    for (idx, label, features) in [
        ("t0", "1", [1.0, 1.0]),
        ("t1", "1", [1.0, 1.0]),
        ("t2", "0", [0.0, 0.0]),
        ("t3", "0", [0.0, 0.0]),
    ] {
        db.add_triple_parts(idx, "http://example.org/x0", &features[0].to_string());
        db.add_triple_parts(idx, "http://example.org/x1", &features[1].to_string());
        db.add_triple_parts(idx, "http://example.org/gold", label);
    }
}

fn clear_materialized_predicate(db: &mut SparqlDatabase, predicate: &str) {
    if let Some(triples) = db.neural_materialized_triples.remove(predicate) {
        for triple in triples {
            db.delete_triple(&triple);
        }
    }
}

fn train_multiclass(db: &mut SparqlDatabase, save_path: &str) {
    let program = format!(
        r#"
PREFIX ex: <http://example.org/>

MODEL "digit_model" {{
    ARCH MLP {{ HIDDEN [16, 8] }}
    OUTPUT EXCLUSIVE {{ "A", "B", "C" }}
}}

NEURAL RELATION ex:predictedDigit USING MODEL "digit_model" {{
    INPUT {{
        ?sample ex:x0 ?x0 .
        ?sample ex:x1 ?x1 .
        ?sample ex:x2 ?x2 .
    }}
    FEATURES {{ ?x0, ?x1, ?x2 }}
}}

TRAIN NEURAL RELATION ex:predictedDigit {{
    DATA {{ ?sample ex:gold ?label . }}
    LABEL ?label
    TARGET {{ ?sample ex:predictedDigit ?label }}
    LOSS cross_entropy
    OPTIMIZER adam
    LEARNING_RATE 0.1
    EPOCHS 80
    BATCH_SIZE 4
    SAVE_TO "{save_path}"
}}
"#,
        save_path = save_path,
    );
    kolibrie::neural_relations::execute_neural_program(db, &program)
        .expect("training program failed");
    clear_materialized_predicate(db, "http://example.org/predictedDigit");
}

fn train_binary(db: &mut SparqlDatabase, save_path: &str) {
    let program = format!(
        r#"
PREFIX ex: <http://example.org/>

MODEL "fraud_model" {{
    ARCH MLP {{ HIDDEN [8, 4] }}
    OUTPUT BINARY {{ true }}
}}

NEURAL RELATION ex:isFraud USING MODEL "fraud_model" {{
    INPUT {{
        ?sample ex:x0 ?x0 .
        ?sample ex:x1 ?x1 .
    }}
    FEATURES {{ ?x0, ?x1 }}
}}

TRAIN NEURAL RELATION ex:isFraud {{
    DATA {{ ?sample ex:gold ?label . }}
    LABEL ?label
    TARGET {{ ?sample ex:isFraud true }}
    LOSS binary_cross_entropy
    OPTIMIZER adam
    LEARNING_RATE 0.1
    EPOCHS 80
    BATCH_SIZE 2
    SAVE_TO "{save_path}"
}}
"#,
        save_path = save_path,
    );
    kolibrie::neural_relations::execute_neural_program(db, &program)
        .expect("binary training program failed");
    clear_materialized_predicate(db, "http://example.org/isFraud");
}

/// Returns the decoded object for the first triple matching (subject, predicate).
fn lookup_object(db: &SparqlDatabase, subject_iri: &str, predicate_iri: &str) -> Option<String> {
    let dict = db.dictionary.read().unwrap();
    let s = *dict.string_to_id.get(subject_iri)?;
    let p = *dict.string_to_id.get(predicate_iri)?;
    for t in &db.triples {
        if t.subject == s && t.predicate == p {
            return dict.decode(t.object).map(str::to_string);
        }
    }
    None
}

fn count_triples_with_predicate(db: &SparqlDatabase, predicate_iri: &str) -> usize {
    let dict = db.dictionary.read().unwrap();
    let Some(&pred_id) = dict.string_to_id.get(predicate_iri) else { return 0; };
    db.triples.iter().filter(|t| t.predicate == pred_id).count()
}

/// Test 1: head-only rule (fraud-style). `?label` is only in CONSTRUCT; WHERE
/// binds only `?sample`. Assert predictions get materialized.
#[test]
fn head_only_output_variable_materializes() {
    let mut db = SparqlDatabase::new();
    populate_multiclass_db(&mut db);
    train_multiclass(&mut db, &tmp_model_path("head_only"));

    let rule = r#"
PREFIX ex: <http://example.org/>

RULE :PredictDigit :-
CONSTRUCT {
    ?sample ex:predictedDigit ?label .
}
WHERE {
    ?sample ex:gold ?gold .
}
ML.PREDICT(MODEL "digit_model",
    INPUT {
        SELECT ?sample ?x0 ?x1 ?x2
        WHERE {
            ?sample ex:x0 ?x0 .
            ?sample ex:x1 ?x1 .
            ?sample ex:x2 ?x2 .
        }
    },
    OUTPUT ?label
)
"#;

    process_rule_definition(rule, &mut db).expect("rule processing failed");

    for (subject, expected_label) in [
        ("s0", "A"),
        ("s2", "B"),
        ("s4", "C"),
    ] {
        let got = lookup_object(&db, subject, "http://example.org/predictedDigit");
        assert_eq!(
            got.as_deref(),
            Some(expected_label),
            "wrong prediction for {}",
            subject
        );
    }
}

/// Test 3: INPUT FILTER preserved — only rows with x0 > 0 get predictions.
#[test]
fn input_filter_preserved() {
    let mut db = SparqlDatabase::new();
    populate_multiclass_db(&mut db);
    train_multiclass(&mut db, &tmp_model_path("filter"));

    let rule = r#"
PREFIX ex: <http://example.org/>

RULE :PredictSelective :-
CONSTRUCT {
    ?sample ex:predictedDigit ?label .
}
WHERE {
    ?sample ex:gold ?gold .
}
ML.PREDICT(MODEL "digit_model",
    INPUT {
        SELECT ?sample ?x0 ?x1 ?x2
        WHERE {
            ?sample ex:x0 ?x0 .
            ?sample ex:x1 ?x1 .
            ?sample ex:x2 ?x2 .
            FILTER (?x0 > 0)
        }
    },
    OUTPUT ?label
)
"#;
    process_rule_definition(rule, &mut db).expect("rule processing failed");

    // Only s0, s1 have x0 > 0, so exactly two predictedDigit triples should exist.
    let count = count_triples_with_predicate(&db, "http://example.org/predictedDigit");
    assert_eq!(count, 2, "FILTER did not restrict predictions correctly");

    // And the filtered-out sample has no prediction.
    assert!(lookup_object(&db, "s2", "http://example.org/predictedDigit").is_none());
}

/// Test 4 + 14: binary always-emit contract + companion `_prob` IRI.
#[test]
fn binary_always_emit_with_probability_companion() {
    let mut db = SparqlDatabase::new();
    populate_binary_db(&mut db);
    train_binary(&mut db, &tmp_model_path("binary"));

    let rule = r#"
PREFIX ex: <http://example.org/>

RULE :Flag :-
CONSTRUCT {
    ?sample ex:isFraud ?flag .
}
WHERE {
    ?sample ex:gold ?gold .
}
ML.PREDICT(MODEL "fraud_model",
    INPUT {
        SELECT ?sample ?x0 ?x1
        WHERE {
            ?sample ex:x0 ?x0 .
            ?sample ex:x1 ?x1 .
        }
    },
    OUTPUT ?flag
)
"#;
    process_rule_definition(rule, &mut db).expect("rule processing failed");

    // Every row, positive and negative, gets `isFraud true` (new ML.PREDICT contract).
    let fraud_count = count_triples_with_predicate(&db, "http://example.org/isFraud");
    assert_eq!(fraud_count, 4, "binary ML.PREDICT should emit for every row");

    // Companion probability predicate (locked IRI rule: append _prob to full IRI).
    let prob_count = count_triples_with_predicate(&db, "http://example.org/isFraud_prob");
    assert_eq!(prob_count, 4);

    // The probability for the positive training rows should be higher than for negatives.
    let prob_pos = lookup_object(&db, "t0", "http://example.org/isFraud_prob")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap();
    let prob_neg = lookup_object(&db, "t2", "http://example.org/isFraud_prob")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap();
    assert!(
        prob_pos > prob_neg,
        "positive row prob {} should exceed negative row prob {}",
        prob_pos,
        prob_neg
    );
}

/// Test 6: Rerun cleanup. Train once, run ML.PREDICT, change one feature row,
/// rerun the rule with new data — old prediction should be gone.
#[test]
fn rerun_cleans_stale_predictions() {
    let mut db = SparqlDatabase::new();
    populate_multiclass_db(&mut db);
    train_multiclass(&mut db, &tmp_model_path("rerun"));

    let rule = r#"
PREFIX ex: <http://example.org/>

RULE :PredictDigit :-
CONSTRUCT {
    ?sample ex:predictedDigit ?label .
}
WHERE {
    ?sample ex:gold ?gold .
}
ML.PREDICT(MODEL "digit_model",
    INPUT {
        SELECT ?sample ?x0 ?x1 ?x2
        WHERE {
            ?sample ex:x0 ?x0 .
            ?sample ex:x1 ?x1 .
            ?sample ex:x2 ?x2 .
        }
    },
    OUTPUT ?label
)
"#;
    process_rule_definition(rule, &mut db).expect("first run failed");
    let after_first = count_triples_with_predicate(&db, "http://example.org/predictedDigit");
    assert_eq!(after_first, 6);

    // Flip s0's feature values so it now looks like a C.
    db.delete_triple_parts("s0", "http://example.org/x0", "1");
    db.delete_triple_parts("s0", "http://example.org/x1", "0");
    db.delete_triple_parts("s0", "http://example.org/x2", "0");
    db.add_triple_parts("s0", "http://example.org/x0", "0");
    db.add_triple_parts("s0", "http://example.org/x1", "0");
    db.add_triple_parts("s0", "http://example.org/x2", "1");

    process_rule_definition(rule, &mut db).expect("second run failed");
    let after_second = count_triples_with_predicate(&db, "http://example.org/predictedDigit");
    assert_eq!(
        after_second, 6,
        "rerun should replace triples, not accumulate them"
    );

    // s0 should now predict C.
    assert_eq!(
        lookup_object(&db, "s0", "http://example.org/predictedDigit").as_deref(),
        Some("C"),
        "s0's prediction should have updated to C"
    );
}

/// Test 11: Preserves non-ML conclusions. CONSTRUCT has both an ML triple and
/// a static one. Both should appear after execution.
#[test]
fn preserves_non_ml_conclusions() {
    let mut db = SparqlDatabase::new();
    populate_multiclass_db(&mut db);
    train_multiclass(&mut db, &tmp_model_path("mixed_conclusions"));

    let rule = r#"
PREFIX ex: <http://example.org/>

RULE :PredictAndAlert :-
CONSTRUCT {
    ?sample ex:predictedDigit ?label ;
            ex:alert "processed" .
}
WHERE {
    ?sample ex:gold ?gold .
}
ML.PREDICT(MODEL "digit_model",
    INPUT {
        SELECT ?sample ?x0 ?x1 ?x2
        WHERE {
            ?sample ex:x0 ?x0 .
            ?sample ex:x1 ?x1 .
            ?sample ex:x2 ?x2 .
        }
    },
    OUTPUT ?label
)
"#;
    process_rule_definition(rule, &mut db).expect("rule processing failed");

    // ML triple from side-effect:
    assert!(lookup_object(&db, "s0", "http://example.org/predictedDigit").is_some());
    // Static triple from Datalog:
    assert!(lookup_object(&db, "s0", "http://example.org/alert").is_some());
}

/// Test 15: Empty INPUT rerun clears stale predictions. Run 1 materializes
/// triples; remove input data; run 2 should remove all prior triples.
#[test]
fn empty_input_rerun_clears_stale() {
    let mut db = SparqlDatabase::new();
    populate_multiclass_db(&mut db);
    train_multiclass(&mut db, &tmp_model_path("empty_rerun"));

    let rule = r#"
PREFIX ex: <http://example.org/>

RULE :PredictDigit :-
CONSTRUCT {
    ?sample ex:predictedDigit ?label .
}
WHERE {
    ?sample ex:gold ?gold .
}
ML.PREDICT(MODEL "digit_model",
    INPUT {
        SELECT ?sample ?x0 ?x1 ?x2
        WHERE {
            ?sample ex:x0 ?x0 .
            ?sample ex:x1 ?x1 .
            ?sample ex:x2 ?x2 .
        }
    },
    OUTPUT ?label
)
"#;
    process_rule_definition(rule, &mut db).expect("first run failed");
    assert!(count_triples_with_predicate(&db, "http://example.org/predictedDigit") > 0);

    // Scrub ALL the feature triples so INPUT returns zero rows.
    let subjects: Vec<&'static str> = vec!["s0", "s1", "s2", "s3", "s4", "s5"];
    for s in subjects {
        for (pred, val) in [
            ("http://example.org/x0", "1"),
            ("http://example.org/x0", "0"),
            ("http://example.org/x1", "1"),
            ("http://example.org/x1", "0"),
            ("http://example.org/x2", "1"),
            ("http://example.org/x2", "0"),
        ] {
            db.delete_triple_parts(s, pred, val);
        }
    }

    process_rule_definition(rule, &mut db).expect("second run failed");
    assert_eq!(
        count_triples_with_predicate(&db, "http://example.org/predictedDigit"),
        0,
        "empty INPUT should wipe previously-materialized predictions"
    );
}

/// Test 16: Unused OUTPUT variable. Rule conclusion does not reference `?label`.
#[test]
fn unused_output_variable_errors() {
    let mut db = SparqlDatabase::new();
    populate_multiclass_db(&mut db);
    train_multiclass(&mut db, &tmp_model_path("unused"));

    let rule = r#"
PREFIX ex: <http://example.org/>

RULE :NoRef :-
CONSTRUCT {
    ?sample ex:tagged "x" .
}
WHERE {
    ?sample ex:gold ?gold .
}
ML.PREDICT(MODEL "digit_model",
    INPUT {
        SELECT ?sample ?x0 ?x1 ?x2
        WHERE {
            ?sample ex:x0 ?x0 .
            ?sample ex:x1 ?x1 .
            ?sample ex:x2 ?x2 .
        }
    },
    OUTPUT ?label
)
"#;
    let result = process_rule_definition(rule, &mut db);
    match result {
        Err(msg) => {
            assert!(
                msg.contains("is not referenced by any conclusion triple"),
                "expected unused-output error, got: {}",
                msg
            );
        }
        Ok(_) => panic!("expected Err for unused OUTPUT variable"),
    }
}

/// Test 10: Model-name mismatch. The rule names a different MODEL string than
/// what the registered NEURAL RELATION uses.
#[test]
fn model_name_mismatch_errors() {
    let mut db = SparqlDatabase::new();
    populate_multiclass_db(&mut db);
    train_multiclass(&mut db, &tmp_model_path("mismatch"));

    let rule = r#"
PREFIX ex: <http://example.org/>

RULE :Mismatch :-
CONSTRUCT {
    ?sample ex:predictedDigit ?label .
}
WHERE {
    ?sample ex:gold ?gold .
}
ML.PREDICT(MODEL "other_model",
    INPUT {
        SELECT ?sample ?x0 ?x1 ?x2
        WHERE {
            ?sample ex:x0 ?x0 .
            ?sample ex:x1 ?x1 .
            ?sample ex:x2 ?x2 .
        }
    },
    OUTPUT ?label
)
"#;
    let result = process_rule_definition(rule, &mut db);
    match result {
        Err(msg) => {
            assert!(
                msg.contains("other_model") && msg.contains("digit_model"),
                "expected model-mismatch error, got: {}",
                msg
            );
        }
        Ok(_) => panic!("expected Err for model-name mismatch"),
    }
}

/// Test 12: Missing anchor in INPUT SELECT.
#[test]
fn missing_anchor_in_input_select_errors() {
    let mut db = SparqlDatabase::new();
    populate_multiclass_db(&mut db);
    train_multiclass(&mut db, &tmp_model_path("missing_anchor"));

    let rule = r#"
PREFIX ex: <http://example.org/>

RULE :MissingAnchor :-
CONSTRUCT {
    ?sample ex:predictedDigit ?label .
}
WHERE {
    ?sample ex:gold ?gold .
}
ML.PREDICT(MODEL "digit_model",
    INPUT {
        SELECT ?x0 ?x1 ?x2
        WHERE {
            ?sample ex:x0 ?x0 .
            ?sample ex:x1 ?x1 .
            ?sample ex:x2 ?x2 .
        }
    },
    OUTPUT ?label
)
"#;
    let result = process_rule_definition(rule, &mut db);
    match result {
        Err(msg) => {
            assert!(
                msg.contains("?sample") && msg.contains("not bound by INPUT"),
                "expected bindings-coverage error, got: {}",
                msg
            );
        }
        Ok(_) => panic!("expected Err for missing anchor in INPUT SELECT"),
    }
}

/// Test 13: Multiple conclusion predicates for one OUTPUT ?var.
#[test]
fn multiple_conclusion_predicates_errors() {
    let mut db = SparqlDatabase::new();
    populate_multiclass_db(&mut db);
    train_multiclass(&mut db, &tmp_model_path("multi_pred"));

    let rule = r#"
PREFIX ex: <http://example.org/>

RULE :MultiPred :-
CONSTRUCT {
    ?sample ex:predictedDigit ?label ;
            ex:otherLabel ?label .
}
WHERE {
    ?sample ex:gold ?gold .
}
ML.PREDICT(MODEL "digit_model",
    INPUT {
        SELECT ?sample ?x0 ?x1 ?x2
        WHERE {
            ?sample ex:x0 ?x0 .
            ?sample ex:x1 ?x1 .
            ?sample ex:x2 ?x2 .
        }
    },
    OUTPUT ?label
)
"#;
    let result = process_rule_definition(rule, &mut db);
    match result {
        Err(msg) => {
            assert!(
                msg.contains("multiple conclusion predicates"),
                "expected single-predicate invariant error, got: {}",
                msg
            );
        }
        Ok(_) => panic!("expected Err for multiple conclusion predicates"),
    }
}

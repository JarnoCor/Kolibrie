//! A module containing useful structs and methods for performing
//! Incremental View Maintenance (IVM) in various ways.
//!
//! There is currently only one way of performing IVM, using a counting-based approach.
//! More methods are coming soon.

mod counting;
mod dred;
mod fbf;
pub mod imars;

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;

pub use counting::CountingMaintenance;
pub use counting::Multiset;
pub use counting::multiset_add;
pub use counting::multiset_subtract;

pub use dred::DRedMaintenance;

pub use fbf::FBFMaintenance;

use shared::dictionary::Dictionary;
use shared::rule::Rule;
use shared::terms::Term;
use shared::terms::TriplePattern;
use shared::triple::Triple;

use crate::reasoning::convert_string_binding_to_u32;
use crate::reasoning::rules::evaluate_filters;
use crate::reasoning::rules::join_premise_with_hash_join;
use crate::reasoning::rules::join_rule;

/// Convert rule evaluation to use the optimized hash join
/// Optimised hash-join: return early when build map is empty (see for loop)
fn find_premise_solutions(dict: &Dictionary, rule: &Rule, all_facts: &Vec<Triple>) -> Vec<HashMap<String, u32>> {
    if rule.premise.is_empty() {
        return Vec::new();
    }

    // Initialise current bindings to an empty vector
    let mut cur_binding_set = vec![BTreeMap::new()];

    // Process each premise with the build hash-table using the optimized join
    for premise in &rule.premise {
        cur_binding_set = join_premise_with_hash_join(premise, all_facts, cur_binding_set, dict);
        if cur_binding_set.is_empty() {
            break;
        }
    }

    // Convert results back to HashMap<String, u32>
    cur_binding_set
        .into_iter()
        .map(|binding| convert_string_binding_to_u32(&binding, dict))
        .collect()
}

/// Construct a new Triple from a conclusion pattern and bound variables
fn construct_triple(
    conclusion: &TriplePattern, 
    vars: &HashMap<String, u32>, 
    dict: &mut Dictionary
) -> Triple {
    let subject = match &conclusion.0 {
        Term::Variable(v) => {
            vars.get(v).copied().unwrap_or_else(|| {
                eprintln!("Warning: Variable '{}' not found in bindings. Available variables: {:?}", v, vars.keys().collect::<Vec<_>>());
                0
            })
        },
        Term::Constant(c) => *c,
        _ => 0
    };
    
    let predicate = match &conclusion.1 {
        Term::Variable(v) => {
            vars.get(v).copied().unwrap_or_else(|| {
                eprintln!("Warning: Variable '{}' not found in bindings. Available variables: {:?}", v, vars.keys().collect::<Vec<_>>());
                0
            })
        },
        Term::Constant(c) => *c,
        _ => 0
    };

    let object = match &conclusion.2 {
        Term::Variable(v) => {
            // Check if this variable is bound in the current context
            if let Some(&bound_value) = vars.get(v) {
                bound_value
            } else {
                // If not bound, create a new placeholder in the dictionary
                dict.encode(&format!("ml_output_placeholder_{}", v))
            }
        },
        Term::Constant(c) => *c,
        _ => 0
    };

    Triple {
        subject,
        predicate,
        object,
    }
}

fn evaluate_rule_with_restrictions(results: &mut Vec<Triple>, dict: &mut Dictionary, rule: &Rule, positive_facts: &HashSet<Triple>, positive_1: &HashSet<Triple>, positive_2: &HashSet<Triple>) {

    let bindings = join_rule(rule, positive_facts, positive_1);
    results.clear();

    for binding in bindings {
        // check if the bindings are disjoint with positive_2 or not
        if check_bindings(rule, &binding, positive_2) {
            // check if the binding adheres to the filters of the rule
            if evaluate_filters(&binding, &rule.filters, dict) {
                for conclusion in &rule.conclusion {
                    let inferred = construct_triple(conclusion, &binding, dict);

                    results.push(inferred);
                }
            }
        }
    }
}

fn check_bindings(rule: &Rule, binding: &HashMap<String, u32>, positive: &HashSet<Triple>) -> bool {
    for triple in &rule.premise {
        let subject = match &triple.0 {
            Term::Variable(v) => {
                binding.get(v).copied().unwrap_or_else(|| {
                    eprintln!("Warning: Variable '{}' not found in bindings. Available variables: {:?}", v, binding.keys().collect::<Vec<_>>());
                    0
                })
            },
            Term::Constant(c) => *c,
            _ => 0
        };

        let predicate = match &triple.1 {
            Term::Variable(v) => {
                binding.get(v).copied().unwrap_or_else(|| {
                    eprintln!("Warning: Variable '{}' not found in bindings. Available variables: {:?}", v, binding.keys().collect::<Vec<_>>());
                    0
                })
            },
            Term::Constant(c) => *c,
            _ => 0
        };

        let object = match &triple.2 {
            Term::Variable(v) => {
                binding.get(v).copied().unwrap_or_else(|| {
                    eprintln!("Warning: Variable '{}' not found in bindings. Available variables: {:?}", v, binding.keys().collect::<Vec<_>>());
                    0
                })
            },
            Term::Constant(c) => *c,
            _ => 0
        };

        let constructed = Triple {
            subject,
            predicate,
            object,
        };

        if positive.contains(&constructed) {
            return true;
        }
    }

    false
}

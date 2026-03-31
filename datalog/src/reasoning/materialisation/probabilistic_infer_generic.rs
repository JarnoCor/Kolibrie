/*
 * Copyright © 2026 Volodymyr Kadzhaia
 * Copyright © 2026 Pieter Bonte
 * KU Leuven — Stream Intelligence Lab, Belgium
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * you can obtain one at https://mozilla.org/MPL/2.0/.
 */

use shared::dictionary::Dictionary;
use shared::probabilistic_rule::ProbabilisticRule;
use shared::probability_store::ProbabilityStore;
use shared::triple::Triple;
use std::collections::HashSet;
use crate::reasoning::Reasoner;

/// Result of a single probabilistic inference round.
pub struct ProbInferResult {
    /// Newly derived triples (not previously in known_facts).
    pub new_facts: HashSet<Triple>,
    /// Whether any existing triple's probability was updated this round.
    pub probability_changed: bool,
}

/// Strategy trait for probabilistic materialisation.
///
/// Similar to `InferenceStrategy` but additionally receives and updates a
/// `ProbabilityStore`, and signals whether probabilities changed (needed
/// for fixpoint detection with monotone convergence).
pub trait ProbabilisticInferenceStrategy {
    fn infer_round(
        &mut self,
        dictionary: &mut Dictionary,
        rules: &[ProbabilisticRule],
        all_facts: &[Triple],
        known_facts: &HashSet<Triple>,
        prob_store: &mut ProbabilityStore,
    ) -> ProbInferResult;
}

impl Reasoner {
    /// Generic driver for probabilistic materialisation.
    ///
    /// Loops until no new facts are derived AND no probability changed
    /// (fixpoint with monotone convergence).
    pub fn infer_with_probabilistic_strategy<S: ProbabilisticInferenceStrategy>(
        &mut self,
        mut strat: S,
    ) -> Vec<Triple> {
        let mut all_facts: Vec<Triple> = self.index_manager.query(None, None, None);
        let mut known_facts: HashSet<Triple> = all_facts.iter().cloned().collect();
        let idx_before_inference = all_facts.len();

        loop {
            let mut dict = self.dictionary.write().unwrap();
            let result = strat.infer_round(
                &mut dict,
                &self.probabilistic_rules,
                &all_facts,
                &known_facts,
                &mut self.probability_store,
            );
            drop(dict);

            let has_new_facts = !result.new_facts.is_empty();

            for fact in result.new_facts {
                if !known_facts.contains(&fact) {
                    known_facts.insert(fact.clone());
                    self.index_manager.insert(&fact);
                    all_facts.push(fact);
                }
            }

            // Terminate when no new facts AND no probability updates
            if !has_new_facts && !result.probability_changed {
                break;
            }
        }

        all_facts.split_off(idx_before_inference)
    }
}

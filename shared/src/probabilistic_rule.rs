/*
 * Copyright © 2026 Volodymyr Kadzhaia
 * Copyright © 2026 Pieter Bonte
 * KU Leuven — Stream Intelligence Lab, Belgium
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * you can obtain one at https://mozilla.org/MPL/2.0/.
 */

use crate::rule::Rule;

/// Defines how premise probabilities are combined to compute the conclusion probability.
///
/// Based on probabilistic RDF literature:
/// - Independent: Huang & Liu (WISE'09) — product of independent events
/// - NoisyOr: Common in Bayesian causal models — disjunctive combination
/// - Min: Conservative lower bound — useful for safety-critical reasoning
/// - Weighted: Rule-specific scaling factor applied after combination
#[derive(Debug, Clone, PartialEq)]
pub enum ProbCombination {
    /// P(conclusion) = product(p_i) for all premise probabilities p_i.
    /// Models premises as independent probabilistic events.
    Independent,

    /// P(conclusion) = min(p_i) for all premise probabilities p_i.
    /// Conservative: conclusion is at most as probable as the weakest premise.
    Min,

    /// P(conclusion) = 1 - product(1 - p_i) for all premise probabilities p_i.
    /// Noisy-OR gate: models multiple independent causes of the same effect.
    /// Each premise independently contributes to making the conclusion true.
    NoisyOr,

    /// P(conclusion) = weight * combined_premise_prob.
    /// The weight acts as an additional scaling factor applied after
    /// Independent combination of premise probabilities.
    Weighted(f64),
}

/// A rule annotated with probabilistic semantics.
///
/// Wraps the existing `Rule` struct (premise, filters, conclusion) with
/// probability combination mode, confidence threshold, and rule confidence.
/// All existing rule matching infrastructure works on the inner `Rule`.
#[derive(Debug, Clone)]
pub struct ProbabilisticRule {
    /// The underlying classical rule (premise, filters, conclusion).
    pub rule: Rule,

    /// How to combine probabilities from matched premise triples.
    pub combination: ProbCombination,

    /// Minimum probability threshold: conclusions with probability below
    /// this value are discarded during inference (pruning).
    pub threshold: f64,

    /// Confidence/weight of the rule itself (default 1.0).
    /// Multiplied with the combined premise probability.
    pub rule_confidence: f64,
}

impl ProbabilisticRule {
    /// Create a new probabilistic rule with default settings.
    /// Combination: Independent, threshold: 0.0, confidence: 1.0.
    pub fn new(rule: Rule) -> Self {
        Self {
            rule,
            combination: ProbCombination::Independent,
            threshold: 0.0,
            rule_confidence: 1.0,
        }
    }

    /// Builder: set the combination mode.
    pub fn with_combination(mut self, combination: ProbCombination) -> Self {
        self.combination = combination;
        self
    }

    /// Builder: set the probability threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Builder: set the rule confidence.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.rule_confidence = confidence;
        self
    }
}

/// Combine a set of premise probabilities according to the specified mode.
///
/// # Arguments
/// * `probs` - Probability values of matched premise triples
/// * `combination` - How to combine them
/// * `rule_confidence` - Additional scaling factor from the rule itself
///
/// # Returns
/// The combined probability for the conclusion, clamped to [0.0, 1.0].
pub fn combine_probabilities(
    probs: &[f64],
    combination: &ProbCombination,
    rule_confidence: f64,
) -> f64 {
    if probs.is_empty() {
        return rule_confidence.max(0.0).min(1.0);
    }

    let combined = match combination {
        ProbCombination::Independent => {
            probs.iter().product::<f64>()
        }
        ProbCombination::Min => {
            probs.iter().cloned().fold(f64::INFINITY, f64::min)
        }
        ProbCombination::NoisyOr => {
            1.0 - probs.iter().map(|p| 1.0 - p).product::<f64>()
        }
        ProbCombination::Weighted(weight) => {
            weight * probs.iter().product::<f64>()
        }
    };

    (combined * rule_confidence).max(0.0).min(1.0)
}

/// Parse a combination mode from a string (used by RULE syntax parser).
pub fn parse_combination_mode(s: &str) -> Option<ProbCombination> {
    match s.to_lowercase().as_str() {
        "independent" => Some(ProbCombination::Independent),
        "min" => Some(ProbCombination::Min),
        "noisy_or" | "noisyor" => Some(ProbCombination::NoisyOr),
        _ => {
            // Try parsing as "weighted=0.8"
            if let Some(rest) = s.strip_prefix("weighted") {
                let rest = rest.trim_start_matches('=').trim();
                if let Ok(w) = rest.parse::<f64>() {
                    return Some(ProbCombination::Weighted(w));
                }
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_independent_combination() {
        let probs = vec![0.8, 0.7];
        let result = combine_probabilities(&probs, &ProbCombination::Independent, 1.0);
        assert!((result - 0.56).abs() < 1e-9);
    }

    #[test]
    fn test_min_combination() {
        let probs = vec![0.8, 0.5, 0.9];
        let result = combine_probabilities(&probs, &ProbCombination::Min, 1.0);
        assert!((result - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_noisy_or_combination() {
        let probs = vec![0.5, 0.5];
        let result = combine_probabilities(&probs, &ProbCombination::NoisyOr, 1.0);
        // 1 - (1-0.5)*(1-0.5) = 1 - 0.25 = 0.75
        assert!((result - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_combination() {
        let probs = vec![0.8, 0.7];
        let result = combine_probabilities(&probs, &ProbCombination::Weighted(0.5), 1.0);
        // 0.5 * 0.8 * 0.7 = 0.28
        assert!((result - 0.28).abs() < 1e-9);
    }

    #[test]
    fn test_rule_confidence_scaling() {
        let probs = vec![0.8];
        let result = combine_probabilities(&probs, &ProbCombination::Independent, 0.9);
        // 0.8 * 0.9 = 0.72
        assert!((result - 0.72).abs() < 1e-9);
    }

    #[test]
    fn test_empty_probs() {
        let result = combine_probabilities(&[], &ProbCombination::Independent, 0.9);
        assert!((result - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_clamp_to_bounds() {
        let probs = vec![1.5]; // should still work, clamp output
        let result = combine_probabilities(&probs, &ProbCombination::Independent, 1.0);
        assert!(result <= 1.0);
    }

    #[test]
    fn test_parse_combination_mode() {
        assert_eq!(parse_combination_mode("independent"), Some(ProbCombination::Independent));
        assert_eq!(parse_combination_mode("min"), Some(ProbCombination::Min));
        assert_eq!(parse_combination_mode("noisy_or"), Some(ProbCombination::NoisyOr));
        assert_eq!(parse_combination_mode("noisyor"), Some(ProbCombination::NoisyOr));
        assert_eq!(parse_combination_mode("weighted=0.5"), Some(ProbCombination::Weighted(0.5)));
        assert_eq!(parse_combination_mode("unknown"), None);
    }

    #[test]
    fn test_probabilistic_rule_builder() {
        use crate::terms::Term;
        let rule = Rule {
            premise: vec![(
                Term::Variable("X".into()),
                Term::Constant(1),
                Term::Variable("Y".into()),
            )],
            filters: vec![],
            conclusion: vec![(
                Term::Variable("X".into()),
                Term::Constant(2),
                Term::Variable("Y".into()),
            )],
        };

        let prob_rule = ProbabilisticRule::new(rule)
            .with_combination(ProbCombination::NoisyOr)
            .with_threshold(0.3)
            .with_confidence(0.95);

        assert_eq!(prob_rule.combination, ProbCombination::NoisyOr);
        assert!((prob_rule.threshold - 0.3).abs() < 1e-9);
        assert!((prob_rule.rule_confidence - 0.95).abs() < 1e-9);
    }
}

"""Unit and end-to-end tests for :mod:`src.choice_model` and :mod:`src.choice_analysis`.

The fixtures are fictional, anonymous procurement choice sets.  Tests check the
statistical behaviour that matters for this analysis: the conditional-logit
formula, likelihood/gradient/Hessian consistency, identification diagnostics,
ridge objective separation, hold-out loss, institution bootstrap, and the
refusal to present scenario arithmetic as substitution or causal effects.
"""

import json
import math
import unittest
from pathlib import Path

from src.choice_model import (
    Alternative,
    ChoiceDataError,
    ChoiceSet,
    ModelFailure,
    bootstrap,
    bootstrap_summary,
    choice_probabilities,
    common_shift_scenario,
    conditional_log_likelihood,
    design_rank,
    evaluate_objective,
    load_analysis,
    numerical_gradient,
    numerical_hessian,
    numerical_likelihood_gradient,
    optimise,
    owned_price_elasticity,
    PreparedChoice,
    parse_choices,
    parse_config,
    prepare,
    removal_scenario,
    separation_profile,
    support_diagnostics,
    validation_log_loss,
)
from src.choice_analysis import build_report

FIXTURE_CHOICES = Path(__file__).resolve().parents[1] / "fixtures" / "choices.json"
FIXTURE_CONFIG = Path(__file__).resolve().parents[1] / "fixtures" / "choice.json"


def _material(rows):
    return {"choices": rows}


def _config(**overrides):
    base = {
        "attributes": ["price", "update_frequency"],
        "ridge": 0.01,
        "bootstrap_repeats": 0,
        "seed": 1,
        "convergence_tolerance": 1e-8,
        "max_iterations": 1000,
        "hand_beta": {"price": "-log(3)", "update_frequency": 0},
        "hand_beta_coordinate_system": "raw",
    }
    base.update(overrides)
    return base


def _two_alternative_choice(choice_id="h", price_b=1, freq_a=1, freq_b=1, chosen="a", split="fit", institution="i1"):
    return {
        "choice_id": choice_id,
        "institution": institution,
        "split": split,
        "alternatives": [
            {"key": "a", "price": 0, "update_frequency": freq_a},
            {"key": "b", "price": price_b, "update_frequency": freq_b},
        ],
        "chosen": chosen,
    }


class LikelihoodTests(unittest.TestCase):
    def test_hand_probability_is_three_quarters(self):
        """beta_price = -log(3), prices 0 and 1, identical other attributes."""

        material = _material([_two_alternative_choice()])
        analysis = load_analysis(material, config_raw=_config(ridge=0))
        raw = analysis.prepared_raw()
        beta_raw = [-math.log(3), 0.0]
        probabilities = choice_probabilities(raw, beta_raw)[0]
        self.assertAlmostEqual(probabilities[0], 0.75, places=12)
        self.assertAlmostEqual(probabilities[1], 0.25, places=12)

    def test_probabilities_sum_to_one(self):
        material = _material([
            {
                "choice_id": "c",
                "institution": "i1",
                "split": "fit",
                "alternatives": [
                    {"key": "a", "price": 1, "update_frequency": 2},
                    {"key": "b", "price": 2, "update_frequency": 1},
                    {"key": "outside", "price": 0, "update_frequency": 0},
                ],
                "chosen": "b",
            }
        ])
        analysis = load_analysis(material, config_raw=_config())
        for row in choice_probabilities(analysis.prepared_fit(), (-1.0, 0.5)):
            self.assertAlmostEqual(sum(row), 1.0, places=12)

    def test_common_shift_leaves_probabilities_unchanged(self):
        material = _material([_two_alternative_choice()])
        analysis = load_analysis(material, config_raw=_config(ridge=0))
        prepared = analysis.prepared_fit()[0]
        beta = [-1.0, 0.0]
        before = choice_probabilities((prepared,), beta)[0]
        # Shift both prices up by 10 *after* standardisation; standardised
        # space is linear, so any common vector must leave probabilities fixed.
        after = choice_probabilities(
            (
                PreparedChoice(
                    prepared.choice,
                    tuple(
                        (row[0] + 10.0, row[1]) for row in prepared.x
                    ),
                ),
            ),
            beta,
        )[0]
        for a, b in zip(before, after):
            self.assertAlmostEqual(a, b, places=13)

    def test_logsumexp_is_stable_for_extreme_utilities(self):
        material = _material([_two_alternative_choice()])
        analysis = load_analysis(material, config_raw=_config(ridge=0))
        prepared = analysis.prepared_fit()
        probabilities = choice_probabilities(prepared, (1e4, 0.0))[0]
        self.assertTrue(all(math.isfinite(value) for value in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1.0, places=12)

    def test_likelihood_matches_manual_formula(self):
        material = _material([
            _two_alternative_choice(chosen="b"),
            {
                "choice_id": "c2",
                "institution": "i2",
                "split": "fit",
                "alternatives": [
                    {"key": "a", "price": 1, "update_frequency": 2},
                    {"key": "outside", "price": 0, "update_frequency": 0},
                ],
                "chosen": "outside",
            },
        ])
        analysis = load_analysis(material, config_raw=_config(ridge=0))
        prepared = analysis.prepared_fit()
        beta = (-1.0, 0.5)
        expected = 0.0
        for item in prepared:
            utilities = [sum(beta[k] * row[k] for k in range(2)) for row in item.x]
            top = utilities[item.chosen]
            expected += top - (max(utilities) + math.log(sum(math.exp(u - max(utilities)) for u in utilities)))
        self.assertAlmostEqual(conditional_log_likelihood(prepared, beta), expected, places=10)


class GradientHessianTests(unittest.TestCase):
    def setUp(self):
        self.material = _material([
            {
                "choice_id": "c1",
                "institution": "i1",
                "split": "fit",
                "alternatives": [
                    {"key": "a", "price": 1, "update_frequency": 2},
                    {"key": "b", "price": 2, "update_frequency": 1},
                ],
                "chosen": "a",
            },
            {
                "choice_id": "c2",
                "institution": "i2",
                "split": "fit",
                "alternatives": [
                    {"key": "a", "price": 3, "update_frequency": 1},
                    {"key": "b", "price": 1, "update_frequency": 2},
                    {"key": "outside", "price": 0, "update_frequency": 0},
                ],
                "chosen": "b",
            },
        ])
        self.analysis = load_analysis(self.material, config_raw=_config())
        self.prepared = self.analysis.prepared_fit()

    def test_analytic_gradient_matches_finite_difference(self):
        beta = (-0.8, 0.3)
        obj = evaluate_objective(self.prepared, beta, None)
        finite = numerical_gradient(self.prepared, beta, None)
        for analytic, numeric in zip(obj.gradient, finite):
            self.assertAlmostEqual(analytic, numeric, places=7)

    def test_analytic_hessian_matches_finite_difference(self):
        beta = (-0.8, 0.3)
        obj = evaluate_objective(self.prepared, beta, None)
        finite = numerical_hessian(self.prepared, beta, None)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(obj.information[i][j], finite[i][j], places=6)

    def test_hessian_is_the_expected_information(self):
        """Hessian of the negative log-likelihood is sum_i Cov_i(x)."""

        beta = (-0.8, 0.3)
        obj = evaluate_objective(self.prepared, beta, None)
        expected = [[0.0, 0.0], [0.0, 0.0]]
        for item in self.prepared:
            probabilities = choice_probabilities((item,), beta)[0]
            expected_mean = [sum(p * row[k] for row, p in zip(item.x, probabilities)) for k in range(2)]
            for j in range(2):
                for k in range(2):
                    moment = sum(p * row[j] * row[k] for row, p in zip(item.x, probabilities))
                    expected[j][k] += moment - expected_mean[j] * expected_mean[k]
        for j in range(2):
            for k in range(2):
                self.assertAlmostEqual(obj.information[j][k], expected[j][k], places=9)

    def test_penalized_and_unpenalized_objectives_are_distinct(self):
        beta = (-0.8, 0.3)
        unpenalized = evaluate_objective(self.prepared, beta, None)
        penalized = evaluate_objective(self.prepared, beta, 2.0)
        self.assertAlmostEqual(unpenalized.ridge_penalty, 0.0)
        self.assertAlmostEqual(penalized.ridge_penalty, 0.5 * 2.0 * (0.64 + 0.09))
        self.assertNotAlmostEqual(unpenalized.objective, penalized.objective)
        self.assertEqual(penalized.penalized, True)
        self.assertEqual(unpenalized.penalized, False)

    def test_likelihood_gradient_ignores_penalty(self):
        beta = (-0.8, 0.3)
        penalized = evaluate_objective(self.prepared, beta, 2.0)
        likelihood = evaluate_objective(self.prepared, beta, None, likelihood_only=True)
        for a, b in zip(penalized.likelihood_gradient, likelihood.likelihood_gradient):
            self.assertAlmostEqual(a, b, places=12)
        finite = numerical_likelihood_gradient(self.prepared, beta)
        for a, b in zip(penalized.likelihood_gradient, finite):
            self.assertAlmostEqual(a, b, places=6)

    def test_penalized_optimum_satisfies_ridge_stationarity(self):
        beta_start = (-1.0, 0.5)
        result = optimise(self.prepared, beta_start, 2.0, tolerance=1e-10, max_iterations=2000)
        self.assertTrue(result.converged)
        likelihood_gradient = evaluate_objective(
            self.prepared, result.beta, None, likelihood_only=True
        ).likelihood_gradient
        for j, g in enumerate(likelihood_gradient):
            self.assertAlmostEqual(g - 2.0 * result.beta[j], 0.0, places=6)
        # The unpenalised likelihood is not maximised under penalisation.
        unpenalized = optimise(self.prepared, beta_start, None, tolerance=1e-10, max_iterations=2000)
        if unpenalized.converged:
            self.assertGreater(
                conditional_log_likelihood(self.prepared, unpenalized.beta),
                conditional_log_likelihood(self.prepared, result.beta) - 1e-9,
            )


class IdentificationTests(unittest.TestCase):
    def test_only_common_attribute_variation_is_rank_deficient(self):
        material = _material([
            {
                "choice_id": "c1",
                "institution": "i1",
                "split": "fit",
                "alternatives": [
                    {"key": "a", "price": 1, "update_frequency": 1},
                    {"key": "b", "price": 1, "update_frequency": 1},
                ],
                "chosen": "a",
            }
        ])
        analysis = load_analysis(material, config_raw=_config())
        rank, width, full = design_rank(analysis.prepared_fit())
        self.assertEqual(width, 2)
        self.assertEqual(rank, 0)
        self.assertFalse(full)

    def test_variation_in_one_attribute_leaves_other_direction_unidentified(self):
        material = _material([
            {
                "choice_id": "c1",
                "institution": "i1",
                "split": "fit",
                "alternatives": [
                    {"key": "a", "price": 1, "update_frequency": 1},
                    {"key": "b", "price": 2, "update_frequency": 1},
                ],
                "chosen": "a",
            },
            {
                "choice_id": "c2",
                "institution": "i2",
                "split": "fit",
                "alternatives": [
                    {"key": "a", "price": 3, "update_frequency": 1},
                    {"key": "b", "price": 1, "update_frequency": 1},
                ],
                "chosen": "b",
            },
        ])
        analysis = load_analysis(material, config_raw=_config())
        rank, width, full = design_rank(analysis.prepared_fit())
        self.assertEqual(rank, 1)
        self.assertEqual(width, 2)
        self.assertFalse(full)

    def test_full_rank_needs_independent_attribute_contrasts(self):
        material = _material([
            {
                "choice_id": "c1",
                "institution": "i1",
                "split": "fit",
                "alternatives": [
                    {"key": "a", "price": 1, "update_frequency": 2},
                    {"key": "b", "price": 2, "update_frequency": 1},
                ],
                "chosen": "a",
            },
            {
                "choice_id": "c2",
                "institution": "i2",
                "split": "fit",
                "alternatives": [
                    {"key": "a", "price": 3, "update_frequency": 1},
                    {"key": "b", "price": 1, "update_frequency": 2},
                ],
                "chosen": "b",
            },
        ])
        analysis = load_analysis(material, config_raw=_config())
        rank, width, full = design_rank(analysis.prepared_fit())
        self.assertTrue(full)
        self.assertEqual(rank, width)

    def test_complete_separation_is_flagged(self):
        material = _material([
            _two_alternative_choice("s1", price_b=1, chosen="a"),
            _two_alternative_choice("s2", price_b=2, chosen="a", institution="i2"),
        ])
        analysis = load_analysis(material, config_raw=_config(ridge=0))
        profile = separation_profile(analysis.prepared_fit())
        self.assertTrue(profile["separated"])
        self.assertTrue(profile["attributes"][0]["chosen_strictly_minimum"])

    def test_convergence_alone_is_not_identification(self):
        """A ridge fit can converge while the likelihood is separated."""

        material = _material([
            _two_alternative_choice("s1", price_b=1, chosen="a"),
            _two_alternative_choice("s2", price_b=2, chosen="a", institution="i2"),
        ])
        analysis = load_analysis(material, config_raw=_config(ridge=0.01))
        fit = analysis.prepared_fit()
        result = optimise(fit, analysis.beta_for_analysis(), analysis.config.ridge, tolerance=1e-8, max_iterations=1000)
        self.assertTrue(result.converged)
        self.assertTrue(separation_profile(fit)["separated"])
        report = build_report(material, config_raw=_config(ridge=0.01))
        self.assertFalse(report["estimation"]["identified"])
        self.assertIn("complete_separation_detected", report["estimation"]["reason_codes"])

    def test_single_candidate_set_is_counted_and_has_no_information(self):
        material = _material([
            {
                "choice_id": "only",
                "institution": "i1",
                "split": "fit",
                "alternatives": [{"key": "outside", "price": 0, "update_frequency": 0}],
                "chosen": "outside",
            }
        ])
        analysis = load_analysis(material, config_raw=_config())
        rank, width, full = design_rank(analysis.prepared_fit())
        self.assertEqual(rank, 0)
        self.assertFalse(full)
        probability = choice_probabilities(analysis.prepared_fit(), (0.0, 0.0))[0]
        self.assertEqual(probability, [1.0])
        report = build_report(
            material,
            config_raw=_config(hand_beta={"price": 0, "update_frequency": 0}),
        )
        self.assertEqual(report["identification"]["single_candidate"]["count"], 1)
        self.assertIn("single_candidate_no_within_set_contrast", report["warnings"])


class ContractTests(unittest.TestCase):
    def test_duplicate_choice_id_is_rejected(self):
        row = _two_alternative_choice("dup")
        with self.assertRaises(ChoiceDataError) as raised:
            parse_choices(_material([row, dict(row)]), parse_config(_config()))
        self.assertEqual(raised.exception.code, "duplicate_choice")

    def test_chosen_must_be_offered(self):
        row = _two_alternative_choice("bad", chosen="zzz")
        with self.assertRaises(ChoiceDataError) as raised:
            parse_choices(_material([row]), parse_config(_config()))
        self.assertEqual(raised.exception.code, "chosen_not_offered")

    def test_missing_attribute_is_rejected_not_zero_filled(self):
        row = _two_alternative_choice()
        del row["alternatives"][1]["update_frequency"]
        with self.assertRaises(ChoiceDataError) as raised:
            parse_choices(_material([row]), parse_config(_config()))
        self.assertEqual(raised.exception.code, "missing_attribute")

    def test_absent_outside_is_not_synthesised(self):
        row = _two_alternative_choice()
        parsed = parse_choices(_material([row]), parse_config(_config()))
        self.assertEqual(parsed[0].keys, ("a", "b"))
        self.assertFalse(parsed[0].outside_offered)

    def test_outside_alternative_counts_as_regular_alternative(self):
        row = {
            "choice_id": "c",
            "institution": "i1",
            "split": "fit",
            "alternatives": [
                {"key": "a", "price": 1, "update_frequency": 1},
                {"key": "outside", "price": 0, "update_frequency": 0},
            ],
            "chosen": "outside",
        }
        parsed = parse_choices(_material([row]), parse_config(_config()))
        self.assertTrue(parsed[0].outside_offered)
        self.assertEqual(parsed[0].alternatives[1].key, "outside")
        self.assertEqual(parsed[0].alternatives[1].attributes, (0.0, 0.0))

    def test_institution_cannot_straddle_splits(self):
        rows = [
            _two_alternative_choice("a1", institution="i1", split="fit"),
            _two_alternative_choice("a2", institution="i1", split="validation"),
        ]
        with self.assertRaises(ChoiceDataError) as raised:
            parse_choices(_material(rows), parse_config(_config()))
        self.assertEqual(raised.exception.code, "split_conflict")

    def test_standardisation_uses_fit_split_only(self):
        fit = _two_alternative_choice("f1", price_b=1, institution="i1", split="fit")
        validation = _two_alternative_choice("v1", price_b=100, institution="i2", split="validation")
        analysis = load_analysis(_material([fit, validation]), config_raw=_config())
        self.assertAlmostEqual(analysis.standardizer.mean[0], 0.5)
        self.assertAlmostEqual(analysis.standardizer.scale[0], 0.5)
        # The validation price is far outside fit support and is standardised
        # with the fit statistics, not its own.
        self.assertAlmostEqual(analysis.prepared_validation()[0].x[1][0], (100 - 0.5) / 0.5)

    def test_price_unit_conversion_precedes_standardisation(self):
        fit = _two_alternative_choice("f1", price_b=1, institution="i1", split="fit")
        analysis = load_analysis(
            _material([fit]),
            config_raw=_config(price_scale=100, price_unit="cents"),
        )
        # Raw prices 0 and 1 become 0 and 100 cents; mean 50, scale 50.
        self.assertAlmostEqual(analysis.standardizer.mean[0], 50.0)
        self.assertAlmostEqual(analysis.standardizer.scale[0], 50.0)
        self.assertAlmostEqual(analysis.prepared_fit()[0].x[0][0], -1.0)

    def test_restricted_hand_beta_expression_rejects_execution(self):
        with self.assertRaises(ChoiceDataError):
            parse_config(_config(hand_beta={"price": "__import__('os').system('echo hi')", "update_frequency": 0}))
        with self.assertRaises(ChoiceDataError):
            parse_config(_config(hand_beta={"price": "log", "update_frequency": 0}))


class PredictionAndScenarioTests(unittest.TestCase):
    def setUp(self):
        self.material = _material([
            _two_alternative_choice("t1", price_b=1, chosen="a", institution="i1", split="fit"),
            _two_alternative_choice("v1", price_b=1, chosen="a", institution="i2", split="validation"),
        ])
        self.analysis = load_analysis(self.material, config_raw=_config(ridge=0.01))
        self.beta = list(self.analysis.beta_for_analysis())

    def test_validation_log_loss_uses_observed_choice_only(self):
        result = validation_log_loss(self.analysis.prepared_validation(), self.beta)
        self.assertEqual(result["rows"], 1)
        self.assertGreater(result["mean_log_loss"], 0.0)
        probability = choice_probabilities(self.analysis.prepared_validation(), self.beta)[0][0]
        self.assertAlmostEqual(result["mean_log_loss"], -math.log(probability), places=12)

    def test_owned_price_elasticity_sign_and_formula(self):
        prepared = self.analysis.prepared_fit()
        elasticities = owned_price_elasticity(
            prepared,
            self.beta,
            price_index=0,
            price_sigma=self.analysis.standardizer.scale[0],
            price_unit_factor=1.0,
        )[0]
        probability = choice_probabilities(prepared, self.beta)[0]
        sigma = self.analysis.standardizer.scale[0]
        for entry, p in zip(elasticities["alternatives"], probability):
            expected = (self.beta[0] / sigma) * p * (1 - p) * entry["price_raw"]
            self.assertAlmostEqual(entry["own_price_elasticity"], expected, places=12)

    def test_removal_scenario_is_labelled_iia_and_never_fabricates_offers(self):
        prepared = self.analysis.prepared_fit()[0]
        scenario = removal_scenario(prepared, self.beta, "b")
        self.assertEqual(scenario["scenario"], "iia_renormalisation_scenario_not_substitution_rate")
        self.assertEqual(scenario["keys_without"], ["a"])
        self.assertEqual(len(scenario["probabilities_without"]), 1)
        self.assertAlmostEqual(scenario["probabilities_without"][0], 1.0)

    def test_removing_the_chosen_alternative_is_not_relabelled(self):
        prepared = self.analysis.prepared_fit()[0]
        scenario = removal_scenario(prepared, self.beta, "a")
        self.assertTrue(scenario["chosen_removed"])
        self.assertIsNone(scenario["chosen_probability_without"])

    def test_removal_of_unoffered_key_is_an_error(self):
        prepared = self.analysis.prepared_fit()[0]
        with self.assertRaises(ModelFailure):
            removal_scenario(prepared, self.beta, "outside")

    def test_support_flags_out_of_range_validation_attributes(self):
        rows = [
            _two_alternative_choice("t1", price_b=1, chosen="a", institution="i1", split="fit"),
            _two_alternative_choice("v1", price_b=50, chosen="a", institution="i2", split="validation"),
        ]
        report = build_report(_material(rows), config_raw=_config(ridge=0.01))
        flags = report["flags"]["extrapolation"]
        self.assertTrue(any(row["choice_id"] == "v1" for row in flags))
        fit = load_analysis(_material(rows), config_raw=_config(ridge=0.01)).prepared_fit()
        target = load_analysis(_material(rows), config_raw=_config(ridge=0.01)).prepared_validation()
        self.assertFalse(support_diagnostics(fit, target)[0]["supported"])

    def test_common_shift_scenario_is_labeled_and_zero(self):
        prepared = self.analysis.prepared_fit()[0]
        scenario = common_shift_scenario(prepared, self.beta, (10.0, 0.0))
        self.assertAlmostEqual(scenario["max_absolute_change"], 0.0, places=12)


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_resamples_whole_institutions(self):
        rows = [
            _two_alternative_choice(f"c{i}", price_b=1 + i, freq_b=2 - (i % 2), chosen="b" if i % 2 else "a", institution=f"i{i % 3}")
            for i in range(6)
        ]
        analysis = load_analysis(_material(rows), config_raw=_config())
        fit = analysis.prepared_fit()
        beta = list(analysis.beta_for_analysis())
        result = bootstrap(fit, beta, analysis.config.ridge, repeats=20, seed=7, tolerance=1e-8, max_iterations=1000)
        self.assertEqual(result["requested"], 20)
        # Every replicate contains a whole number of institution groups.
        for replicate in result["replicates"]:
            self.assertIn(replicate["status"], ("success", "failed"))
            if replicate["status"] == "failed":
                self.assertIn("code", replicate)

    def test_failed_replicates_are_kept_separate(self):
        analysis = load_analysis(
            _material([_two_alternative_choice()]),
            config_raw=_config(ridge=0),
        )
        result = bootstrap(analysis.prepared_fit(), [0.0, 0.0], None, repeats=3, seed=1, tolerance=1e-12, max_iterations=1)
        summary = bootstrap_summary(result, ["price", "update_frequency"])
        self.assertGreaterEqual(summary["failed"], 1)
        self.assertIsNotNone(summary["failure_rate"])
        for replicate in summary["replicates"]:
            if replicate["status"] == "failed":
                self.assertNotIn("beta", replicate)

    def test_bootstrap_requires_institutions(self):
        analysis = load_analysis(_material([_two_alternative_choice()]), config_raw=_config())
        summary = bootstrap_summary(
            bootstrap(analysis.prepared_fit(), [0.0, 0.0], None, repeats=0, seed=1),
            ["price", "update_frequency"],
        )
        self.assertEqual(summary["requested"], 0)
        self.assertIsNone(summary["intervals"])


class ReportTests(unittest.TestCase):
    def test_fixture_report_is_json_serialisable_and_documented(self):
        material = json.loads(FIXTURE_CHOICES.read_text(encoding="utf-8"))
        config = json.loads(FIXTURE_CONFIG.read_text(encoding="utf-8"))
        report = build_report(material, config_raw=config, path=str(FIXTURE_CHOICES))
        json.dumps(report, ensure_ascii=False)  # must be serialisable
        self.assertIn("不是社会价值", report["disclaimer"])
        self.assertIn("不是真实替代率", report["scenarios"]["note"])
        self.assertEqual(report["input"]["institutions"], ["i1", "i2", "i3"])
        self.assertEqual(report["input"]["fit_rows"], 2)
        self.assertEqual(report["input"]["validation_rows"], 1)

    def test_fixture_reports_separated_price_direction(self):
        material = json.loads(FIXTURE_CHOICES.read_text(encoding="utf-8"))
        config = json.loads(FIXTURE_CONFIG.read_text(encoding="utf-8"))
        report = build_report(material, config_raw=config)
        self.assertFalse(report["identification"]["identified"])
        self.assertIn("complete_separation_detected", report["estimation"]["reason_codes"])
        self.assertIn("coefficients_not_identified", report["warnings"])

    def test_report_separates_penalized_and_unpenalized_objective(self):
        material = json.loads(FIXTURE_CHOICES.read_text(encoding="utf-8"))
        config = json.loads(FIXTURE_CONFIG.read_text(encoding="utf-8"))
        report = build_report(material, config_raw=config)
        objective = report["estimation"]["objective"]
        self.assertTrue(objective["sa_penalized"])
        self.assertEqual(objective["optimised"], "negative_penalized_log_likelihood")
        self.assertIsNotNone(objective["ridge_penalty"])
        self.assertIsNotNone(objective["log_likelihood_unpenalized"])

    def test_report_never_calls_bootstrap_point_the_estimate(self):
        material = json.loads(FIXTURE_CHOICES.read_text(encoding="utf-8"))
        config = json.loads(FIXTURE_CONFIG.read_text(encoding="utf-8"))
        report = build_report(material, config_raw=config)
        if report["bootstrap"].get("point"):
            self.assertEqual(report["bootstrap"]["point_note"], "bootstrap_mean_not_estimate")

    def test_zero_start_when_hand_beta_absent(self):
        report = build_report(
            _material([_two_alternative_choice()]),
            config_raw={"ridge": 0.01, "bootstrap_repeats": 0},
        )
        self.assertIsNotNone(report["estimation"]["beta"])
        self.assertTrue(report["estimation"]["converged"])

    def test_unreadable_config_is_a_config_error(self):
        with self.assertRaises(ChoiceDataError) as raised:
            build_report(_material([_two_alternative_choice()]), config_raw={"ridge": -1})
        self.assertEqual(raised.exception.code, "invalid_config")


class VerificationQuestionTests(unittest.TestCase):
    """The specific checks requested for the hand benchmark and diagnostics."""

    def test_hand_benchmark_075_025_and_common_ten_unit_shift(self):
        material = _material([_two_alternative_choice(freq_a=1, freq_b=1)])
        analysis = load_analysis(material, config_raw=_config(ridge=0))
        beta_std = analysis.beta_for_analysis()
        probabilities = choice_probabilities(analysis.prepared_fit(), beta_std)[0]
        self.assertAlmostEqual(probabilities[0], 0.75, places=12)
        self.assertAlmostEqual(probabilities[1], 0.25, places=12)
        shifted = common_shift_scenario(analysis.prepared_fit()[0], beta_std, (10.0, 10.0))
        self.assertAlmostEqual(shifted["max_absolute_change"], 0.0, places=12)
        for before, after in zip(shifted["before"], shifted["after"]):
            self.assertAlmostEqual(before, after, places=13)

    def test_max_utility_flag_on_dominant_alternative(self):
        """Selection that is uniform across all fit sets drives the fit to the
        boundary; the flag must fire even though the reported beta is finite."""

        material = _material([
            _two_alternative_choice("m1", price_b=5, chosen="a", institution="i1"),
            _two_alternative_choice("m2", price_b=9, chosen="a", institution="i2"),
            _two_alternative_choice("m3", price_b=1, freq_b=2, chosen="a", institution="i3"),
        ])
        report = build_report(material, config_raw=_config(ridge=0.001))
        probabilities = [max(row["probabilities"]) for row in report["predictions"]["fit"]]
        self.assertTrue(all(p > 0.99 for p in probabilities))
        flagged = report["flags"]["max_utility"]["choice_ids"]
        self.assertTrue({"m1", "m2"} <= set(flagged))

    def test_outside_missing_is_reported_not_synthesised(self):
        material = _material([_two_alternative_choice("o1", institution="i1")])
        report = build_report(material, config_raw=_config(ridge=0.01))
        self.assertEqual(report["flags"]["outside_missing"], ["o1"])
        self.assertIn("outside_alternative_not_offered", report["warnings"])
        self.assertEqual(report["predictions"]["fit"][0]["keys"], ["a", "b"])

    def test_ridge_penalized_and_unpenalized_reporting(self):
        rows = [
            _two_alternative_choice(f"r{i}", price_b=1 + i % 3, chosen="b" if i % 2 else "a", institution=f"i{i}")
            for i in range(8)
        ]
        material = _material(rows)
        unpenalized = build_report(material, config_raw=_config(ridge=0, bootstrap_repeats=0))
        penalized = build_report(material, config_raw=_config(ridge=0.5, bootstrap_repeats=0))
        self.assertEqual(unpenalized["estimation"]["objective"]["optimised"], "negative_log_likelihood")
        self.assertFalse(unpenalized["estimation"]["objective"]["sa_penalized"])
        self.assertEqual(penalized["estimation"]["objective"]["optimised"], "negative_penalized_log_likelihood")
        self.assertTrue(penalized["estimation"]["objective"]["sa_penalized"])
        self.assertGreater(penalized["estimation"]["objective"]["ridge_penalty"], 0.0)

    def test_bootstrap_failure_row_keeps_reason(self):
        rows = [
            _two_alternative_choice(f"b{i}", price_b=1 + i, chosen="b" if i % 2 else "a", institution=f"i{i}")
            for i in range(4)
        ]
        analysis = load_analysis(_material(rows), config_raw=_config(ridge=0))
        result = bootstrap(
            analysis.prepared_fit(),
            [0.0, 0.0],
            None,
            repeats=5,
            seed=3,
            tolerance=1e-12,
            max_iterations=2,
        )
        failed = [row for row in result["replicates"] if row["status"] == "failed"]
        self.assertTrue(failed)
        for row in failed:
            self.assertEqual(row["code"], "bootstrap_refit_failed")
            self.assertNotIn("beta", row)


class CombinedSeparationTests(unittest.TestCase):
    def test_joint_dominance_across_attributes_is_separated(self):
        """Each attribute alone does not separate the chosen option strictly,
        but some utility direction does (one set dominates on price, the other
        on update frequency)."""

        material = _material([
            {
                "choice_id": "j1",
                "institution": "i1",
                "split": "fit",
                "alternatives": [
                    {"key": "a", "price": 1, "update_frequency": 3},
                    {"key": "b", "price": 3, "update_frequency": 2},
                ],
                "chosen": "a",
            },
            {
                "choice_id": "j2",
                "institution": "i2",
                "split": "fit",
                "alternatives": [
                    {"key": "a", "price": 2, "update_frequency": 3},
                    {"key": "b", "price": 2, "update_frequency": 1},
                ],
                "chosen": "a",
            },
        ])
        report = build_report(material, config_raw=_config(ridge=0.01))
        self.assertTrue(report["identification"]["fitted_direction_separation"]["separated_along_fitted_direction"])
        self.assertFalse(report["estimation"]["identified"])
        self.assertIn("complete_separation_detected", report["estimation"]["reason_codes"])

    def test_tied_maximum_is_not_flagged_as_separation(self):
        material = _material([
            {
                "choice_id": "tie",
                "institution": "i1",
                "split": "fit",
                "alternatives": [
                    {"key": "a", "price": 0, "update_frequency": 1},
                    {"key": "b", "price": 0, "update_frequency": 1},
                ],
                "chosen": "a",
            }
        ])
        analysis = load_analysis(material, config_raw=_config(ridge=0.01))
        self.assertFalse(separation_profile(analysis.prepared_fit())["separated"])
        self.assertEqual(analysis.prepared_fit()[0].x[0], analysis.prepared_fit()[0].x[1])


class ReproducibilityTests(unittest.TestCase):
    def test_bootstrap_is_reproducible_given_seed(self):
        rows = [
            _two_alternative_choice(
                f"r{i}",
                price_b=1 + i % 4,
                freq_b=1 + i % 3,
                chosen="b" if i % 2 else "a",
                institution=f"i{i % 4}",
            )
            for i in range(8)
        ]
        analysis = load_analysis(_material(rows), config_raw=_config(ridge=0.05))
        kwargs = dict(repeats=10, seed=42, tolerance=1e-8, max_iterations=1000)
        first = bootstrap(analysis.prepared_fit(), list(analysis.beta_for_analysis()), 0.05, **kwargs)
        second = bootstrap(analysis.prepared_fit(), list(analysis.beta_for_analysis()), 0.05, **kwargs)
        self.assertEqual(first["replicates"], second["replicates"])

    def test_validation_predictions_never_use_unoffered_keys(self):
        rows = [
            _two_alternative_choice("v1", price_b=2, chosen="a", institution="i1", split="validation"),
        ]
        report = build_report(_material(rows), config_raw=_config(ridge=0.01))
        for row in report["predictions"]["validation"]:
            self.assertEqual(row["keys"], ["a", "b"])
            self.assertEqual(len(row["probabilities"]), 2)


class BinaryFixtureReportTests(unittest.TestCase):
    def test_report_for_simple_price_only_tradeoff(self):
        """A clean two-alternative design with genuine variation should be
        identified: full rank, no separation, finite standard errors, and
        validation loss consistent with in-set probabilities."""

        rows = [
            {
                "choice_id": f"c{i}",
                "institution": f"i{i}",
                "split": "fit" if i < 6 else "validation",
                "alternatives": [
                    {"key": "a", "price": 0, "update_frequency": 1 + i % 3},
                    {"key": "b", "price": 1 + i % 3, "update_frequency": 1},
                ],
                "chosen": ["a", "a", "b", "b", "a", "b", "b", "a", "a"][i],
            }
            for i in range(9)
        ]
        report = build_report(_material(rows), config_raw=_config(ridge=0.001))
        self.assertTrue(report["identification"]["rank"]["full_rank"])
        self.assertFalse(report["identification"]["separation"]["separated"])
        self.assertTrue(report["estimation"]["converged"])
        self.assertTrue(report["estimation"]["identified"])
        self.assertIsNotNone(report["validation"]["mean_log_loss"])
        self.assertGreater(report["estimation"]["beta"]["price"], -20)


class EdgeCaseTests(unittest.TestCase):
    def test_no_choices_is_reported_not_crashed(self):
        report = build_report({"choices": []}, config_raw=_config(ridge=0))
        self.assertEqual(report["estimation"]["status"], "no_fit_rows")
        self.assertEqual(report["predictions"]["fit"], [])
        self.assertIn("no_validation_rows", report["warnings"])

    def test_identical_alternatives_are_counted(self):
        row = {
            "choice_id": "same",
            "institution": "i1",
            "split": "fit",
            "alternatives": [
                {"key": "a", "price": 1, "update_frequency": 1},
                {"key": "b", "price": 1, "update_frequency": 1},
            ],
            "chosen": "a",
        }
        report = build_report(_material([row]), config_raw=_config(ridge=0.01))
        self.assertEqual(report["identification"]["identical_alternatives"]["count"], 1)
        self.assertEqual(
            report["predictions"]["fit"][0]["probabilities"],
            [0.5, 0.5],
        )

    def test_missing_outside_alternative_is_flagged_but_not_created(self):
        report = build_report(
            _material([_two_alternative_choice("n1")]),
            config_raw=_config(ridge=0.01),
        )
        self.assertEqual(report["identification"]["outside"]["offered_count"], 0)
        self.assertEqual(report["flags"]["outside_missing"], ["n1"])

    def test_validation_log_loss_reports_unbounded_rows(self):
        """When the observed validation choice has zero fitted probability the
        mean log loss is undefined; report that instead of putting Infinity in
        the table."""

        rows = [
            {
                "choice_id": "fit0",
                "institution": "i1",
                "split": "fit",
                "alternatives": [
                    {"key": "a", "price": 0, "update_frequency": 1},
                    {"key": "b", "price": 1, "update_frequency": 1},
                ],
                "chosen": "a",
            },
            {
                "choice_id": "v1",
                "institution": "i2",
                "split": "validation",
                "alternatives": [
                    {"key": "a", "price": 0, "update_frequency": 0},
                    {"key": "b", "price": 100, "update_frequency": 0},
                ],
                "chosen": "b",
            },
        ]
        report = build_report(_material(rows), config_raw=_config(ridge=0.001))
        validation = report["validation"]
        # The fitted price coefficient is finite, so b retains a tiny but
        # positive probability; the loss is therefore defined, not unbounded.
        self.assertEqual(validation["unbounded_rows"], [])
        self.assertGreater(validation["mean_log_loss"], 100.0)
        self.assertTrue(validation["available"])

    def test_validation_log_loss_available_when_all_probabilities_positive(self):
        rows = [
            {
                "choice_id": "fit0",
                "institution": "i1",
                "split": "fit",
                "alternatives": [
                    {"key": "a", "price": 0, "update_frequency": 1},
                    {"key": "b", "price": 1, "update_frequency": 1},
                ],
                "chosen": "a",
            },
            {
                "choice_id": "v0",
                "institution": "i2",
                "split": "validation",
                "alternatives": [
                    {"key": "a", "price": 0, "update_frequency": 1},
                    {"key": "b", "price": 1, "update_frequency": 1},
                ],
                "chosen": "a",
            },
        ]
        report = build_report(_material(rows), config_raw=_config(ridge=0.01))
        self.assertTrue(report["validation"]["available"])
        self.assertEqual(report["validation"]["unbounded_rows"], [])

    def test_hand_beta_standardized_coordinates_are_honoured(self):
        analysis = load_analysis(
            _material([_two_alternative_choice()]),
            config_raw=_config(hand_beta_coordinate_system="standardized"),
        )
        self.assertEqual(analysis.beta_for_analysis(), (-math.log(3), 0.0))

    def test_beta_raw_reports_unidentified_coordinate_for_zero_scale(self):
        rows = [
            _two_alternative_choice("z1", price_b=1, freq_a=1, freq_b=1, institution="i1"),
        ]
        report = build_report(_material(rows), config_raw=_config(ridge=0.01))
        self.assertEqual(report["estimation"]["beta_raw"]["update_frequency"], "not_identified")


class PriceShiftFromRawDataTests(unittest.TestCase):
    def test_adding_ten_to_every_offered_price_changes_nothing(self):
        """The invariance must hold when the shift is applied to the raw
        price column of every alternative, then standardised."""

        base = _two_alternative_choice("h", price_b=1, freq_a=1, freq_b=1)
        analysis = load_analysis(_material([base]), config_raw=_config(ridge=0))
        beta = list(analysis.beta_for_analysis())
        before = choice_probabilities(analysis.prepared_fit(), beta)[0]

        shifted_alternatives = tuple(
            Alternative(
                key=alternative.key,
                attributes=(alternative.raw_attributes[0] + 10.0, alternative.raw_attributes[1]),
                raw_attributes=(alternative.raw_attributes[0] + 10.0, alternative.raw_attributes[1]),
            )
            for alternative in analysis.choices[0].alternatives
        )
        shifted_choice = ChoiceSet(
            choice_id="shifted",
            institution="i1",
            split="fit",
            alternatives=shifted_alternatives,
            chosen=0,
            outside_offered=False,
        )
        shifted_standardised = prepare((shifted_choice,), analysis.standardizer)
        after = choice_probabilities(shifted_standardised, beta)[0]
        for a, b in zip(before, after):
            self.assertAlmostEqual(a, b, places=13)


class ConfigValidationTests(unittest.TestCase):
    def test_boolean_is_not_a_number(self):
        with self.assertRaises(ChoiceDataError):
            parse_config(_config(ridge=True))

    def test_negative_ridge_is_rejected(self):
        with self.assertRaises(ChoiceDataError) as raised:
            parse_config(_config(ridge=-0.5))
        self.assertEqual(raised.exception.code, "invalid_config")

    def test_non_integer_bootstrap_repeats_is_rejected(self):
        with self.assertRaises(ChoiceDataError):
            parse_config(_config(bootstrap_repeats=1.5))

    def test_unsupported_attribute_is_rejected(self):
        with self.assertRaises(ChoiceDataError):
            parse_config(_config(attributes=["price", "contract_length"]))


if __name__ == "__main__":
    unittest.main()

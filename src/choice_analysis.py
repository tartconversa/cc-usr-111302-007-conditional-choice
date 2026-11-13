"""Analysis orchestration and command line entry point for the choice model.

The report deliberately separates four things that are easy to conflate:

1. what the data can identify (rank, separation, support);
2. what the optimiser did (convergence of which objective);
3. what the fitted model predicts within the offered sets;
4. scenario arithmetic that holds only under the model's IIA assumption.

Nothing in the report is presented as social value, willingness to pay, or a
causal price effect, and no individual-level sensitive attribute is inferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .choice_model import (
    STATUS_CODES,
    AnalysisInput,
    ChoiceDataError,
    ModelFailure,
    bootstrap,
    bootstrap_summary,
    common_shift_scenario,
    curvature_diagnostics,
    design_rank,
    directional_separation,
    evaluate_objective,
    in_set_predictions,
    load_analysis,
    numerical_gradient,
    numerical_hessian,
    numerical_likelihood_gradient,
    optimise,
    owned_price_elasticity,
    removal_scenario,
    separation_profile,
    standard_errors,
    support_diagnostics,
    validation_log_loss,
)
from .choice_reader import load


DISCLAIMER = (
    "这些是在给定候选集合与属性下的条件Logit概率与情景算术，不是社会价值、"
    "支付意愿或因果价格效应；机构标签只用于整组划分与重抽样。"
)


def _fit_scope(analysis: AnalysisInput):
    return analysis.prepared_fit()


def _initial_beta(analysis: AnalysisInput) -> tuple[float, ...]:
    """Starting values are the declared hand_beta, else zero; never a data-driven start."""

    return analysis.beta_for_analysis()


def _assemble_estimation(analysis: AnalysisInput, fit_items: Sequence[Any]) -> dict[str, Any]:
    """Fit and diagnose, keeping optimiser convergence distinct from identification."""

    config = analysis.config
    ridge = config.ridge
    initial = _initial_beta(analysis)
    rank, width, full_rank = design_rank(fit_items)
    separation = separation_profile(fit_items)
    checks: dict[str, Any] = {}
    directional = None
    reasons: list[str] = []

    result = None
    if fit_items:
        result = optimise(
            fit_items,
            initial,
            ridge,
            tolerance=config.convergence_tolerance,
            max_iterations=config.max_iterations,
        )

    if result is not None and fit_items:
        directional = directional_separation(fit_items, result.beta)

    identified = full_rank and not separation["separated"] and not (directional or {}).get(
        "separated_along_fitted_direction", False
    )
    if not full_rank:
        reasons.append(STATUS_CODES["rank_deficient"])
    if separation["separated"]:
        reasons.append(STATUS_CODES["separation"])
    if (
        directional
        and directional["separated_along_fitted_direction"]
        and STATUS_CODES["separation"] not in reasons
    ):
        reasons.append(STATUS_CODES["separation"])
    if result is None:
        reasons.append("no_rows_in_scope")

    diagnosis = None
    standard_error = None
    likelihood_curvature = None
    if result is not None and result.converged:
        diagnosis = curvature_diagnostics(result.hessian)
        likelihood_curvature = curvature_diagnostics(
            evaluate_objective(fit_items, result.beta, None, likelihood_only=True).likelihood_information
        )
        standard_error = standard_errors(result.hessian)
        if diagnosis["non_positive"] or likelihood_curvature["non_positive"]:
            identified = False
            reasons.append(STATUS_CODES["rank_deficient"])
    elif result is not None and result.message:
        reasons.append(result.message)

    if result is not None and result.converged and fit_items:
        analytic = evaluate_objective(fit_items, result.beta, ridge).gradient
        finite = numerical_gradient(fit_items, result.beta, ridge)
        checks["objective_gradient_max_abs_difference"] = max(
            (abs(a - b) for a, b in zip(analytic, finite)), default=0.0
        )
        likelihood_analytic = evaluate_objective(fit_items, result.beta, None, likelihood_only=True).likelihood_gradient
        likelihood_finite = numerical_likelihood_gradient(fit_items, result.beta)
        checks["likelihood_gradient_max_abs_difference"] = max(
            (abs(a - b) for a, b in zip(likelihood_analytic, likelihood_finite)), default=0.0
        )
        finite_hessian = numerical_hessian(fit_items, result.beta, ridge)
        checks["hessian_max_abs_difference"] = max(
            (
                abs(result.hessian[i][j] - finite_hessian[i][j])
                for i in range(width)
                for j in range(width)
            ),
            default=0.0,
        )

    beta = list(result.beta) if result is not None else None
    names = list(config.attributes)
    beta_raw = analysis.beta_to_raw(result.beta) if result is not None else None
    return {
        "status": (
            "identified"
            if identified
            else ("converged_but_not_identified" if fit_items and result is not None and result.converged
                  else ("not_identified" if fit_items else "no_fit_rows"))
        ),
        "identified": bool(identified),
        "reason_codes": reasons,
        "converged": bool(result.converged) if result is not None else False,
        "iterations": result.iterations if result is not None else 0,
        "beta": dict(zip(names, beta)) if beta is not None else None,
        "beta_raw": (
            {name: (value if value is not None else "not_identified")
             for name, value in zip(names, beta_raw)}
            if beta_raw is not None else None
        ),
        "beta_coordinate_system": "standardized",
        "beta_raw_note": "raw 列由标准化系数按 fit 划分 sigma 与价格单位换算得到；sigma=0 的坐标不可识别。",
        "standard_errors": dict(zip(names, standard_error)) if standard_error else None,
        "objective": {
            # "sa" refers to the ridge penalty on squared coefficients, which
            # is the option declared in the config; the likelihood term is
            # kept separate below so a penalised fit is never called unpenalised.
            "optimised": "negative_penalized_log_likelihood" if ridge else "negative_log_likelihood",
            "ridge": ridge,
            "sa_penalized": bool(ridge),
            "value": result.objective if result is not None else None,
            "log_likelihood_unpenalized": result.log_likelihood if result is not None else None,
            "ridge_penalty": result.ridge_penalty if result is not None else None,
        },
        "gradient_norm": result.gradient_norm if result is not None else None,
        "ridge": ridge,
        "curvature": {
            "objective": diagnosis,
            "likelihood_unpenalized": likelihood_curvature,
            "note": "curvature 只描述惩罚目标或未惩罚似然在给定 beta 处的局部形状；"
                    "完全分离时大样本正态近似不适用。",
        },
        "checks": checks,
        "rank": {"difference_matrix_rank": rank, "columns": width, "full_rank": full_rank},
        "separation": separation,
        "fitted_direction_separation": directional,
        "caveat": (
            "优化收敛（或ridge惩罚下可解）不等于所有系数被识别；"
            "rank或分离诊断未通过时不得宣称识别。"
        ),
    }


def _distribution_counts(analysis: AnalysisInput) -> dict[str, Any]:
    singles = [choice.choice_id for choice in analysis.choices if choice.count == 1]
    outside_offered = [choice.choice_id for choice in analysis.choices if choice.outside_offered]
    outside_missing = [choice.choice_id for choice in analysis.choices if not choice.outside_offered]
    identical = [
        choice.choice_id
        for choice in analysis.choices
        if len({alternative.attributes for alternative in choice.alternatives}) == 1
    ]
    return {
        "single_candidate": {"count": len(singles), "choice_ids": singles},
        "outside": {
            "offered_count": len(outside_offered),
            "offered_choice_ids": outside_offered,
            "missing_count": len(outside_missing),
            "missing_choice_ids": outside_missing,
        },
        "identical_alternatives": {"count": len(identical), "choice_ids": identical},
    }


def _predictions(analysis: AnalysisInput, beta: Sequence[float] | None) -> dict[str, Any]:
    if beta is None:
        return {"fit": [], "validation": []}
    fit = in_set_predictions(analysis.prepared_fit(), beta)
    validation = in_set_predictions(analysis.prepared_validation(), beta)
    return {"fit": fit, "validation": validation}


def _validation(analysis: AnalysisInput, beta: Sequence[float] | None) -> dict[str, Any]:
    if beta is None:
        return {
            "rows": 0,
            "mean_log_loss": None,
            "sum_log_loss": None,
            "unbounded_rows": [],
            "available": False,
        }
    prepared = analysis.prepared_validation()
    if not prepared:
        return {
            "rows": 0,
            "mean_log_loss": None,
            "sum_log_loss": None,
            "unbounded_rows": [],
            "available": False,
        }
    result = validation_log_loss(prepared, beta)
    result["available"] = result["mean_log_loss"] is not None
    result["definition"] = "mean negative log probability of the observed choice on validation sets"
    return result


def _elasticities(analysis: AnalysisInput, beta: Sequence[float] | None) -> list[dict[str, Any]]:
    if beta is None or "price" not in analysis.config.attributes:
        return []
    price_index = list(analysis.config.attributes).index("price")
    standardizer = analysis.standardizer
    prepared = analysis.prepared_fit() + analysis.prepared_validation()
    return owned_price_elasticity(
        prepared,
        beta,
        price_index=price_index,
        price_sigma=standardizer.scale[price_index] or 1.0,
        price_unit_factor=analysis.price_units.factor,
    )


def _remove_scenarios(analysis: AnalysisInput, beta: Sequence[float] | None) -> dict[str, Any]:
    """Scenario arithmetic after deleting one offered alternative.

    These are renormalisations under the model's IIA assumption; they are
    reported as scenarios, never as an estimated substitution rate or a
    counterfactual market outcome.  Deletion of the *chosen* alternative is
    kept separate because the observed choice then has no target.
    """

    if beta is None:
        return {"scenarios": [], "chosen_deletions": [], "note": "model_not_fitted"}
    scenarios = []
    chosen_deletions = []
    for item in analysis.prepared_fit() + analysis.prepared_validation():
        for key in item.choice.keys:
            scenario = removal_scenario(item, beta, key)
            if scenario["chosen_removed"]:
                chosen_deletions.append(scenario)
            else:
                scenarios.append(scenario)
    return {
        "scenarios": scenarios,
        "chosen_deletions": chosen_deletions,
        "note": "删除方案后重算的概率只是IIA假设下的情景算术，不是真实替代率。",
    }


def _flags(analysis: AnalysisInput, beta: Sequence[float] | None) -> dict[str, Any]:
    flags: dict[str, Any] = {
        "max_utility": {"count": 0, "choice_ids": []},
        "extrapolation": [],
        "extrapolation_note": "超出 fit 支撑的属性只做标记；不把外推预测当作可支持的结论。",
        "outside_missing": [],
        "common_shift_check": None,
    }
    if beta is None or not analysis.choices:
        return flags
    prepared = analysis.prepared_fit() + analysis.prepared_validation()
    probabilities = in_set_predictions(prepared, beta)
    maximum = [row for row in probabilities if max(row["probabilities"]) >= 0.99]
    flags["max_utility"] = {
        "count": len(maximum),
        "choice_ids": [row["choice_id"] for row in maximum],
        "threshold": 0.99,
        "meaning": "inside the offered set the model puts almost all weight on one alternative; "
                   "the conditional-logit fit can approach a boundary here",
    }
    flags["outside_missing"] = [
        choice.choice_id for choice in analysis.choices if not choice.outside_offered
    ]
    extrapolation = support_diagnostics(analysis.prepared_fit(), prepared)
    flags["extrapolation"] = [row for row in extrapolation if not row["supported"]]
    if analysis.prepared_fit():
        first = analysis.prepared_fit()[0]
        shift = tuple(0.1 for _ in analysis.config.attributes)
        scenario = common_shift_scenario(first, beta, shift)
        flags["common_shift_check"] = {
            "choice_id": scenario["choice_id"],
            "delta": shift,
            "max_absolute_change": scenario["max_absolute_change"],
            "meaning": STATUS_CODES["all_shift_invariant"],
        }
    return flags


def build_report(
    material: Any,
    *,
    config_raw: Mapping[str, Any] | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    analysis = load_analysis(material, config_raw=config_raw)
    config = analysis.config
    fit_items = _fit_scope(analysis)
    estimation = _assemble_estimation(analysis, fit_items)
    names = list(config.attributes)
    beta = None
    if estimation["beta"] is not None:
        beta = [estimation["beta"][name] for name in names]

    counts = _distribution_counts(analysis)
    warnings: list[str] = []
    if counts["single_candidate"]["count"]:
        warnings.append(STATUS_CODES["single_candidate"])
    if counts["outside"]["missing_count"]:
        warnings.append(STATUS_CODES["outside_missing"])
    if not estimation["identified"] and fit_items:
        warnings.append("coefficients_not_identified")
    if not analysis.validation_choices:
        warnings.append("no_validation_rows")

    boot: dict[str, Any]
    if beta is None or not fit_items or not estimation["converged"]:
        boot = {
            "requested": config.bootstrap_repeats,
            "successful": 0,
            "failed": 0,
            "replicates": [],
            "skipped": STATUS_CODES["not_fitted"],
            "reason": "fit_not_converged" if fit_items else STATUS_CODES["no_choices"],
        }
    else:
        raw_boot = bootstrap(
            fit_items,
            beta,
            config.ridge if config.ridge else None,
            repeats=config.bootstrap_repeats,
            seed=config.seed,
            tolerance=config.convergence_tolerance,
            max_iterations=config.max_iterations,
        )
        boot = bootstrap_summary(raw_boot, names)
        boot["note"] = "whole_institution_resampling; failures_kept_separate"
        boot["inference_caveat"] = (
            "只有重拟合本身通过识别诊断的复制才提供系数的抽样信息；"
            "unidentified_rate 与 identified 计数必须与区间一起阅读。"
        )
        if not estimation["identified"]:
            boot["identification_caveat"] = (
                "系数未通过识别诊断；这些重拟合区间只描述惩罚拟合的抽样波动，"
                "不能据此宣称系数被识别。"
            )
        if config.ridge:
            boot["basis"] = "ridge_penalized_refits"
        else:
            boot["basis"] = "unpenalized_refits"

    report = {
        "input": {
            "path": path,
            "choice_rows": len(analysis.choices),
            "institutions": list(analysis.institutions),
            "fit_rows": len(analysis.fit_choices),
            "validation_rows": len(analysis.validation_choices),
            "attributes": names,
            "price_units": {
                "label": analysis.price_units.label,
                "factor_to_analysis_scale": analysis.price_units.factor,
            },
        },
        "model": {
            "formula": "exp(beta'x_ij) / sum_k exp(beta'x_ik)",
            "alternative_specific_intercepts": False,
            "attributes": names,
            "beta_coordinate_system": (
                "raw" if config.hand_beta_coordinate_system == "raw" else "standardized"
            ),
            "standardization": {
                "fit_on": "fit_split_only",
                "mean": dict(zip(names, analysis.standardizer.mean)),
                "scale": dict(zip(names, analysis.standardizer.scale)),
                "fit_alternatives": analysis.standardizer.fit_count,
            },
            "note": "同一选择内所有属性整体平移不改变概率；价格单位换算先于标准化。",
        },
        "identification": {
            "rank": estimation["rank"],
            "separation": estimation["separation"],
            "fitted_direction_separation": estimation["fitted_direction_separation"],
            "single_candidate": counts["single_candidate"],
            "outside": counts["outside"],
            "identical_alternatives": counts["identical_alternatives"],
            "identified": estimation["identified"],
        },
        "estimation": {
            key: value
            for key, value in estimation.items()
            if key not in ("rank", "separation", "fitted_direction_separation")
        },
        "predictions": {
            **_predictions(analysis, beta),
            "interpretation": "概率只对给定候选集合定义；不是市场份额预测，也不外推到未展示方案。",
        },
        "validation": _validation(analysis, beta),
        "elasticities": _elasticities(analysis, beta),
        "scenarios": _remove_scenarios(analysis, beta),
        "bootstrap": boot,
        "flags": _flags(analysis, beta),
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }
    return report


def render(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, default=_default)


def summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Compact view: what was identified, what the model predicts, what it does not claim."""

    estimation = report["estimation"]
    bootstrap = report["bootstrap"]
    return {
        "identified": estimation["identified"],
        "status": estimation["status"],
        "reason_codes": estimation["reason_codes"],
        "converged": estimation["converged"],
        "objective": estimation["objective"],
        "beta": estimation["beta"],
        "beta_raw": estimation["beta_raw"],
        "standard_errors": estimation["standard_errors"],
        "rank": report["identification"]["rank"],
        "bootstrap": {
            key: bootstrap.get(key)
            for key in ("requested", "successful", "failed", "identified", "unidentified_rate", "basis")
        },
        "validation_log_loss": report["validation"]["mean_log_loss"],
        "warnings": report["warnings"],
        "disclaimer": report["disclaimer"],
    }


def _default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"无法序列化的类型: {type(value).__name__}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="对匿名合成采购选择表做条件Logit分析（不是社会价值或因果价格效应估计）"
    )
    parser.add_argument("choices", help="choice 表路径（由 src.choice_reader 读取）")
    parser.add_argument("config", nargs="?", help="可选 config JSON 路径；缺省用选择表内 config")
    parser.add_argument("--output", help="可选 JSON 输出路径")
    parser.add_argument("--summary", action="store_true", help="只打印精简摘要")
    args = parser.parse_args(argv)

    material = load(args.choices)
    config_raw = None
    if args.config:
        config_raw = load(args.config)
    try:
        report = build_report(material, config_raw=config_raw, path=args.choices)
    except ChoiceDataError as error:
        print(json.dumps({"ok": False, "error_code": error.code, "error": str(error)}, ensure_ascii=False))
        return 2
    except ModelFailure as error:
        print(json.dumps({"ok": False, "error_code": error.code, "error": str(error)}, ensure_ascii=False))
        return 3
    text = render(summary(report) if args.summary else report)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

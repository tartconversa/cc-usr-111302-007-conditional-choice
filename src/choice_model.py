"""Conditional-logit analysis for anonymous, synthetic procurement choice sets.

The model is a baseline conditional logit with alternative attributes only:

    P(chosen = j | set i) = exp(beta' x_ij) / sum_k exp(beta' x_ik)

There are no alternative-specific intercepts and no institution covariates.
Institution labels are used only to keep train/validation groups separate and
for whole-group bootstrap resampling.  Nothing here infers personal or
sensitive characteristics.

Prices are converted to the analysis unit *before* any fit-sample
standardisation.  Standardisation statistics are estimated on the fit split
only and are then applied unchanged to validation alternatives.

The module has no third-party dependencies; all linear algebra needed for the
small attribute models is implemented locally.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

ATTRIBUTE_KEYS = ("price", "update_frequency")
OUTSIDE_KEY = "outside"
PRICE_UNIT_LABEL = "minor_unit_compatibility_scale"

STATUS_CODES = {
    "no_choices": "no_rows_in_scope",
    "missing_attribute": "alternative_missing_attribute",
    "rank_deficient": "design_matrix_rank_deficient",
    "separation": "complete_separation_detected",
    "invalid_beta": "non_finite_starting_beta",
    "optimizer_failed": "optimizer_failed_to_converge",
    "bootstrap_failed": "bootstrap_refit_failed",
    "single_candidate": "single_candidate_no_within_set_contrast",
    "outside_missing": "outside_alternative_not_offered",
    "all_shift_invariant": "common_shift_leaves_probabilities_unchanged",
    "not_fitted": "model_not_fitted",
    "invalid_config": "invalid_configuration",
    "duplicate_choice": "duplicate_choice_key",
    "split_unexpected": "unexpected_split_label",
    "institution_missing": "missing_institution_label",
    "alternative_missing_key": "alternative_missing_key",
    "duplicate_alternative": "duplicate_alternative_key_within_set",
    "chosen_not_offered": "chosen_key_not_in_alternatives",
    "not_offered": "alternative_not_offered_in_choice_set",
    "split_conflict": "institution_labelled_in_both_splits",
}


class ChoiceDataError(ValueError):
    """Raised when the input contract in :mod:`src.choice_reader` is violated."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ModelFailure(RuntimeError):
    """Raised when an analysis cannot produce a defensible estimate."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Alternative:
    key: str
    attributes: tuple[float, ...]
    raw_attributes: tuple[float, ...] = ()

    def raw_attribute(self, index: int) -> float:
        return self.raw_attributes[index]


@dataclass(frozen=True)
class ChoiceSet:
    choice_id: str
    institution: str
    split: str
    alternatives: tuple[Alternative, ...]
    chosen: int
    outside_offered: bool

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(alternative.key for alternative in self.alternatives)

    @property
    def count(self) -> int:
        return len(self.alternatives)


@dataclass(frozen=True)
class PriceUnits:
    factor: float
    label: str


@dataclass(frozen=True)
class Standardizer:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    fit_count: int

    def apply(self, attributes: Sequence[float]) -> tuple[float, ...]:
        return tuple(
            (value - mean) / scale if scale else 0.0
            for value, mean, scale in zip(attributes, self.mean, self.scale)
        )

@dataclass(frozen=True)
class Config:
    attributes: tuple[str, ...]
    ridge: float | None
    bootstrap_repeats: int
    seed: int
    convergence_tolerance: float
    max_iterations: int
    hand_beta: tuple[float, ...] | None
    hand_beta_coordinate_system: str
    price_factor: float = 1.0
    price_label: str = PRICE_UNIT_LABEL


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChoiceDataError("missing_attribute", f"{field_name} 必须为数值")
    result = float(value)
    if not math.isfinite(result):
        raise ChoiceDataError("missing_attribute", f"{field_name} 必须为有限数值")
    return result


def _hand_beta_expression(value: Any) -> float:
    """Evaluate the small arithmetic expression form allowed by the config.

    Config expressions such as ``"-log(3)"`` are parsed with :mod:`ast` and
    only unary +/- and binary + - * / ** over numbers and ``log`` are allowed.
    Anything else is rejected instead of being handed to ``eval``.
    """

    if isinstance(value, bool):
        raise ChoiceDataError("invalid_config", "hand_beta 值不能为布尔")
    if isinstance(value, (int, float)):
        return _number(value, "hand_beta")
    if not isinstance(value, str) or not value.strip():
        raise ChoiceDataError("invalid_config", "hand_beta 值必须为数值或受限表达式")

    allowed_nodes = (
        ast.Expression,
        ast.Constant,
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Call,
        ast.Name,
        ast.Load,
    )
    tree = ast.parse(value.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ChoiceDataError("invalid_config", f"hand_beta 表达式含不允许的语法: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id != "log":
            raise ChoiceDataError("invalid_config", f"hand_beta 表达式含未知名称: {node.id}")
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) != "log":
            raise ChoiceDataError("invalid_config", "hand_beta 只允许调用 log")
        if isinstance(node, ast.Call) and len(node.args) != 1:
            raise ChoiceDataError("invalid_config", "log 只接受一个参数")
    value = eval(compile(tree, "<hand_beta>", "eval"), {"__builtins__": {}}, {"log": math.log})
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChoiceDataError("invalid_config", "hand_beta 表达式必须求值为数值")
    result = float(value)
    if not math.isfinite(result):
        raise ChoiceDataError("invalid_config", "hand_beta 表达式结果必须有限")
    return result


def _parse_price_units(raw: Mapping[str, Any]) -> PriceUnits:
    factor = raw.get("price_scale", 1.0)
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or float(factor) <= 0:
        raise ChoiceDataError("invalid_config", "price_scale 必须为正数")
    label = raw.get("price_unit", PRICE_UNIT_LABEL)
    if not isinstance(label, str) or not label:
        raise ChoiceDataError("invalid_config", "price_unit 必须为非空字符串")
    return PriceUnits(float(factor), label)


def _parse_config(raw: Mapping[str, Any]) -> Config:
    attributes = tuple(raw.get("attributes", ATTRIBUTE_KEYS))
    if not attributes:
        raise ChoiceDataError("invalid_config", "attributes 不可为空")
    if len(set(attributes)) != len(attributes):
        raise ChoiceDataError("invalid_config", "attributes 不可重复")
    unsatisfied = [name for name in attributes if name not in ATTRIBUTE_KEYS]
    if unsatisfied:
        raise ChoiceDataError("invalid_config", f"当前分析器只支持属性 {ATTRIBUTE_KEYS}: {unsatisfied}")

    units = _parse_price_units(raw)
    raw_ridge = raw.get("ridge")
    if raw_ridge is None:
        ridge = None
    else:
        ridge_value = _number(raw_ridge, "ridge")
        if ridge_value < 0:
            raise ChoiceDataError("invalid_config", "ridge 必须为非负数")
        ridge = ridge_value if ridge_value > 0 else None

    repeats = raw.get("bootstrap_repeats", 200)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 0:
        raise ChoiceDataError("invalid_config", "bootstrap_repeats 必须为非负整数")
    seed = raw.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ChoiceDataError("invalid_config", "seed 必须为整数")
    raw_tolerance = raw.get("convergence_tolerance", 1e-8)
    if isinstance(raw_tolerance, bool) or not isinstance(raw_tolerance, (int, float)):
        raise ChoiceDataError("invalid_config", "convergence_tolerance 必须为数值")
    tolerance = float(raw_tolerance)
    raw_iterations = raw.get("max_iterations", 1000)
    if isinstance(raw_iterations, bool) or not isinstance(raw_iterations, int):
        raise ChoiceDataError("invalid_config", "max_iterations 必须为整数")
    iterations = raw_iterations
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ChoiceDataError("invalid_config", "convergence_tolerance 必须为正数")
    if iterations < 1:
        raise ChoiceDataError("invalid_config", "max_iterations 必须为正整数")

    hand_raw = raw.get("hand_beta")
    hand_beta: tuple[float, ...] | None = None
    coordinate_system = str(raw.get("hand_beta_coordinate_system", "raw"))
    if coordinate_system not in ("raw", "standardized"):
        raise ChoiceDataError("invalid_config", "hand_beta_coordinate_system 必须为 raw 或 standardized")
    if hand_raw is not None:
        if not isinstance(hand_raw, Mapping):
            raise ChoiceDataError("invalid_config", "hand_beta 必须为属性名到数值的映射")
        missing = [name for name in attributes if name not in hand_raw]
        if missing:
            raise ChoiceDataError("invalid_config", f"hand_beta 缺少属性: {missing}")
        hand_beta = tuple(_hand_beta_expression(hand_raw[name]) for name in attributes)

    return Config(
        attributes=attributes,
        ridge=ridge,
        bootstrap_repeats=repeats,
        seed=seed,
        convergence_tolerance=tolerance,
        max_iterations=iterations,
        hand_beta=hand_beta,
        hand_beta_coordinate_system=coordinate_system,
        price_factor=units.factor,
        price_label=units.label,
    )


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _zero(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def _rank(matrix: Sequence[Sequence[float]], tolerance: float = 1e-10) -> int:
    """Column rank of a small matrix via partial-pivoted elimination."""

    if not matrix:
        return 0
    rows = [list(map(float, row)) for row in matrix]
    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    for row in rows:
        row.extend([0.0] * (n_cols - len(row)))
    scale = max((abs(value) for row in rows for value in row), default=0.0)
    if scale == 0.0:
        return 0
    threshold = tolerance * scale
    rank = 0
    for column in range(n_cols):
        pivot = max(range(rank, n_rows), key=lambda index: abs(rows[index][column]), default=-1)
        if pivot < 0 or abs(rows[pivot][column]) <= threshold:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        for index in range(rank + 1, n_rows):
            factor = rows[index][column] / pivot_value
            if factor:
                for j in range(column, n_cols):
                    rows[index][j] -= factor * rows[rank][j]
        rank += 1
        if rank == n_rows:
            break
    return rank


def _inverse(matrix: Sequence[Sequence[float]]) -> list[list[float]] | None:
    size = len(matrix)
    augmented = [
        [float(value) for value in row] + [1.0 if i == j else 0.0 for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda index: abs(augmented[index][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for index in range(size):
            if index == column:
                continue
            factor = augmented[index][column]
            if factor:
                augmented[index] = [
                    value - factor * base for value, base in zip(augmented[index], augmented[column])
                ]
    return [row[size:] for row in augmented]


def _symmetric_eigenvalues(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Jacobi eigenvalue iteration for a small symmetric matrix."""

    size = len(matrix)
    values = [[float(x) for x in row] for row in matrix]
    for _ in range(100):
        off_diagonal = 0.0
        pivot = (0, 1)
        for i in range(size):
            for j in range(i + 1, size):
                if abs(values[i][j]) > off_diagonal:
                    off_diagonal = abs(values[i][j])
                    pivot = (i, j)
        if off_diagonal <= 1e-12:
            break
        i, j = pivot
        angle = 0.5 * math.atan2(2 * values[i][j], values[j][j] - values[i][i])
        c, s = math.cos(angle), math.sin(angle)
        for k in range(size):
            if k in (i, j):
                continue
            a_ki, a_kj = values[k][i], values[k][j]
            values[k][i] = values[i][k] = c * a_ki - s * a_kj
            values[k][j] = values[j][k] = s * a_ki + c * a_kj
        a_ii, a_jj, a_ij = values[i][i], values[j][j], values[i][j]
        values[i][i] = c * c * a_ii - 2 * s * c * a_ij + s * s * a_jj
        values[j][j] = s * s * a_ii + 2 * s * c * a_ij + c * c * a_jj
        values[i][j] = values[j][i] = 0.0
    return [values[i][i] for i in range(size)]


# ---------------------------------------------------------------------------
# Data contract and preparation
# ---------------------------------------------------------------------------


class AnalysisInput:
    def __init__(self, config: Config, price_units: PriceUnits, choices: tuple[ChoiceSet, ...]) -> None:
        self.config = config
        self.price_units = price_units
        self.choices = choices
        self.standardizer = fit_standardizer(choices, config.attributes)
        self._cache: dict[str, tuple[PreparedChoice, ...]] = {}

    @property
    def fit_choices(self) -> tuple[ChoiceSet, ...]:
        return tuple(choice for choice in self.choices if choice.split == "fit")

    @property
    def validation_choices(self) -> tuple[ChoiceSet, ...]:
        return tuple(choice for choice in self.choices if choice.split == "validation")

    @property
    def institutions(self) -> tuple[str, ...]:
        return tuple(sorted({choice.institution for choice in self.choices}))

    def prepared_fit(self) -> tuple[PreparedChoice, ...]:
        return self._cached("fit", lambda: prepare(self.fit_choices, self.standardizer))

    def prepared_validation(self) -> tuple[PreparedChoice, ...]:
        return self._cached("validation", lambda: prepare(self.validation_choices, self.standardizer))

    def prepared_raw(self) -> tuple[PreparedChoice, ...]:
        return self._cached("raw", lambda: prepare_raw(self.choices))

    def _cached(self, key: str, build) -> tuple[PreparedChoice, ...]:
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def beta_to_raw(self, beta: Sequence[float]) -> tuple[float, ...]:
        """Convert a standardised-coordinate beta to raw attribute units.

        beta_raw[k] = beta_std[k] * sigma[k] / price_unit_factor[k], where
        sigma is the fit-split scale and the price factor is the pre-
        standardisation unit conversion.  A zero fit scale means the raw
        coefficient is not identified; None is returned for that coordinate.
        """

        converted: list[float | None] = []
        for j, attribute in enumerate(self.config.attributes):
            scale = self.standardizer.scale[j]
            if scale == 0:
                converted.append(None)
                continue
            factor = self.price_units.factor if attribute == "price" else 1.0
            converted.append(beta[j] * scale / factor)
        return tuple(converted)

    def beta_for_analysis(self) -> tuple[float, ...]:
        """Starting values in standardised coordinates.

        ``hand_beta`` is a declared starting point when present; otherwise the
        fit starts at zero.  Starting values are never reported as estimates
        and never override the fitted optimum.
        """

        if self.config.hand_beta is None:
            return tuple(0.0 for _ in self.config.attributes)
        if self.config.hand_beta_coordinate_system == "standardized":
            return self.config.hand_beta
        raw = self.config.hand_beta
        converted = []
        for j, attribute in enumerate(self.config.attributes):
            scale = self.standardizer.scale[j]
            if scale == 0:
                # A constant fit-split attribute carries no within-set contrast,
                # so its raw starting value could be any finite number without
                # changing any probability.  Use zero and let the rank
                # diagnostic report the missing identification.
                converted.append(0.0)
                continue
            factor = self.price_units.factor if attribute == "price" else 1.0
            converted.append(raw[j] * scale / factor)
        return tuple(converted)


def parse_config(raw: Mapping[str, Any] | None) -> Config:
    return _parse_config(raw or {})


def parse_choices(
    material: Any,
    config: Config,
    *,
    price_units: PriceUnits | None = None,
) -> tuple[ChoiceSet, ...]:
    """Validate the choice table and convert prices to analysis units.

    ``material`` is the object returned by :func:`src.choice_reader.load`.
    Prices are multiplied by the declared unit factor *before* any
    standardisation.  The outside alternative, when present, is a regular
    alternative and is never synthesised when it is absent.
    """

    if not isinstance(material, Mapping) or "choices" not in material:
        raise ChoiceDataError("missing_alternative_key", "输入必须为含 choices 列表的对象")
    rows = material["choices"]
    if not isinstance(rows, list):
        raise ChoiceDataError("missing_alternative_key", "choices 必须为列表")
    units = price_units or PriceUnits(1.0, PRICE_UNIT_LABEL)
    if isinstance(material.get("price_units"), Mapping) and price_units is None:
        units = _parse_price_units(material["price_units"])

    seen_ids: set[str] = set()
    choices: list[ChoiceSet] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ChoiceDataError("missing_alternative_key", f"第 {index} 条 choice 不是对象")
        choice_id = row.get("choice_id")
        if not isinstance(choice_id, str) or not choice_id:
            raise ChoiceDataError("missing_alternative_key", f"第 {index} 条 choice_id 缺失或非字符串")
        if choice_id in seen_ids:
            raise ChoiceDataError("duplicate_choice", f"choice_id 重复: {choice_id}")
        seen_ids.add(choice_id)

        institution = row.get("institution")
        if not isinstance(institution, str) or not institution:
            raise ChoiceDataError("institution_missing", f"{choice_id} 缺少 institution 标签")
        split = row.get("split")
        if split not in ("fit", "validation"):
            raise ChoiceDataError("split_unexpected", f"{choice_id} 的 split 必须为 fit 或 validation")

        alternatives = row.get("alternatives")
        if not isinstance(alternatives, list) or not alternatives:
            raise ChoiceDataError("missing_alternative_key", f"{choice_id} 缺少非空 alternatives 列表")
        built: list[Alternative] = []
        keys: set[str] = set()
        for position, raw_alternative in enumerate(alternatives):
            if not isinstance(raw_alternative, Mapping):
                raise ChoiceDataError("missing_alternative_key", f"{choice_id} 第 {position} 个方案不是对象")
            key = raw_alternative.get("key")
            if not isinstance(key, str) or not key:
                raise ChoiceDataError("alternative_missing_key", f"{choice_id} 第 {position} 个方案缺 key")
            if key in keys:
                raise ChoiceDataError("duplicate_alternative", f"{choice_id} 内方案键重复: {key}")
            keys.add(key)
            converted: list[float] = []
            raw_values: list[float] = []
            for attribute in config.attributes:
                if attribute not in raw_alternative:
                    raise ChoiceDataError(
                        "missing_attribute", f"{choice_id}/{key} 缺少属性 {attribute}"
                    )
                value = _number(raw_alternative[attribute], f"{choice_id}/{key}/{attribute}")
                raw_values.append(value)
                converted.append(value * units.factor if attribute == "price" else value)
            built.append(
                Alternative(
                    key=key,
                    attributes=tuple(converted),
                    raw_attributes=tuple(raw_values),
                )
            )

        chosen = row.get("chosen")
        if not isinstance(chosen, str) or not chosen:
            raise ChoiceDataError("missing_alternative_key", f"{choice_id} 的 chosen 必须为方案键字符串")
        chosen_index = next((i for i, alternative in enumerate(built) if alternative.key == chosen), None)
        if chosen_index is None:
            raise ChoiceDataError("chosen_not_offered", f"{choice_id} 的 chosen 不在 alternatives 中: {chosen}")

        choices.append(
            ChoiceSet(
                choice_id=choice_id,
                institution=institution,
                split=split,
                alternatives=tuple(built),
                chosen=chosen_index,
                outside_offered=OUTSIDE_KEY in keys,
            )
        )
    split_map: dict[str, set[str]] = {}
    for choice in choices:
        split_map.setdefault(choice.institution, set()).add(choice.split)
    conflicts = sorted(name for name, values in split_map.items() if len(values) > 1)
    if conflicts:
        raise ChoiceDataError(
            "split_conflict",
            f"同一机构同时出现在 fit 与 validation，整组划分会泄漏: {conflicts}",
        )
    return tuple(choices)


def load_analysis(
    material: Any,
    *,
    config_raw: Mapping[str, Any] | None = None,
) -> AnalysisInput:
    raw = config_raw if config_raw is not None else (material.get("config") if isinstance(material, Mapping) else None)
    if raw is not None and not isinstance(raw, Mapping):
        raise ChoiceDataError("invalid_config", "config 必须为对象")
    config = parse_config(raw)
    units = PriceUnits(config.price_factor, config.price_label)
    return AnalysisInput(config, units, parse_choices(material, config, price_units=units))


def fit_standardizer(choices: Iterable[ChoiceSet], attributes: Sequence[str]) -> Standardizer:
    """Estimate attribute means and scales on the fit split only."""

    width = len(attributes)
    values = [alternative.attributes for choice in choices if choice.split == "fit" for alternative in choice.alternatives]
    count = len(values)
    if count == 0:
        return Standardizer(tuple([0.0] * width), tuple([0.0] * width), 0)
    means = tuple(sum(value[j] for value in values) / count for j in range(width))
    scales = []
    for j in range(width):
        variance = sum((value[j] - means[j]) ** 2 for value in values) / count
        scale = math.sqrt(variance)
        scales.append(scale if scale > 0 else 0.0)
    return Standardizer(means, tuple(scales), count)


@dataclass(frozen=True)
class PreparedChoice:
    choice: ChoiceSet
    x: tuple[tuple[float, ...], ...]

    @property
    def chosen(self) -> int:
        return self.choice.chosen


def prepare(choices: Iterable[ChoiceSet], standardizer: Standardizer) -> tuple[PreparedChoice, ...]:
    return tuple(
        PreparedChoice(
            choice=choice,
            x=tuple(standardizer.apply(alternative.attributes) for alternative in choice.alternatives),
        )
        for choice in choices
    )


def prepare_raw(choices: Iterable[ChoiceSet]) -> tuple[PreparedChoice, ...]:
    return tuple(
        PreparedChoice(
            choice=choice,
            x=tuple(alternative.raw_attributes for alternative in choice.alternatives),
        )
        for choice in choices
    )


# ---------------------------------------------------------------------------
# Conditional likelihood
# ---------------------------------------------------------------------------


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ModelFailure(STATUS_CODES["no_choices"], "空集合没有对数分母")
    maximum = max(values)
    if not _is_finite(maximum):
        return maximum
    total = sum(math.exp(value - maximum) for value in values)
    return maximum + math.log(total)


def _utilities(beta: Sequence[float], x: Sequence[Sequence[float]]) -> list[float]:
    return [sum(beta[k] * row[k] for k in range(len(beta))) for row in x]


def conditional_log_likelihood(prepared: Iterable[PreparedChoice], beta: Sequence[float]) -> float:
    """Total conditional log-likelihood, summed over choice sets."""

    total = 0.0
    for item in prepared:
        utility = _utilities(beta, item.x)
        total += utility[item.chosen] - _logsumexp(utility)
    return total


def choice_probabilities(prepared: Iterable[PreparedChoice], beta: Sequence[float]) -> list[list[float]]:
    probabilities: list[list[float]] = []
    for item in prepared:
        utility = _utilities(beta, item.x)
        denominator = _logsumexp(utility)
        probabilities.append([math.exp(value - denominator) for value in utility])
    return probabilities


@dataclass(frozen=True)
class ObjectiveValue:
    log_likelihood: float
    ridge_penalty: float
    objective: float
    gradient: tuple[float, ...]          # d(objective)/dbeta, i.e. minimisation gradient
    information: tuple[tuple[float, ...], ...]  # d2(objective)/dbeta2
    likelihood_gradient: tuple[float, ...] = ()
    likelihood_information: tuple[tuple[float, ...], ...] = ()
    ridge: float | None = None
    penalized: bool = False


def evaluate_objective(
    prepared: Iterable[PreparedChoice],
    beta: Sequence[float],
    ridge: float | None,
    *,
    likelihood_only: bool = False,
) -> ObjectiveValue:
    """Gradient and Hessian of the log-likelihood (and of the fitting objective).

    At a penalised optimum the *likelihood* gradient is not zero; the gradient
    of the minimised objective (likelihood minus the ridge penalty) is.  Both
    are returned so neither is mistaken for the other:

    * ``likelihood_gradient`` / ``likelihood_information`` ignore the penalty;
    * ``gradient`` / ``information`` are the derivative of ``objective``.

    ``likelihood_only=True`` skips the penalty entirely.
    """

    width = len(beta)
    log_likelihood = 0.0
    gradient = [0.0] * width
    information = _zero(width, width)
    for item in prepared:
        utility = _utilities(beta, item.x)
        denominator = _logsumexp(utility)
        probabilities = [math.exp(value - denominator) for value in utility]
        expected = [0.0] * width
        second_moment = _zero(width, width)
        for alternative, probability in zip(item.x, probabilities):
            for j in range(width):
                expected[j] += probability * alternative[j]
                for k in range(width):
                    second_moment[j][k] += probability * alternative[j] * alternative[k]
        log_likelihood += utility[item.chosen] - denominator
        chosen_row = item.x[item.chosen]
        for j in range(width):
            gradient[j] += chosen_row[j] - expected[j]
            for k in range(width):
                information[j][k] += second_moment[j][k] - expected[j] * expected[k]

    # Log-likelihood derivatives before the penalty is applied.  These are
    # derivatives of the log-likelihood itself (maximisation direction), so
    # at a penalised optimum ``likelihood_gradient == ridge * beta``.
    likelihood_gradient = tuple(gradient)
    likelihood_information = tuple(tuple(row) for row in information)

    penalty = 0.0
    penalized = not likelihood_only and ridge is not None and ridge > 0
    if penalized:
        penalty = 0.5 * ridge * sum(value * value for value in beta)
        for j in range(width):
            gradient[j] -= ridge * beta[j]
        for j in range(width):
            information[j][j] += ridge

    return ObjectiveValue(
        log_likelihood=log_likelihood,
        ridge_penalty=penalty,
        objective=-(log_likelihood - penalty),
        gradient=tuple(-value for value in gradient),
        information=tuple(tuple(row) for row in information),
        likelihood_gradient=likelihood_gradient,
        likelihood_information=likelihood_information,
        ridge=ridge if penalized else None,
        penalized=penalized,
    )


def numerical_gradient(
    prepared: Iterable[PreparedChoice],
    beta: Sequence[float],
    ridge: float | None,
    step: float = 1e-6,
) -> list[float]:
    """Central finite differences of the minimised objective (test helper)."""

    base = list(beta)
    result: list[float] = []
    for j in range(len(base)):
        up = list(base)
        down = list(base)
        up[j] += step
        down[j] -= step
        result.append(
            (evaluate_objective(prepared, up, ridge).objective
             - evaluate_objective(prepared, down, ridge).objective) / (2 * step)
        )
    return result


def numerical_likelihood_gradient(
    prepared: Iterable[PreparedChoice],
    beta: Sequence[float],
    step: float = 1e-6,
) -> list[float]:
    """Central finite differences of d(log_likelihood)/dbeta, penalty-free."""

    base = list(beta)
    result: list[float] = []
    for j in range(len(base)):
        up = list(base)
        down = list(base)
        up[j] += step
        down[j] -= step
        likelihood_up = conditional_log_likelihood(prepared, up)
        likelihood_down = conditional_log_likelihood(prepared, down)
        result.append((likelihood_up - likelihood_down) / (2 * step))
    return result


def numerical_hessian(
    prepared: Iterable[PreparedChoice],
    beta: Sequence[float],
    ridge: float | None,
    step: float = 1e-5,
) -> list[list[float]]:
    """Central finite differences of the gradient, for analytic checks."""

    width = len(beta)
    base = list(beta)
    result = _zero(width, width)
    for j in range(width):
        up = list(base)
        down = list(base)
        up[j] += step
        down[j] -= step
        gradient_up = evaluate_objective(prepared, up, ridge).gradient
        gradient_down = evaluate_objective(prepared, down, ridge).gradient
        for k in range(width):
            result[k][j] = (gradient_up[k] - gradient_down[k]) / (2 * step)
    return result


# ---------------------------------------------------------------------------
# Identification diagnostics
# ---------------------------------------------------------------------------


def within_set_differences(prepared: Iterable[PreparedChoice]) -> list[list[float]]:
    """Enumerate x_ij - x_ik for every within-set pair in the fit scope.

    A conditional-logit beta is identified only from these differences; a
    rank-deficient difference matrix means some coefficient directions are not
    identified, regardless of what an optimiser reports.
    """

    rows: list[list[float]] = []
    for item in prepared:
        for j in range(1, item.choice.count):
            rows.append([item.x[j][k] - item.x[0][k] for k in range(len(item.x[0]))])
    return rows


def design_rank(prepared: Iterable[PreparedChoice]) -> tuple[int, int, bool]:
    items = tuple(prepared)
    rows = within_set_differences(items)
    width = len(items[0].x[0]) if items and items[0].x else 0
    rank = _rank(rows) if rows else 0
    return rank, width, width > 0 and rank == width


def separation_profile(prepared: Iterable[PreparedChoice]) -> dict[str, Any]:
    """Detect complete separation along each standardised attribute.

    For every attribute direction, compare the chosen alternative's value with
    the other offered alternatives.  If the chosen value is uniformly the
    maximum (or uniformly the minimum) wherever the direction varies within a
    set, the likelihood can be driven to a boundary by sending the coefficient
    to +/- infinity and no finite maximum-likelihood estimate identifies it.
    """

    items = tuple(prepared)
    width = len(items[0].x[0]) if items else 0
    profile: dict[str, Any] = {"attributes": [], "separated": False}
    for j in range(width):
        always_max = True
        always_min = True
        contrasting = 0
        for item in items:
            values = [row[j] for row in item.x]
            chosen_value = values[item.chosen]
            others = [value for index, value in enumerate(values) if index != item.chosen]
            if not others:
                continue
            if all(value == chosen_value for value in others):
                # Neutral set for this attribute: no within-set contrast.
                continue
            contrasting += 1
            # Strict dominance matters: if another alternative ties the
            # chosen value at the top, probability mass can still split
            # between the tied alternatives and the likelihood stays bounded.
            if not all(value < chosen_value for value in others):
                always_max = False
            if not all(value > chosen_value for value in others):
                always_min = False
        attribute = {
            "index": j,
            "contrasting_sets": contrasting,
            "chosen_strictly_maximum": always_max and contrasting > 0,
            "chosen_strictly_minimum": always_min and contrasting > 0,
        }
        profile["attributes"].append(attribute)
        if attribute["chosen_strictly_maximum"] or attribute["chosen_strictly_minimum"]:
            profile["separated"] = True
    return profile


def directional_separation(
    prepared: Iterable[PreparedChoice],
    beta: Sequence[float],
) -> dict[str, Any]:
    """Check strict dominance along the fitted utility direction.

    Per-attribute checks can miss separation along a combination of attributes
    (for example chosen alternatives that dominate jointly on price and
    frequency).  If in every contrasting set the chosen alternative has
    strictly the largest fitted utility, the likelihood has a boundary
    supremum and the finite fitted beta is not an MLE.
    """

    items = tuple(prepared)
    contrasting = 0
    dominant = 0
    for item in items:
        utilities = _utilities(beta, item.x)
        chosen_value = utilities[item.chosen]
        others = [value for index, value in enumerate(utilities) if index != item.chosen]
        if not others:
            continue
        if all(value == chosen_value for value in others):
            continue
        contrasting += 1
        if all(value < chosen_value for value in others):
            dominant += 1
    return {
        "contrasting_sets": contrasting,
        "chosen_strictly_dominant_sets": dominant,
        "separated_along_fitted_direction": bool(contrasting) and dominant == contrasting,
    }


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------


def _solve_linear(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float] | None:
    size = len(matrix)
    augmented = [[float(value) for value in row] for row in matrix]
    for index, value in enumerate(vector):
        augmented[index].append(float(value))
    for column in range(size):
        pivot = max(range(column, size), key=lambda index: abs(augmented[index][column]))
        if abs(augmented[pivot][column]) <= 1e-14:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for index in range(size):
            if index == column:
                continue
            factor = augmented[index][column]
            if factor:
                augmented[index] = [
                    value - factor * base for value, base in zip(augmented[index], augmented[column])
                ]
    return [augmented[index][size] for index in range(size)]


@dataclass(frozen=True)
class OptimisationResult:
    beta: tuple[float, ...]
    log_likelihood: float
    ridge_penalty: float
    objective: float
    converged: bool
    iterations: int
    message: str = ""
    hessian: tuple[tuple[float, ...], ...] = ()
    gradient_norm: float = 0.0


def optimise(
    prepared: Iterable[PreparedChoice],
    initial: Sequence[float],
    ridge: float | None,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
) -> OptimisationResult:
    """Damped Newton maximisation of the (optionally ridge-penalised) likelihood.

    Convergence is declared only when the gradient norm of the minimised
    objective falls below ``tolerance``; hitting the iteration limit or a
    non-finite step is reported as a failure, never as an estimate.
    """

    items = tuple(prepared)
    beta = [float(value) for value in initial]
    if not items:
        return OptimisationResult(
            beta=tuple(beta), log_likelihood=0.0, ridge_penalty=0.0, objective=0.0,
            converged=False, iterations=0, message=STATUS_CODES["no_choices"],
        )
    if any(not _is_finite(value) for value in beta):
        raise ModelFailure(STATUS_CODES["invalid_beta"], "初始 beta 含非有限值")

    current = evaluate_objective(items, beta, ridge)
    iterations = 0
    stalled = False
    for iterations in range(1, max_iterations + 1):
        gradient_norm = math.sqrt(sum(value * value for value in current.gradient))
        if gradient_norm <= tolerance:
            break
        direction = _solve_linear(current.information, [-value for value in current.gradient])
        if direction is None or any(not _is_finite(value) for value in direction):
            return OptimisationResult(
                beta=tuple(beta), log_likelihood=current.log_likelihood,
                ridge_penalty=current.ridge_penalty, objective=current.objective,
                converged=False, iterations=iterations - 1,
                message=STATUS_CODES["rank_deficient"],
                hessian=current.information, gradient_norm=gradient_norm,
            )
        slope = sum(current.gradient[j] * direction[j] for j in range(len(beta)))
        if slope > 0:
            # The Newton solve did not yield a descent direction (possible
            # with a numerically indefinite information matrix); fall back to
            # steepest descent.
            direction = [-value for value in current.gradient]
            slope = -sum(value * value for value in current.gradient)
        armijo_bound = 1e-4 * slope
        candidate_value = None
        candidate = beta
        for trial_scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            trial = [beta[j] + trial_scale * direction[j] for j in range(len(beta))]
            trial_value = evaluate_objective(items, trial, ridge)
            if trial_value.objective <= current.objective + trial_scale * armijo_bound:
                candidate, candidate_value = trial, trial_value
                break
        if candidate_value is None:
            # The Armijo condition could not be met at floating-point
            # resolution.  That is a legitimate optimum only when the gradient
            # is already small; otherwise it is a stalled optimiser, not an
            # estimate.
            stalled = True
            break
        beta, current = candidate, candidate_value
    final_norm = math.sqrt(sum(value * value for value in current.gradient))
    # Acceptance is either (a) the absolute gradient norm below tolerance,
    # or (b) the Newton decrement g' H^{-1} g below a floor scaled to the
    # objective magnitude (machine precision limits the achievable decrease
    # when the curvature is very large).  This is convergence of the
    # optimiser only; identification is decided separately by the rank and
    # separation diagnostics.
    decrement = None
    solved = _solve_linear(current.information, current.gradient) if current.information else None
    if solved is not None:
        decrement = sum(current.gradient[j] * solved[j] for j in range(len(beta)))
    decrement_floor = tolerance * max(1.0, abs(current.objective))
    converged = (final_norm <= tolerance or (
        decrement is not None and abs(decrement) <= decrement_floor
    )) and not (stalled and final_norm > tolerance)
    return OptimisationResult(
        beta=tuple(beta), log_likelihood=current.log_likelihood,
        ridge_penalty=current.ridge_penalty, objective=current.objective,
        converged=converged, iterations=iterations,
        message="" if converged else STATUS_CODES["optimizer_failed"],
        hessian=current.information, gradient_norm=final_norm,
    )


def curvature_diagnostics(information: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Eigenvalue-based condition of the converged information matrix."""

    values = _symmetric_eigenvalues(information)
    largest = max((abs(value) for value in values), default=0.0)
    smallest = min(values, default=0.0)
    return {
        "eigenvalues": list(values),
        "largest_absolute": largest,
        "smallest": smallest,
        "non_positive": any(value <= 1e-10 for value in values),
        "condition_estimate": (abs(largest) / abs(smallest)) if smallest and smallest > 1e-10 else None,
    }


def standard_errors(information: Sequence[Sequence[float]]) -> list[float] | None:
    inverted = _inverse(information)
    if inverted is None:
        return None
    diagonal = [inverted[j][j] for j in range(len(inverted))]
    if any(value <= 0 or not _is_finite(value) for value in diagonal):
        return None
    return [math.sqrt(value) for value in diagonal]


# ---------------------------------------------------------------------------
# Prediction, losses, elasticities and counterfactuals
# ---------------------------------------------------------------------------


def validation_log_loss(prepared: Iterable[PreparedChoice], beta: Sequence[float]) -> dict[str, Any]:
    """Mean negative log probability of the observed choice on validation sets."""

    items = tuple(prepared)
    if not items:
        return {"rows": 0, "mean_log_loss": None, "sum_log_loss": None, "unbounded_rows": []}
    probabilities = choice_probabilities(items, beta)
    losses = []
    unbounded = []
    for item, row in zip(items, probabilities):
        probability = row[item.chosen]
        if probability <= 0:
            losses.append(float("inf"))
            unbounded.append(item.choice.choice_id)
        else:
            losses.append(-math.log(probability))
    return {
        "rows": len(items),
        "mean_log_loss": (None if unbounded else sum(losses) / len(losses)),
        "sum_log_loss": (None if unbounded else sum(losses)),
        "unbounded_rows": unbounded,
        "note": (
            "验证集上有方案的观测选择概率为 0，对数损失无界；不以 Infinity 填表。"
            if unbounded else ""
        ),
    }


def in_set_predictions(prepared: Iterable[PreparedChoice], beta: Sequence[float]) -> list[dict[str, Any]]:
    """Reported probabilities are conditional on exactly the offered set."""

    items = tuple(prepared)
    probabilities = choice_probabilities(items, beta)
    output = []
    for item, row in zip(items, probabilities):
        output.append(
            {
                "choice_id": item.choice.choice_id,
                "institution": item.choice.institution,
                "split": item.choice.split,
                "keys": list(item.choice.keys),
                "chosen": item.choice.alternatives[item.chosen].key,
                "probabilities": row,
                "chosen_probability": row[item.chosen],
            }
        )
    return output


def owned_price_elasticity(
    prepared: Iterable[PreparedChoice],
    beta: Sequence[float],
    *,
    price_index: int,
    price_sigma: float = 1.0,
    price_unit_factor: float = 1.0,
) -> list[dict[str, Any]]:
    """Own-price elasticity of each alternative with respect to its own price.

    For a conditional logit the own-price elasticity of alternative j is
    ``beta_price * p_j * (1 - p_j) * price_j`` in the coefficient's own units.
    With a standardised price attribute the coefficient is

        beta_std = beta_raw * sigma,    price_std = (price_raw * factor - mean) / sigma

    so the elasticity at the observed price is

        (beta_std / sigma) * p_j * (1 - p_j) * price_raw * factor.

    It is reported at observed prices, conditional on the offered set.  It is
    a local property of the fitted probability model, not a causal price
    effect or a welfare measure.
    """

    items = tuple(prepared)
    sigma = price_sigma if price_sigma > 0 else 1.0
    factor = price_unit_factor if price_unit_factor > 0 else 1.0
    output = []
    for item, row in zip(items, choice_probabilities(items, beta)):
        entries = []
        for index, (alternative, probability) in enumerate(zip(item.choice.alternatives, row)):
            raw_price = alternative.raw_attributes[price_index]
            price = raw_price * factor
            entries.append(
                {
                    "key": alternative.key,
                    "price_raw": raw_price,
                    "price_analysis_units": price,
                    "probability": probability,
                    "own_price_elasticity": (beta[price_index] / sigma) * probability * (1 - probability) * price,
                }
            )
        output.append(
            {
                "choice_id": item.choice.choice_id,
                "institution": item.choice.institution,
                "alternatives": entries,
            }
        )
    return output


def removal_scenario(
    choice: PreparedChoice,
    beta: Sequence[float],
    key: str,
) -> dict[str, Any]:
    """Scenario probabilities after deleting one offered alternative.

    This is a purely mechanical renormalisation under the IIA assumption of
    the conditional-logit model; it is *not* an estimated substitution rate or
    a causal effect of removing the product.  The selected alternative is
    never fractionalised and the set is not padded with unoffered options.
    """

    keys = choice.choice.keys
    if key not in keys:
        raise ModelFailure(STATUS_CODES["not_offered"], f"{choice.choice.choice_id} 未提供方案 {key}")
    index = keys.index(key)
    full = choice_probabilities((choice,), beta)[0]
    if choice.chosen == index:
        return {
            "choice_id": choice.choice.choice_id,
            "removed": key,
            "chosen_removed": True,
            "scenario": "iia_renormalisation_scenario_not_substitution_rate",
            "probabilities_without": [],
            "keys_without": [value for value in keys if value != key],
            "chosen_probability_without": None,
        }
    kept = [j for j in range(choice.choice.count) if j != index]
    kept_probabilities = [full[j] for j in kept]
    total = sum(kept_probabilities)
    if total == 0:
        raise ModelFailure(STATUS_CODES["separation"], "删除后其余方案概率为零，无法归一")
    conditional = [value / total for value in kept_probabilities]
    chosen_key = choice.choice.alternatives[choice.chosen].key
    chosen_probability = conditional[kept.index(choice.chosen)]
    return {
        "choice_id": choice.choice.choice_id,
        "removed": key,
        "chosen_removed": False,
        "scenario": "iia_renormalisation_scenario_not_substitution_rate",
        "probabilities_without": conditional,
        "keys_without": [keys[j] for j in kept],
        "chosen_probability_without": chosen_probability,
        "iia_ratio_shift": total,
    }


def common_shift_scenario(
    choice: PreparedChoice,
    beta: Sequence[float],
    delta: Sequence[float],
) -> dict[str, Any]:
    """Probabilities after adding the same vector to every offered alternative."""

    shifted = tuple(
        tuple(row[k] + delta[k] for k in range(len(delta))) for row in choice.x
    )
    shifted_choice = PreparedChoice(choice.choice, shifted)
    before = choice_probabilities((choice,), beta)[0]
    after = choice_probabilities((shifted_choice,), beta)[0]
    return {
        "choice_id": choice.choice.choice_id,
        "before": before,
        "after": after,
        "max_absolute_change": max(abs(a - b) for a, b in zip(before, after)),
    }


def support_diagnostics(
    fit: Iterable[PreparedChoice],
    target: Iterable[PreparedChoice],
) -> list[dict[str, Any]]:
    """Flag attributes outside the fit-split support.

    Extrapolation has no supported interpretation here; the flag is reported
    rather than silently used to make a counterfactual claim.
    """

    fit_items = tuple(fit)
    width = len(fit_items[0].x[0]) if fit_items else 0
    bounds = []
    for j in range(width):
        values = [alternative.raw_attributes[j] for item in fit_items for alternative in item.choice.alternatives]
        bounds.append((min(values), max(values)) if values else (0.0, 0.0))
    output = []
    for item in target:
        outside: list[dict[str, Any]] = []
        for alternative in item.choice.alternatives:
            for j in range(width):
                value = alternative.raw_attributes[j]
                low, high = bounds[j]
                if value < low or value > high:
                    outside.append(
                        {
                            "attribute": j,
                            "key": alternative.key,
                            "value": value,
                            "fit_min": low,
                            "fit_max": high,
                        }
                    )
        output.append(
            {
                "choice_id": item.choice.choice_id,
                "flags": outside,
                "supported": not outside,
            }
        )
    return output


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _rng(seed: int):
    state = seed & ((1 << 64) - 1)

    def next_int(upper: int) -> int:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 33) % upper

    return next_int


def _resample_groups(
    prepared: Sequence[PreparedChoice],
    institutions: Sequence[str],
    next_int,
) -> tuple[PreparedChoice, ...]:
    """Whole-institution resampling with replacement.

    The same institution's choice sets stay together; the replicate inherits
    all of its rows, and a replicate may repeat an institution group.
    """

    by_institution: dict[str, tuple[PreparedChoice, ...]] = {}
    for item in prepared:
        by_institution.setdefault(item.choice.institution, ())
        by_institution[item.choice.institution] += (item,)
    drawn = []
    for _ in range(len(institutions)):
        institution = institutions[next_int(len(institutions))]
        drawn.extend(by_institution[institution])
    return tuple(drawn)


def bootstrap(
    prepared: Sequence[PreparedChoice],
    beta: Sequence[float],
    ridge: float | None,
    *,
    repeats: int,
    seed: int,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
) -> dict[str, Any]:
    """Whole-institution refits.  Each failed replicate is reported separately.

    No successful estimate is substituted for a failed replicate; the failed
    row keeps its reason code so coverage is visible rather than averaged away.
    """

    items = tuple(prepared)
    institutions = sorted({item.choice.institution for item in items})
    if not items or not institutions:
        return {"requested": repeats, "successful": 0, "failed": 0, "replicates": []}
    next_int = _rng(seed)
    replicates: list[dict[str, Any]] = []
    for index in range(repeats):
        sample = _resample_groups(items, institutions, next_int)
        try:
            result = optimise(sample, beta, ridge, tolerance=tolerance, max_iterations=max_iterations)
        except ModelFailure as failure:
            replicates.append({"replicate": index, "status": "failed", "code": failure.code, "detail": str(failure)})
            continue
        if not result.converged:
            replicates.append(
                {
                    "replicate": index,
                    "status": "failed",
                    "code": STATUS_CODES["bootstrap_failed"],
                    "detail": result.message,
                }
            )
            continue
        rank, width, full_rank = design_rank(sample)
        separated = separation_profile(sample)["separated"]
        directional = directional_separation(sample, result.beta)
        replicate_identified = (
            full_rank
            and not separated
            and not directional["separated_along_fitted_direction"]
        )
        replicates.append(
            {
                "replicate": index,
                "status": "success",
                "beta": list(result.beta),
                "identified": replicate_identified,
                "reason_codes": (
                    [] if replicate_identified
                    else [
                        code
                        for code, flag in (
                            (STATUS_CODES["rank_deficient"], not full_rank),
                            (STATUS_CODES["separation"], separated or directional["separated_along_fitted_direction"]),
                        )
                        if flag
                    ]
                ),
            }
        )
    successful = [row for row in replicates if row["status"] == "success"]
    identified = [row for row in successful if row.get("identified")]
    return {
        "requested": repeats,
        "successful": len(successful),
        "failed": repeats - len(successful),
        "identified": len(identified),
        "replicates": replicates,
    }


def bootstrap_summary(result: Mapping[str, Any], parameter_names: Sequence[str]) -> dict[str, Any]:
    successful_rows = [row for row in result.get("replicates", []) if row["status"] == "success"]
    successful = [row["beta"] for row in successful_rows]
    output: dict[str, Any] = {
        "requested": result.get("requested", 0),
        "successful": result.get("successful", 0),
        "failed": result.get("failed", 0),
        "identified": result.get("identified", 0),
        "replicates": result.get("replicates", []),
    }
    if not successful:
        output["point"] = None
        output["standard_error"] = None
        output["intervals"] = None
        output["failure_rate"] = 1.0 if result.get("requested") else None
        return output
    width = len(successful[0])
    means = [sum(row[j] for row in successful) / len(successful) for j in range(width)]
    spread = [
        math.sqrt(sum((row[j] - means[j]) ** 2 for row in successful) / len(successful)) for j in range(width)
    ]
    ordered = [sorted(row[j] for row in successful) for j in range(width)]

    def quantile(values: Sequence[float], probability: float) -> float:
        position = probability * (len(values) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return values[low]
        return values[low] + (values[high] - values[low]) * (position - low)

    output["point"] = dict(zip(parameter_names, means))
    output["standard_error"] = dict(zip(parameter_names, spread))
    output["intervals"] = {
        name: {"low": quantile(values, 0.025), "high": quantile(values, 0.975)}
        for name, values in zip(parameter_names, ordered)
    }
    identified = [row["beta"] for row in successful_rows if row.get("identified")]
    output["identified_only"] = None
    if identified:
        width_id = len(identified[0])
        means_id = [sum(row[j] for row in identified) / len(identified) for j in range(width_id)]
        ordered_id = [sorted(row[j] for row in identified) for j in range(width_id)]
        output["identified_only"] = {
            "rows": len(identified),
            "mean": dict(zip(parameter_names, means_id)),
            "intervals": {
                name: {"low": quantile(values, 0.025), "high": quantile(values, 0.975)}
                for name, values in zip(parameter_names, ordered_id)
            },
            "note": "仅通过识别诊断的重拟合复制；样本量小，不替代正式区间。",
        }
    output["failure_rate"] = result.get("failed", 0) / result["requested"] if result.get("requested") else None
    output["unidentified_rate"] = (
        (len(successful_rows) - result.get("identified", 0)) / result["requested"]
        if result.get("requested") else None
    )
    output["point_note"] = "bootstrap_mean_not_estimate"
    output["intervals_basis"] = (
        "所有成功的惩罚重拟合；identified 计数单独给出，区间不因未识别而删去重拟合。"
    )
    return output

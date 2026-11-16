"""Train on the prepared Austin cohort, then score the untouched time holdout.

Run this script with `validate` followed by `test`. The test stage reuses the
same fitted models and the thresholds recorded by the validation stage.
"""

import argparse
import hashlib
import json
from functools import partial
from importlib import import_module
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits


pilot = import_module("5001_首次实验")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/prepared/猫狗入所配对表_2022–2024.csv"
RESULTS = ROOT / "results/formal"
MODELS = RESULTS / "models.joblib"
PROTOCOL = RESULTS / "validation_protocol.json"
RECALL_TARGET = 0.70
BOOTSTRAP_DRAWS = 1000
KITTEN = "Cat <3 mo"
BASE = (*pilot.BASE_CATEGORIES, "log_age")
CONTEXT = tuple(pilot.INTAKE_CONTEXT)
AGE_FEATURES = ("age_band", "species_age_band")
CONTINUOUS = (*BASE, *CONTEXT)
BANDS = (*pilot.FEATURES, "age_band")
SPECS = {
    "historical": ("historical", (), None),
    "logit_bands": ("logit", BANDS, None),
    "bagging_bands": ("bagging", BANDS, 42),
    "mlp_bands": ("mlp", BANDS, 42),
    "bayes_bands": ("bayes", BANDS, None),
    "logit_cont": ("logit", CONTINUOUS, None),
    "bagging_cont": ("bagging", CONTINUOUS, 42),
    "mlp_cont": ("mlp", CONTINUOUS, 42),
    "logit_base": ("logit", BASE, None),
    "logit_a": ("logit", (*BASE, *AGE_FEATURES), None),
    "logit_b_with_bands": ("logit", (*CONTINUOUS, "age_band"), None),
    "logit_ab": ("logit", (*CONTINUOUS, *AGE_FEATURES), None),
    "bagging_ab": ("bagging", (*CONTINUOUS, *AGE_FEATURES), 42),
    "tree_cont": ("tree", CONTINUOUS, 42),
    "forest_cont": ("forest", CONTINUOUS, 42),
    "mlp_bands_seed77": ("mlp", BANDS, 77),
    "mlp_bands_seed123": ("mlp", BANDS, 123),
    "mlp_cont_seed77": ("mlp", CONTINUOUS, 77),
    "mlp_cont_seed123": ("mlp", CONTINUOUS, 123),
}
WINDOW_MODELS = ("historical", "logit_ab", "bagging_cont")
ERROR_MODELS = ("logit_ab", "bagging_cont", "mlp_cont", "forest_cont", "bayes_bands")
AP_COMPARISONS = (
    ("bagging_bands", "logit_bands"),
    ("mlp_bands", "bagging_bands"),
    ("bayes_bands", "logit_bands"),
    ("bagging_cont", "logit_ab"),
    ("mlp_cont", "bagging_cont"),
    ("logit_cont", "logit_bands"),
    ("bagging_cont", "bagging_bands"),
    ("mlp_cont", "mlp_bands"),
    ("logit_a", "logit_base"),
    ("logit_ab", "logit_cont"),
    ("logit_cont", "logit_base"),
    ("logit_ab", "logit_a"),
    ("logit_ab", "logit_b_with_bands"),
    ("bagging_ab", "bagging_cont"),
    ("bagging_cont", "tree_cont"),
    ("forest_cont", "bagging_cont"),
)


def load_splits(names):
    raw = pd.read_csv(
        DATA,
        dtype={"adopted_30d": "Int8", "adopted_60d": "Int8", "outcome_id": "Int64",
               "seen_in_earlier_split": bool},
    )
    eligible = raw.loc[~raw["seen_in_earlier_split"] & raw["split"].isin(names)].copy()
    if not eligible["episode_id"].is_unique or eligible.groupby("animal_id")["split"].nunique().gt(1).any():
        raise ValueError("Episode keys must be unique and animal IDs must stay within one split.")
    categorical = [*pilot.FEATURES, *AGE_FEATURES]
    eligible[categorical] = eligible[categorical].fillna("Unknown")
    return {
        name: eligible.loc[eligible["split"].eq(name)].reset_index(drop=True)
        for name in names
    }


def analysis_rows(frame, cohort):
    required = ["adopted_30d"] if cohort == "primary" else ["adopted_30d", "adopted_60d"]
    return frame.dropna(subset=required).reset_index(drop=True)


def experiment_groups():
    return (
        ("primary", 30, tuple(SPECS)),
        ("common_windows", 30, WINDOW_MODELS),
        ("common_windows", 60, WINDOW_MODELS),
    )


def feature_pipeline(method, features, estimator):
    if method == "bayes":
        return make_pipeline(
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64),
            FunctionTransformer(partial(np.add, 1)),
            estimator,
        )
    categories = [field for field in features if field != "log_age"]
    transforms = [("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categories)]
    if "log_age" in features:
        age = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
        transforms.append(("age", age, ["log_age"]))
    return make_pipeline(ColumnTransformer(transforms), estimator)


def fit_experiment(train, target, run_id):
    method, fields, seed = SPECS[run_id]
    if method == "historical":
        return float(train[target].mean()), {"train_prior": float(train[target].mean())}

    x = train[list(fields)]
    y = train[target].to_numpy(dtype=int)
    if method == "mlp":
        # Retry only when the optimizer hits its limit. Labels from validation
        # do not participate in that decision.
        attempts = []
        for limit in (3000, 6000):
            classifier = MLPClassifier(
                hidden_layer_sizes=(16,), activation="tanh", solver="lbfgs", alpha=0.5,
                max_iter=limit, max_fun=limit * 10, random_state=seed,
            )
            model = feature_pipeline(method, fields, classifier)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x, y)
            problems = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
            attempts.append({"limit": limit, "iterations": int(classifier.n_iter_), "warnings": problems})
            if not problems:
                break
        return model, {"optimizer_attempts": attempts}

    if method == "logit":
        classifier = pilot.logistic()
    elif method == "tree":
        classifier = DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=seed)
    elif method in {"bagging", "forest"}:
        classifier = pilot.forest().set_params(
            bootstrap=True, max_features=None if method == "bagging" else "sqrt", random_state=seed
        )
    else:
        classifier = CategoricalNB(alpha=1.0)
    model = feature_pipeline(method, fields, classifier)
    model.fit(x, y)
    return model, {}


def run_key(cohort, horizon, run_id):
    return f"{cohort}/{horizon}/{run_id}"


def predict(rows, run_id, model):
    method, fields, _ = SPECS[run_id]
    if method == "historical":
        return np.full(len(rows), model)
    return model.predict_proba(rows[list(fields)])[:, 1]


def validation_threshold(y, probabilities):
    positive_scores = np.sort(probabilities[y == 1])
    needed = int(np.ceil(RECALL_TARGET * len(positive_scores)))
    return float(positive_scores[-needed])


def groups_for(rows):
    groups = [("all", "All", rows.index), ("pairing", "Clear first outcome", rows.index[rows["outcome_id"].notna() & rows["outcome_type"].notna()])]
    months = rows["intake_month"]
    earlier_half = months.le("2024-03") if months.iat[0] < "2024-07" else months.le("2024-10")
    for tag, subset in rows.groupby(earlier_half):
        period = subset["intake_month"]
        groups.append(("period", f"{period.min()} to {period.max()}", subset.index))
    for field in ("animal_type", "species_age_band", "intake_type", "intake_condition", "intake_month"):
        groups.extend((field, label, subset.index) for label, subset in rows.groupby(field))
    return groups


def prediction_metrics(rows, horizon, probabilities, run_id, cutoff, cohort):
    y = rows[f"adopted_{horizon}d"].astype(int)
    method, _, seed = SPECS[run_id]
    for kind, name, indexes in groups_for(rows):
        if len(indexes) == 0:
            continue
        p = probabilities[indexes]
        reference = pilot.score(y.iloc[indexes], p)
        tuned = {}
        if cutoff is not None:
            score = pilot.score(y.iloc[indexes], p, threshold=cutoff)
            tuned = {f"v70_{metric}": score[metric] for metric in ("tp", "fp", "fn", "precision", "recall")}
        yield {
            "cohort": cohort, "window_days": horizon, "run": run_id, "method": method, "seed": seed,
            "group_type": kind, "group": name, "threshold": pilot.THRESHOLD,
            "validation_recall_threshold": cutoff, **reference, **tuned,
        }


def paired_ap(rows, probabilities, comparisons):
    """Resample evaluation animals so every comparison uses the same draws."""
    clusters = list(rows.groupby("animal_id", sort=False).indices.values())
    y = rows["adopted_30d"].to_numpy(dtype=int)
    names = sorted({name for pair in comparisons for name in pair})
    point = {name: average_precision_score(y, probabilities[name]) for name in names}
    rng = np.random.default_rng(pilot.SEED)
    draws = np.empty((BOOTSTRAP_DRAWS, len(comparisons)))
    for j in range(BOOTSTRAP_DRAWS):
        draw = rng.integers(len(clusters), size=len(clusters))
        index = np.concatenate([clusters[i] for i in draw])
        values = {name: average_precision_score(y[index], probabilities[name][index]) for name in names}
        draws[j] = [values[a] - values[b] for a, b in comparisons]
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return pd.DataFrame(
        [
            {"run": a, "reference": b, "delta_ap": point[a] - point[b], "ci_low": lo, "ci_high": hi}
            for (a, b), lo, hi in zip(comparisons, low, high)
        ]
    )


def bayes_dependence(train):
    result = []
    for species in ("All", "Cat", "Dog"):
        subset = train if species == "All" else train.loc[train["animal_type"].eq(species)]
        for label, rows in subset.groupby("adopted_30d"):
            young = rows["age_band"].eq("<3 mo")
            neonatal = rows["intake_condition"].eq("Neonatal")
            result.append({
                "animal_type": species, "adopted_30d": int(label), "n": len(rows),
                "young": int(young.sum()), "neonatal": int(neonatal.sum()),
                "observed_both": int((young & neonatal).sum()),
                "expected_both_if_independent": float(young.mean() * neonatal.mean() * len(rows)),
            })
    return pd.DataFrame(result)


def trace_kitten_errors(rows, probabilities):
    y = rows["adopted_30d"].astype(int)
    days = rows["days_to_outcome"]
    outcomes = rows["outcome_type"].fillna("No unambiguous exit")
    outcomes = outcomes.mask(outcomes.eq("Adoption") & days.between(31, 60), "Adoption on days 31–60")
    outcomes = outcomes.mask(outcomes.eq("Adoption") & days.gt(60), "Adoption after day 60")
    kittens = rows["species_age_band"].eq(KITTEN)
    score_rows = []
    for name in ERROR_MODELS:
        yes = probabilities[name] >= pilot.THRESHOLD
        fp = kittens & y.eq(0) & yes
        fn = kittens & y.eq(1) & ~yes
        for result, counts in (
            ("false_positive_outcome", outcomes.loc[fp].value_counts()),
            ("false_negative_intake_type", rows.loc[fn, "intake_type"].value_counts()),
        ):
            score_rows.extend(
                {"run": name, "error": result, "category": category, "n": int(n)}
                for category, n in counts.items()
            )
    return pd.DataFrame(score_rows)


def window_reclassification(rows, predictions):
    y30 = rows["adopted_30d"].eq(1)
    y60 = rows["adopted_60d"].eq(1)
    results = []
    for name in WINDOW_MODELS[1:]:
        for group, subset in {"All": rows.index, KITTEN: rows.index[rows["species_age_band"].eq(KITTEN)]}.items():
            fp30 = (predictions[30][name] >= pilot.THRESHOLD) & ~y30
            fp60 = (predictions[60][name] >= pilot.THRESHOLD) & ~y60
            results.append({
                "run": name, "group": group, "n": len(subset),
                "positive_30d": int(y30.iloc[subset].sum()), "positive_60d": int(y60.iloc[subset].sum()),
                "false_positive_30d": int(fp30.iloc[subset].sum()),
                "original_false_positives_adopted_d31_to_60": int((fp30 & y60).iloc[subset].sum()),
                "false_positive_60d_after_refit": int(fp60.iloc[subset].sum()),
            })
    return pd.DataFrame(results)


def case_studies(rows, probabilities):
    fields = [*pilot.FEATURES, "age_upon_intake"]
    columns = ["episode_id", "intake_day", *fields, "outcome_type", "days_to_outcome", "adopted_30d", "adopted_60d"]
    kittens = rows.loc[rows["species_age_band"].eq(KITTEN)].copy()
    output = []
    adoption = kittens.loc[kittens["outcome_type"].eq("Adoption") & kittens["days_to_outcome"].le(60)].copy()
    profiles = adoption.groupby(fields)["adopted_30d"]
    early = profiles.transform("sum")
    support = np.minimum(early, profiles.transform("size") - early)
    if support.gt(0).any():
        profile = adoption.loc[support.idxmax(), fields]
        matching = adoption.loc[adoption[fields].eq(profile).all(axis=1)]
        pair = matching.drop_duplicates("adopted_30d").sort_values("adopted_30d", ascending=False)
        output.append(pair[columns].assign(case="same_features", profile_records=int(len(matching))))
    a, b = probabilities["bagging_cont"], probabilities["mlp_cont"]
    gap = np.abs(a - b)
    for preferred, other, name in ((a, b, "bagging_only"), (b, a, "mlp_only")):
        differing = kittens.index[(preferred[kittens.index] >= 0.5) & (other[kittens.index] < 0.5)]
        if len(differing):
            index = differing[np.argmax(gap[differing])]
            output.append(rows.loc[[index], columns].assign(case=name, profile_records=pd.NA))
    table = pd.concat(output) if output else rows.loc[[], columns].assign(case="", profile_records=pd.NA)
    for name in ERROR_MODELS:
        table[name] = probabilities[name][table.index]
    return table


def save_table(path, name, frame):
    frame.to_csv(path / f"{name}.csv", index=False, encoding="utf-8-sig")


def evaluate(split, frame, models, thresholds):
    path = RESULTS / split
    path.mkdir(parents=True, exist_ok=True)
    metrics = []
    window_probs = {}
    window_rows = None
    for cohort, horizon, names in experiment_groups():
        rows = analysis_rows(frame, cohort)
        if cohort == "common_windows":
            window_rows = rows
        p = {name: predict(rows, name, models[run_key(cohort, horizon, name)]) for name in names}
        for name, values in p.items():
            if not (np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))):
                raise ValueError(f"Invalid probabilities for {name}")
            cutoff = thresholds[run_key(cohort, horizon, name)]
            metrics.extend(prediction_metrics(rows, horizon, values, name, cutoff, cohort))
        predictions = rows[["episode_id", "animal_id", "adopted_30d", "adopted_60d"]].assign(**p)
        save_table(path, f"predictions_{cohort}_{horizon}d", predictions)
        if cohort == "primary":
            save_table(path, "ap_comparisons", paired_ap(rows, p, AP_COMPARISONS))
            save_table(path, "kitten_error_outcomes", trace_kitten_errors(rows, p))
            save_table(path, "case_studies", case_studies(rows, p))
        else:
            window_probs[horizon] = p
    save_table(path, "window_reclassification", window_reclassification(window_rows, window_probs))
    table = pd.DataFrame(metrics)
    save_table(path, "metrics", table)
    overall = table.loc[table["group_type"].eq("all"), ["cohort", "run", "window_days", "n", "ap", "brier", "recall", "fp"]]
    print(overall.round(4).to_string(index=False), flush=True)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    return {
        "data_sha256": fingerprint(DATA),
        "experiment_sha256": fingerprint(Path(__file__)),
        "pilot_sha256": fingerprint(Path(pilot.__file__)),
        "versions": {"scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }


def validate():
    sets = load_splits(("train", "validation"))
    models, thresholds, settings = {}, {}, {}
    for cohort, horizon, names in experiment_groups():
        train = analysis_rows(sets["train"], cohort)
        validation = analysis_rows(sets["validation"], cohort)
        target = f"adopted_{horizon}d"
        print(f"{cohort} {horizon}d · train {len(train):,}, validation {len(validation):,}", flush=True)
        for name in names:
            key = run_key(cohort, horizon, name)
            start = perf_counter()
            model, details = fit_experiment(train, target, name)
            model_probability = predict(validation, name, model)
            method, features, seed = SPECS[name]
            settings[key] = {"method": method, "features": features, "seed": seed,
                             "training_rows": len(train), "training_positive_rows": int(train[target].sum()),
                             "fit_seconds": round(perf_counter() - start, 2), **details}
            if method != "historical":
                settings[key]["parameters"] = model[-1].get_params(deep=False)
            models[key] = model
            thresholds[key] = None if method == "historical" else validation_threshold(
                validation[target].to_numpy(dtype=int), model_probability
            )
            print(f"  {name} {settings[key]['fit_seconds']:.1f}s", details, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    evaluate("validation", sets["validation"], models, thresholds)
    save_table(RESULTS / "validation", "naive_bayes_independence", bayes_dependence(analysis_rows(sets["train"], "primary")))
    model_temp = MODELS.with_suffix(".tmp")
    joblib.dump(models, model_temp, compress=3)
    model_temp.replace(MODELS)
    protocol = {
        "inputs": inputs(), "model_sha256": fingerprint(MODELS),
        "validation_recall_target": RECALL_TARGET, "reference_threshold": pilot.THRESHOLD,
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": pilot.SEED,
        "ap_comparisons": AP_COMPARISONS, "thresholds": thresholds, "settings": settings,
    }
    protocol_temp = PROTOCOL.with_suffix(".tmp")
    protocol_temp.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol_temp.replace(PROTOCOL)
    print(f"Validation models and thresholds recorded in {PROTOCOL.name}", flush=True)


def test():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["inputs"] != inputs() or protocol["model_sha256"] != fingerprint(MODELS):
        raise ValueError("The experiment inputs or fitted models differ from the validation run.")
    frame = load_splits(("train", "validation", "test"))["test"]
    models = joblib.load(MODELS)
    evaluate("test", frame, models, protocol["thresholds"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "test"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        (validate if args.stage == "validate" else test)()

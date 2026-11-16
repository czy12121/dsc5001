"""DSC5001 pilot on the City of Austin's January–July 2024 data snapshots.

Run with Python, pandas, NumPy, scikit-learn and Matplotlib. Paths are relative
to this script. The source queries are recorded in data/sources.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
THRESHOLD = 0.5
HORIZON = pd.Timedelta(days=30)
AGE_BANDS = ["<3 mo", "3–<6 mo", "6–<12 mo", "1–<7 yr", "7+ yr"]
BASE_CATEGORIES = ["animal_type", "sex_upon_intake"]
INTAKE_CONTEXT = ["intake_type", "intake_condition"]
FEATURES = BASE_CATEGORIES + INTAKE_CONTEXT
MAIN_MODELS = ["Historical rate", "Logistic regression", "Decision tree", "Random forest"]


def load_episodes():
    intakes = pd.read_json(ROOT / "data/intakes_2024_Jan_Jul.json", convert_dates=False)
    outcomes = pd.read_json(ROOT / "data/outcomes_2024_Jan_Jul.json", convert_dates=False)

    # The archive mixes local timestamps with midnight values carrying an offset.
    # Preserve the reported local date and treat midnight outcomes as date-only.
    intakes["intake_at"] = pd.to_datetime(intakes["datetime"].str[:19], format="ISO8601")
    outcomes["outcome_at"] = pd.to_datetime(outcomes["datetime"].str[:19], format="ISO8601")
    intakes["intake_day"] = intakes["intake_at"].dt.normalize()
    outcomes["outcome_day"] = outcomes["outcome_at"].dt.normalize()
    outcomes["date_only"] = outcomes["outcome_at"].eq(outcomes["outcome_day"])

    if intakes.duplicated(["animal_id", "intake_day"]).any():
        raise ValueError("Resolve animal IDs with more than one intake on the same date.")
    if outcomes.duplicated(["animal_id", "outcome_day"]).any():
        raise ValueError("Resolve animal IDs with more than one outcome on the same date.")
    if not intakes["intake_at"].notna().all() or not outcomes["outcome_at"].notna().all():
        raise ValueError("Resolve missing timestamps before matching the snapshots.")

    intakes = intakes.sort_values("intake_at")
    intakes["next_intake_at"] = intakes.groupby("animal_id")["intake_at"].shift(-1)
    repeat_days = intakes.loc[intakes.duplicated("animal_id"), ["animal_id", "intake_day"]]
    outcomes = outcomes.merge(
        repeat_days.assign(repeat_day=True),
        left_on=["animal_id", "outcome_day"],
        right_on=["animal_id", "intake_day"],
        how="left",
    )
    outcomes["ambiguous_day"] = outcomes["date_only"] & outcomes["repeat_day"].eq(True)
    end_of_day = outcomes["outcome_day"] + pd.Timedelta(days=1, seconds=-1)
    outcomes["order_at"] = outcomes["outcome_at"].where(~outcomes["date_only"], end_of_day)

    episode = pd.merge_asof(
        intakes,
        outcomes[["animal_id", "animal_type", "order_at", "outcome_day", "outcome_type", "ambiguous_day"]]
        .sort_values("order_at"),
        left_on="intake_at",
        right_on="order_at",
        by=["animal_id", "animal_type"],
        direction="forward",
    )
    before_next_intake = episode["next_intake_at"].isna() | episode["order_at"].lt(episode["next_intake_at"])
    matched = episode["order_at"].notna() & before_next_intake
    episode["days_to_outcome"] = (episode["outcome_day"] - episode["intake_day"]).dt.days.where(matched)
    within_30 = episode["days_to_outcome"].between(0, 30)
    next_in_30 = episode["next_intake_at"].dt.normalize().le(episode["intake_day"] + HORIZON)
    episode["label_unknown"] = (within_30 & (episode["ambiguous_day"].eq(True) | episode["outcome_type"].isna())) | (
        ~matched & next_in_30
    )
    episode["matched_outcome"] = matched & ~episode["ambiguous_day"].eq(True)
    episode["outcome_type"] = episode["outcome_type"].where(episode["matched_outcome"])
    episode["adopted_30d"] = (within_30 & episode["outcome_type"].eq("Adoption")).astype(int)
    episode["intake_month"] = episode["intake_day"].dt.strftime("%Y-%m")

    age_parts = episode["age_upon_intake"].str.extract(r"^(\d+) (day|week|month|year)s?$")
    units = age_parts[1].map({"day": 1 / 365.25, "week": 7 / 365.25, "month": 1 / 12, "year": 1})
    age_years = pd.to_numeric(age_parts[0]) * units
    # A recorded age of zero years has no finer precision.
    age_years = age_years.mask(age_parts[1].eq("year") & age_parts[0].eq("0"))
    episode["log_age"] = np.log1p(age_years * 365.25)
    episode["age_band"] = pd.cut(
        age_years, bins=[0, 0.25, 0.5, 1, 7, np.inf], labels=AGE_BANDS, right=False
    ).astype("str").fillna("Unknown")
    episode["species_age_band"] = episode["animal_type"] + " " + episode["age_band"]
    return episode, {"intake_rows": len(intakes), "outcome_rows": len(outcomes), "date_only_outcomes": int(outcomes["date_only"].sum())}


def make_splits(episodes):
    train_all = episodes.loc[episodes["intake_month"].isin(["2024-01", "2024-02", "2024-03"])]
    validation_all = episodes.loc[episodes["intake_month"].isin(["2024-05", "2024-06"])]
    seen = validation_all["animal_id"].isin(train_all["animal_id"])
    train = train_all.loc[~train_all["label_unknown"]].copy()
    validation = validation_all.loc[~validation_all["label_unknown"] & ~seen].copy().reset_index(drop=True)
    for rows in [train, validation]:
        if rows["adopted_30d"].nunique() != 2:
            raise ValueError("Both label values are needed in each split.")
    assert not set(train["animal_id"]).intersection(validation["animal_id"])
    audit = {
        "train": split_counts(train_all, train),
        "validation": split_counts(validation_all, validation),
        "validation_rows_seen_in_train": int(seen.sum()),
        "validation_unique_animals_seen_in_train": int(validation_all.loc[seen, "animal_id"].nunique()),
    }
    return train, validation, audit


def split_counts(before, kept):
    return {
        "raw_rows": len(before),
        "label_unknown": int(before["label_unknown"].sum()),
        "kept_rows": len(kept),
        "positive_rows": int(kept["adopted_30d"].sum()),
        "rows_linked_to_exit_by_jul31": int(kept["matched_outcome"].sum()),
        "rows_with_unknown_age": int(kept["log_age"].isna().sum()),
    }


def logistic():
    return LogisticRegression(C=1.0, max_iter=1000)


def forest():
    return RandomForestClassifier(
        n_estimators=350, max_depth=5, min_samples_leaf=25, random_state=SEED, n_jobs=2
    )


def fit_probabilities(train, validation):
    experiments = [
        ("Historical rate", DummyClassifier(strategy="prior"), FEATURES),
        ("Logistic regression", logistic(), FEATURES),
        ("Decision tree", DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=SEED), FEATURES),
        ("Random forest", forest(), FEATURES),
        ("Logistic + age bands", logistic(), FEATURES + ["age_band"]),
        ("Logistic + age bands and interaction", logistic(), FEATURES + ["age_band", "species_age_band"]),
        ("Logistic without intake context", logistic(), BASE_CATEGORIES),
        ("Forest without intake context", forest(), BASE_CATEGORIES),
    ]
    probabilities = {}
    for name, model, categories in experiments:
        transform = ColumnTransformer(
            [
                ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categories),
                ("age", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), ["log_age"]),
            ]
        )
        pipeline = make_pipeline(transform, model)
        columns = categories + ["log_age"]
        pipeline.fit(train[columns], train["adopted_30d"])
        probabilities[name] = pipeline.predict_proba(validation[columns])[:, 1]
    return pd.DataFrame(probabilities, index=validation.index)


def score(y, probability, threshold=THRESHOLD):
    predicted = probability >= threshold
    positive = y == 1
    tp = int((predicted & positive).sum())
    fp = int((predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    tn = int((~predicted & ~positive).sum())
    return {
        "n": len(y),
        "positives": tp + fn,
        "prevalence": float(y.mean()),
        "mean_probability": float(probability.mean()),
        "ap": average_precision_score(y, probability) if 0 < tp + fn < len(y) else np.nan,
        "macro_f1": f1_score(y, predicted, average="macro", labels=[0, 1], zero_division=0),
        "brier": brier_score_loss(y, probability),
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def score_subgroups(train, validation, probabilities):
    rows = []
    for column in ["animal_type", "species_age_band", "intake_type", "intake_month"]:
        training_groups = train.groupby(column)["adopted_30d"].agg(["size", "mean"])
        for value, group in validation.groupby(column):
            size = int(training_groups["size"].get(value, 0))
            rate = float(training_groups["mean"].get(value, np.nan))
            models = MAIN_MODELS[1:] + ["Logistic + age bands and interaction"]
            if column == "intake_type":
                models = models + ["Forest without intake context"]
            for model in models:
                result = score(group["adopted_30d"], probabilities.loc[group.index, model])
                rows.append({"grouping": column, "group": value, "model": model, "train_n": size, "train_rate": rate, **result})
    return pd.DataFrame(rows)


def paired_ap_intervals(validation, probabilities, n_draws=1000):
    comparisons = [
        ("Random forest", "Logistic regression"),
        ("Random forest", "Decision tree"),
        ("Logistic + age bands", "Logistic regression"),
        ("Logistic + age bands and interaction", "Logistic + age bands"),
        ("Logistic regression", "Logistic without intake context"),
        ("Random forest", "Forest without intake context"),
    ]
    names = sorted({name for pair in comparisons for name in pair})
    clusters = list(validation.groupby("animal_id").indices.values())
    y = validation["adopted_30d"].to_numpy()
    p = {name: probabilities[name].to_numpy() for name in names}
    delta = np.empty((n_draws, len(comparisons)))
    rng = np.random.default_rng(SEED)
    for row in range(n_draws):
        draws = rng.integers(len(clusters), size=len(clusters))
        index = np.concatenate([clusters[i] for i in draws])
        aps = {name: average_precision_score(y[index], p[name][index]) for name in names}
        delta[row] = [aps[a] - aps[b] for a, b in comparisons]
    aps = {name: average_precision_score(y, p[name]) for name in names}
    low, high = np.quantile(delta, [0.025, 0.975], axis=0)
    return pd.DataFrame(
        [
            {"model": a, "reference": b, "ap_difference": aps[a] - aps[b], "ci_low": lo, "ci_high": hi}
            for (a, b), lo, hi in zip(comparisons, low, high)
        ]
    )


def example_pair(validation, probabilities):
    """Find two kittens with the same inputs and different adoption timing."""
    columns = FEATURES + ["age_upon_intake"]
    candidates = validation.loc[
        validation["outcome_type"].eq("Adoption")
        & validation["species_age_band"].eq("Cat <3 mo")
        & probabilities["Random forest"].ge(THRESHOLD)
    ].copy()
    by_profile = candidates.groupby(columns)["adopted_30d"]
    timely = by_profile.transform("sum")
    late = by_profile.transform("size") - timely
    candidates["both_count"] = np.minimum(timely, late)
    candidates = candidates.loc[candidates["both_count"].gt(0)].sort_values(
        ["both_count", "intake_at"], ascending=[False, True]
    )
    if candidates.empty:
        return candidates
    profile = candidates.iloc[0][columns]
    same_profile = candidates.loc[candidates[columns].eq(profile).all(axis=1)]
    pair = same_profile.drop_duplicates("adopted_30d").sort_values("adopted_30d", ascending=False)
    return pair[
        ["animal_id", "intake_at", *columns, "outcome_type", "days_to_outcome", "adopted_30d"]
    ].join(probabilities[MAIN_MODELS[1:]])


def plot_diagnostics(scores, groups, false_positives):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#C5CED8",
        "text.color": "#1C2B39",
        "axes.labelcolor": "#1C2B39",
        "xtick.color": "#435467",
        "ytick.color": "#435467",
    })
    fig, axes = plt.subplot_mosaic(
        [["models", "ages"], ["errors", "errors"]],
        figsize=(12.4, 7.4),
        gridspec_kw={"height_ratios": [1.25, 0.85]},
        layout="constrained",
    )
    fig.suptitle("奥斯汀动物收容所 · 30 天领养预测", fontsize=18, fontweight="bold")

    ax = axes["models"]
    models = scores.set_index("model").loc[MAIN_MODELS]
    y = np.arange(len(models))
    for metric, offset, label, color in [
        ("ap", -0.16, "平均精确率 AP", "#166F65"),
        ("macro_f1", 0.16, "宏平均 F1", "#8B70AD"),
    ]:
        ax.barh(y + offset, models[metric], height=0.26, label=label, color=color)
        for position, value in zip(y + offset, models[metric]):
            ax.text(value + 0.013, position, f"{value:.3f}", va="center", fontsize=9)
    ax.set_yticks(y, ["历史比例", "逻辑回归", "单棵决策树", "随机森林"])
    ax.set_xlim(0, 0.83)
    ax.invert_yaxis()
    ax.set_title(f"2024 年 5–6 月验证集（{int(models['n'].iloc[0]):,} 条）", loc="left", fontsize=13)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)

    ax = axes["ages"]
    rates = groups.loc[
        groups["grouping"].eq("species_age_band") & groups["model"].eq("Random forest")
    ].set_index("group")
    x = np.arange(len(AGE_BANDS))
    for species, label, color, side in [("Cat", "猫", "#166F65", -1), ("Dog", "狗", "#91521A", 1)]:
        points = rates.reindex([f"{species} {band}" for band in AGE_BANDS])
        xx = x + side * 0.21
        ax.bar(xx, points["prevalence"], width=0.36, label=label, color=color)
        for at, (_, row) in zip(xx, points.iterrows()):
            if pd.notna(row["prevalence"]):
                ax.annotate(
                    f"{row['prevalence']:.0%}\n(n={int(row['n']):,})",
                    (at, row["prevalence"]), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=color,
                )
    ax.set_xticks(x, ["<3 月", "3–<6 月", "6–<12 月", "1–<7 岁", "≥7 岁"])
    ax.set_ylim(0, 0.9)
    ax.set_xlim(-0.55, len(x) - 0.45)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("年龄与物种对应的实际 30 天领养比例", loc="left", fontsize=13)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)

    ax = axes["errors"]
    kittens = false_positives.loc[false_positives["species_age_band"].eq("Cat <3 mo")]["recorded_outcome"]
    outcomes = kittens.value_counts()
    keys = ["Adoption", "No linked exit by July 31", "Transfer"]
    counts = [int(outcomes.get(key, 0)) for key in keys]
    counts.append(len(kittens) - sum(counts))
    labels = ["超过 30 天才领养", "截至 7 月底未配到离所", "转移", "其他去向"]
    bars = ax.barh(labels, counts, height=0.63, color=["#166F65", "#8FA1AF", "#B48554", "#AD9DBA"])
    ax.bar_label(bars, padding=5, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts + [1]) * 1.16)
    ax.set_xlabel("入所记录数")
    ax.set_title(f"随机森林误报的未满 3 月龄猫（{len(kittens)} 条）", loc="left", fontsize=13)
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    fig.savefig(RESULTS / "validation_diagnostics.png", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    episodes, audit = load_episodes()
    train, validation, split_audit = make_splits(episodes)
    audit.update(split_audit)
    probabilities = fit_probabilities(train, validation)
    scores = pd.DataFrame(
        [{"model": model, **score(validation["adopted_30d"], probabilities[model])} for model in probabilities]
    )
    groups = score_subgroups(train, validation, probabilities)
    intervals = paired_ap_intervals(validation, probabilities)
    cases = example_pair(validation, probabilities)
    errors = validation.loc[probabilities["Random forest"].ge(THRESHOLD) & validation["adopted_30d"].eq(0)].copy()
    errors["recorded_outcome"] = errors["outcome_type"].fillna("No linked exit by July 31")
    error_counts = errors.groupby(["species_age_band", "recorded_outcome"]).size().rename("false_positives").reset_index()

    (RESULTS / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    for filename, table in [
        ("model_scores.csv", scores),
        ("subgroup_scores.csv", groups),
        ("ap_comparisons.csv", intervals),
        ("example_pair.csv", cases),
        ("forest_false_positives.csv", error_counts),
    ]:
        table.to_csv(RESULTS / filename, index=False, encoding="utf-8-sig")
    plot_diagnostics(scores, groups, errors)

    print(json.dumps(audit, indent=2))
    print(scores[["model", "ap", "macro_f1", "brier", "precision", "recall"]].round(4).to_string(index=False))
    print(intervals.round(4).to_string(index=False))
    print(cases.round({model: 4 for model in MAIN_MODELS[1:] if model in cases}).to_string(index=False))


if __name__ == "__main__":
    main()

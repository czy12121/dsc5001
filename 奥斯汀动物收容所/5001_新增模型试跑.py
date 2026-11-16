"""Compare two additional model families on the original Austin pilot split.

Run with Python, NumPy, pandas and scikit-learn. The first pilot module supplies
the paired intake records, time split and metrics.
"""

from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from threadpoolctl import threadpool_limits


pilot = import_module("5001_首次实验")
ROOT = Path(__file__).resolve().parent
TARGET = "adopted_30d"
FIELDS = [*pilot.FEATURES, "age_band"]
MLP_SEEDS = (42, 77, 123)
AGE_BANDS = "Age bands"
CONTINUOUS_AGE = "Continuous age"


def fit_discriminative_models(train, validation, age_representation, extra_categories=()):
    continuous = age_representation == CONTINUOUS_AGE
    categories = [*(pilot.FEATURES if continuous else FIELDS), *extra_categories]
    columns = [*categories, "log_age"] if continuous else categories
    transformers = [("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categories)]
    if continuous:
        age = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
        transformers.append(("age", age, ["log_age"]))
    encoder = ColumnTransformer(transformers)
    x_train = encoder.fit_transform(train[columns])
    x_validation = encoder.transform(validation[columns])
    y = train[TARGET]
    bagged_trees = RandomForestClassifier(
        n_estimators=350,
        bootstrap=True,
        max_features=None,
        max_depth=5,
        min_samples_leaf=25,
        random_state=pilot.SEED,
        n_jobs=2,
    )
    models = [
        ("Logistic regression", None, pilot.logistic()),
        ("Bagging", pilot.SEED, bagged_trees),
    ]
    scores = {}
    for name, seed, model in models:
        model.fit(x_train, y)
        scores[(age_representation, name, seed)] = model.predict_proba(x_validation)[:, 1]

    # A smooth activation helps the small L-BFGS model converge on this pilot.
    with threadpool_limits(limits=1):
        for seed in MLP_SEEDS:
            mlp = MLPClassifier(
                hidden_layer_sizes=(16,),
                activation="tanh",
                solver="lbfgs",
                alpha=0.5,
                max_iter=3000,
                max_fun=30_000,
                random_state=seed,
            )
            mlp.fit(x_train, y)
            print(f"{age_representation}: MLP seed {seed}, {mlp.n_iter_} iterations")
            scores[(age_representation, "MLP", seed)] = mlp.predict_proba(x_validation)[:, 1]
    return scores


def fit_naive_bayes(train, validation, extra_categories=()):
    fields = [*FIELDS, *extra_categories]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    # Reserve code zero for unseen levels. Laplace smoothing covers this code
    # along with the categories seen in training.
    train_codes = encoder.fit_transform(train[fields]) + 1
    validation_codes = encoder.transform(validation[fields]) + 1
    model = CategoricalNB(alpha=1.0)
    model.fit(train_codes, train[TARGET])
    return model.predict_proba(validation_codes)[:, 1]


def evaluate(validation, predictions):
    groups = {"All": validation, "Cat <3 mo": validation.loc[validation["species_age_band"].eq("Cat <3 mo")]}
    groups.update(dict(tuple(validation.groupby("intake_month"))))
    rows = []
    for group, subset in groups.items():
        for (age_representation, model, seed), probabilities in predictions.items():
            metrics = pilot.score(subset[TARGET], probabilities[subset.index])
            rows.append({"group": group, "age_representation": age_representation, "model": model, "seed": seed, **metrics})
    return pd.DataFrame(rows)


def show_naive_bayes_check(train):
    few_months = train["age_band"].eq("<3 mo")
    neonatal = train["intake_condition"].eq("Neonatal")
    for label, group in train.groupby(TARGET):
        observed_both = (few_months & neonatal).loc[group.index].mean()
        independent_both = few_months.loc[group.index].mean() * neonatal.loc[group.index].mean()
        print(
            f"Target {label}: n={len(group)}, age <3 months and neonatal "
            f"observed={observed_both:.2%}, expected under independence={independent_both:.2%}"
        )


def show_example(validation, predictions):
    example = pd.Series(
        {
            "animal_type": "Cat",
            "sex_upon_intake": "Intact Male",
            "intake_type": "Stray",
            "intake_condition": "Normal",
            "age_band": "<3 mo",
        }
    )
    with_profile = validation[FIELDS].eq(example).all(axis=1)
    group = validation.loc[with_profile, TARGET]
    print(f"Example profile: {group.sum()} / {len(group)} validation records adopted within 30 days")
    if group.empty:
        return
    for (age_representation, model, seed), values in predictions.items():
        if age_representation == AGE_BANDS:
            print(f"  {model} {seed if seed is not None else ''}: {values[group.index[0]]:.2%}")


def main():
    episodes, _ = pilot.load_episodes()
    train, validation, _ = pilot.make_splits(episodes)
    print(f"Train: {len(train):,}, validation: {len(validation):,}. Adoption prior: {train[TARGET].mean():.2%}")
    predictions = {
        (AGE_BANDS, "Historical rate", None): np.full(len(validation), train[TARGET].mean()),
        (AGE_BANDS, "Categorical naive Bayes", None): fit_naive_bayes(train, validation),
        **fit_discriminative_models(train, validation, AGE_BANDS),
        **fit_discriminative_models(train, validation, CONTINUOUS_AGE),
    }
    scores = evaluate(validation, predictions)
    path = ROOT / "results" / "model_family_pilot.csv"
    path.parent.mkdir(exist_ok=True)
    scores.to_csv(path, index=False, encoding="utf-8-sig")
    all_scores = ["group", "age_representation", "model", "seed", "ap", "brier", "mean_probability", "tp", "fp", "fn"]
    print(scores.loc[scores["group"].isin(["All", "Cat <3 mo"]), all_scores].round(4).to_string(index=False))
    show_naive_bayes_check(train)
    show_example(validation, predictions)


if __name__ == "__main__":
    main()

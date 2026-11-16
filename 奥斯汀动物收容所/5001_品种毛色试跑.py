"""Test the contribution of the official intake Breed and Color fields.

Use the first pilot's labels and time split, and the second pilot's model settings.
"""

from importlib import import_module
from pathlib import Path

import pandas as pd


family = import_module("5001_新增模型试跑")
pilot = family.pilot
ROOT = Path(__file__).resolve().parent
MIN_CATEGORY_COUNT = 10
FEATURE_SETS = {
    "Original five": (),
    "+ Breed": ("breed",),
    "+ Color": ("color",),
    "+ Breed and Color": ("breed", "color"),
}


def attach_intake_fields(episodes):
    names = {
        "Animal ID": "animal_id",
        "DateTime": "intake_at",
        "Animal Type": "animal_type",
        "Breed": "breed",
        "Color": "color",
    }
    path = ROOT / "data/full/austin_intakes_2013_2025.csv"
    intakes = pd.read_csv(path, usecols=list(names), dtype="string").rename(columns=names)
    intakes["intake_at"] = pd.to_datetime(intakes["intake_at"], format="%m/%d/%Y %I:%M:%S %p")
    selected = intakes["intake_at"].between(episodes["intake_at"].min(), episodes["intake_at"].max())
    intakes = intakes.loc[selected & intakes["animal_type"].isin(episodes["animal_type"].unique())]
    joined = episodes.merge(
        intakes, on=["animal_id", "intake_at", "animal_type"], how="left", validate="one_to_one", indicator=True
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Some pilot records did not match the full intake CSV.")
    return joined.drop(columns="_merge")


def pool_rare_categories(train, validation):
    for column in ("breed", "color"):
        train_values = train[column].str.strip().fillna("Unknown").replace("", "Unknown")
        val_values = validation[column].str.strip().fillna("Unknown").replace("", "Unknown")
        counts = train_values.value_counts()
        common = counts[counts.ge(MIN_CATEGORY_COUNT)].index
        print(
            f"{column}: {len(counts)} raw categories in train; kept {len(common)}; "
            f"pooled {int((~train_values.isin(common)).sum())} train and "
            f"{int((~val_values.isin(common)).sum())} validation records "
            f"({int((~val_values.isin(counts.index)).sum())} unseen in train)"
        )
        train[column] = train_values.where(train_values.isin(common), "Other")
        validation[column] = val_values.where(val_values.isin(common), "Other")


def main():
    episodes, _ = pilot.load_episodes()
    train, validation, _ = pilot.make_splits(attach_intake_fields(episodes))
    print(f"Train: {len(train):,}; validation: {len(validation):,}")
    pool_rare_categories(train, validation)

    runs = []
    for name, fields in FEATURE_SETS.items():
        print(name)
        predictions = family.fit_discriminative_models(train, validation, family.CONTINUOUS_AGE, fields)
        predictions[(family.AGE_BANDS, "Categorical naive Bayes", None)] = family.fit_naive_bayes(
            train, validation, fields
        )
        runs.append(family.evaluate(validation, predictions).assign(feature_set=name))

    results = pd.concat(runs, ignore_index=True)
    path = ROOT / "results" / "breed_color_pilot.csv"
    path.parent.mkdir(exist_ok=True)
    results.to_csv(path, encoding="utf-8-sig", index=False)
    means = results.groupby(["group", "model", "feature_set"])[["ap", "brier"]].mean()
    print(means["ap"].unstack("feature_set").reindex(columns=list(FEATURE_SETS)).round(4).to_string())
    print("Overall Brier scores (lower is better):")
    print(means.loc["All", "brier"].unstack("feature_set").reindex(columns=list(FEATURE_SETS)).round(4).to_string())


if __name__ == "__main__":
    main()

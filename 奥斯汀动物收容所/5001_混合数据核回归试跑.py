"""Pilot mixed-data kernel regression on the fixed 30-day validation split."""

from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from threadpoolctl import threadpool_limits


formal = import_module("5001_正式实验")
TARGET = "adopted_30d"
AGE_BANDWIDTH = 0.35  # In log(1 + age in days) units.
CATEGORY_MISMATCH_WEIGHT = 0.35
BATCH_SIZE = 256
RESULTS = formal.ROOT / "results" / "mixed_kernel_pilot.csv"
FEATURE_SETS = {
    "age_bands": formal.BANDS,
    "base": formal.BASE,
    "base_a": (*formal.BASE, *formal.AGE_FEATURES),
    "base_b": formal.CONTINUOUS,
    "base_ab": (*formal.CONTINUOUS, *formal.AGE_FEATURES),
}


def kernel_probabilities(train, evaluation, features):
    fields = list(features)
    categories = [field for field in fields if field != "log_age"]
    uses_continuous_age = "log_age" in fields
    history = train[fields + [TARGET]].copy()
    queries = evaluation[fields].copy()

    if uses_continuous_age:
        median_age = history["log_age"].median()
        if pd.isna(median_age):
            median_age = 0.0
        for frame in (history, queries):
            frame["age_missing"] = frame["log_age"].isna()
            frame["log_age"] = frame["log_age"].fillna(median_age)
        fields.append("age_missing")
        categories.append("age_missing")

    # Equal feature vectors have equal weights for any query.
    references = (
        history.groupby(fields, sort=False, dropna=False)[TARGET]
        .agg(total="size", adopted="sum")
        .reset_index()
    )
    totals = references["total"].to_numpy(dtype=float)
    adoptions = references["adopted"].to_numpy(dtype=float)
    reference_codes = {}
    query_codes = {}
    for field in categories:
        levels = pd.Index(references[field].unique())
        reference_codes[field] = levels.get_indexer(references[field])
        query_codes[field] = levels.get_indexer(queries[field])
    if uses_continuous_age:
        reference_ages = references["log_age"].to_numpy(dtype=float)
        query_ages = queries["log_age"].to_numpy(dtype=float)

    probabilities = np.empty(len(queries), dtype=float)
    log_mismatch = np.log(CATEGORY_MISMATCH_WEIGHT)
    for start in range(0, len(queries), BATCH_SIZE):
        batch = slice(start, min(start + BATCH_SIZE, len(queries)))
        log_weights = np.zeros((batch.stop - start, len(references)))
        for field in categories:
            different = query_codes[field][batch, None] != reference_codes[field][None, :]
            log_weights += log_mismatch * different
        if uses_continuous_age:
            age_distance = (query_ages[batch, None] - reference_ages[None, :]) / AGE_BANDWIDTH
            log_weights -= 0.5 * age_distance**2

        # A common rescaling of weights cancels in the weighted average.
        log_weights -= log_weights.max(axis=1, keepdims=True)
        weights = np.exp(log_weights)
        probabilities[batch] = (weights @ adoptions) / (weights @ totals)
    return probabilities


def main():
    splits = formal.load_splits(("train", "validation"))
    train = formal.analysis_rows(splits["train"], "primary")
    validation = formal.analysis_rows(splits["validation"], "primary")
    labels = validation[TARGET].to_numpy(dtype=int)
    metrics = []
    for name, fields in FEATURE_SETS.items():
        probabilities = kernel_probabilities(train, validation, fields)
        metrics.append({
            "run": name,
            "features": " ".join(fields),
            "training_n": len(train),
            "validation_n": len(validation),
            "validation_positives": int(labels.sum()),
            "age_bandwidth": AGE_BANDWIDTH if "log_age" in fields else None,
            "category_mismatch_weight": CATEGORY_MISMATCH_WEIGHT,
            "ap": average_precision_score(labels, probabilities),
            "brier": brier_score_loss(labels, probabilities),
        })

    results = pd.DataFrame(metrics)
    RESULTS.parent.mkdir(exist_ok=True)
    results.to_csv(RESULTS, index=False)
    print(results[["run", "training_n", "validation_n", "ap", "brier"]].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    ))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

"""Build and audit the 2022–2024 Austin intake episodes. Requires pandas and NumPy."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "full"
DESTINATION = ROOT / "data" / "prepared"
START = pd.Timestamp("2022-01-01")
END = pd.Timestamp("2025-01-01")
HORIZONS = (30, 60)
SPLITS = {
    "train": (START, pd.Timestamp("2023-11-01")),
    "validation": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-07-01")),
    "test": (pd.Timestamp("2024-09-01"), END),
}
KEY = ["animal_id", "animal_type"]
INTAKE_FIELDS = {
    "Animal ID": "animal_id",
    "Animal Type": "animal_type",
    "DateTime": "intake_datetime_raw",
    "Age upon Intake": "age_upon_intake",
    "Sex upon Intake": "sex_upon_intake",
    "Intake Type": "intake_type",
    "Intake Condition": "intake_condition",
}
OUTCOME_FIELDS = {
    "Animal ID": "animal_id",
    "Animal Type": "animal_type",
    "DateTime": "outcome_datetime_raw",
    "Outcome Type": "outcome_type",
}
AGE_BANDS = ["<3 mo", "3–<6 mo", "6–<12 mo", "1–<7 yr", "7+ yr"]


def load_archive():
    intake = pd.read_csv(
        SOURCE / "austin_intakes_2013_2025.csv", usecols=list(INTAKE_FIELDS), dtype="string"
    ).rename(columns=INTAKE_FIELDS)
    outcome = pd.read_csv(
        SOURCE / "austin_outcomes_2013_2025.csv", usecols=list(OUTCOME_FIELDS), dtype="string"
    ).rename(columns=OUTCOME_FIELDS)
    for frame in (intake, outcome):
        for column in frame.columns:
            frame[column] = frame[column].str.strip().replace("", pd.NA)

    intake["intake_at"] = pd.to_datetime(
        intake["intake_datetime_raw"], format="%m/%d/%Y %I:%M:%S %p"
    ).astype("datetime64[ns]")
    # Offsets in this export occur on midnight values. Work with the dates
    # reported by the source so a day-level event keeps its recorded date.
    outcome["outcome_at"] = pd.to_datetime(
        outcome["outcome_datetime_raw"].str[:19], format="ISO8601"
    ).astype("datetime64[ns]")
    for frame, clock in ((intake, "intake_at"), (outcome, "outcome_at")):
        if frame[[*KEY, clock]].isna().any().any():
            raise ValueError(f"Source rows are missing a key or date in {clock}.")
    archive_end = outcome["outcome_at"].max().normalize()
    intake = intake.loc[intake["animal_type"].isin(["Cat", "Dog"])].copy()
    outcome = outcome.loc[outcome["animal_type"].isin(["Cat", "Dog"])].copy()
    intake_period = intake["intake_at"].ge(START) & intake["intake_at"].lt(END)
    cohort_ids = intake.loc[intake_period, "animal_id"].unique()
    return (
        intake.loc[intake["animal_id"].isin(cohort_ids)].copy(),
        outcome.loc[outcome["animal_id"].isin(cohort_ids)].copy(),
        archive_end,
    )


def prepare_events(intakes, outcomes):
    # Collapse only events whose task-relevant fields agree. Keep conflicting
    # times in the timeline to delimit the neighbouring intake episodes.
    intake_key = [*KEY, "intake_at"]
    events = intakes.drop_duplicates().copy()
    events["intake_conflict"] = events.duplicated(intake_key, keep=False)
    events = events.drop_duplicates(intake_key).sort_values("intake_at", kind="stable")
    events["intake_day"] = events["intake_at"].dt.normalize()
    by_animal = events.groupby(KEY, sort=False)
    events["previous_intake_at"] = by_animal["intake_at"].shift()
    events["next_intake_at"] = by_animal["intake_at"].shift(-1)
    events["episode_id"] = events["animal_id"] + "_" + events["intake_at"].dt.strftime("%Y%m%dT%H%M%S")

    exit_events = outcomes.drop_duplicates(subset=[*KEY, "outcome_at", "outcome_type"]).copy()
    exit_events["outcome_day"] = exit_events["outcome_at"].dt.normalize()
    exit_events["outcome_date_only"] = exit_events["outcome_at"].eq(exit_events["outcome_day"])
    end_of_day = exit_events["outcome_day"] + pd.Timedelta(days=1, nanoseconds=-1)
    exit_events["latest_possible_outcome_at"] = exit_events["outcome_at"].where(
        ~exit_events["outcome_date_only"], end_of_day
    )
    exit_events["outcome_id"] = pd.array(exit_events.index + 1, dtype="Int64")
    return events, exit_events


def link_outcomes(intakes, outcomes):
    clock_fields = [*KEY, "intake_at", "episode_id", "previous_intake_at"]
    timeline = intakes[clock_fields].sort_values("intake_at", kind="stable")
    exit_events = outcomes.sort_values("latest_possible_outcome_at", kind="stable")
    links = pd.merge_asof(
        exit_events,
        timeline,
        left_on="latest_possible_outcome_at",
        right_on="intake_at",
        by=KEY,
        direction="backward",
    )
    # A midnight outcome on a new intake day could be before or after that
    # intake. When there is an earlier episode, keep both as possible owners.
    ambiguous_day = (
        links["outcome_date_only"]
        & links["intake_at"].gt(links["outcome_day"])
        & links["previous_intake_at"].notna()
    )
    # At the exact time of a repeat intake, an outcome could close either stay.
    same_time = (
        ~links["outcome_date_only"]
        & links["outcome_at"].eq(links["intake_at"])
        & links["previous_intake_at"].notna()
    )
    links["ambiguous_assignment"] = ambiguous_day | same_time
    # More than one same-day intake is unusual but possible. Pair a date-only
    # event tentatively with every stay whose time window intersects that day.
    ambiguities = links.loc[links["ambiguous_assignment"]].drop(
        columns=["intake_at", "episode_id", "previous_intake_at"]
    ).merge(timeline, on=KEY)
    overlaps = ambiguities["intake_at"].le(ambiguities["latest_possible_outcome_at"])
    next_by_episode = intakes.set_index("episode_id")["next_intake_at"]
    following = ambiguities["episode_id"].map(next_by_episode)
    overlaps &= following.isna() | following.ge(ambiguities["outcome_at"])
    candidates = pd.concat(
        [
            links.loc[links["episode_id"].notna() & ~links["ambiguous_assignment"]],
            ambiguities.loc[overlaps],
        ],
        ignore_index=True,
    )
    candidates["days_to_outcome"] = (candidates["outcome_day"] - candidates["intake_at"].dt.normalize()).dt.days
    return candidates.sort_values(["days_to_outcome", "outcome_at", "outcome_id"], kind="stable")


def possible_outcome_summary(episodes, candidates):
    by_episode = candidates.groupby("episode_id", sort=False)
    episodes["candidate_outcomes"] = episodes["episode_id"].map(by_episode.size()).fillna(0).astype(int)
    uncertain = candidates.loc[candidates["ambiguous_assignment"]]
    episodes["uncertain_assignment_count"] = episodes["episode_id"].map(
        uncertain.groupby("episode_id").size()
    ).fillna(0).astype(int)
    first = first_possible_outcomes(candidates)
    clear = first.loc[
        ~first["ambiguous_assignment"] & ~first["episode_id"].duplicated(keep=False)
    ]
    fields = [
        "episode_id", "outcome_id", "outcome_at", "outcome_datetime_raw", "outcome_date_only",
        "outcome_type", "days_to_outcome",
    ]
    return episodes.merge(clear[fields], on="episode_id", how="left", validate="one_to_one")


def first_possible_outcomes(candidates):
    # A day-level timestamp can overlap multiple possible first events.
    earliest_end = candidates.groupby("episode_id")["latest_possible_outcome_at"].transform("min")
    return candidates.loc[candidates["outcome_at"].le(earliest_end)]


def add_labels(episodes, candidates, archive_end):
    # Evaluate each window separately so an omission on day 45 does not turn a
    # fully observed 30-day negative into an unknown.
    for horizon in HORIZONS:
        nearby = candidates.loc[candidates["days_to_outcome"].le(horizon)]
        ends_at = episodes["intake_day"] + pd.Timedelta(days=horizon)
        first_events = first_possible_outcomes(nearby).copy()
        first_events["is_adoption"] = first_events["outcome_type"].eq("Adoption").fillna(False)
        first_events["uncertain"] = first_events["ambiguous_assignment"] | first_events["outcome_type"].isna()
        first_by_episode = first_events.groupby("episode_id")
        disagreement = first_by_episode["is_adoption"].nunique().gt(1)
        unclear = first_by_episode["uncertain"].any() | disagreement
        unclear_outcome = episodes["episode_id"].isin(unclear.index[unclear])
        observed_exit = episodes["episode_id"].isin(unclear.index) & ~unclear_outcome
        adopted = episodes["episode_id"].isin(first_events.loc[first_events["is_adoption"], "episode_id"])
        unexplained_repeat = episodes["next_intake_at"].dt.normalize().le(ends_at) & ~observed_exit
        incomplete_followup = ends_at.gt(archive_end) & ~observed_exit
        unknown = episodes["intake_conflict"] | unclear_outcome | unexplained_repeat | incomplete_followup
        episodes[f"adopted_{horizon}d"] = adopted.astype("Int8").mask(unknown)
        episodes[f"label_{horizon}d_basis"] = np.select(
            [
                episodes["intake_conflict"].to_numpy(dtype=bool),
                unclear_outcome.to_numpy(dtype=bool),
                unexplained_repeat.to_numpy(dtype=bool),
                incomplete_followup.to_numpy(dtype=bool),
                observed_exit.to_numpy(dtype=bool),
            ],
            [
                "conflicting_intake", "unclear_outcome", "unexplained_return",
                "incomplete_followup", "recorded_exit",
            ],
            default="no_exit_recorded_in_window",
        )
    return episodes


def add_features_and_splits(episodes):
    age = episodes["age_upon_intake"].str.lower().str.extract(r"^(\d+) (day|week|month|year)s?$")
    unit_days = age[1].map({"day": 1.0, "week": 7.0, "month": 365.25 / 12, "year": 365.25})
    days = (pd.to_numeric(age[0]) * unit_days).mask(age[0].eq("0") & age[1].eq("year"))
    episodes["age_days"] = days
    episodes["log_age"] = np.log1p(days)
    episodes["age_band"] = pd.cut(
        days, bins=[0, 365.25 / 4, 365.25 / 2, 365.25, 7 * 365.25, np.inf],
        labels=AGE_BANDS, right=False
    ).astype("string").fillna("Unknown")
    episodes["species_age_band"] = episodes["animal_type"] + " " + episodes["age_band"]
    episodes["intake_month"] = episodes["intake_day"].dt.strftime("%Y-%m")
    episodes["split"] = "followup_gap"
    episodes["seen_in_earlier_split"] = False
    earlier_ids = set()
    for split, (start, stop) in SPLITS.items():
        period = episodes["intake_day"].ge(start) & episodes["intake_day"].lt(stop)
        ids = episodes.loc[period, "animal_id"]
        episodes.loc[period, "split"] = split
        episodes.loc[period, "seen_in_earlier_split"] = ids.isin(earlier_ids)
        earlier_ids.update(ids)
    return episodes


def audit(episodes, raw_intakes, candidates, archive_end):
    if episodes["episode_id"].duplicated().any() or episodes["outcome_id"].dropna().duplicated().any():
        raise ValueError("Episodes and their unambiguous outcomes must be unique.")
    if episodes["days_to_outcome"].dropna().lt(0).any() or candidates["days_to_outcome"].lt(0).any():
        raise ValueError("An exit precedes its intake.")
    label_30 = episodes["adopted_30d"]
    label_60 = episodes["adopted_60d"]
    if (label_30.notna() & label_60.notna() & label_30.gt(label_60)).any():
        raise ValueError("A known 60-day label is lower than the 30-day label.")

    raw_cohort = raw_intakes.loc[raw_intakes["intake_at"].ge(START) & raw_intakes["intake_at"].lt(END)]
    counts = []
    seen = set()
    for split, (start, stop) in SPLITS.items():
        raw_rows = raw_cohort["intake_at"].ge(start) & raw_cohort["intake_at"].lt(stop)
        period = episodes.loc[episodes["split"].eq(split)]
        eligible = period.loc[~period["seen_in_earlier_split"]]
        if seen.intersection(eligible["animal_id"]):
            raise ValueError("An eligible animal appears in multiple splits.")
        seen.update(eligible["animal_id"])
        for horizon in HORIZONS:
            target = f"adopted_{horizon}d"
            labelled = eligible.loc[eligible[target].notna()]
            if set(labelled[target]) != {0, 1}:
                raise ValueError(f"Both labels are required in {split} at {horizon} days.")
            counts.append({
                "split": split,
                "window_days": horizon,
                "raw_intakes": int(raw_rows.sum()),
                "unique_intake_events": len(period),
                "animals_already_in_earlier_split_rows": int(period["seen_in_earlier_split"].sum()),
                "unknown_labels_after_id_filter": int(eligible[target].isna().sum()),
                "labelled_rows": len(labelled),
                "positive_rows": int(labelled[target].sum()),
                "positive_rate": float(labelled[target].mean()),
                "cat_rows": int(labelled["animal_type"].eq("Cat").sum()),
                "dog_rows": int(labelled["animal_type"].eq("Dog").sum()),
                "kitten_rows": int(labelled["species_age_band"].eq("Cat <3 mo").sum()),
                "labelled_without_exit_candidate": int(labelled["candidate_outcomes"].eq(0).sum()),
                "labelled_without_unambiguous_pair": int(labelled["outcome_id"].isna().sum()),
            })
    counts = pd.DataFrame(counts)
    summary = {
        "source_csv_rows_in_cohort": len(raw_cohort),
        "cohort_unique_intake_events": len(episodes),
        "removed_duplicate_intake_rows": len(raw_cohort) - len(raw_cohort.drop_duplicates()),
        "conflicting_intake_events": int(episodes["intake_conflict"].sum()),
        "cohort_rows_with_no_candidate_outcome": int(episodes["candidate_outcomes"].eq(0).sum()),
        "cohort_rows_with_one_unambiguous_outcome": int(episodes["outcome_id"].notna().sum()),
        "cohort_rows_with_multiple_candidates": int(episodes["candidate_outcomes"].gt(1).sum()),
        "cohort_rows_whose_outcome_can_belong_to_more_than_one_intake": int(episodes["uncertain_assignment_count"].gt(0).sum()),
        "cohort_matched_outcomes_with_date_precision": int(episodes["outcome_date_only"].eq(True).sum()),
        "cohort_unknown_ages": int(episodes["age_days"].isna().sum()),
        "coverage_last_outcome_date": str(archive_end.date()),
        "followup_gap_rows": int(episodes["split"].eq("followup_gap").sum()),
    }
    return summary, counts


def main():
    intakes, outcomes, archive_end = load_archive()
    intake_events, outcome_events = prepare_events(intakes, outcomes)
    candidates = link_outcomes(intake_events, outcome_events)
    episodes = intake_events.loc[intake_events["intake_at"].ge(START) & intake_events["intake_at"].lt(END)].copy()
    candidates = candidates.loc[candidates["episode_id"].isin(episodes["episode_id"])]
    episodes = possible_outcome_summary(episodes, candidates)
    episodes = add_labels(episodes, candidates, archive_end)
    episodes = add_features_and_splits(episodes)
    checks, split_counts = audit(episodes, intakes, candidates, archive_end)
    DESTINATION.mkdir(exist_ok=True)
    episodes.sort_values(["intake_at", "animal_id"], kind="stable").to_csv(
        DESTINATION / "猫狗入所配对表_2022–2024.csv", index=False, encoding="utf-8-sig"
    )
    split_counts.to_csv(DESTINATION / "训练验证测试样本数.csv", index=False, encoding="utf-8-sig")
    (DESTINATION / "数据质检.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(split_counts.to_string(index=False))


if __name__ == "__main__":
    main()

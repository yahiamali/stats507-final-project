from __future__ import annotations

from typing import Iterable

import pandas as pd


TEAM_CONTEXT_COLUMNS = [
    "team_wins",
    "team_losses",
    "team_sos",
    "team_srs",
    "tournament_seed",
]

PLAYER_AGG_MEAN_COLUMNS = [
    "games_played",
    "games_started",
    "minutes_per_game",
    "pts_per_game",
    "trb_per_game",
    "ast_per_game",
    "stl_per_game",
    "blk_per_game",
    "tov_per_game",
    "fg_pct",
    "three_p_pct",
    "ft_pct",
    "efg_pct",
    "ts_pct",
    "three_par",
    "ftr",
    "per",
    "orb_pct",
    "drb_pct",
    "trb_pct",
    "ast_pct",
    "stl_pct",
    "blk_pct",
    "tov_pct",
    "usg_pct",
    "ows",
    "dws",
    "ws",
    "ws_40",
    "obpm",
    "dbpm",
    "bpm",
    "completeness_score",
]

PLAYER_AGG_SUM_COLUMNS = [
    "pts_per_game",
    "trb_per_game",
    "ast_per_game",
    "stl_per_game",
    "blk_per_game",
    "tov_per_game",
]

GROUP_KEYS = ["season_year", "team_slug", "team"]


def _safe_numeric_columns(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def build_team_features(players: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player-season rows into one row per team-season.
    """
    df = players.copy()

    mean_cols = _safe_numeric_columns(df, PLAYER_AGG_MEAN_COLUMNS)
    sum_cols = _safe_numeric_columns(df, PLAYER_AGG_SUM_COLUMNS)
    context_cols = _safe_numeric_columns(df, TEAM_CONTEXT_COLUMNS)

    team_features = (
        df.groupby(GROUP_KEYS, dropna=False)
        .agg({**{col: "mean" for col in mean_cols}, **{col: "first" for col in context_cols}})
        .reset_index()
    )

    count_df = (
        df.groupby(GROUP_KEYS, dropna=False)["player_name"]
        .count()
        .reset_index(name="roster_size")
    )

    sum_df = (
        df.groupby(GROUP_KEYS, dropna=False)[sum_cols]
        .sum()
        .reset_index()
        .rename(columns={col: f"{col}_sum" for col in sum_cols})
    )

    team_features = team_features.merge(count_df, on=GROUP_KEYS, how="left")
    team_features = team_features.merge(sum_df, on=GROUP_KEYS, how="left")
    team_features = team_features.rename(columns={col: f"{col}_mean" for col in mean_cols})

    return team_features


def build_matchup_dataset(
    matchups: pd.DataFrame,
    team_features: pd.DataFrame,
    include_target: bool = True,
    augment_mirror: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Merge team-level features onto matchup rows and convert them into difference features:
    feature_diff = team_a_feature - team_b_feature

    If include_target=True, target y = 1 when team_a wins, else 0.
    """
    games = matchups.copy()
    features = team_features.copy()

    a_features = features.add_suffix("_a")
    b_features = features.add_suffix("_b")

    merged = games.merge(
        a_features,
        left_on=["season_year", "team_a_slug"],
        right_on=["season_year_a", "team_slug_a"],
        how="left",
    )
    merged = merged.merge(
        b_features,
        left_on=["season_year", "team_b_slug"],
        right_on=["season_year_b", "team_slug_b"],
        how="left",
    )

    a_base_cols = [
        c for c in features.columns if c not in {"season_year", "team_slug", "team"}
    ]

    diff_cols = []
    for base_col in a_base_cols:
        a_col = f"{base_col}_a"
        b_col = f"{base_col}_b"
        diff_col = f"diff_{base_col}"
        merged[diff_col] = merged[a_col] - merged[b_col]
        diff_cols.append(diff_col)

    merged["seed_gap_raw"] = merged["team_a_seed"] - merged["team_b_seed"]
    diff_cols.append("seed_gap_raw")

    if include_target:
        merged = merged.loc[merged["winner_slug"].notna()].copy()
        merged["target"] = (merged["winner_slug"] == merged["team_a_slug"]).astype(int)

    keep_cols = [
        "season_year",
        "round",
        "region",
        "team_a_slug",
        "team_a_name",
        "team_a_seed",
        "team_b_slug",
        "team_b_name",
        "team_b_seed",
    ] + diff_cols

    if include_target:
        keep_cols.append("target")

    modeled = merged[keep_cols].copy()

    if include_target and augment_mirror:
        swap_pairs = {
            "team_a_slug": "team_b_slug",
            "team_a_name": "team_b_name",
            "team_a_seed": "team_b_seed",
        }
        mirrored = modeled.copy()
        for left, right in swap_pairs.items():
            mirrored[left], mirrored[right] = modeled[right], modeled[left]

        for col in diff_cols:
            mirrored[col] = -1.0 * mirrored[col]
        mirrored["target"] = 1 - mirrored["target"]

        modeled = pd.concat([modeled, mirrored], ignore_index=True)

    return modeled, diff_cols


def split_by_season(
    df: pd.DataFrame,
    train_end_year: int = 2023,
    val_year: int = 2024,
    test_year: int = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df.loc[df["season_year"] <= train_end_year].copy()
    val_df = df.loc[df["season_year"] == val_year].copy()
    test_df = df.loc[df["season_year"] == test_year].copy()
    return train_df, val_df, test_df


def get_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].copy()
    y = df["target"].copy()
    return X, y

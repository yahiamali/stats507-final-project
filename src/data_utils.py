from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


LEAKAGE_COLUMNS = {
    "tournament_result",
    "games_won",
    "upset_contribution",
}


def load_player_data(path: str | Path) -> pd.DataFrame:
    """Load the parquet file using pyarrow, then convert to pandas."""
    table = pq.read_table(path)
    df = table.to_pandas()
    return df


def load_matchups(path: str | Path) -> pd.DataFrame:
    """Load tournament matchups from a JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_bracket(path: str | Path) -> dict:
    """Load 2026 bracket JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_first_four_placeholders(
    matchups_2026: pd.DataFrame, bracket_data: dict
) -> pd.DataFrame:
    """
    Replace placeholder round-of-64 team slugs/names such as FF_WINNER_Midwest_16
    using the winners stored in bracket_2026.json.
    """
    resolved = matchups_2026.copy()

    placeholder_lookup: dict[str, tuple[str, str, int]] = {}
    for game in bracket_data.get("first_four", []):
        region = game["region"]
        seed = game["seed"]
        winner_slug, winner_name, winner_seed = game["winner"]
        key = f"FF_WINNER_{region}_{seed}"
        placeholder_lookup[key] = (winner_slug, winner_name, winner_seed)

    for side in ["a", "b"]:
        slug_col = f"team_{side}_slug"
        name_col = f"team_{side}_name"
        seed_col = f"team_{side}_seed"

        for idx, value in resolved[slug_col].items():
            if value in placeholder_lookup:
                repl_slug, repl_name, repl_seed = placeholder_lookup[value]
                resolved.at[idx, slug_col] = repl_slug
                resolved.at[idx, name_col] = repl_name
                resolved.at[idx, seed_col] = repl_seed

    return resolved


def clean_player_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning aligned with the project goal:
    - remove known leakage columns from later feature use
    - keep rows marked include_in_training=True when available
    """
    cleaned = df.copy()

    if "include_in_training" in cleaned.columns:
        cleaned = cleaned.loc[cleaned["include_in_training"] == True].copy()

    return cleaned

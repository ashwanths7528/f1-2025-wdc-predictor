import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Enable FastF1 cache (folder "cache" in your project)
fastf1.Cache.enable_cache("cache")


# -------------------------------
# 1. Basic utilities
# -------------------------------

def points_for_position(position: int) -> int:
    """Standard F1 points (no sprint / FL extras)."""
    table = {
        1: 25,
        2: 18,
        3: 15,
        4: 12,
        5: 10,
        6: 8,
        7: 6,
        8: 4,
        9: 2,
        10: 1
    }
    return table.get(position, 0)


def fetch_season_results(season: int) -> pd.DataFrame:
    """
    Use FastF1 to get race results for a given season.
    Returns rows with: season, round, race, driver, constructor, position, points.
    """
    rows = []

    for round_no in range(1, 30):  # 30 is a safe upper bound
        try:
            session = fastf1.get_session(season, round_no, "R")  # "R" = Race
        except Exception as e:
            # No more rounds in this season
            print(f"[{season}] Stopping at round {round_no}: {e}")
            break

        event_name = session.event['EventName']
        print(f"[{season}] Loading round {round_no}: {event_name}")

        try:
            session.load()
        except Exception as e:
            print(f"  Skipping {event_name} (couldn't load): {e}")
            continue

        results = session.results
        if results is None or results.empty:
            print(f"  No results for {event_name}, stopping.")
            break

        for _, r in results.iterrows():
            classified = r['ClassifiedPosition']

            # Skip non-finishers etc. (e.g. 'R', 'W')
            try:
                pos_int = int(classified)
            except (TypeError, ValueError):
                continue

            pts = points_for_position(pos_int)

            rows.append({
                "season": season,
                "round": int(round_no),
                "race": event_name,
                "driver": r['LastName'],        # can change to FullName if you want
                "constructor": r['TeamName'],
                "position": pos_int,
                "points": pts
            })

    df = pd.DataFrame(rows)
    return df


# -------------------------------
# 2. Training snapshots (past seasons)
# -------------------------------

def build_training_snapshot(df_season: pd.DataFrame, races_remaining: int = 2) -> pd.DataFrame:
    """
    Build a 'with N races left' snapshot for a completed season.

    For each driver we compute:
      - current_points   (at snapshot round)
      - gap_to_leader
      - avg_points_per_race
      - constructor at snapshot
      - champion (1 if WDC that year, else 0)
    """
    season = df_season["season"].iloc[0]
    max_round = df_season["round"].max()
    snapshot_round = max_round - races_remaining  # e.g. if 22 races, snapshot at round 20

    if snapshot_round < 1:
        raise ValueError(f"Season {season}: not enough rounds for races_remaining={races_remaining}")

    # Points up to snapshot_round
    up_to_snapshot = df_season[df_season["round"] <= snapshot_round]
    current_points = (
        up_to_snapshot
        .groupby("driver")["points"]
        .sum()
        .reset_index()
        .rename(columns={"points": "current_points"})
    )

    # Final points at end of season
    final_points = (
        df_season
        .groupby("driver")["points"]
        .sum()
        .reset_index()
        .rename(columns={"points": "final_points"})
    )

    # Champion = driver with most final_points
    champion_driver = final_points.sort_values("final_points", ascending=False).iloc[0]["driver"]

    # Constructor at snapshot (last race before snapshot)
    last_race_info = (
        up_to_snapshot
        .sort_values(["driver", "round"])
        .groupby("driver")
        .tail(1)[["driver", "constructor"]]
    )

    snap = current_points.merge(last_race_info, on="driver", how="left")
    snap = snap.merge(final_points, on="driver", how="left")

    leader_points = snap["current_points"].max()
    snap["gap_to_leader"] = leader_points - snap["current_points"]
    snap["avg_points_per_race"] = snap["current_points"] / snapshot_round
    snap["races_remaining"] = races_remaining
    snap["season"] = season

    # Target: is this driver the eventual WDC?
    snap["champion"] = (snap["driver"] == champion_driver).astype(int)

    return snap

def fetch_sprint_points_for_season(season: int) -> pd.DataFrame:
    """
    Fetch sprint points for a given season using FastF1.
    We look at 'S' (Sprint) sessions and use the official 'Points' column.
    Returns a DataFrame with columns: season, round, driver, sprint_points.
    """
    rows = []

    for round_no in range(1, 30):  # safe upper bound
        try:
            sprint_session = fastf1.get_session(season, round_no, "S")  # 'S' = Sprint
        except Exception:
            # No sprint for this weekend (or no such round) -> skip
            continue

        event_name = sprint_session.event['EventName']
        print(f"[{season}] Loading round {round_no}: {event_name} (Sprint)")

        try:
            sprint_session.load()
        except Exception as e:
            print(f"  Skipping {event_name} Sprint (couldn't load): {e}")
            continue

        sprint_results = sprint_session.results
        if sprint_results is None or sprint_results.empty:
            print(f"  No Sprint results for {event_name}, skipping sprint.")
            continue

        for _, r in sprint_results.iterrows():
            pts = r.get("Points", 0.0)
            if pd.isna(pts):
                pts = 0.0

            rows.append({
                "season": season,
                "round": int(round_no),
                "driver": r["LastName"],
                "sprint_points": float(pts),
            })

    return pd.DataFrame(rows)



def prepare_training_data(train_seasons, races_remaining=2):
    all_snaps = []

    for s in train_seasons:
        print(f"\n=== Processing training season {s} ===")
        df_season = fetch_season_results(s)

        if df_season.empty:
            print(f"Season {s}: no data, skipping.")
            continue

        snap = build_training_snapshot(df_season, races_remaining=races_remaining)
        all_snaps.append(snap)

    full = pd.concat(all_snaps, ignore_index=True)

    # One-hot encode constructor (team)
    full = pd.get_dummies(full, columns=["constructor"], prefix="team")

    feature_cols = [
        "current_points",
        "gap_to_leader",
        "avg_points_per_race",
        "races_remaining",
    ] + [c for c in full.columns if c.startswith("team_")]

    X = full[feature_cols]
    y = full["champion"]

    return X, y, feature_cols


# -------------------------------
# 3. LIVE snapshot for 2025 (22 races done, 2 left)
# -------------------------------

def build_live_snapshot_2025(df_season: pd.DataFrame,
                             races_completed: int = 22,
                             races_remaining: int = 2) -> pd.DataFrame:
    """
    Build snapshot for the ongoing 2025 season with 2 races remaining.
    We assume 22 races are completed (your info).
    """
    season = df_season["season"].iloc[0]

    # Keep only races up to the completed round (22)
    snapshot_round = races_completed
    up_to_snapshot = df_season[df_season["round"] <= snapshot_round]

    last_completed = up_to_snapshot["round"].max()
    if pd.isna(last_completed):
        raise RuntimeError(f"No race data found for season {season}.")
    if last_completed < snapshot_round:
        print(f"Warning: data only up to round {last_completed}, "
              f"but races_completed is {races_completed}. Using {last_completed} instead.")
        snapshot_round = last_completed
        up_to_snapshot = df_season[df_season["round"] <= snapshot_round]

    # Current points at snapshot
        # --- Race points at snapshot (as before) ---
    current_points = (
        up_to_snapshot
        .groupby("driver")["points"]
        .sum()
        .reset_index()
        .rename(columns={"points": "current_points"})
    )

    # --- Add sprint points ONLY for 2025 ---
    if season == 2025:
        sprint_df = fetch_sprint_points_for_season(season)
        if not sprint_df.empty:
            # Only up to the snapshot round (22)
            sprint_up_to = sprint_df[sprint_df["round"] <= snapshot_round]

            sprint_points = (
                sprint_up_to
                .groupby("driver")["sprint_points"]
                .sum()
                .reset_index()
                .rename(columns={"sprint_points": "sprint_points_so_far"})
            )

            # Merge sprint points into current_points
            current_points = current_points.merge(
                sprint_points, on="driver", how="left"
            )
            # Drivers with no sprint points get 0
            current_points["sprint_points_so_far"] = current_points["sprint_points_so_far"].fillna(0.0)

            # Total points = race points + sprint points
            current_points["current_points"] = (
                current_points["current_points"] + current_points["sprint_points_so_far"]
            )
        else:
            print("⚠ No sprint data found for 2025; using race points only.")


    # Constructor at snapshot
    last_race_info = (
        up_to_snapshot
        .sort_values(["driver", "round"])
        .groupby("driver")
        .tail(1)[["driver", "constructor"]]
    )

    snap = current_points.merge(last_race_info, on="driver", how="left")

    leader_points = snap["current_points"].max()
    snap["gap_to_leader"] = leader_points - snap["current_points"]
    snap["avg_points_per_race"] = snap["current_points"] / snapshot_round
    snap["races_remaining"] = races_remaining
    snap["season"] = season

    return snap


def prepare_2025_features(feature_cols,
                          races_completed=22,
                          races_remaining=2):
    """Fetch 2025 data and build feature matrix compatible with training columns."""
    live_season = 2025
    print(f"\n=== Building LIVE snapshot for season {live_season} ===")
    df_2025 = fetch_season_results(live_season)

    if df_2025.empty:
        raise RuntimeError("No data for 2025; cannot build live snapshot.")

    snap_2025 = build_live_snapshot_2025(
        df_2025,
        races_completed=races_completed,
        races_remaining=races_remaining,
    )

    # One-hot encode constructor
    snap_2025 = pd.get_dummies(snap_2025, columns=["constructor"], prefix="team")

    # Ensure all team_ columns used in training exist in snap_2025
    for col in feature_cols:
        if col.startswith("team_") and col not in snap_2025.columns:
            snap_2025[col] = 0

    # Extra teams in 2025 but not in training are ignored (not in feature_cols)
    X_live = snap_2025[feature_cols]

    return snap_2025, X_live


# -------------------------------
# 4. Train model + predict WDC
# -------------------------------

def train_wdc_model(X, y):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)

    y_val_pred = model.predict(X_val)
    print("\nValidation accuracy:", accuracy_score(y_val, y_val_pred))
    print("\nValidation classification report:\n")
    print(classification_report(y_val, y_val_pred))

    return model


def predict_wdc(model, snap_live: pd.DataFrame, X_live: pd.DataFrame) -> pd.DataFrame:
    probs = model.predict_proba(X_live)[:, 1]  # probability champion=1

    out = snap_live[["season", "driver", "current_points", "gap_to_leader"]].copy()
    out["wdc_probability"] = probs

    out = out.sort_values("wdc_probability", ascending=False).reset_index(drop=True)
    return out


def main():
    # 1) Train on completed seasons INCLUDING 2024
    train_seasons = [2022, 2023, 2024]
    X_train, y_train, feature_cols = prepare_training_data(train_seasons, races_remaining=2)
    print(f"\nTraining samples: {len(X_train)}")
    model = train_wdc_model(X_train, y_train)

    # 2) Build 2025 snapshot (22 done, 2 to go) and predict
    snap_2025, X_live_2025 = prepare_2025_features(
        feature_cols,
        races_completed=22,
        races_remaining=2,
    )

    print("\n=== WDC prediction for 2025 with 2 races remaining (after round 22) ===")
    predictions = predict_wdc(model, snap_2025, X_live_2025)
    print(predictions.head(10))
    
if __name__ == "__main__":
    main()

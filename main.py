import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import fastf1

def fetch_f1_results(season=2023):
    """
    Use FastF1 to get race results for a given season.
    We loop over rounds until FastF1 tells us there is no such round.
    Note: FastF1 has full data from around 2018 onwards.
    """
    rows = []

    for round_no in range(1, 30):  # 30 is just a safe upper bound
        try:
            # You can use numeric round numbers directly
            session = fastf1.get_session(season, round_no, "R")  # "R" = race
        except Exception as e:
            print(f"Stopping at round {round_no}: {e}")
            break

        event_name = session.event['EventName']
        print(f"Loading {season} round {round_no}: {event_name}")

        try:
            session.load()   # actually fetch data from F1/jolpica servers
        except Exception as e:
            print(f"  Skipping {event_name} (couldn't load): {e}")
            continue

        results = session.results
        if results is None or results.empty:
            print(f"  No results for {event_name}, stopping.")
            break

        # session.results is already a pandas DataFrame with many columns
        # e.g. DriverNumber, Abbreviation, TeamName, GridPosition, Position, etc. :contentReference[oaicite:1]{index=1}
        for _, r in results.iterrows():
             grid = r['GridPosition']
             classified = r['ClassifiedPosition']

            # Skip rows with missing grid/classified values
             if pd.isna(grid) or pd.isna(classified):
                continue

            # Convert grid safely
             try:
                grid_int = int(grid)
             except (TypeError, ValueError):
                continue

            # Convert classified position safely
            # Some values can be 'R', 'W', etc. -> skip those
             try:
                pos_int = int(classified)
             except (TypeError, ValueError):
                # e.g. 'R' for retired -> we ignore those for this simple model
                continue
            
             rows.append({
                "season": season,
                "round": round_no,
                "race": event_name,
                "driver": r['LastName'],           # or r['FullName'] if you prefer
                "constructor": r['TeamName'],
                "grid": int(r['GridPosition']),
                "position": int(r['ClassifiedPosition']),
            })

    df = pd.DataFrame(rows)
    return df

    """
    Fetch race results for a given season from the Ergast API.
    Returns a pandas DataFrame.
    """
    data = []

    for round_no in range(1, rounds + 1):
        url = f"http://ergast.com/api/f1/{season}/{round_no}/results.json"
        response = requests.get(url)
        
        # Basic error handling
        if response.status_code != 200:
            print(f"Failed to fetch round {round_no}, status code: {response.status_code}")
            continue

        json_data = response.json()
        races = json_data["MRData"]["RaceTable"]["Races"]
        if not races:
            # No race (maybe season has fewer rounds than 'rounds')
            print(f"No race data for round {round_no}, stopping.")
            break

        race = races[0]
        race_name = race["raceName"]

        for result in race["Results"]:
            data.append({
                "season": season,
                "round": round_no,
                "race": race_name,
                "driver": result["Driver"]["familyName"],
                "constructor": result["Constructor"]["name"],
                "grid": int(result["grid"]),
                "position": int(result["position"])
            })

    df = pd.DataFrame(data)
    return df


def add_podium_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary 'podium' column: 1 if position <= 3, else 0.
    """
    df["podium"] = df["position"].apply(lambda x: 1 if x <= 3 else 0)
    return df


def encode_constructor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode constructor (team) as numeric using one-hot encoding.
    """
    constructor_dummies = pd.get_dummies(df["constructor"], prefix="team")
    df_encoded = pd.concat([df, constructor_dummies], axis=1)
    return df_encoded


def train_model(df: pd.DataFrame):
    """
    Train a simple logistic regression model to predict whether
    a driver will finish on the podium based on grid + team.
    """
    # Select features: grid position and team one-hot columns
    feature_cols = ["grid"] + [col for col in df.columns if col.startswith("team_")]
    X = df[feature_cols]
    y = df["podium"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Model accuracy:", acc)
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))

    return model, feature_cols


def show_some_predictions(df: pd.DataFrame, model, feature_cols):
    """
    Show a few rows with actual vs predicted podium results.
    """
    X_all = df[feature_cols]
    df["predicted_podium"] = model.predict(X_all)

    print("\nSample predictions (1=podium, 0=not podium):\n")
    print(df[["race", "driver", "constructor", "grid", "position", "podium", "predicted_podium"]].head(15))


def main():
    print("Fetching F1 data...")
    df = fetch_f1_results(season=2023) 
    print(f"Total rows fetched: {len(df)}")

    df = add_podium_column(df)
    df = encode_constructor(df)

    print("Training model...")
    model, feature_cols = train_model(df)

    show_some_predictions(df, model, feature_cols)


if __name__ == "__main__":
    main()

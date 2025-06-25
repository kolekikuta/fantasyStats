from nba_api.stats.endpoints import playergamelogs, scoreboardv2, commonallplayers, scheduleleaguev2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime, timedelta
import requests



import joblib
import pandas as pd

### ------------------ Data Retrieval ------------------ ###
#import data from NBA API
def fetch_historical_data():
    all_data = []
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
    for season in seasons:
        logs = playergamelogs.PlayerGameLogs(season_nullable=season).get_data_frames()[0]
        logs = clean_data(logs)
        all_data.append(logs)
    return pd.concat(all_data, ignore_index=True)

# Get today's data
def getTodaysData():
    currYear = datetime.now().year
    currMonth = datetime.now().month

    if currMonth < 10:
        season_nullable = f"{currYear-1}-{currYear%100:02d}"
    else:
        season_nullable = f"{currYear}-{currYear%100+1:02d}"
    player_game_logs = playergamelogs.PlayerGameLogs(season_nullable=season_nullable, last_n_games_nullable=5)

    data_df = player_game_logs.get_data_frames()[0]
    data_df = clean_data(data_df)

    joblib.dump(data_df, 'data_last_five.pkl')

    return preprocess_data(data_df)


def getUpcomingMatchups():
    print("Fetching upcoming matchups...")
    players_df = commonallplayers.CommonAllPlayers(is_only_current_season=1).get_data_frames()[0]
    players_df = players_df[["PERSON_ID", "DISPLAY_FIRST_LAST", "TEAM_ID", "TEAM_NAME"]]

    matchup_records = []

    today = datetime.now()
    monday = today - timedelta(days=today.weekday())

    teams_df = pd.read_json('teams.json', orient='records')

    schedule = scheduleleaguev2.ScheduleLeagueV2()
    games_df = schedule.season_games.get_data_frame()
    weeks_df = schedule.season_weeks.get_data_frame()

    # Check if today's date is within the range of weeks
    weeks_df["startDate"] = pd.to_datetime(weeks_df["startDate"]).dt.tz_localize(None)
    weeks_df["endDate"]   = pd.to_datetime(weeks_df["endDate"]).dt.tz_localize(None)
    last_week = weeks_df.sort_values("weekNumber", ascending=True).iloc[-1]
    last_start = last_week["startDate"]
    last_end   = last_week["endDate"]

    if today > last_end:
        # off-season: use the final week
        window_start = last_start
        window_end   = last_end
        print(f"Off-season detected — using last season week {last_week['weekNumber']} ({last_start.date()} → {last_end.date()})")
    else:
        # in-season: rolling Mon→Sun
        window_start = today - timedelta(days=today.weekday())
        window_end   = window_start + timedelta(days=6)
        print(f"In-season: using week window {window_start.date()} → {window_end.date()}")

    games_df['gameDate'] = pd.to_datetime(games_df['gameDate'])

    mask = (games_df['gameDate'] >= window_start) & (games_df['gameDate'] < window_end)
    week_games = games_df.loc[mask].copy()

    matchups = (
        week_games
        .loc[:, ['gameDate',
                 'homeTeam_teamTricode',
                 'awayTeam_teamTricode']]
        .rename(columns={
            'homeTeam_teamTricode': 'HOME',
            'awayTeam_teamTricode': 'AWAY',
            'gameDate': 'GAME_DATE'
        })
    )

    for _, g in week_games.iterrows():
        game_date = g["gameDate"].strftime("%Y-%m-%d")
        home_id   = g["homeTeam_teamId"]
        away_id   = g["awayTeam_teamId"]
        home_abbr = g["homeTeam_teamTricode"]
        away_abbr = g["awayTeam_teamTricode"]

        # roster slices
        home_players = players_df[players_df["TEAM_ID"] == home_id]
        away_players = players_df[players_df["TEAM_ID"] == away_id]

        for _, p in home_players.iterrows():
            matchup_records.append({
                "PLAYER_NAME": p["DISPLAY_FIRST_LAST"],
                "MATCHUP":     f"{home_abbr} vs. {away_abbr}",
                "GAME_DATE":   game_date
            })
        for _, p in away_players.iterrows():
            matchup_records.append({
                "PLAYER_NAME": p["DISPLAY_FIRST_LAST"],
                "MATCHUP":     f"{away_abbr} @ {home_abbr}",
                "GAME_DATE":   game_date
            })

    # test case for breaks or end of season, no upcoming matchups
    if not matchup_records:
        print("No matchups found for the upcoming week.")
        return None


    matchups_df = pd.DataFrame(matchup_records)
    matchups_df.sort_values(by=['PLAYER_NAME', 'GAME_DATE'], inplace=True)
    matchups_df.reset_index(drop=True, inplace=True)

    print("Upcoming matchups fetched successfully.")
    return matchups_df

def buildFeatureSet():
    print("Building feature set for all players...")
    currYear = datetime.now().year
    currMonth = datetime.now().month

    if currMonth < 10:
        season_nullable = f"{currYear-1}-{currYear%100:02d}"
    else:
        season_nullable = f"{currYear}-{currYear%100+1:02d}"
    player_game_logs = playergamelogs.PlayerGameLogs(season_nullable=season_nullable, last_n_games_nullable=10)

    data_df = player_game_logs.get_data_frames()[0]
    data_df = clean_data(data_df)

    stats = ["PTS","AST","REB","FG_PCT","FT_PCT","FG3M","STL","BLK","TOV"]
    rolling_means = (
        data_df
        .groupby("PLAYER_NAME")[stats]
        .mean()
        .reset_index()
        .rename(columns={s: f"AVG_{s}_PAST_5" for s in stats})
    )

    matchups = getUpcomingMatchups()
    matchups_with_avgs = matchups.merge(
        rolling_means,
        on="PLAYER_NAME",
        how="left"
    )

    matchups_with_avgs[["TEAM", "OPPONENT", "HOME"]] = matchups_with_avgs["MATCHUP"].apply(split_matchup)
    matchups_with_avgs["GAME_DATE"] = pd.to_datetime(matchups_with_avgs["GAME_DATE"])

    features_df = matchups_with_avgs.copy()
    features_df['TIMESTAMP'] = matchups_with_avgs['GAME_DATE'].astype('int64') / 1e9  # UNIX time

    features_df = pd.concat([
        features_df,
        one_hot_encode(matchups_with_avgs, "TEAM"),
        one_hot_encode(matchups_with_avgs, "OPPONENT")
    ], axis=1)

    features_df["PLAYER_NAME"] = matchups_with_avgs["PLAYER_NAME"]

    features_df.to_pickle('features.pkl')
    print("Feature set built successfully.")

    return features_df

def loadData():
    print("Loading data...")
    try:
        data_df = pd.read_pickle('data.pkl')
        print("Data loaded successfully from pickle file.")
        return data_df
    except:
        print("No data found. Fetching from NBA API...")
        data_df = fetch_historical_data()
        if data_df is not None:
            data_df.to_pickle('data.pkl')
            return data_df
        else:
            print("Failed to retrieve data.")
            return None


def clean_data(df):
    drop_cols = [
        "SEASON_YEAR", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "WL", "BLKA", "PF", "PFD", "DD2", "TD3",
        "GP_RANK", "W_RANK", "L_RANK", "W_PCT_RANK", "MIN_RANK", "FGM_RANK", "FGA_RANK", "FG_PCT_RANK",
        "FG3M_RANK", "FG3A_RANK", "FG3_PCT_RANK", "FTM_RANK", "FTA_RANK", "FT_PCT_RANK", "OREB_RANK",
        "DREB_RANK", "REB_RANK", "AST_RANK", "TOV_RANK", "STL_RANK", "BLK_RANK", "BLKA_RANK", "PF_RANK",
        "PFD_RANK", "PTS_RANK", "PLUS_MINUS_RANK", "NBA_FANTASY_PTS_RANK", "DD2_RANK", "TD3_RANK",
        "NICKNAME", "PLUS_MINUS", "WNBA_FANTASY_PTS", "WNBA_FANTASY_PTS_RANK", "MIN_SEC"
    ]
    return df.drop(columns=drop_cols)

### ------------------ Data Preprocessing ------------------ ###
def preprocess_data(data_df):
    print("Pre-processing data...")

    # Apply matchup split just once to data_df
    data_df[["TEAM", "OPPONENT", "HOME"]] = data_df["MATCHUP"].apply(split_matchup)
    data_df["GAME_DATE"] = pd.to_datetime(data_df["GAME_DATE"])

    features_df = pd.DataFrame(index=data_df.index)
    features_df['TIMESTAMP'] = data_df['GAME_DATE'].astype('int64') / 1e9  # UNIX time

    # Rolling average stats per player
    for stat in ["PTS", "AST", "REB", "FG_PCT", "FT_PCT", "FG3M", "STL", "BLK", "TOV"]:
        features_df[f"AVG_{stat}_PAST_5"] = data_df.groupby("PLAYER_NAME")[stat].transform(lambda x: x.rolling(5, min_periods=1).mean())
    features_df["AVG_FG_PCT_PAST_5"] = data_df.groupby("PLAYER_NAME")["FG_PCT"].transform(lambda x: x.rolling(5, min_periods=1).mean())

    # One-hot encode TEAM and OPPONENT columns
    features_df = pd.concat([
        features_df,
        one_hot_encode(data_df, "TEAM"),
        one_hot_encode(data_df, "OPPONENT")
    ], axis=1)

    # for training only
    #train_cols = features_df.columns.tolist()
    #joblib.dump(train_cols, 'train_columns.pkl')

    # Add target + identifier
    features_df["PLAYER_NAME"] = data_df["PLAYER_NAME"]
    #features_df["NBA_FANTASY_PTS"] = data_df["NBA_FANTASY_PTS"]


    features_df.to_pickle('features.pkl')
    print("Data pre-processed successfully.")

    return features_df


def one_hot_encode(df, col):
    enc_opp = joblib.load('opp_encoder.pkl') if col == "OPPONENT" else joblib.load('team_encoder.pkl')
    cols = joblib.load('opp_columns.pkl') if col == "OPPONENT" else joblib.load('team_columns.pkl')

    encoded = enc_opp.transform(df[[col]])

    return pd.DataFrame(encoded, columns=cols, index=df.index)

def one_hot_encode_train(df):
    enc_opp = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    opp_train = enc_opp.fit_transform(df[["OPPONENT"]])
    opp_cols  = enc_opp.get_feature_names_out(["OPPONENT"])
    df = pd.concat([ df.drop(columns=["OPPONENT"]),
                          pd.DataFrame(opp_train, columns=opp_cols, index=df.index)
                        ], axis=1)

    # persist both model and enc_opp and opp_cols
    joblib.dump(enc_opp,  'opp_encoder.pkl')
    joblib.dump(opp_cols, 'opp_columns.pkl')

    enc_team = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    team_train = enc_team.fit_transform(df[["TEAM"]])
    team_cols  = enc_team.get_feature_names_out(["TEAM"])
    df = pd.concat([ df.drop(columns=["TEAM"]),
                          pd.DataFrame(team_train, columns=team_cols, index=df.index)
                        ], axis=1)
    # persist both model and enc_team and team_cols
    joblib.dump(enc_team,  'team_encoder.pkl')
    joblib.dump(team_cols, 'team_columns.pkl')

    return

#utility function
def split_matchup(row):
    if "vs." in row:
        team, opponent = row.split(" vs. ")
        return pd.Series([team, opponent, 1])  # 1 = home
    elif "@" in row:
        team, opponent = row.split(" @ ")
        return pd.Series([team, opponent, 0])  # 0 = away
    else:
        return pd.Series([None, None, None])


### ------------------ Model Training & Prediction ------------------ ###
def trainModel(data_df):
    print("Training model...")
    X = data_df.drop(columns=["NBA_FANTASY_PTS", "PLAYER_NAME"])
    y = data_df["NBA_FANTASY_PTS"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'nba_fantasy_model.pkl')

    print("Model trained successfully.")

    return model, X_test, y_test

def test(model, X_test, y_test):
    print("Testing model...")
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)

    print("RMSE:", rmse)
    print("MAE:", mae)

def predict(X_test):
    try:
        model = loadModel()
        if model is None:
            return None
    except Exception as e:
        print("Error loading model:", e)
        return None

    print("Making predictions...")

    X_original = X_test.copy()

    if "PLAYER_NAME" in X_test.columns:
        X_test = X_test.drop(columns=["PLAYER_NAME"])

    train_cols = joblib.load('train_columns.pkl')
    X_test_aligned = X_test.reindex(columns=train_cols, fill_value=0)

    # Get predictions
    preds = model.predict(X_test_aligned)
    results_df = pd.DataFrame({
        "PLAYER_NAME": X_original["PLAYER_NAME"].values,
        "GAME_DATE":   pd.to_datetime(X_original["GAME_DATE"], unit="s") \
                            .dt.strftime("%Y-%m-%d"),
        "PREDICTED_FANTASY_PTS": preds
    })

    # Round and sort if you like
    results_df["PREDICTED_FANTASY_PTS"] = results_df["PREDICTED_FANTASY_PTS"].round()

    # --- NEW AGGREGATIONS ---
    # 1. Weekly sum per player
    weekly_sum = (
        results_df
        .groupby("PLAYER_NAME")["PREDICTED_FANTASY_PTS"]
        .sum()
        .reset_index()
        .rename(columns={"PREDICTED_FANTASY_PTS": "WEEKLY_SUM"})
    )

    # 2. Next-game projection: pick the earliest GAME_DATE per player
    #    First, sort by date
    results_by_date = results_df.sort_values(["PLAYER_NAME", "GAME_DATE"])
    #    Then drop duplicates keeping the first per player
    next_game = (
        results_by_date
        .drop_duplicates("PLAYER_NAME", keep="first")
        .loc[:, ["PLAYER_NAME", "PREDICTED_FANTASY_PTS"]]
        .rename(columns={"PREDICTED_FANTASY_PTS": "NEXT_GAME_PTS"})
    )

    # 3. Merge them together
    summary = pd.merge(weekly_sum, next_game, on="PLAYER_NAME")

    # Optionally, join back the full schedule if you need the date:
    # summary = summary.merge(
    #     results_by_date[["PLAYER_NAME","GAME_DATE"]].drop_duplicates("PLAYER_NAME"),
    #     on="PLAYER_NAME"
    # )

    # Sort by weekly sum descending
    summary = summary.sort_values("WEEKLY_SUM", ascending=False).reset_index(drop=True)

    # Persist or return
    summary.to_json('predictions.json', orient='records', force_ascii=False, lines=True)
    print("Predictions made successfully.")
    return summary

def loadModel():
    print("Loading model...")
    try:
        model = joblib.load('nba_fantasy_model.pkl')
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return None
"""
def buildFeatureSet(data_df, processed_df, games_per_player=1):
    print(f"Building feature set for all players...")

    if data_df is None or processed_df is None:
        print("Missing data for building feature set.")
        return None

    # Make sure GAME_DATE is datetime for proper sorting
    data_df["GAME_DATE"] = pd.to_datetime(data_df["GAME_DATE"])

    # Sort data by date (latest first)
    data_df = data_df.sort_values(by="GAME_DATE", ascending=False)

    # Get the indices of the N most recent games for each player
    recent_indices = (
        data_df.groupby("PLAYER_NAME")
               .head(games_per_player)
               .index
    )

    # Filter the processed feature set to only include those games
    selected_features = processed_df.loc[recent_indices]

    print(f"Feature set built for {selected_features['PLAYER_NAME'].nunique()} players.")
    selected_features.drop(columns=["NBA_FANTASY_PTS"], inplace=True, errors='ignore')

    return selected_features
"""




#data_df = loadData()
#processed_df = preprocess_data(data_df)
#x = buildFeatureSet(data_df, processed_df)
#x = buildFeatureSet()
#model, X_test, y_test = trainModel(processed_df)
#test(model, X_test, y_test)
#predictions = predict(x)

from flask import Flask, request, jsonify
from nba_api.stats.endpoints import playergamelogs
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime



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

    # One-hot encode TEAM and OPPONENT columns
    features_df = pd.concat([
        features_df,
        one_hot_encode(data_df, "TEAM"),
        one_hot_encode(data_df, "OPPONENT")
    ], axis=1)

    # Add target + identifier
    #eatures_df["NBA_FANTASY_PTS"] = data_df["NBA_FANTASY_PTS"]
    features_df["PLAYER_NAME"] = data_df["PLAYER_NAME"]

    features_df.to_pickle('features.pkl')
    print("Data pre-processed successfully.")
    return features_df


def one_hot_encode(df, col):
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    transformed = enc.fit_transform(df[[col]])
    return pd.DataFrame(transformed, columns=enc.get_feature_names_out([col]), index=df.index)


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
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print("MSE:", mse)
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

    # Get predictions
    predictions = model.predict(X_test)

    # Make sure PLAYER_NAME is in the test set
    if "PLAYER_NAME" not in X_original.columns:
        print("Error: PLAYER_NAME column is missing in X_test.")
        return None

    # Create a DataFrame with names and predictions
    results_df = pd.DataFrame({
        "PLAYER_NAME": X_original["PLAYER_NAME"].values,
        "PREDICTED_FANTASY_PTS": predictions
    })

    results_df = results_df.sort_values(by="PREDICTED_FANTASY_PTS", ascending=False).reset_index(drop=True)
    results_df["PREDICTED_FANTASY_PTS"] = results_df["PREDICTED_FANTASY_PTS"].round()

    results_df.to_json('predictions.json', orient='records', force_ascii=False, lines=True)
    print("Predictions made successfully.")

    return results_df

def loadModel():
    print("Loading model...")
    try:
        model = joblib.load('nba_fantasy_model.pkl')
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return None

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






#data_df = loadData()
#processed_df = preprocess_data(data_df)
#x = buildFeatureSet(data_df, processed_df)
#X = processed_df.drop(columns=["NBA_FANTASY_PTS", "PLAYER_NAME"])
#y = processed_df["NBA_FANTASY_PTS"]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#model = loadModel()
#model, X_test, y_test = trainModel(processed_df)
#test(model, X_test, y_test)
#predictions = predict(x)

#getUpcomingMatchups()
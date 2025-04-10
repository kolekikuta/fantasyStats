from flask import Flask, request, jsonify
from nba_api.stats.endpoints import playergamelogs
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder


import joblib
import pandas as pd



#import data from NBA API
def getHistoricalData():
    data_df = pd.DataFrame()
    try:
        for season in ["2021-22", "2022-23", "2023-24", "2024-25"]:
            print("Retrieving data for season: ", season)
            player_game_logs = playergamelogs.PlayerGameLogs(season_nullable=season)
            temp_df = player_game_logs.get_data_frames()[0]
            temp_df = temp_df.drop(["SEASON_YEAR", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "WL", "BLKA", "PF", "PFD",
                                  "DD2", "TD3", "GP_RANK", "W_RANK", "L_RANK", "W_PCT_RANK", "MIN_RANK", "FGM_RANK", "FGA_RANK", "FG_PCT_RANK", "FG3M_RANK", "FG3A_RANK", "FG3_PCT_RANK",
                                  "FTM_RANK", "FTA_RANK", "FT_PCT_RANK", "OREB_RANK", "DREB_RANK", "REB_RANK", "AST_RANK", "TOV_RANK", "STL_RANK", "BLK_RANK", "BLKA_RANK", "PF_RANK",
                                 "PFD_RANK", "PTS_RANK", "PLUS_MINUS_RANK", "NBA_FANTASY_PTS_RANK", "DD2_RANK", "TD3_RANK", "NICKNAME", "PLUS_MINUS",
                                 "WNBA_FANTASY_PTS", "WNBA_FANTASY_PTS_RANK", "MIN_SEC"], axis=1)

            data_df = pd.concat([data_df, temp_df], ignore_index=True)
    except Exception as e:
        print("Error retrieving player game logs:", e)
        return None

    print("Data retrieved successfully.")
    print("Pre-processing data...")

    #split matchup column into home and away teams and one hot encode
    data_df[["TEAM", "OPPONENT", "HOME"]] = data_df["MATCHUP"].apply(split_matchup)

    team_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = team_encoder.fit_transform(data_df[["TEAM"]])
    encoded_df = pd.DataFrame(encoded, columns=team_encoder.get_feature_names_out(["TEAM"]), index=data_df.index)
    data_df = pd.concat([data_df.drop(columns=["TEAM"]), encoded_df], axis=1)

    opponent_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = opponent_encoder.fit_transform(data_df[["OPPONENT"]])
    encoded_df = pd.DataFrame(encoded, columns=opponent_encoder.get_feature_names_out(["OPPONENT"]), index=data_df.index)
    data_df = pd.concat([data_df.drop(columns=["OPPONENT"]), encoded_df], axis=1)

    #convert datetime to timestamp
    data_df["GAME_DATE"] = pd.to_datetime(data_df["GAME_DATE"])
    data_df['TIMESTAMP'] = data_df['GAME_DATE'].values.astype('int64') / 10**9   # Convert to seconds since epoch

    print("Data pre-processed successfully.")
    return data_df

def split_matchup(row):
    if "vs." in row:
        team, opponent = row.split(" vs. ")
        return pd.Series([team, opponent, 1])  # 1 = home
    elif "@" in row:
        team, opponent = row.split(" @ ")
        return pd.Series([team, opponent, 0])  # 0 = away
    else:
        return pd.Series([None, None, None])

def trainModel(data_df):
    print("Training model...")
    X = data_df.drop(columns=["PLAYER_NAME", "NBA_FANTASY_PTS", "MATCHUP", "GAME_DATE"])
    y = data_df["NBA_FANTASY_PTS"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'nba_fantasy_model.pkl')

    print("Model trained successfully.")

    return model, X_test, y_test


def predict(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print("Predictions:", predictions)
    print("MSE:", mse)
    print("MAE:", mae)

data_df = getHistoricalData()
model, X_test, y_test = trainModel(data_df)
predict(model, X_test, y_test)

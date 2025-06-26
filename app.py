from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import fantasy
import pandas as pd
import joblib
import json

app = Flask(__name__)
CORS(app)






@app.route('/')
def index():
    return render_template("index.html")

@app.route('/player')
def player_page():
    player_name = request.args.get("name")
    if not player_name:
        return "Player name not provided", 400


    try:
        with open("data_last_five.json", "r", encoding="utf-8") as f:
            content = json.load(f)

        fetch_date_str = content.get("metadata", {}).get("generate_date", None)
        fetch_date = datetime.strptime(fetch_date_str, "%Y-%m-%d %H:%M:%S") if fetch_date_str else None
        if datetime.now() - fetch_date > pd.Timedelta(days=1):
            print("Game logs older than 24 hours. Fetching...")
            dataLastFive_df = fantasy.getLastFive()
        else:
            print("Using cached game logs.")
            dataLastFive_df = pd.DataFrame(content["data"])
    except Exception as e:
        print("No valid data found. Fetching new.", e)
        dataLastFive_df = fantasy.getLastFive()

    dataLastFive_df["GAME_DATE"] = pd.to_datetime(dataLastFive_df["GAME_DATE"])
    dataLastFive_df = dataLastFive_df[dataLastFive_df["PLAYER_NAME"] == player_name].sort_values("GAME_DATE", ascending=False)

    with open("predictions.json", "r", encoding="utf-8") as f:
        content = json.load(f)
    prediction_df = pd.DataFrame(content["data"])

    fantasy_row = prediction_df[prediction_df["PLAYER_NAME"] == player_name]
    fantasy_value = fantasy_row["NEXT_GAME_PTS"].values[0] if not fantasy_row.empty else None
    fantasy_value_week = fantasy_row["WEEKLY_SUM"].values[0] if not fantasy_row.empty else None

    if dataLastFive_df.empty:
        return f"No data found for player: {player_name}", 404

    return render_template("player.html",
                           player_name=player_name,
                           games=dataLastFive_df.to_dict(orient="records"),
                           fantasy_value=fantasy_value,
                           fantasy_value_week=fantasy_value_week)

@app.route('/api/predictions')
def predictions():

    try:
        with open("predictions.json", "r", encoding="utf-8") as f:
            content = json.load(f)

        generate_date_str = content.get("metadata", {}).get("generate_date", None)
        generate_date = datetime.strptime(generate_date_str, "%Y-%m-%d %H:%M:%S") if generate_date_str else None
        if datetime.now() - generate_date > pd.Timedelta(days=1):
            print("Predictions are older than 24 hours. Recomputing...")
            features = fantasy.buildFeatureSet()
            predictions_df = fantasy.predict(features)
            return jsonify(predictions_df.to_dict(orient="records"))
        else:
            print("Using cached predictions.")
            return jsonify(content["data"])
    except Exception as e:
        print("No valid predictions found. Generating new.", e)
        features = fantasy.buildFeatureSet()
        predictions_df = fantasy.predict(features)
        return jsonify(predictions_df.to_dict(orient="records"))

@app.route('/api/data')
def data():
    data = joblib.load('data_last_five.pkl')
    # Sort by player name and game date
    data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"])
    data.sort_values(["PLAYER_NAME", "GAME_DATE"], inplace=True)

    # Group by player
    grouped = data.groupby("PLAYER_NAME")

    result = []
    for player, group in grouped:
        result.append({
            "player": player,
            "games": group.to_dict(orient="records")
        })

    return jsonify(result)





if __name__ == '__main__':
    app.run(debug=True)
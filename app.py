from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import fantasy
import pandas as pd
import joblib

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

    data = joblib.load('data_last_five.pkl')
    df = pd.DataFrame(data)

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df[df["PLAYER_NAME"] == player_name].sort_values("GAME_DATE", ascending=False)

    prediction_df = pd.read_json('predictions.json', lines=True)

    fantasy_row = prediction_df[prediction_df["PLAYER_NAME"] == player_name]
    fantasy_value = fantasy_row["NEXT_GAME_PTS"].values[0] if not fantasy_row.empty else None
    fantasy_value_week = fantasy_row["WEEKLY_SUM"].values[0] if not fantasy_row.empty else None

    if df.empty:
        return f"No data found for player: {player_name}", 404

    return render_template("player.html",
                           player_name=player_name,
                           games=df.to_dict(orient="records"),
                           fantasy_value=fantasy_value,
                           fantasy_value_week=fantasy_value_week)

@app.route('/api/predictions')
def predictions():

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
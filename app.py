from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sqlalchemy import func
from db import SessionLocal, PlayerPrediction
from datetime import datetime, timezone, timedelta
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
        # Fetch last five games using the updated function
        dataLastFive_df = fantasy.getLastFive(player_name)
    except Exception as e:
        print(f"Error fetching last five games: {e}")
        return f"Error fetching game logs for {player_name}", 500

    if dataLastFive_df.empty:
        return f"No game logs found for player: {player_name}", 404

    # Get fantasy prediction values from JSON cache
    try:
        with open("predictions.json", "r", encoding="utf-8") as f:
            content = json.load(f)
        prediction_df = pd.DataFrame(content["data"])
    except Exception as e:
        print(f"Error loading predictions.json: {e}")
        prediction_df = pd.DataFrame()

    fantasy_row = prediction_df[prediction_df["PLAYER_NAME"] == player_name]
    fantasy_value = fantasy_row["NEXT_GAME_PTS"].values[0] if not fantasy_row.empty else None
    fantasy_value_week = fantasy_row["WEEKLY_SUM"].values[0] if not fantasy_row.empty else None

    return render_template(
        "player.html",
        player_name=player_name,
        games=dataLastFive_df.to_dict(orient="records"),
        fantasy_value=fantasy_value,
        fantasy_value_week=fantasy_value_week
    )


@app.route('/api/predictions')
def predictions():
    try:
        with SessionLocal() as db:
            # Check the most recent prediction time
            latest_prediction = db.query(func.max(PlayerPrediction.generate_date)).scalar()

            if not latest_prediction or datetime.now(timezone.utc) - latest_prediction > timedelta(days=1):
                print("Predictions are older than 24 hours. Recomputing...")
                features = fantasy.buildFeatureSet()
                fantasy.predict(features)  # This writes new data to DB

            # Load fresh predictions from DB
            predictions = db.query(PlayerPrediction).all()
            prediction_data = [
                {
                    "PLAYER_NAME": p.player_name,
                    "WEEKLY_SUM": round(p.weekly_sum, 1),
                    "NEXT_GAME_PTS": round(p.next_game_pts, 1),
                }
                for p in predictions
            ]

            return jsonify(prediction_data)

    except Exception as e:
        print("Error fetching/generating predictions:", e)
        return jsonify({"error": "Failed to load predictions"}), 500

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
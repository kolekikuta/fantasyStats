from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sqlalchemy import func
from db import SessionLocal, PlayerPrediction
from datetime import datetime, timezone, timedelta
import fantasy
import pandas as pd
import joblib
import json
from memory import print_memory_usage

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

    # Get fantasy prediction values from the database
    fantasy_value = None
    fantasy_value_week = None

    try:
        with SessionLocal() as db:
            prediction = (
                db.query(PlayerPrediction)
                .filter(PlayerPrediction.player_name == player_name)
                .order_by(PlayerPrediction.generate_date.desc())  # in case of duplicates
                .first()
            )

            if prediction:
                fantasy_value = round(prediction.next_game_pts, 1)
                fantasy_value_week = round(prediction.weekly_sum, 1)

    except Exception as e:
        print(f"Error loading prediction from database: {e}")

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
            latest_prediction = db.query(func.max(PlayerPrediction.generate_date)).scalar().replace(tzinfo=timezone.utc)

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




if __name__ == '__main__':
    app.run(debug=True)
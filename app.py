from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore, auth
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import atexit
import fantasy
import pandas as pd

#Initialize Firestore DB
cred = credentials.Certificate('nbafantasy-d9051-firebase-adminsdk-fbsvc-474ddbefc0.json')
firebase_admin.initialize_app(cred)
db = firestore.client()


app = Flask(__name__)
CORS(app)

# Function to update predictions at midnight
def update_predictions():
    #load model from firebase storage
    model = fantasy.loadModel()
    if model is None:
        print("Model not found. Cannot update predictions.")
        return
    #load datasets
    
    #build feature set

    #run predictions

    #update in firestore
    print("Updating predictions...")

scheduler = BackgroundScheduler()
#update at midnight
scheduler.add_job(func=update_predictions, trigger="cron", hour=0, minute=0)
scheduler.start()

atexit.register(lambda: scheduler.shutdown())



@app.route('/')
def index():
    return "Welcome to the Fantasy Basketball Prediction API!"

#Signup endpoint to create new users
@app.route('/signup', methods=['POST'])
def signUp():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    try:
        #Create user in firebase auth
        user = auth.create_user(email=email, password=password)

        #Add user to Firestore
        db.collection('users').document(user.uid).set({
            'email': email,
            'name': name,
            "createdAt": firestore.SERVER_TIMESTAMP
        })
        return jsonify({"message": "User created", "uid": user.uid})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Login endpoint to authenticate users
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    try:
        user = auth.get_user_by_email(email)
        # Note: Password verification is done on the client side with Firebase SDK
        return jsonify({"message": "Login successful", "uid": user.uid})
    except auth.AuthError as e:
        return jsonify({"error": str(e)}), 400

#Get Predictd Fantasy Points for all players from db
@app.route('/table', methods=['GET'])
def get_table():
    try:
        players_ref = db.collection('players')
        players = players_ref.stream()
        player_list = []
        for player in players:
            player_data = player.to_dict()
            player_data['id'] = player.id
            player_list.append(player_data)
            player_list.sort(key=lambda x: x.get('PREDICTED_FANTASY_PTS', 0), reverse=True)
        return jsonify(player_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to update players' fantasy values in db
@app.route('/update', methods=['POST'])
def update_players():
    try:
        data = request.get_json()
        players_ref = db.collection('players')

        for player in data:
            player_id = player.get('id')
            new_value = player.get('fantasy_value')

            if player_id is not None and new_value is not None:
                players_ref.document(player_id).set({
                    "fantasy_value": new_value
                }, merge=True)


        return jsonify({"message": "Players updated successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
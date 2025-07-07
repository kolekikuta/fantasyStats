# 🏀 fantasyStats

**fantasyStats** is a full-stack Flask web application that provides NBA player analytics and fantasy point predictions based on recent performance. The app scrapes game logs, builds a machine learning model, and offers dynamic insights into a player's fantasy value.

## 🚀 Features

- 📈 **Fantasy Point Predictions**
  Predicts player fantasy points using a trained machine learning model based on the last 5 games.

- 🧠 **ML Model Pipeline**
  - Feature engineering from recent player performance
  - Categorical encoding and scaling
  - Trained regression model (Random Forrest Regressor)

- 📊 **Last 5 Game Logs per Player**
  - Automatically fetches from NBA API if data is outdated or missing
  - Stored in a SQLite database for fast access and caching

- 🌐 **Web Interface**
  - Search any NBA player and view their:
    - Last 5 games
    - Predicted fantasy points for the next game
    - Total predicted fantasy points for the week

- 🗃️ **Database-Backed Storage**
  - SQLAlchemy models for players, game logs, and predictions
  - Automatically updates and checks data freshness

- ⚙️ **REST API Endpoints**
  - `/api/predictions`: Get all player predictions
  - `/api/data`: Grouped last-five game logs (for development use)
  - `/player?name=Player Name`: Render a player-specific page

## 🛠️ Tech Stack

- **Backend**: Flask, SQLAlchemy
- **Frontend**: Jinja2 templates, HTML/CSS
- **Data**: NBA API via `nba_api`
- **ML**: scikit-learn, pandas
- **Storage**: SQLite (or Postgres-compatible)

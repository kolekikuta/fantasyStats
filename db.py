from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime, timezone

DATABASE_URL = "sqlite:///nba_data.db"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PlayerPrediction(Base):
    __tablename__ = "player_predictions"

    id = Column(Integer, primary_key=True, index=True)
    player_name = Column(String)
    weekly_sum = Column(Float)
    next_game_pts = Column(Float)
    generate_date = Column(DateTime, default=datetime.now(timezone.utc))

class PlayerGameLog(Base):
    __tablename__ = 'player_game_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, index=True)
    player_name = Column(String, index=True)
    game_id = Column(String)
    game_date = Column(DateTime, index=True)
    matchup = Column(String)
    min = Column(Float)
    fgm = Column(Integer)
    fga = Column(Integer)
    fg_pct = Column(Float)
    fg3m = Column(Integer)
    fg3a = Column(Integer)
    fg3_pct = Column(Float)
    ftm = Column(Integer)
    fta = Column(Integer)
    ft_pct = Column(Float)
    oreb = Column(Integer)
    dreb = Column(Integer)
    reb = Column(Integer)
    ast = Column(Integer)
    tov = Column(Integer)
    stl = Column(Integer)
    blk = Column(Integer)
    pts = Column(Integer)
    nba_fantasy_pts = Column(Float)
    available_flag = Column(Integer)
    generate_date = Column(DateTime, default=datetime.now(timezone.utc))

class NBATeams(Base):
    __tablename__ = "nba_teams"

    team_id = Column(Integer, primary_key=True, index=True)
    abbreviation = Column(String, index=True)
    teamName = Column(String)

class NBAPlayers(Base):
    __tablename__ = "nba_players"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=False, index=True)

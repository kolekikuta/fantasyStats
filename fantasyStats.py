from collections import OrderedDict
import pandas as pd
import csv

from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import playergamelogs

playerDict = {}
predictList = []


def main():
    #define dict
    #add player
    #find player
    #sort players
    #fantasy totals
    #player dict holds game dict, holds stats dict
    #add game
    #playerlist = player: game:dict(game) game = stats:dict(stats) stats = points etc
    
    stripStats(getPlayerGameLogs())
    makePredictList()
    exportCSV()
    
def addPlayer(name, date, p, r, a, s, b, to, fgm, fga, ftm, fta, threepm):
    stats = {"Points": p,
             "Rebounds": r,
             "Assists": a,
             "Steals": s,
             "Blocks": b,
             "Turnovers": to,
             "FGM": fgm,
             "FGA": fga,
             "FTM": ftm,
             "FTA": fta,
             "3PM": threepm}
    if name in playerDict:
        playerDict[name][date] = stats
    else:
        playerDict[name] = {date: stats}
    


            
def fantasyPoints(player, game):
    return ((playerDict[player][game]["Points"]) + 
            (playerDict[player][game]["Rebounds"]) + 
            (playerDict[player][game]["Assists"] * 2) + 
            (playerDict[player][game]["Steals"] * 4) +
            (playerDict[player][game]["Blocks"] * 4) +
            (playerDict[player][game]["Turnovers"] * -2) +
            (playerDict[player][game]["FGM"] * 2) +
            (playerDict[player][game]["FGA"] * -1) +
            (playerDict[player][game]["FTM"]) +
            (playerDict[player][game]["FTA"] * -1) +
            (playerDict[player][game]["3PM"])
            )
    
#least squares linear regression line    
def predict(player):
    n = len(playerDict[player])
    sx = 0
    sy = 0
    sxy = 0
    sx2 = 0
    x = 1
    sortGames = sorted(playerDict[player].keys())

    for game in sortGames:
        sx += x
        sy += fantasyPoints(player, game)
        sxy += x * fantasyPoints(player, game)
        sx2 += pow(x,2)
        x += 1
    
    if n==1:
        return sy
    else:
        m = (n*sxy - sx*sy)/(n*sx2 - pow(sx,2))
        b = (sy - m*sx)/n
        return m * (x+1) + b

def makePredictList():
    global predictList
    for player in playerDict:
        predictList.append({"Player Name": player, "Predicted Fantasy Value": round(predict(player),2)})
    predictList = sorted(predictList, key=lambda x: x["Predicted Fantasy Value"], reverse=True)
        
def printPlayerDict():
    for player in playerDict:
        print(player, end=":\n")
        for date in playerDict[player]:
            print(date, end=":\n")
            #for stat in playerDict[player][date]:
                #print(stat, playerDict[player][date][stat])
            print("Fantasy Points:", fantasyPoints(player, date))

def printPredictList():
    for player in predictList:
        name = player["Player Name"]
        value = player["Predicted Fantasy Value"]
        print(f"{name} {value:.2f}")

def getPlayerGameLogs():
    player_game_logs = playergamelogs.PlayerGameLogs(season_nullable="2021-22", last_n_games_nullable=20)
    pgl_df = player_game_logs.get_data_frames()[0]
    pgl_df = pgl_df.drop(["SEASON_YEAR", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "MATCHUP", "WL", "FG_PCT", "FG3_PCT", "FT_PCT", "OREB", "DREB", "BLKA", "PF", "PFD",
                          "DD2", "TD3", "GP_RANK", "W_RANK", "L_RANK", "W_PCT_RANK", "MIN_RANK", "FGM_RANK", "FGA_RANK", "FG_PCT_RANK", "FG3M_RANK", "FG3A_RANK", "FG3_PCT_RANK",
                          "FTM_RANK", "FTA_RANK", "FT_PCT_RANK", "OREB_RANK", "DREB_RANK", "REB_RANK", "AST_RANK", "TOV_RANK", "STL_RANK", "BLK_RANK", "BLKA_RANK", "PF_RANK",
                         "PFD_RANK", "PTS_RANK", "PLUS_MINUS_RANK", "NBA_FANTASY_PTS_RANK", "DD2_RANK", "TD3_RANK", "VIDEO_AVAILABLE_FLAG", "NICKNAME", "PLUS_MINUS",
                         "WNBA_FANTASY_PTS", "WNBA_FANTASY_PTS_RANK"], axis=1)
    return pgl_df

def stripStats(pgl_df):
    for index in pgl_df.index:
        name = pgl_df["PLAYER_NAME"][index]
        date = pgl_df["GAME_DATE"][index][0:10]
        addPlayer(name, date, pgl_df["PTS"][index], pgl_df["REB"][index], pgl_df["AST"][index], pgl_df["STL"][index], pgl_df["BLK"][index], pgl_df["TOV"][index], pgl_df["FGM"][index], pgl_df["FGA"][index], pgl_df["FTM"][index], pgl_df["FTA"][index], pgl_df["FG3M"][index])

def exportCSV():
    fields = ["Player Name", "Predicted Fantasy Value"]
            
    with open('list.csv', 'w', newline='') as csvfile:
        fields = ["Player Name", "Predicted Fantasy Value"]
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictList)
    print("Export Successful")

    
main()
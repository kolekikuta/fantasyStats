from espn_api.basketball import League
import datetime




def getFreeAgents(leagueId):
    current_year = datetime.datetime.now().year
    try:
        league = League(league_id=leagueId, year=current_year, espn_s2='your_espn_s2', swid='your_swid')
        free_agents = league.free_agents()
    except Exception as e:
        print("Error retrieving free agents:", e)
        return None

    return free_agents

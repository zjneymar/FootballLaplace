import pandas as pd
import numpy as np
from scipy.stats import poisson
def get_team_form(df, team_name, n_last=5):
    """计算某队最近 n_last 场主/客场攻防均值"""
    # 主场数据
    home_games = df[df['HomeTeam'] == team_name].tail(n_last)
    home_scored = home_games['FTHG'].mean() if len(home_games) > 0 else np.nan
    home_conceded = home_games['FTAG'].mean() if len(home_games) > 0 else np.nan

   # 客场数据：筛选该队作为客队的比赛，取最近 n_last 场（修正 AwayTime → AwayTeam）
    away_games = df[df['AwayTeam'] == team_name].tail(n_last)
    away_scored = away_games['FTAG'].mean() if len(away_games) > 0 else np.nan
    away_conceded = away_games['FTHG'].mean() if len(away_games) > 0 else np.nan

   # 联赛平均兜底（英超近10年大致值）
    league_avg_home_goals = 1.4   # 主场场均进球
    league_avg_away_goals = 1.1   # 客场场均进球
    league_avg_home_conceded = 1.2  # 主场场均失球（即客队在客场的平均进球）
    league_avg_away_conceded = 1.4  # 客场场均失球（即主队在主场的平均进球）

 
    return {
    'home_attack': home_scored if not np.isnan(home_scored) else league_avg_home_goals,
    'home_defense': home_conceded if not np.isnan(home_conceded) else league_avg_home_conceded,
    'away_attack': away_scored if not np.isnan(away_scored) else league_avg_away_goals,
    'away_defense': away_conceded if not np.isnan(away_conceded) else league_avg_away_conceded,
   }

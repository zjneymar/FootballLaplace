import pandas as pd
import numpy as np

# 联赛积分榜数据路径
LEAGUE_TABLE_FILE = "data/league-chemp.csv"


def load_league_table():
    """加载联赛积分榜数据"""
    # 处理分号分隔和引号的CSV格式
    df = pd.read_csv(LEAGUE_TABLE_FILE, sep=';', quotechar='"')
    return df


def get_league_averages():
    """
    计算联赛场均 xG 和 xGA

    返回:
        tuple: (league_avg_xG, league_avg_xGA)
    """
    league_df = load_league_table()
    # xG 是累积值，需要除以比赛场次得到场均值
    league_avg_xG = (league_df['xG'] / league_df['matches']).mean()
    league_avg_xGA = (league_df['xGA'] / league_df['matches']).mean()
    return league_avg_xG, league_avg_xGA


def get_team_form(team_name, n_last=None, end_idx=None):
    """
    根据联赛积分榜数据获取球队场均 xG 和 xGA

    参数:
        team_name: 球队名称
        n_last: 保留参数（用于兼容旧接口）
        end_idx: 保留参数（用于兼容旧接口）

    返回:
        dict: 包含球队场均 xG, xGA
    """
    league_df = load_league_table()

    # 查找球队
    team_data = league_df[league_df['team'] == team_name]

    if len(team_data) == 0:
        # 球队未找到，使用联赛平均值
        league_avg_xG, league_avg_xGA = get_league_averages()
        return {
            'xG': league_avg_xG,
            'xGA': league_avg_xGA,
            'league_avg_xG': league_avg_xG,
            'league_avg_xGA': league_avg_xGA
        }

    team_data = team_data.iloc[0]
    # xG 是累积值，需要除以比赛场次得到场均值
    team_xG = team_data['xG'] / team_data['matches']
    team_xGA = team_data['xGA'] / team_data['matches']

    # 获取联赛平均值（场均）
    league_avg_xG, league_avg_xGA = get_league_averages()

    return {
        'xG': team_xG,
        'xGA': team_xGA,
        'league_avg_xG': league_avg_xG,
        'league_avg_xGA': league_avg_xGA
    }


def predict_lambdas_xg(home_team, away_team):
    """
    使用 xG 数据预测比赛期望进球

    参数:
        home_team: 主队名称
        away_team: 客队名称

    返回:
        tuple: (lambda_home, lambda_away)
    """
    # 获取主队和客队的 xG, xGA
    home_form = get_team_form(home_team)
    away_form = get_team_form(away_team)

    # 获取联赛平均值
    league_avg_xG = home_form['league_avg_xG']
    league_avg_xGA = home_form['league_avg_xGA']

    # 计算期望进球
    # λ_home = 主队场均xG × (客队场均xGA / 联赛平均xGA)
    lambda_home = home_form['xG'] * (away_form['xGA'] / league_avg_xGA)

    # λ_away = 客队场均xG × (主队场均xGA / 联赛平均xGA)
    lambda_away = away_form['xG'] * (home_form['xGA'] / league_avg_xGA)

    return lambda_home, lambda_away


if __name__ == "__main__":
    # 测试
    home_team = "Arsenal"
    away_team = "Manchester City"

    home_form = get_team_form(home_team)
    away_form = get_team_form(away_team)

    print(f"{home_team}: xG={home_form['xG']:.2f}, xGA={home_form['xGA']:.2f}")
    print(f"{away_team}: xG={away_form['xG']:.2f}, xGA={away_form['xGA']:.2f}")
    print(f"联赛平均: xG={home_form['league_avg_xG']:.2f}, xGA={home_form['league_avg_xGA']:.2f}")

    lambda_home, lambda_away = predict_lambdas_xg(home_team, away_team)
    print(f"\n预期进球: {home_team} {lambda_home:.2f} - {lambda_away:.2f} {away_team}")

import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from models.poisson_match_prob import poisson_match_prob


def load_test_data(filepath):
    """加载test.csv数据"""
    df = pd.read_csv(filepath)
    # 重命名列以统一格式
    df = df.rename(columns={
        'home_team_name': 'HomeTeam',
        'away_team_name': 'AwayTeam',
        'home_team_goal_count': 'FTHG',
        'away_team_goal_count': 'FTAG'
    })
    # 计算实际结果
    df['FTR'] = df.apply(
        lambda x: 'H' if x['FTHG'] > x['FTAG'] else ('D' if x['FTHG'] == x['FTAG'] else 'A'),
        axis=1
    )
    return df


def get_league_averages(df):
    """
    计算联赛场均 xG 和 xGA（使用team_a_xg和team_b_xg）
    """
    # 遍历每场比赛，计算比赛前的累积xG/xGA
    all_xg = []
    all_xga = []
    min_games = 5

    for i in range(len(df)):
        home_team = df.iloc[i]['HomeTeam']
        away_team = df.iloc[i]['AwayTeam']

        if i >= min_games:
            # 主队历史数据
            home_history = df.iloc[:i]
            home_home = home_history[home_history['HomeTeam'] == home_team]
            home_away = home_history[home_history['AwayTeam'] == home_team]
            # xG: 主场用team_a_xg，客场用team_b_xg
            home_xg = home_home['team_a_xg'].sum() + home_away['team_b_xg'].sum()
            # xGA: 主场用team_b_xg，客场用team_a_xg
            home_xga = home_home['team_b_xg'].sum() + home_away['team_a_xg'].sum()
            home_matches = len(home_home) + len(home_away)

            if home_matches >= min_games:
                all_xg.append(home_xg / home_matches)
                all_xga.append(home_xga / home_matches)

            # 客队历史数据
            away_home = home_history[home_history['HomeTeam'] == away_team]
            away_away = home_history[home_history['AwayTeam'] == away_team]
            away_xg = away_home['team_a_xg'].sum() + away_away['team_b_xg'].sum()
            away_xga = away_home['team_b_xg'].sum() + away_away['team_a_xg'].sum()
            away_matches = len(away_home) + len(away_away)

            if away_matches >= min_games:
                all_xg.append(away_xg / away_matches)
                all_xga.append(away_xga / away_matches)

    league_avg_xG = np.mean(all_xg) if all_xg else 1.3
    league_avg_xGA = np.mean(all_xga) if all_xga else 1.3
    return league_avg_xG, league_avg_xGA


def get_team_xg_before_match(df, team_name, match_idx):
    """获取某队在特定比赛之前的累积 xG 和 xGA（使用team_a_xg和team_b_xg）"""
    df_history = df.iloc[:match_idx]

    if len(df_history) == 0:
        return None, None, 0

    # 该队作为主队的比赛
    home_games = df_history[df_history['HomeTeam'] == team_name]
    # 该队作为客队的比赛
    away_games = df_history[df_history['AwayTeam'] == team_name]

    # xG: 主场用team_a_xg，客场用team_b_xg
    xg = home_games['team_a_xg'].sum() + away_games['team_b_xg'].sum()
    # xGA: 主场用team_b_xg（客队xG），客场用team_a_xg（主队xG）
    xga = home_games['team_b_xg'].sum() + away_games['team_a_xg'].sum()
    matches_played = len(home_games) + len(away_games)

    return xg, xga, matches_played


def backtest(filepath="data/test.csv", min_games=5):
    """
    回测预测策略的准确性（使用test.csv中的xG数据）
    """
    df = load_test_data(filepath)

    # 计算联赛平均
    league_avg_xG, league_avg_xGA = get_league_averages(df)

    print(f"联赛平均: xG={league_avg_xG:.2f}, xGA={league_avg_xGA:.2f}")

    results = []

    for i in range(len(df)):
        match = df.iloc[i]
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        actual_result = match['FTR']
        actual_goals_home = match['FTHG']
        actual_goals_away = match['FTAG']

        # 获取历史比赛的xG数据（team_a_xg=主队xG, team_b_xg=客队xG）
        home_xg, home_xga, home_matches = get_team_xg_before_match(df, home_team, i)
        away_xg, away_xga, away_matches = get_team_xg_before_match(df, away_team, i)

        if home_matches < min_games or away_matches < min_games:
            continue

        # 场均xG/xGA
        home_avg_xG = home_xg / home_matches
        home_avg_xGA = home_xga / home_matches
        away_avg_xG = away_xg / away_matches
        away_avg_xGA = away_xga / away_matches

        # 使用xG模型计算期望进球
        lambda_home = home_avg_xG * (away_avg_xGA / league_avg_xGA)
        lambda_away = away_avg_xG * (home_avg_xGA / league_avg_xGA)

        # 泊松概率预测
        pred = poisson_match_prob(lambda_home, lambda_away)

        # 预测结果
        probs = {'H': pred['home_win'], 'D': pred['draw'], 'A': pred['away_win']}
        predicted_result = max(probs, key=probs.get)

        results.append({
            'home_team': home_team,
            'away_team': away_team,
            'actual_result': actual_result,
            'predicted_result': predicted_result,
            'actual_goals': (actual_goals_home, actual_goals_away),
            'predicted_lambda': (lambda_home, lambda_away),
            'prob_home_win': pred['home_win'],
            'prob_draw': pred['draw'],
            'prob_away_win': pred['away_win'],
            'home_xg': home_xg,
            'away_xg': away_xg,
        })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        return {
            'total_matches': 0, 'correct_predictions': 0, 'accuracy': 0,
            'home_precision': 0, 'draw_precision': 0, 'away_precision': 0,
            'home_actual_count': 0, 'draw_actual_count': 0, 'away_actual_count': 0,
            'score_accuracy': 0, 'total_goals_accuracy': 0, 'mae': 0,
            'details': results_df
        }

    total = len(results_df)

    # 准确率计算
    correct = (results_df['actual_result'] == results_df['predicted_result']).sum()
    accuracy = correct / total if total > 0 else 0

    home_correct = ((results_df['actual_result'] == 'H') & (results_df['predicted_result'] == 'H')).sum()
    draw_correct = ((results_df['actual_result'] == 'D') & (results_df['predicted_result'] == 'D')).sum()
    away_correct = ((results_df['actual_result'] == 'A') & (results_df['predicted_result'] == 'A')).sum()

    home_actual = (results_df['actual_result'] == 'H').sum()
    draw_actual = (results_df['actual_result'] == 'D').sum()
    away_actual = (results_df['actual_result'] == 'A').sum()

    home_precision = home_correct / home_actual if home_actual > 0 else 0
    draw_precision = draw_correct / draw_actual if draw_actual > 0 else 0
    away_precision = away_correct / away_actual if away_actual > 0 else 0

    # 比分预测
    results_df['score_match'] = results_df.apply(
        lambda x: x['actual_goals'][0] == int(x['predicted_lambda'][0]) and
                  x['actual_goals'][1] == int(x['predicted_lambda'][1]),
        axis=1
    )
    score_accuracy = results_df['score_match'].sum() / total if total > 0 else 0

    # 总进球数
    results_df['actual_total'] = results_df['actual_goals'].apply(lambda x: x[0] + x[1])
    results_df['predicted_total'] = results_df['predicted_lambda'].apply(lambda x: round(x[0] + x[1]))
    results_df['total_goals_match'] = results_df['actual_total'] == results_df['predicted_total']
    total_goals_accuracy = results_df['total_goals_match'].sum() / total if total > 0 else 0

    # MAE
    results_df['lambda_total'] = results_df['predicted_lambda'].apply(lambda x: x[0] + x[1])
    mae = (results_df['actual_total'] - results_df['lambda_total']).abs().mean()

    return {
        'total_matches': total,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'home_precision': home_precision,
        'draw_precision': draw_precision,
        'away_precision': away_precision,
        'home_actual_count': home_actual,
        'draw_actual_count': draw_actual,
        'away_actual_count': away_actual,
        'score_accuracy': score_accuracy,
        'total_goals_accuracy': total_goals_accuracy,
        'mae': mae,
        'details': results_df
    }


def print_backtest_report(report):
    """打印回测报告"""
    print("\n" + "=" * 60)
    print("回测报告 (使用test.csv中的xG数据)")
    print("=" * 60)
    print(f"总比赛场次: {report['total_matches']}")
    print(f"正确预测场次: {report['correct_predictions']}")
    print(f"胜平负准确率: {report['accuracy']:.1%}")
    print("-" * 60)
    print("分类准确率:")
    print(f"  主胜命中率: {report['home_precision']:.1%} ({report['home_actual_count']}场主胜)")
    print(f"  平局命中率: {report['draw_precision']:.1%} ({report['draw_actual_count']}场平局)")
    print(f"  客胜命中率: {report['away_precision']:.1%} ({report['away_actual_count']}场客胜)")
    print("-" * 60)
    print(f"比分完全匹配率: {report['score_accuracy']:.1%}")
    print(f"总进球数准确率: {report['total_goals_accuracy']:.1%}")
    print(f"总进球数平均绝对误差: {report['mae']:.2f} 球")
    print("=" * 60)


if __name__ == "__main__":
    report = backtest("data/test.csv", min_games=5)
    print_backtest_report(report)

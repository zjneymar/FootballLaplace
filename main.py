# 足球比赛预测：历史场均进球 + 双泊松模型
# 依赖: pandas, numpy, scipy
import pandas as pd
import numpy as np
from scipy.stats import poisson
from utils.load_and_prepare_data import load_and_prepare_data
from models.poisson_match_prob import poisson_match_prob
from models.predict_lambdas import predict_lambdas


# ========================
# 配置区
# ========================
CSV_FILE = "data/E0.csv"          # 比赛数据文件路径(必须放在data文件夹中数据参考例子)
HOME_TEAM = "Brentford"        # 主队名称（必须和 CSV 中一致）
AWAY_TEAM = "Brighton"        # 客队名称
N_LAST = 10                   # 使用最近 N 场主/客场数据


if __name__ == "__main__":
    df = load_and_prepare_data(CSV_FILE)
    # 检查球队是否存在
    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    if HOME_TEAM not in all_teams:
        print(f"❌ 错误: 主队 '{HOME_TEAM}' 不在数据中！可用球队示例: {sorted(list(all_teams))[:5]}...")
        exit()
    if AWAY_TEAM not in all_teams:
        print(f"❌ 错误: 客队 '{AWAY_TEAM}' 不在数据中！")
        exit()

    # 预测 λ
    print(f"\n 计算 {HOME_TEAM} vs {AWAY_TEAM} 的期望进球...")
    lambda1, lambda2 = predict_lambdas(df, HOME_TEAM, AWAY_TEAM, N_LAST)

    # 泊松预测
    result = poisson_match_prob(lambda1, lambda2)

    # 输出结果
    print("\n" + "="*50)
    print(f"⚽ 预测: {HOME_TEAM} vs {AWAY_TEAM}")
    print("="*50)
    print(f"主队期望进球 (λ₁): {result['lambda1']:.2f}")
    print(f"客队期望进球 (λ₂): {result['lambda2']:.2f}")
    print(f"最可能比分: {result['most_likely_score']} ({result['most_likely_prob']:.1%})")
    print("-"*50)
    print(f"胜平负概率:")
    print(f"  主胜: {result['home_win']:.1%}")
    print(f"  平局: {result['draw']:.1%}")
    print(f"  客胜: {result['away_win']:.1%}")
    print("\n✅ 预测完成！")
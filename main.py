# 足球比赛预测：xG + 双泊松模型
# 依赖: pandas, numpy, scipy

# Fix Windows console encoding
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from models.poisson_match_prob import poisson_match_prob
from models.xg_model import predict_lambdas


# ========================
# 配置区
# ========================
HOME_TEAM = "Sunderland"  # 主队名称
AWAY_TEAM = "Fulham"          # 客队名称


if __name__ == "__main__":
    # 预测 λ
    print(f"\n计算 {HOME_TEAM} vs {AWAY_TEAM} 的期望进球...")
    lambda1, lambda2 = predict_lambdas(HOME_TEAM, AWAY_TEAM)

    # 泊松预测
    result = poisson_match_prob(lambda1, lambda2)

    # 输出结果
    print("\n" + "="*50)
    print(f"预测: {HOME_TEAM} vs {AWAY_TEAM}")
    print("="*50)
    print(f"主队期望进球: {result['lambda1']:.2f}")
    print(f"客队期望进球: {result['lambda2']:.2f}")
    print(f"最可能比分: {result['most_likely_score']} ({result['most_likely_prob']:.1%})")
    print("-"*50)
    print(f"胜平负概率:")
    print(f"  主胜: {result['home_win']:.1%}")
    print(f"  平局: {result['draw']:.1%}")
    print(f"  客胜: {result['away_win']:.1%}")
    print("\n预测完成！")

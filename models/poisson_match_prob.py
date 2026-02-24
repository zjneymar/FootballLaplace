import numpy as np
from scipy.stats import poisson

# Dixon-Coles 模型的相关系数
DC_RHO = 0.92


def dixon_coles_prob(x, y, lambda1, lambda2, rho=DC_RHO):
    """
    Dixon-Coles 修正后的比分概率

    参数:
        x: 主队进球数
        y: 客队进球数
        lambda1: 主队期望进球
        lambda2: 客队期望进球
        rho: Dixon-Coles 相关系数

    返回:
        比分 (x,y) 的概率
    """
    # 标准泊松概率
    poisson_xy = poisson.pmf(x, lambda1) * poisson.pmf(y, lambda2)

    # 计算 κ
    kappa = lambda1 + lambda2 + lambda1 * lambda2

    # 计算 τ(x,y) 修正系数
    if x == 0 and y == 0:
        tau = 1 - lambda1 * lambda2 * (1 - rho) / kappa
    elif x == 1 and y == 0:
        tau = 1 + (1 - rho) / kappa
    elif x == 0 and y == 1:
        tau = 1 + (1 - rho) / kappa
    elif x == 1 and y == 1:
        tau = 1 - (1 - rho) / kappa
    else:
        tau = 1

    return tau * poisson_xy


def poisson_match_prob(lambda1, lambda2, max_goals=6, rho=DC_RHO):
    """
    用 Dixon-Coles 模型计算胜平负概率和最可能比分

    参数:
        lambda1: 主队期望进球
        lambda2: 客队期望进球
        max_goals: 最大计算进球数
        rho: Dixon-Coles 相关系数

    返回:
        包含预测结果的字典
    """
    # 计算修正后的概率矩阵
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_matrix[i, j] = dixon_coles_prob(i, j, lambda1, lambda2, rho)

    # 归一化（确保概率和为1）
    prob_matrix = prob_matrix / prob_matrix.sum()

    # 计算胜平负概率
    home_win = np.sum(np.tril(prob_matrix, -1))
    draw = np.sum(np.diag(prob_matrix))
    away_win = np.sum(np.triu(prob_matrix, 1))

    # 找出最可能的比分
    i_max, j_max = np.unravel_index(prob_matrix.argmax(), prob_matrix.shape)
    most_likely_score = f"{i_max}:{j_max}"
    most_likely_prob = prob_matrix[i_max, j_max]

    return {
        'home_win': home_win,
        'draw': draw,
        'away_win': away_win,
        'most_likely_score': most_likely_score,
        'most_likely_prob': most_likely_prob,
        'lambda1': lambda1,
        'lambda2': lambda2,
        'prob_matrix': prob_matrix
    }
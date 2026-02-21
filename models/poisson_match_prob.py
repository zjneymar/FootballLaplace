import pandas as pd
import numpy as np
from scipy.stats import poisson
def poisson_match_prob(lambda1, lambda2, max_goals=6):
    """用泊松分布计算胜平负概率和最可能比分"""
    home_probs = [poisson.pmf(i, lambda1) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(j, lambda2) for j in range(max_goals + 1)]

    prob_matrix = np.outer(home_probs, away_probs)

    home_win = np.sum(np.tril(prob_matrix, -1))
    draw = np.sum(np.diag(prob_matrix))
    away_win = np.sum(np.triu(prob_matrix, 1))

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
import pandas as pd
import numpy as np
from scipy.stats import poisson
from models.get_team_form import get_team_form
def predict_lambdas(df, home_team, away_team, n_last=5):
    """预测主客队期望进球 λ1, λ2"""
    form_h = get_team_form(df, home_team, n_last)
    form_a = get_team_form(df, away_team, n_last)

    lambda1 = (form_h['home_attack'] + form_a['away_defense']) / 2
    lambda2 = (form_a['away_attack'] + form_h['home_defense']) / 2

    return lambda1, lambda2
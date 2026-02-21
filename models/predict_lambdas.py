from models.xg_model import predict_lambdas_xg, get_team_form


def predict_lambdas(df, home_team, away_team, n_last=5, Endidx=None):
    """
    预测主客队期望进球 λ1, λ2 (使用 xG 数据)

    参数:
        df: 保留参数（用于兼容旧接口）
        home_team: 主队名称
        away_team: 客队名称
        n_last: 保留参数（用于兼容旧接口）
        Endidx: 保留参数（用于兼容旧接口）

    返回:
        tuple: (lambda_home, lambda_away)
    """
    return predict_lambdas_xg(home_team, away_team)

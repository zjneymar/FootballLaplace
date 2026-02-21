import pandas as pd
import numpy as np
from scipy.stats import poisson
def load_and_prepare_data(csv_file):
    """加载并预处理数据"""
    df = pd.read_csv(csv_file)
    # 尝试解析日期（兼容多种格式）
    date_cols = ['Date', 'date', 'DATE']
    date_col = None
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(date_col).reset_index(drop=True)
    else:
        print("未找到日期列，按原始顺序处理")
    return df

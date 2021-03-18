import numpy as np, pandas as pd
from scipy.special import logit, expit
from scipy.special import logit, expit

def shift_logit(df, cols, shift = .1):
    """
    shifts non-NaN vals in df[cols] up by 'shift' percents, then logit transforms 
    """
    transf = df.copy()
    transf[cols] = logit((transf[cols] + shift) / 100)
    
    return transf


def inv_transform(df, cols, shift = .1):
    """
    backtransforms a shifted logit tranformed df[cols]
    """
    inv = df.copy()
    inv[cols] = (expit(inv[cols]) + shift) * 100
    
    return inv

def guess_mean_val(df, grouped_on, cols):
    c = df.copy()
    
    mean_cols = c.groupby(grouped_on).transform('mean')[cols]
    mean_cols.columns = [f'mean_{i}' for i in mean_cols.columns]
    
    c[mean_cols.columns] = mean_cols
    
    for col in cols:
        c.loc[(c[col].isna()),col] = c[f'mean_{col}']
    
    return c.drop(columns = mean_cols.columns)

# def guess_mean_val(df, cols):
#     c = df.copy()
#     for col in cols:
#         m = c[col].mean()
#         c.loc[(c[col].isna()),col] = m
#     return c
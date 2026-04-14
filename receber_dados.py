import pandas as pd
import numpy as np
from scipy import stats

def eh_categorica(coluna, limite_unique=10):
    dados_limpos = coluna.dropna()
    n_unique = dados_limpos.nunique()
    return n_unique <= limite_unique

def eh_normal(coluna):
    dados_limpos = coluna.dropna()
    if len(dados_limpos) < 20:
        return False
    try:
        stat, p_value = stats.shapiro(dados_limpos)
        return p_value > 0.05
    except:
        return False

def preencher(dados):
    for coluna in dados.columns:
        if dados[coluna].isnull().any():
            if eh_categorica(dados[coluna]):
                valor = dados[coluna].mode()[0]
                dados[coluna] = dados[coluna].fillna(valor)
            else:
                if eh_normal(dados[coluna]):
                    valor = dados[coluna].mean()
                else:
                    valor = dados[coluna].median()
                dados[coluna] = dados[coluna].fillna(valor)
    return dados
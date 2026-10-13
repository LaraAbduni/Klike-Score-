"""Leitura dos dados e definição do cenário de previsão."""

from pathlib import Path

import numpy as np
import pandas as pd


PASTA = Path(__file__).resolve().parent
ARQUIVO = PASTA / "dados" / "klike_challenge_dataset.csv"

# Tudo que estaria disponível antes de veicular o vídeo.
CATEGORICAS = ["platform", "category", "objective", "target_audience_age", "format", "text_density", "is_retargeting", "has_subtitle", "has_cta", "has_hook", "has_face"]
NUMERICAS = ["video_duration_s", "music_voice_ratio"]
CRIATIVO = CATEGORICAS + NUMERICAS

# Só entram no exercício diagnóstico, depois da veiculação.
DEPOIS = [
    "impressions", "clicks", "ctr", "cpc", "spend", "conversions",
    "revenue", "roas", "avg_watch_time_s", "engagement_rate",
]


def carregar():
    df = pd.read_csv(ARQUIVO, parse_dates=["date"])
    if df["campaign_id"].duplicated().any():
        raise ValueError("campaign_id deveria identificar uma única linha")
    if df["date"].isna().any() or df["klike_score"].isna().any():
        raise ValueError("data e score precisam estar preenchidos")
    return df


def separar(df):
    """Últimos ~20% para teste; mantém um mesmo dia de um lado só."""
    corte = df.sort_values("date").iloc[int(len(df) * 0.8)]["date"]
    treino = df.loc[df.date < corte].sort_values(["date", "campaign_id"]).copy()
    teste = df.loc[df.date >= corte].sort_values(["date", "campaign_id"]).copy()
    return treino, teste


def atributos(df, diagnostico=False):
    colunas = CRIATIVO + (DEPOIS if diagnostico else [])
    x = df[colunas].copy()
    # Assim booleanos e faltantes entram juntos no OneHotEncoder.
    for coluna in ["is_retargeting", "has_subtitle", "has_cta", "has_hook", "has_face"]:
        x[coluna] = x[coluna].map({True: "sim", False: "não"})
    # Uma transformação por linha, sem estatísticas aprendidas do teste.
    x["log_duracao"] = np.log1p(x.pop("video_duration_s"))
    return x

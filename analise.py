"""Análise descritiva do período de treino. Rode: python analise.py"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dados import carregar, separar


PASTA = Path(__file__).resolve().parent
FIGURAS = PASTA / "figuras"
RESULTADOS = PASTA / "resultados"
PLATAFORMAS = ["Meta", "TikTok", "LinkedIn"]
CORES = {"Meta": "#4267A5", "TikTok": "#009F92", "LinkedIn": "#D18646"}
METRICAS = ["ctr", "engagement_rate", "roas", "klike_score", "conversoes_por_mil"]


def salvar(nome):
    plt.savefig(FIGURAS / nome, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def aparencia():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def resumo_numerico(df):
    colunas = [
        "video_duration_s", "impressions", "clicks", "ctr", "cpc", "spend",
        "conversions", "revenue", "roas", "avg_watch_time_s",
        "engagement_rate", "klike_score", "conversoes_por_mil",
    ]
    saida = {}
    for coluna in colunas:
        x = df[coluna].dropna()
        saida[coluna] = {
            "n": int(x.size),
            "media": float(x.mean()),
            "mediana": float(x.median()),
            "p25": float(x.quantile(0.25)),
            "p75": float(x.quantile(0.75)),
            "p95": float(x.quantile(0.95)),
            "min": float(x.min()),
            "max": float(x.max()),
        }
    return saida


def checar_qualidade(df):
    """Marca problemas, sem apagar campanhas ou substituir métricas no EDA."""
    origem = df.drop(columns=["conversoes_por_mil"], errors="ignore")
    faltantes = {
        nome: {"n": int(qtd), "pct": round(100 * qtd / len(df), 1)}
        for nome, qtd in origem.isna().sum().items()
        if qtd
    }
    # O IQR sinaliza pontos para inspeção. Em gasto e impressões, não equivale a erro.
    colunas = [
        "video_duration_s", "music_voice_ratio", "impressions", "clicks",
        "ctr", "cpc", "spend", "conversions", "revenue", "roas",
        "avg_watch_time_s", "engagement_rate", "klike_score",
    ]
    extremos = {}
    for coluna in colunas:
        serie = df[coluna].dropna()
        q1, q3 = serie.quantile([0.25, 0.75])
        intervalo = q3 - q1
        abaixo, acima = q1 - 1.5 * intervalo, q3 + 1.5 * intervalo
        mascara = df[coluna].notna() & ((df[coluna] < abaixo) | (df[coluna] > acima))
        extremos[coluna] = {
            "n": int(mascara.sum()),
            "limites": [round(float(abaixo), 5), round(float(acima), 5)],
            "exemplos": df.loc[mascara, "campaign_id"].head(5).tolist(),
        }

    def casos(mascara):
        return {"n": int(mascara.sum()), "ids": df.loc[mascara, "campaign_id"].head(10).tolist()}

    # Há pequenos desvios pela geração/arredondamento dos contadores. Acima de
    # 0,1 ponto percentual no CTR, a divergência já muda a interpretação.
    inconsistencias = {
        "ctr_vs_clicks_impressions": casos((df["ctr"] - df["clicks"] / df["impressions"]).abs() > 0.001),
        "cpc_vs_spend_clicks": casos(
            df["cpc"].notna() & ((df["cpc"] - df["spend"] / df["clicks"]).abs() > 0.011)
        ),
        "roas_vs_revenue_spend": casos(
            df["revenue"].notna() & ((df["roas"] - df["revenue"] / df["spend"]).abs() > 0.011)
        ),
        "watch_time_maior_que_duracao": casos(
            df["avg_watch_time_s"].notna() & df["video_duration_s"].notna()
            & (df["avg_watch_time_s"] > df["video_duration_s"])
        ),
        "conversions_maior_que_clicks": casos(df["conversions"] > df["clicks"]),
        "valores_negativos": casos((df[["impressions", "clicks", "spend", "conversions"]] < 0).any(axis=1)),
    }
    return {
        "faltantes": faltantes,
        "ids_repetidos": int(df["campaign_id"].duplicated().sum()),
        "linhas_repetidas": int(df.duplicated().sum()),
        "outliers_iqr": extremos,
        "inconsistencias": inconsistencias,
        "valores_no_teto": {
            "ctr_15pct": int((df["ctr"] == 0.15).sum()),
            "engajamento_10pct": int((df["engagement_rate"] == 0.10).sum()),
        },
    }


def correlacoes(df):
    """Correlação não é efeito causal. Guarda o tamanho de cada par válido."""
    atributos = ["video_duration_s", "music_voice_ratio"]
    # Tempo assistido já é um resultado da exibição; fica separado dos atributos do criativo.
    atributos_pos_exibicao = ["avg_watch_time_s"]
    saida = {"atributos_do_criativo": {}, "metrica_pos_exibicao": {}}
    for chave, colunas in [
        ("atributos_do_criativo", atributos),
        ("metrica_pos_exibicao", atributos_pos_exibicao),
    ]:
        for coluna in colunas:
            saida[chave][coluna] = {}
            for metrica in METRICAS:
                pares = df[[coluna, metrica]].dropna()
                saida[chave][coluna][metrica] = {
                    "n": len(pares),
                    "pearson": round(float(pares[coluna].corr(pares[metrica], method="pearson")), 3),
                    "spearman": round(float(pares[coluna].corr(pares[metrica], method="spearman")), 3),
                }
    return saida


def comparacoes(df):
    """Medianas por subgrupo, com n visível para evitar conclusões de células raras."""
    saida = {}
    for atributo in ["has_hook", "has_face", "has_cta", "has_subtitle", "format", "text_density"]:
        saida[atributo] = {}
        for plataforma in PLATAFORMAS:
            parte = df[df["platform"] == plataforma]
            grupos = {}
            for valor, grupo in parte.groupby(atributo, dropna=False):
                nome = "faltante" if pd.isna(valor) else str(valor)
                grupos[nome] = {
                    "n": len(grupo),
                    "medianas": {
                        metrica: None if pd.isna(grupo[metrica].median()) else round(float(grupo[metrica].median()), 4)
                        for metrica in METRICAS
                    },
                    "n_validos": {metrica: int(grupo[metrica].notna().sum()) for metrica in METRICAS},
                }
            saida[atributo][plataforma] = grupos
    return saida


def desenhar_score(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    partes = [df.loc[df["platform"] == p, "klike_score"].dropna() for p in PLATAFORMAS]
    caixa = ax.boxplot(partes, positions=np.arange(3), widths=0.5, patch_artist=True,
                       showfliers=True, flierprops={"marker": ".", "alpha": 0.35})
    for elemento, nome in zip(caixa["boxes"], PLATAFORMAS):
        elemento.set(facecolor=CORES[nome], alpha=0.66, edgecolor="#374151")
    for elemento in caixa["medians"]:
        elemento.set(color="#1f2937", linewidth=2)
    ax.set_xticks(range(3), [f"{p}\n(n={len(x)})" for p, x in zip(PLATAFORMAS, partes)])
    ax.set_ylabel("Klike Score (0–100)")
    ax.set_ylim(0, 105)
    ax.set_title("Distribuição do score por plataforma")
    ax.text(0, -0.19, "Caixa: metade central dos criativos. Linha: mediana. Pontos: valores extremos.",
            transform=ax.transAxes, fontsize=9, color="#5b6470")
    ax.yaxis.grid(True, color="#e5e7eb")
    fig.tight_layout()
    salvar("01_score_por_plataforma.png")


def barras_binarias(df, atributo, metrica, arquivo, titulo, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    largura = 0.32
    centros = np.arange(len(PLATAFORMAS))
    maximo = 0.0
    for i, valor in enumerate([False, True]):
        alturas = []
        rotulos = []
        for plataforma in PLATAFORMAS:
            x = df.loc[(df["platform"] == plataforma) & (df[atributo] == valor), metrica].dropna()
            alturas.append(100 * x.median() if len(x) else 0)
            rotulos.append(len(x))
        maximo = max(maximo, *alturas)
        barras = ax.bar(centros + (i - 0.5) * largura, alturas, width=largura - 0.04,
                        label="Sim" if valor else "Não", color="#009F92" if valor else "#A7B3C4")
        for barra, n in zip(barras, rotulos):
            ax.annotate(f"{barra.get_height():.1f}%\nn={n}",
                        (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(centros, PLATAFORMAS)
    ax.set_ylim(0, maximo * 1.28)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    nome = {"has_hook": "Hook inicial", "has_face": "Rosto no vídeo"}[atributo]
    ax.legend(title=nome, frameon=False, ncol=2)
    ax.yaxis.grid(True, color="#e5e7eb")
    ax.set_axisbelow(True)
    fig.tight_layout()
    salvar(arquivo)


def desenhar_texto(df):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    grupos = ["low", "medium", "high"]
    posicoes = np.arange(len(grupos))
    largura = 0.25
    maximo = 0
    for i, plataforma in enumerate(PLATAFORMAS):
        y, ns = [], []
        for densidade in grupos:
            score = df.loc[(df["platform"] == plataforma) & (df["text_density"] == densidade), "klike_score"]
            y.append(score.median())
            ns.append(score.notna().sum())
        maximo = max(maximo, *y)
        barras = ax.bar(posicoes + (i - 1) * largura, y, width=largura - 0.025,
                        color=CORES[plataforma], label=plataforma)
        for barra, n in zip(barras, ns):
            ax.annotate(f"{barra.get_height():.1f}\n(n={n})",
                        (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(posicoes, ["Pouco", "Médio", "Muito"])
    ax.set_ylim(0, max(100, maximo * 1.2))
    ax.set_xlabel("Texto na tela")
    ax.set_ylabel("Klike Score mediano (0–100)")
    ax.set_title("Densidade de texto e score, em cada plataforma")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.yaxis.grid(True, color="#e5e7eb")
    ax.set_axisbelow(True)
    fig.tight_layout()
    salvar("04_texto_e_score.png")


def desenhar_roas(df):
    # ROAS depende fortemente do objetivo. A tabela mantém também o n por célula.
    tabela = df.pivot_table(index="objective", columns="platform", values="roas", aggfunc="median")
    n = df.pivot_table(index="objective", columns="platform", values="roas", aggfunc="count")
    ordem = [o for o in ["awareness", "traffic", "engagement", "conversions", "app_install"] if o in tabela.index]
    tabela = tabela.reindex(index=ordem, columns=PLATAFORMAS)
    n = n.reindex(index=ordem, columns=PLATAFORMAS)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    img = ax.imshow(tabela.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto", vmin=0)
    for i in range(len(ordem)):
        for j in range(3):
            valor = tabela.iloc[i, j]
            if pd.notna(valor):
                cor = "white" if valor > tabela.stack().max() * 0.55 else "#1f2937"
                ax.text(j, i, f"{valor:.2f}x\n(n={int(n.iloc[i, j])})", ha="center", va="center", color=cor)
    ax.set_xticks(range(3), PLATAFORMAS)
    ax.set_yticks(range(len(ordem)), ordem)
    ax.set_title("ROAS mediano varia com plataforma e objetivo")
    ax.set_xlabel("Plataforma")
    ax.set_ylabel("Objetivo da campanha")
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ROAS mediano (vezes)")
    fig.tight_layout()
    salvar("05_roas_por_objetivo.png")


def main():
    FIGURAS.mkdir(exist_ok=True)
    RESULTADOS.mkdir(exist_ok=True)
    aparencia()
    treino, _ = separar(carregar())
    df = treino.copy()
    # Uma razão por impressão fica sem sentido nas linhas em que impressões e
    # CTR não batem. Conservamos essas linhas para as outras análises.
    impressoes_ok = (df["ctr"] - df["clicks"] / df["impressions"]).abs() <= 0.001
    df["conversoes_por_mil"] = np.where(
        impressoes_ok, 1000 * df["conversions"] / df["impressions"], np.nan
    )

    perfil = {
        "periodo": [str(df["date"].min().date()), str(df["date"].max().date())],
        "n": len(df),
        "plataformas": df["platform"].value_counts().to_dict(),
        "objetivos": df["objective"].value_counts().to_dict(),
        "categorias": df["category"].value_counts().to_dict(),
        "medidas": resumo_numerico(df),
        "qualidade": checar_qualidade(df),
        "correlacoes": correlacoes(df),
        "comparacoes": comparacoes(df),
        "por_plataforma": {
            p: {"n": int(len(g)), "medianas": {m: round(float(g[m].median()), 4) for m in METRICAS}}
            for p, g in df.groupby("platform")
        },
        "por_objetivo_plataforma": [
            {"plataforma": p, "objetivo": o, "n": len(g),
             "ctr_mediano": round(float(g["ctr"].median()), 4),
             "roas_mediano": round(float(g["roas"].median()), 3)}
            for (p, o), g in df.groupby(["platform", "objective"])
        ],
    }
    destino = RESULTADOS / "eda.json"
    destino.write_text(json.dumps(perfil, indent=2, ensure_ascii=False), encoding="utf-8")

    desenhar_score(df)
    barras_binarias(df, "has_hook", "ctr", "02_hook_e_ctr.png",
                    "CTR com e sem hook por plataforma", "CTR mediano (%)")
    barras_binarias(df, "has_face", "engagement_rate", "03_rosto_e_engajamento.png",
                    "Engajamento com e sem rosto", "Engajamento mediano (%)")
    desenhar_texto(df)
    desenhar_roas(df)

    qualidade = perfil["qualidade"]
    print(f"EDA: {len(df)} campanhas de treino, de {perfil['periodo'][0]} a {perfil['periodo'][1]}.")
    print(f"Score mediano: {df['klike_score'].median():.1f}. Campos com faltantes: {len(qualidade['faltantes'])}.")
    print("Inconsistências nas identidades das métricas:")
    for nome, dado in qualidade["inconsistencias"].items():
        print(f"  {nome}: {dado['n']}")
    print("Medianas por plataforma (CTR, ROAS, score):")
    for p in PLATAFORMAS:
        x = perfil["por_plataforma"][p]
        m = x["medianas"]
        print(f"  {p}: n={x['n']}, CTR={100*m['ctr']:.2f}%, ROAS={m['roas']:.2f}x, score={m['klike_score']:.1f}")
    print(f"Gráficos em {FIGURAS} e tabelas em {destino}")


if __name__ == "__main__":
    main()

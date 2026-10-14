"""Sugestões de testes para uma campanha, com evidência do conjunto de treino."""

import argparse

import joblib
import pandas as pd

from dados import PASTA, atributos, carregar, separar


ALTERACOES = [
    ("has_hook", True, "Abrir o vídeo com um hook nos primeiros 3 segundos"),
    ("has_face", True, "Testar uma versão que mostre um rosto humano"),
    ("has_subtitle", True, "Incluir legendas legíveis ao longo do vídeo"),
    ("has_cta", True, "Adicionar um CTA explícito"),
    ("text_density", "medium", "Testar densidade média de texto na tela"),
]


def pares(treino, campanha, coluna, novo, minimo=10):
    """Busca um contexto com duas alternativas minimamente observadas."""
    # Inconsistências graves entre CTR e contagens não servem para estimar CTR.
    base = treino.loc[(treino.ctr - treino.clicks / treino.impressions).abs() <= 0.001]
    if "campaign_id" in campanha:
        base = base.loc[base.campaign_id != campanha["campaign_id"]]

    contextos = [
        ["platform", "category", "objective"],
        ["platform", "category"],
        ["platform", "objective"],
        ["platform"],
    ]
    for campos in contextos:
        grupo = base
        for campo in campos:
            grupo = grupo.loc[grupo[campo] == campanha[campo]]
        antes = grupo.loc[grupo[coluna] == campanha[coluna], "ctr"].dropna()
        depois = grupo.loc[grupo[coluna] == novo, "ctr"].dropna()
        if len(antes) >= minimo and len(depois) >= minimo:
            return {
                "contexto": " / ".join(str(campanha[c]) for c in campos),
                "n_atual": len(antes), "n_alternativa": len(depois),
                "ctr_atual_pct": 100 * antes.median(),
                "ctr_alternativa_pct": 100 * depois.median(),
                "diferenca_pp": 100 * (depois.median() - antes.median()),
            }
    return None


def recomendar(campanha, treino, modelo, limite=4):
    """Retorna até `limite` mudanças apoiadas por pares e pelo modelo.

    Pode devolver menos de três se não houver três mudanças positivas com suporte.
    Isso é deliberado: não inventamos uma promessa para completar uma lista.
    """
    if not isinstance(campanha, pd.Series):
        campanha = pd.Series(campanha)
    exigidas = ["platform", "category", "objective", "campaign_id"]
    if any(pd.isna(campanha.get(c)) for c in exigidas):
        raise ValueError("campanha precisa ter ID, plataforma, categoria e objetivo")

    original = float(modelo.predict(atributos(pd.DataFrame([campanha])))[0])
    sugestoes = []
    for coluna, novo, frase in ALTERACOES:
        atual = campanha.get(coluna)
        if pd.isna(atual) or atual == novo:
            continue
        comparacao = pares(treino, campanha, coluna, novo)
        if comparacao is None or comparacao["diferenca_pp"] <= 0:
            continue
        variante = campanha.copy()
        variante[coluna] = novo
        delta = float(modelo.predict(atributos(pd.DataFrame([variante])))[0] - original)
        if delta <= 0:
            continue
        sugestoes.append({
            "acao": frase, "atributo": coluna,
            "de": str(atual), "para": str(novo),
            "delta_score_modelo": round(delta, 2),
            **{k: round(v, 3) if isinstance(v, float) else v
               for k, v in comparacao.items()},
        })
    return sorted(sugestoes, key=lambda s: s["delta_score_modelo"], reverse=True)[:limite]


def main():
    parser = argparse.ArgumentParser(description="Sugestões para testar em um criativo")
    parser.add_argument("--id", default="KLK-0169", help="campaign_id do CSV")
    args = parser.parse_args()
    df = carregar()
    treino, _ = separar(df)
    linhas = df.loc[df.campaign_id == args.id]
    if linhas.empty:
        parser.error(f"campanha {args.id} não encontrada")
    caminho = PASTA / "artefatos" / "modelo.joblib"
    if not caminho.exists():
        parser.error("rode `python modelo.py` antes de pedir recomendações")
    campanha = linhas.iloc[0]
    modelo = joblib.load(caminho)
    sugestoes = recomendar(campanha, treino, modelo)
    print(f"Campanha {args.id} | {campanha.platform} / {campanha.category} / "
          f"{campanha.objective} | score observado {campanha.klike_score:.1f}")
    for i, s in enumerate(sugestoes, start=1):
        print(f"\n{i}. {s['acao']} ({s['de']} -> {s['para']}).")
        print(f"   Cenário do modelo: +{s['delta_score_modelo']:.1f} pontos no score.")
        print(f"   Pares {s['contexto']}: CTR mediano {s['ctr_atual_pct']:.2f}% "
              f"(n={s['n_atual']}) vs. {s['ctr_alternativa_pct']:.2f}% "
              f"(n={s['n_alternativa']}); diferença observada "
              f"+{s['diferenca_pp']:.2f} ponto(s) percentual(is).")
    if len(sugestoes) < 3:
        print("\nNão há três mudanças positivas com pares suficientes para este criativo.")
    print("\nDiferenças observacionais e cenários do modelo; testar as versões em A/B "
          "antes de atribuir causalidade.")


if __name__ == "__main__":
    main()

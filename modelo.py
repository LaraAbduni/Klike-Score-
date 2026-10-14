"""Prevê o score de um criativo antes da veiculação."""

import json

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dados import CATEGORICAS, PASTA, atributos, carregar, separar


def montar_modelo(regressor):
    cat = Pipeline([
        ("faltantes", SimpleImputer(strategy="most_frequent")),
        ("dummies", OneHotEncoder(handle_unknown="ignore")),
    ])
    num = Pipeline([
        ("faltantes", SimpleImputer(strategy="median", add_indicator=True)),
        ("escala", StandardScaler()),
    ])
    preparo = ColumnTransformer([
        ("cat", cat, CATEGORICAS),
        ("num", num, ["music_voice_ratio", "log_duracao"]),
    ])
    return Pipeline([("preparo", preparo), ("regressao", regressor)])


def blocos_temporais(treino):
    """Folds crescentes por dia; nenhum dia aparece em treino e validação."""
    dias = np.sort(treino.date.unique())
    for dias_treino, dias_validacao in TimeSeriesSplit(n_splits=5).split(dias):
        it = np.flatnonzero(treino.date.isin(dias[dias_treino]))
        iv = np.flatnonzero(treino.date.isin(dias[dias_validacao]))
        yield it, iv


def avaliar_cv(modelo, x, y, folds):
    resultados = cross_validate(
        modelo, x, y, cv=folds,
        scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
        n_jobs=1,
    )
    return {
        nome: {"media": round(float(valores.mean()), 3),
               "desvio": round(float(valores.std(ddof=1)), 3)}
        for nome, valores in {
            "mae": -resultados["test_mae"],
            "rmse": -resultados["test_rmse"],
            "r2": resultados["test_r2"],
        }.items()
    }


def main():
    df = carregar()
    treino, teste = separar(df)
    x_treino, x_teste = atributos(treino), atributos(teste)
    y_treino, y_teste = treino.klike_score, teste.klike_score
    folds = list(blocos_temporais(treino))

    candidatos = {
        "mediana": DummyRegressor(strategy="median"),
        "ridge": montar_modelo(Ridge(alpha=1.0)),
        "floresta": montar_modelo(RandomForestRegressor(
            n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1,
        )),
    }
    cv = {nome: avaliar_cv(m, x_treino, y_treino, folds)
          for nome, m in candidatos.items()}
    vencedor = min(cv, key=lambda nome: cv[nome]["mae"]["media"])
    modelo = candidatos[vencedor].fit(x_treino, y_treino)

    # Só agora abrimos o período final. Não usamos esse erro para escolher modelo.
    pred = modelo.predict(x_teste)
    teste_metricas = {
        "mae": round(float(mean_absolute_error(y_teste, pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_teste, pred))), 3),
        "r2": round(float(r2_score(y_teste, pred)), 3),
    }
    erro_plataforma = {
        plataforma: {"n": int(mask.sum()),
                     "mae": round(float(mean_absolute_error(y_teste[mask], pred[mask])), 3)}
        for plataforma in teste.platform.unique()
        for mask in [teste.platform == plataforma]
    }
    importancia = permutation_importance(
        modelo, x_teste, y_teste, scoring="neg_mean_absolute_error",
        n_repeats=30, random_state=42, n_jobs=1,
    )
    ranking = sorted([
        {"atributo": coluna, "aumento_mae": round(float(media), 3),
         "dp": round(float(dp), 3)}
        for coluna, media, dp in zip(x_teste.columns,
                                     importancia.importances_mean,
                                     importancia.importances_std)
    ], key=lambda linha: linha["aumento_mae"], reverse=True)

    pasta = PASTA / "resultados"
    pasta.mkdir(exist_ok=True)
    (PASTA / "artefatos").mkdir(exist_ok=True)
    joblib.dump(modelo, PASTA / "artefatos" / "modelo.joblib")
    resumo = {
        "cenario": "criativo antes da veiculação",
        "corte": str(teste.date.min().date()),
        "n_treino": len(treino), "n_teste": len(teste),
        "cv_cinco_blocos_temporais": cv,
        "modelo_escolhido": vencedor,
        "teste_final": teste_metricas,
        "erro_teste_por_plataforma": erro_plataforma,
        "importancia_permutacao_teste_delta_mae": ranking,
    }
    (pasta / "modelo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Corte:", resumo["corte"], "| treino:", len(treino), "| teste:", len(teste))
    for nome, m in cv.items():
        print(f"{nome:9s} | MAE CV {m['mae']['media']:.2f} ± {m['mae']['desvio']:.2f} pontos")
    print(f"Escolhido: {vencedor} | teste: MAE {teste_metricas['mae']:.2f}, "
          f"RMSE {teste_metricas['rmse']:.2f}, R² {teste_metricas['r2']:.3f}")
    print("Importância (aumento do MAE ao embaralhar cada atributo):")
    for item in ranking[:6]:
        print(f"  {item['atributo']}: {item['aumento_mae']:.2f}")


if __name__ == "__main__":
    main()

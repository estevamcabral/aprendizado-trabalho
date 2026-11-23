
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score
)

# =============================================================================
# A. DADOS
# =============================================================================

tabela = pd.read_csv("saida/dados_tratados.csv", index_col=0)

# Utilizar somente as 23 atividades, assim como no Ward
X = tabela[[c for c in tabela.columns if c.startswith("Atividade")]]

# =============================================================================
# B. ESCOLHA DO NUMERO DE GRUPOS
#    Testar k de 2 a 10 utilizando inercia e silhueta
# =============================================================================

resultados = []

for k in range(2, 11):

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    grupos_k = modelo.fit_predict(X)

    inercia = modelo.inertia_
    silhueta = silhouette_score(X, grupos_k)

    davies = davies_bouldin_score(X, grupos_k)

    resultados.append({
        "k": k,
        "inercia": inercia,
        "silhueta": silhueta,
        "davies_bouldin": davies
    })

resultados = pd.DataFrame(resultados).set_index("k")

print("Teste de k = 2 a 10:")
print(resultados.round(3).to_string())

# =============================================================================
# C. GRAFICOS PARA ESCOLHA DE K
# =============================================================================

fig, eixos = plt.subplots(1, 3, figsize=(16, 4))

eixos[0].plot(resultados.index, resultados["inercia"], marker="o")
eixos[0].set_title("Cotovelo (inércia)")
eixos[0].set_xlabel("Número de grupos (k)")
eixos[0].set_ylabel("Inércia")

eixos[1].plot(resultados.index, resultados["silhueta"], marker="o")
eixos[1].set_title("Silhueta média")
eixos[1].set_xlabel("Número de grupos (k)")
eixos[1].set_ylabel("Silhueta")

eixos[2].plot(resultados.index, resultados["davies_bouldin"], marker="o")
eixos[2].set_title("Davies-Bouldin")
eixos[2].set_xlabel("Número de grupos (k)")
eixos[2].set_ylabel("Índice")

plt.tight_layout()
plt.savefig("saida/kmeans_escolha_k.png", dpi=150)
plt.close()

# =============================================================================
# D. AGRUPAMENTO FINAL
#    Inicialmente, utilizar 7 grupos para comparar com Ward.
# =============================================================================

modelo = KMeans(
    n_clusters=7,
    random_state=42,
    n_init=20
)

grupos = modelo.fit_predict(X) + 1

pd.Series(
    grupos,
    index=X.index,
    name="grupo"
).to_csv("saida/kmeans_grupos.csv")

print("\nIsolados em cada grupo:")
print(pd.Series(grupos).value_counts().sort_index().to_string())

# =============================================================================
# E. METRICAS DE AVALIACAO
# =============================================================================

print("\nAvaliação do agrupamento K-Means (7 grupos):")

print(
    "Silhueta:",
    round(silhouette_score(X, grupos), 3)
)

print(
    "Davies-Bouldin:",
    round(davies_bouldin_score(X, grupos), 3)
)

# Comparação com GRAM
colunas_gram = tabela[
    ["GRAM_Levedura", "GRAM_Negativo", "GRAM_Positivo"]
]

gram = colunas_gram.idxmax(axis=1).where(
    colunas_gram.sum(axis=1) == 1
)

tem_gram = gram.notna().values

print(
    "ARI com o GRAM:",
    round(
        adjusted_rand_score(
            gram[tem_gram],
            grupos[tem_gram]
        ),
        3
    )
)

# Comparação com o agrupamento hierárquico
grupos_ward = pd.read_csv(
    "saida/hierarquico_grupos.csv",
    index_col=0
)["grupo"]

grupos_kmeans = pd.Series(grupos, index=X.index)

print(
    "ARI entre K-Means e Ward:",
    round(
        adjusted_rand_score(
            grupos_ward.loc[X.index],
            grupos_kmeans
        ),
        3
    )
)

# =============================================================================
# F. ESTABILIDADE
#    Reamostrar 80% dos isolados e comparar com a partição original.
# =============================================================================

sorteio = np.random.default_rng(42)
comparacoes = []

for _ in range(20):

    amostra = sorteio.choice(
        len(X),
        size=int(0.8 * len(X)),
        replace=False
    )

    X_amostra = X.iloc[amostra]

    modelo_amostra = KMeans(
        n_clusters=7,
        random_state=42,
        n_init=20
    )

    grupos_amostra = modelo_amostra.fit_predict(X_amostra)

    comparacoes.append(
        adjusted_rand_score(
            grupos[amostra],
            grupos_amostra
        )
    )

print(
    "Estabilidade (ARI médio em 20 sorteios de 80%):",
    round(np.mean(comparacoes), 3)
)

# =============================================================================
# G. CARACTERIZACAO DOS GRUPOS
# =============================================================================

print("\nCaracterização dos grupos:")

# Tamanho
print("\nG1. Tamanho de cada grupo:")

print(
    pd.Series(grupos)
    .value_counts()
    .sort_index()
    .to_string()
)

# Perfil médio das atividades
perfil = X.groupby(grupos).mean().T.round(1)

print("\nG2. Perfil médio das atividades:")
print(perfil.to_string())

print("\nAs 3 atividades mais altas de cada grupo:")

for g in perfil.columns:

    mais_altas = perfil[g].sort_values(
        ascending=False
    ).head(3)

    print(
        f"Grupo {g}: " +
        ", ".join(
            f"{a} ({v:+.1f})"
            for a, v in mais_altas.items()
        )
    )

# Moda da morfologia
morfologia = [
    "Cores", "Borda", "Textura",
    "Forma", "Brilho", "Muco", "Relevo"
]

moda = {}

for c in morfologia:

    proporcao = tabela.filter(
        regex=f"^{c}_"
    ).groupby(grupos).mean()

    codigo = proporcao.idxmax(axis=1).str.replace(
        c + "_", "", regex=False
    )

    moda[c] = (
        codigo + " (" +
        (proporcao.max(axis=1) * 100)
        .round()
        .astype(int)
        .astype(str) + "%)"
    )

print("\nG3. Moda da morfologia:")
print(
    pd.DataFrame(moda)
    .rename_axis("grupo")
    .to_string()
)

print("\nAgrupamento K-Means concluído!")

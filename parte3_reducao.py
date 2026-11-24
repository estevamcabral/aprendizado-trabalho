# PARTE 3 - REDUCAO DE DIMENSIONALIDADE (PCA x t-SNE) E COMPARACAO COM O ESPACO ORIGINAL
# Rodar dentro da pasta do projeto, depois do parte1.py, parte2_hierarquico.py e parte2_kmeans.py:
#   python parte3_reducao.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline

SEMENTE = 42
VIZINHOS = 10   # vizinhos usados no trustworthiness e na estabilidade do t-SNE

# =============================================================================
# A. DADOS: as 23 atividades ja padronizadas (o "estado inicial") e os grupos da Parte 2
# =============================================================================
tabela = pd.read_csv("saida/dados_tratados.csv", index_col=0)
X = tabela[[c for c in tabela.columns if c.startswith("Atividade")]]
particoes = {"Ward": pd.read_csv("saida/hierarquico_grupos.csv", index_col=0)["grupo"].loc[X.index].values,
             "K-Means": pd.read_csv("saida/kmeans_grupos.csv", index_col=0)["grupo"].loc[X.index].values}
k_parte2 = {nome: len(np.unique(g)) for nome, g in particoes.items()}   # k escolhido na Parte 2 (Ward 7, K-Means 5)
print("k de cada algoritmo na Parte 2:", k_parte2)
colunas_gram = tabela[["GRAM_Levedura", "GRAM_Negativo", "GRAM_Positivo"]]
gram = colunas_gram.idxmax(axis=1).str.replace("GRAM_", "").where(colunas_gram.sum(axis=1) == 1)
tem_gram = gram.notna().values

# =============================================================================
# B. PCA (LINEAR): quanta informacao cada eixo carrega
# =============================================================================
pca = PCA(random_state=SEMENTE).fit(X)
variancia = pd.DataFrame({"variancia": pca.explained_variance_ratio_,
                          "acumulada": np.cumsum(pca.explained_variance_ratio_)},
                         index=range(1, X.shape[1] + 1)).rename_axis("componente")
d_escolhido = int((variancia["acumulada"] >= 0.80).idxmax())   # menor d que guarda 80% da variancia
print("B1. Variancia explicada pelo PCA:")
print(variancia.round(3).to_string())
print(f"\n  {d_escolhido} componentes guardam {variancia.loc[d_escolhido, 'acumulada']:.1%} da variancia "
      f"(PC1+PC2 = {variancia.loc[2, 'acumulada']:.1%})")

# B2. Erro de reconstrucao MSE(d): projeta em d dimensoes, reconstroi e compara com o original
mse = pd.Series({d: np.mean((X.values - (p := PCA(n_components=d).fit(X)).inverse_transform(p.transform(X))) ** 2)
                 for d in range(1, X.shape[1] + 1)}, name="MSE")
print("\nB2. Erro de reconstrucao (MSE) por numero de componentes:")
print(mse.round(3).to_string())

fig, eixos = plt.subplots(1, 2, figsize=(13, 4))
eixos[0].bar(variancia.index, variancia["variancia"], label="por componente")
eixos[0].plot(variancia.index, variancia["acumulada"], marker="o", color="C1", label="acumulada")
eixos[0].axhline(0.80, ls="--", color="gray")
eixos[0].axvline(d_escolhido, ls=":", color="gray")
eixos[0].set_title("PCA - variância explicada")
eixos[0].set_xlabel("componente")
eixos[0].legend()
eixos[1].plot(mse.index, mse.values, marker="o")
eixos[1].axvline(d_escolhido, ls=":", color="gray")
eixos[1].set_title("PCA - erro de reconstrução MSE(d)")
eixos[1].set_xlabel("número de componentes (d)")
plt.tight_layout()
plt.savefig("saida/reducao_pca_variancia_mse.png", dpi=150)
plt.close()

# B3. Pesos (loadings): quais atividades formam cada eixo -> e o que torna o PCA interpretavel
pesos = pd.DataFrame(pca.components_[:3].T, index=X.columns, columns=["PC1", "PC2", "PC3"]).round(2)
print("\nB3. Pesos das atividades nos 3 primeiros componentes:")
print(pesos.to_string())
for pc in pesos.columns:
    maiores = pesos[pc].abs().sort_values(ascending=False).head(5).index
    print(f"  {pc}: " + ", ".join(f"{a} ({pesos.loc[a, pc]:+.2f})" for a in maiores))

# =============================================================================
# C. t-SNE (NAO LINEAR) e projecoes 2D lado a lado
# =============================================================================
pca2 = PCA(n_components=2).fit_transform(X)
pca_d = PCA(n_components=d_escolhido).fit_transform(X)
tsne2 = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEMENTE).fit_transform(X)
pd.DataFrame(np.column_stack([pca2, tsne2]), index=X.index,
             columns=["PCA_1", "PCA_2", "tSNE_1", "tSNE_2"]).to_csv("saida/reducao_coordenadas.csv")

cores_gram = gram.map({"Negativo": 0, "Positivo": 1, "Levedura": 2}).fillna(3).values
pinturas = {**{f"grupos {n}": g for n, g in particoes.items()}, "GRAM": cores_gram}
fig, eixos = plt.subplots(len(pinturas), 2, figsize=(13, 5.5 * len(pinturas)))
for linha, (nome, cor) in enumerate(pinturas.items()):
    for coluna, (metodo, Z) in enumerate({"PCA": pca2, "t-SNE (perplexity=30)": tsne2}.items()):
        ax = eixos[linha, coluna]
        pontos = ax.scatter(Z[:, 0], Z[:, 1], c=cor, cmap="tab10", vmin=0, vmax=9, s=12)
        ax.set_title(f"{metodo} - cor = {nome}")
        rotulos = ["Negativo", "Positivo", "Levedura", "sem GRAM"] if nome == "GRAM" else None
        ax.legend(pontos.legend_elements()[0], rotulos or [f"grupo {g}" for g in np.unique(cor)],
                  fontsize=7, loc="best")
eixos[0, 0].set_xlabel(f"PC1 ({variancia.loc[1, 'variancia']:.0%})")
eixos[0, 0].set_ylabel(f"PC2 ({variancia.loc[2, 'variancia']:.0%})")
plt.tight_layout()
plt.savefig("saida/reducao_projecoes_2d.png", dpi=150)
plt.close()

# =============================================================================
# D. METRICAS DA PROJECAO
# =============================================================================
# D1. Trustworthiness: vizinhos na figura tambem eram vizinhos nas 23 atividades? (0 a 1)
print(f"\nD1. Trustworthiness ({VIZINHOS} vizinhos, 1 = vizinhanca preservada):")
for nome, Z in {"PCA 2D": pca2, f"PCA {d_escolhido}D": pca_d, "t-SNE 2D": tsne2}.items():
    print(f"  {nome}: {trustworthiness(X, Z, n_neighbors=VIZINHOS):.3f}")


def vizinhos(Z):
    return NearestNeighbors(n_neighbors=VIZINHOS + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]


def sobreposicao(A, B):   # fracao media de vizinhos em comum entre duas figuras (1 = mesma vizinhanca)
    return np.mean([len(set(a) & set(b)) / VIZINHOS for a, b in zip(vizinhos(A), vizinhos(B))])


# D2. Estabilidade do t-SNE: sementes diferentes e perplexity diferente
print("\nD2. Estabilidade do t-SNE:")
sementes = [TSNE(perplexity=30, init="random", random_state=s).fit_transform(X) for s in range(5)]
pares = [sobreposicao(sementes[i], sementes[j]) for i in range(5) for j in range(i + 1, 5)]
print(f"  5 sementes (perplexity=30, init aleatorio): vizinhos em comum entre figuras = {np.mean(pares):.3f}")
perplexidades = [5, 15, 30, 50]
figuras = {p: TSNE(perplexity=p, init="pca", random_state=SEMENTE).fit_transform(X) for p in perplexidades}
for p, Z in figuras.items():
    print(f"  perplexity={p:>2}: trustworthiness = {trustworthiness(X, Z, n_neighbors=VIZINHOS):.3f}, "
          f"vizinhos em comum com perplexity=30 = {sobreposicao(Z, tsne2):.3f}")
print("  (PCA e deterministico: mesma figura com qualquer semente)")

fig, eixos = plt.subplots(1, len(perplexidades), figsize=(5 * len(perplexidades), 4.5))
for ax, (p, Z) in zip(eixos, figuras.items()):
    ax.scatter(Z[:, 0], Z[:, 1], c=particoes["Ward"], cmap="tab10", vmin=0, vmax=9, s=8)
    ax.set_title(f"t-SNE perplexity={p} (cor = Ward)")
plt.tight_layout()
plt.savefig("saida/reducao_tsne_perplexity.png", dpi=150)
plt.close()

# =============================================================================
# E. ESTADO INICIAL x ESPACO REDUZIDO
# =============================================================================
# E1. Utilidade da reducao: um classificador (kNN, 5 vizinhos) tenta prever o GRAM a partir de cada espaco.
#     Validacao cruzada estratificada 5-fold; o PCA e ajustado dentro de cada dobra (sem vazamento).
#     O t-SNE nao projeta pontos novos, entao e calculado com todos os isolados (resultado otimista).
print("\nE1. Utilidade: acuracia balanceada do kNN prevendo o GRAM (5-fold):")
dobras = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
knn = KNeighborsClassifier(n_neighbors=5)
espacos_clf = {"Original (23)": (knn, X.values),
               "PCA 2D": (make_pipeline(PCA(n_components=2), knn), X.values),
               f"PCA {d_escolhido}D": (make_pipeline(PCA(n_components=d_escolhido), knn), X.values),
               "t-SNE 2D": (knn, tsne2)}
utilidade = pd.Series({nome: cross_val_score(m, Z[tem_gram], gram[tem_gram], cv=dobras,
                                             scoring="balanced_accuracy").mean()
                       for nome, (m, Z) in espacos_clf.items()}, name="acuracia_balanceada")
print(utilidade.round(3).to_string())
print(f"  (chute pela classe mais comum = {1 / 3:.3f} de acuracia balanceada)")

# E2. Agrupamento no espaco reduzido x no espaco original (mesmos algoritmos e mesmo k da Parte 2)
#     Silhueta "propria" = medida no espaco onde o agrupamento foi feito;
#     silhueta "original" = os mesmos grupos medidos nas 23 atividades (comparacao justa entre espacos).
algoritmos = {"Ward": lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward"),
              "K-Means": lambda k: KMeans(n_clusters=k, random_state=SEMENTE, n_init=20)}
espacos = {"Original (23)": X.values, f"PCA {d_escolhido}D": pca_d, "PCA 2D": pca2, "t-SNE 2D": tsne2}
sorteio = np.random.default_rng(SEMENTE)
amostras = [sorteio.choice(len(X), size=int(0.8 * len(X)), replace=False) for _ in range(20)]
linhas = []
for nome_alg, criar in algoritmos.items():
    for nome_esp, Z in espacos.items():
        k = k_parte2[nome_alg]
        grupos = criar(k).fit_predict(Z) + 1
        estabilidade = np.mean([adjusted_rand_score(grupos[a], criar(k).fit_predict(Z[a])) for a in amostras])
        linhas.append({"algoritmo": f"{nome_alg} (k={k})", "espaco": nome_esp,
                       "silhueta_propria": silhouette_score(Z, grupos),
                       "silhueta_original": silhouette_score(X, grupos),
                       "davies_bouldin": davies_bouldin_score(Z, grupos),
                       "ARI_GRAM": adjusted_rand_score(gram[tem_gram], grupos[tem_gram]),
                       "ARI_vs_original": adjusted_rand_score(particoes[nome_alg], grupos),
                       "estabilidade": estabilidade})
comparacao = pd.DataFrame(linhas).set_index(["algoritmo", "espaco"])
print("\nE2. Agrupamento (k da Parte 2) no espaco original x reduzido:")
print(comparacao.round(3).to_string())
print("  ARI_vs_original = concordancia com os grupos da Parte 2 (1 = mesmos grupos)")
comparacao.round(4).to_csv("saida/reducao_comparacao_agrupamento.csv")

# E3. Mesmo teste de k = 2 a 10 da Parte 2, agora no espaco PCA reduzido, para comparar com o original
print(f"\nE3. Silhueta por k: original x PCA {d_escolhido}D x t-SNE 2D:")
silhuetas = pd.DataFrame({(alg, esp): [silhouette_score(Z, criar(k).fit_predict(Z)) for k in range(2, 11)]
                          for alg, criar in algoritmos.items() for esp, Z in espacos.items() if esp != "PCA 2D"},
                         index=pd.Index(range(2, 11), name="k"))
print(silhuetas.round(3).to_string())

fig, eixos = plt.subplots(1, 2, figsize=(13, 4))
for ax, alg in zip(eixos, algoritmos):
    for esp in silhuetas[alg].columns:
        ax.plot(silhuetas.index, silhuetas[(alg, esp)], marker="o", label=esp)
    ax.axvline(k_parte2[alg], ls=":", color="gray")
    ax.set_title(f"{alg} - silhueta média por k")
    ax.set_xlabel("número de grupos (k)")
    ax.legend()
plt.tight_layout()
plt.savefig("saida/reducao_silhueta_por_k.png", dpi=150)
plt.close()

# PARTE 4 - SENSIBILIDADE AO PRE-PROCESSAMENTO
# Refaz o pre-processamento das atividades mudando uma decisao por vez e repete o agrupamento (Ward k=7 e K-Means k=5)
# e o PCA. Rodar dentro da pasta do projeto, depois das partes 1 e 2: python parte4_sensibilidade.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SEMENTE = 42
atividades = [f"Atividade {i:02d}" for i in range(1, 24)]
cauda_longa = ["Atividade 02", "Atividade 14", "Atividade 15", "Atividade 23"]
binarias = ["Atividade 01", "Atividade 03", "Atividade 04", "Atividade 16"]

# =============================================================================
# A. MESMA LIMPEZA DA PARTE 1 (so a parte das atividades)
# =============================================================================
dados = pd.read_csv("Dados_REG.csv", sep="\t", dtype=str, keep_default_na=False)
dados = dados.apply(lambda c: c.str.strip()).replace({"NA": np.nan, "": np.nan, "#NÚM!": np.nan})
repetido = dados["n°"].duplicated(keep=False)
dados.loc[repetido, "n°"] = dados.loc[repetido, "n°"] + ["_a", "_b"]
dados = dados.set_index("n°")
limpas = dados[atividades].replace({"Positivo": "1", "Negativo": "0"}).apply(pd.to_numeric).clip(lower=0)
limpas = limpas.fillna(limpas.median())


def preparar(log=True, padronizar_binarias=True):
    ativ = limpas.copy()
    if log:
        ativ[cauda_longa] = np.log1p(ativ[cauda_longa])
    colunas = atividades if padronizar_binarias else [c for c in atividades if c not in binarias]
    ativ[colunas] = StandardScaler().fit_transform(ativ[colunas])
    return ativ


variantes = {"Base (log + padronização)": preparar(),
             "Sem log": preparar(log=False),
             "Binárias sem padronizar": preparar(padronizar_binarias=False)}

tabela = pd.read_csv("saida/dados_tratados.csv", index_col=0)
assert np.allclose(variantes["Base (log + padronização)"].loc[tabela.index].values, tabela[atividades].values), \
    "a versao base deveria ser igual ao saida/dados_tratados.csv"
colunas_gram = tabela[["GRAM_Levedura", "GRAM_Negativo", "GRAM_Positivo"]]
gram = colunas_gram.idxmax(axis=1).where(colunas_gram.sum(axis=1) == 1)
tem_gram = gram.notna().values

# =============================================================================
# B. AGRUPAMENTO E PCA EM CADA VERSAO
# =============================================================================
algoritmos = {"Ward": (7, lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward")),
              "K-Means": (5, lambda k: KMeans(n_clusters=k, random_state=SEMENTE, n_init=20))}
sorteio = np.random.default_rng(SEMENTE)
amostras = [sorteio.choice(len(tabela), size=int(0.8 * len(tabela)), replace=False) for _ in range(20)]
base = {alg: criar(k).fit_predict(variantes["Base (log + padronização)"].loc[tabela.index])
        for alg, (k, criar) in algoritmos.items()}

linhas, silhuetas = [], {}
for nome, ativ in variantes.items():
    X = ativ.loc[tabela.index].values
    variancia = np.cumsum(PCA().fit(X).explained_variance_ratio_)
    for alg, (k, criar) in algoritmos.items():
        grupos = criar(k).fit_predict(X)
        silhuetas[(alg, nome)] = [silhouette_score(X, criar(kk).fit_predict(X)) for kk in range(2, 11)]
        linhas.append({"algoritmo": f"{alg} (k={k})", "versao": nome,
                       "silhueta": silhouette_score(X, grupos),
                       "davies_bouldin": davies_bouldin_score(X, grupos),
                       "ARI_GRAM": adjusted_rand_score(gram[tem_gram], grupos[tem_gram]),
                       "estabilidade": np.mean([adjusted_rand_score(grupos[a], criar(k).fit_predict(X[a]))
                                                for a in amostras]),
                       "ARI_vs_base": adjusted_rand_score(base[alg], grupos),
                       "melhor_k_silhueta": int(np.argmax(silhuetas[(alg, nome)]) + 2),
                       "tamanhos": "/".join(map(str, sorted(np.bincount(grupos), reverse=True))),
                       "PCA_PC1+PC2": variancia[1],
                       "PCA_d_80%": int(np.argmax(variancia >= 0.80) + 1)})

resultado = pd.DataFrame(linhas).set_index(["algoritmo", "versao"])
pd.set_option("display.width", 250)
print("Sensibilidade ao pre-processamento (mesmo k da Parte 2):")
print(resultado.round(3).to_string())
print("\nARI_vs_base = concordancia com os grupos da versao base (1 = mesmos grupos)")
print("tamanhos = numero de isolados em cada grupo, do maior para o menor")
resultado.round(4).to_csv("saida/sensibilidade_resultados.csv")

fig, eixos = plt.subplots(1, 2, figsize=(13, 4))
for ax, (alg, (k, _)) in zip(eixos, algoritmos.items()):
    for nome in variantes:
        ax.plot(range(2, 11), silhuetas[(alg, nome)], marker="o", label=nome)
    ax.axvline(k, ls=":", color="gray")
    ax.set_title(f"{alg} - silhueta média por k em cada pré-processamento")
    ax.set_xlabel("número de grupos (k)")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("saida/sensibilidade_silhueta_por_k.png", dpi=150)
plt.close()

# PARTE 2 - AGRUPAMENTO HIERARQUICO (WARD)
# Rodar dentro da pasta do projeto, depois do parte1.py: python parte2_hierarquico.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

# =============================================================================
# A. DADOS
# =============================================================================
tabela = pd.read_csv("saida/dados_tratados.csv", index_col=0)
X = tabela[[c for c in tabela.columns if c.startswith("Atividade")]]   # as 23 atividades

# =============================================================================
# B. ESCOLHA DO NUMERO DE GRUPOS
#    roda o ALGORITMO com k = 2 a 10 e mede cada resultado com as METRICAS inercia e silhueta
# =============================================================================
resultados = []
for k in range(2, 11):
    grupos_k = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)                     # algoritmo
    inercia = sum(((X[grupos_k == g] - X[grupos_k == g].mean()) ** 2).values.sum() for g in range(k))  # metrica
    resultados.append({"k": k, "inercia": inercia, "silhueta": silhouette_score(X, grupos_k)})          # metrica
resultados = pd.DataFrame(resultados).set_index("k")
print("Teste de k = 2 a 10:")
print(resultados.round(3).to_string())

fig, eixos = plt.subplots(1, 2, figsize=(12, 4))
eixos[0].plot(resultados.index, resultados["inercia"], marker="o")
eixos[0].set_title("Cotovelo (inércia)")
eixos[0].set_xlabel("número de grupos (k)")
eixos[1].plot(resultados.index, resultados["silhueta"], marker="o")
eixos[1].set_title("Silhueta média")
eixos[1].set_xlabel("número de grupos (k)")
eixos[1].margins(y=0.15)
for k, s in resultados["silhueta"].items():
    eixos[1].annotate(f"{s:.3f}", (k, s), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("saida/hierarquico_escolha_k.png")

# =============================================================================
# C. ALGORITMO: agrupamento final com 7 grupos (7 = maior silhueta no teste da parte B)
# =============================================================================
modelo = AgglomerativeClustering(n_clusters=7, linkage="ward")
grupos = modelo.fit_predict(X) + 1

pd.Series(grupos, index=X.index, name="grupo").to_csv("saida/hierarquico_grupos.csv")
print("\nIsolados em cada grupo:")
print(pd.Series(grupos).value_counts().sort_index().to_string())

# =============================================================================
# D. METRICAS: avaliacao do agrupamento final
# =============================================================================
print("\nAvaliacao do agrupamento final (7 grupos):")

# D1. Metricas internas: usam so os dados e os grupos
print("  Silhueta:", round(silhouette_score(X, grupos), 3), " (de -1 a 1, quanto maior melhor)")
print("  Davies-Bouldin:", round(davies_bouldin_score(X, grupos), 3), " (quanto menor melhor, 0 = perfeito)")

# D2. Metrica externa: compara os grupos com o GRAM (os 2 isolados sem GRAM ficam de fora)
colunas_gram = tabela[["GRAM_Levedura", "GRAM_Negativo", "GRAM_Positivo"]]
gram = colunas_gram.idxmax(axis=1).where(colunas_gram.sum(axis=1) == 1)
tem_gram = gram.notna().values
print("  ARI com o GRAM:", round(adjusted_rand_score(gram[tem_gram], grupos[tem_gram]), 3),
      " (1 = grupos iguais ao GRAM, 0 = sem relacao)")

# D3. Estabilidade: 20 vezes, sorteia 80% dos isolados, roda o algoritmo de novo e compara com o final
sorteio = np.random.default_rng(42)
comparacoes = []
for _ in range(20):
    amostra = sorteio.choice(len(X), size=int(0.8 * len(X)), replace=False)
    grupos_amostra = AgglomerativeClustering(n_clusters=7, linkage="ward").fit_predict(X.iloc[amostra])
    comparacoes.append(adjusted_rand_score(grupos[amostra], grupos_amostra))
print("  Estabilidade (ARI medio em 20 sorteios de 80%):", round(np.mean(comparacoes), 3),
      " (1 = grupos sempre iguais)")

# =============================================================================
# E. CARACTERIZACAO: o que os isolados de cada grupo tem em comum
# =============================================================================
print("\nCaracterizacao dos grupos:")

# E1. Tamanho: quantos isolados tem cada grupo
print("\nE1. Tamanho de cada grupo:")
print(pd.Series(grupos).value_counts().sort_index().to_string())

# E2. Perfil medio das atividades: media de cada atividade dentro do grupo
#     (valores padronizados: 0 = media de todos os isolados, positivo = acima, negativo = abaixo)
perfil = X.groupby(grupos).mean().T.round(1)
print("\nE2. Perfil medio das atividades (0 = media geral):")
print(perfil.to_string())
print("\nAs 3 atividades mais altas de cada grupo:")
for g in perfil.columns:
    mais_altas = perfil[g].sort_values(ascending=False).head(3)
    print(f"  grupo {g}: " + ", ".join(f"{a} ({v:+.1f})" for a, v in mais_altas.items()))

# E3. Moda da morfologia: o codigo mais comum de cada caracteristica dentro do grupo
morfologia = ["Cores", "Borda", "Textura", "Forma", "Brilho", "Muco", "Relevo"]
moda = {}
for c in morfologia:
    proporcao = tabela.filter(regex=f"^{c}_").groupby(grupos).mean()   # % dos isolados com cada codigo
    codigo = proporcao.idxmax(axis=1).str.replace(c + "_", "")
    moda[c] = codigo + " (" + (proporcao.max(axis=1) * 100).round().astype(int).astype(str) + "%)"
print("\nE3. Moda da morfologia (codigo mais comum e % dos isolados do grupo com ele):")
print(pd.DataFrame(moda).rename_axis("grupo").to_string())

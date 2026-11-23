import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

# =============================================================================
# CONFIGURACAO
# =============================================================================

SEMENTE = 42      
N_INIT = 20       
FAIXA_K = range(2, 11)
N_SORTEIOS = 20   
FRACAO = 0.8      

SAIDA = Path("saida")
ARQ_DADOS = SAIDA / "dados_tratados.csv"
ARQ_WARD = SAIDA / "hierarquico_grupos.csv"
MORFOLOGIA = ["Cores", "Borda", "Textura", "Forma", "Brilho", "Muco", "Relevo"]


class Tee:
    """Escreve na tela e em arquivo ao mesmo tempo."""

    def __init__(self, *destinos):
        self.destinos = destinos

    def write(self, texto):
        for d in self.destinos:
            d.write(texto)

    def flush(self):
        for d in self.destinos:
            d.flush()


def titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def ajustar(X, k):
    return KMeans(n_clusters=k, random_state=SEMENTE, n_init=N_INIT).fit(X)


# =============================================================================
# A. DADOS
# =============================================================================

def carregar_dados():
    """Le a tabela tratada e separa as 23 atividades (entrada do algoritmo).

    O GRAM e a morfologia ficam de fora do agrupamento: o GRAM e usado como
    referencia externa e a morfologia serve para descrever os grupos.
    """
    tabela = pd.read_csv(ARQ_DADOS, index_col=0)
    X = tabela[[c for c in tabela.columns if c.startswith("Atividade")]]

    colunas_gram = tabela[["GRAM_Levedura", "GRAM_Negativo", "GRAM_Positivo"]]
    gram = colunas_gram.idxmax(axis=1).where(colunas_gram.sum(axis=1) == 1)
    gram = gram.str.replace("GRAM_", "", regex=False)
    return tabela, X, gram


# =============================================================================
# B. ESCOLHA DE K
# =============================================================================

def escolher_k(X):
    linhas = []
    for k in FAIXA_K:
        modelo = ajustar(X, k)
        linhas.append({
            "k": k,
            "inercia": modelo.inertia_,
            "silhueta": silhouette_score(X, modelo.labels_),
            "davies_bouldin": davies_bouldin_score(X, modelo.labels_),
        })
    res = pd.DataFrame(linhas).set_index("k")

    titulo("ESCOLHA DE K (k = 2 a 10)")
    print(res.round(3).to_string())
    print(f"\nMaior silhueta:       k = {res['silhueta'].idxmax()} "
          f"({res['silhueta'].max():.3f})")
    print(f"Menor Davies-Bouldin: k = {res['davies_bouldin'].idxmin()} "
          f"({res['davies_bouldin'].min():.3f})")

    res.round(4).to_csv(SAIDA / "kmeans_escolha_k.csv")

    k_escolhido = int(res["silhueta"].idxmax())
    if k_escolhido != int(res["davies_bouldin"].idxmin()):
        print("\nATENÇÃO: silhueta e Davies-Bouldin indicam k diferentes; "
              f"usando k = {k_escolhido} (maior silhueta).")
    else:
        print(f"\nSilhueta e Davies-Bouldin concordam: k = {k_escolhido}.")

    fig, eixos = plt.subplots(1, 3, figsize=(16, 4))
    graficos = [
        ("inercia", "Cotovelo (inércia)", "Inércia"),
        ("silhueta", "Silhueta média (maior é melhor)", "Silhueta"),
        ("davies_bouldin", "Davies-Bouldin (menor é melhor)", "Índice"),
    ]
    for eixo, (coluna, tit, rotulo) in zip(eixos, graficos):
        eixo.plot(res.index, res[coluna], marker="o")
        eixo.set_title(tit)
        eixo.set_xlabel("Número de grupos (k)")
        eixo.set_ylabel(rotulo)
    plt.tight_layout()
    plt.savefig(SAIDA / "kmeans_escolha_k.png", dpi=150)
    plt.close()
    return k_escolhido


# =============================================================================
# C. AVALIACAO
# =============================================================================

def estabilidade(X, grupos, k):
    """Reamostra FRACAO dos isolados N_SORTEIOS vezes e refaz o K-Means.

    Devolve:
      - ARI medio entre cada reamostragem e a particao original (geral)
      - por grupo: fracao media dos isolados do grupo que continuam juntos
        (ficam todos no mesmo grupo da nova particao)
    """
    sorteio = np.random.default_rng(SEMENTE)
    aris = []
    juntos = {g: [] for g in range(1, k + 1)}

    for _ in range(N_SORTEIOS):
        amostra = sorteio.choice(len(X), size=int(FRACAO * len(X)), replace=False)
        novos = ajustar(X.iloc[amostra], k).labels_
        originais = grupos[amostra]

        aris.append(adjusted_rand_score(originais, novos))

        cruzada = pd.crosstab(originais, novos)
        for g in cruzada.index:
            juntos[g].append(cruzada.loc[g].max() / cruzada.loc[g].sum())

    return float(np.mean(aris)), {g: float(np.mean(v)) for g, v in juntos.items()}


def tabela_por_grupo(X, grupos, gram, estab_grupo):
    s = silhouette_samples(X, grupos)
    tab = pd.DataFrame({"grupo": grupos, "s": s}).groupby("grupo")["s"].agg(
        isolados="size",
        silhueta_media="mean",
        silhueta_negativa=lambda v: int((v < 0).sum()),
    )
    tab["estabilidade_%"] = [round(100 * estab_grupo[g]) for g in tab.index]

    gram_s = pd.Series(gram.values, index=X.index)
    predominante = []
    for g in tab.index:
        freq = gram_s[grupos == g].dropna().value_counts(normalize=True)
        predominante.append(f"{freq.iloc[0]:.0%} {freq.index[0]}")
    tab["GRAM_predominante"] = predominante
    return tab.round(3)


# =============================================================================
# D. CARACTERIZACAO
# =============================================================================

def caracterizar(tabela, X, grupos):
    perfil = X.groupby(grupos).mean().T
    perfil.columns = [f"Grupo {g}" for g in perfil.columns]

    print("\nPerfil médio das atividades (desvios-padrão; atividades 02, 14, 15 "
          "e 23 estão em log):")
    print(perfil.round(1).to_string())

    print("\nAtividades mais altas e mais baixas de cada grupo:")
    for g in perfil.columns:
        ordem = perfil[g].sort_values(ascending=False)
        altas = ", ".join(f"{a.replace('Atividade ', '')} ({v:+.1f})"
                          for a, v in ordem.head(3).items())
        baixas = ", ".join(f"{a.replace('Atividade ', '')} ({v:+.1f})"
                           for a, v in ordem.tail(3)[::-1].items())
        print(f"  {g}: altas -> {altas} | baixas -> {baixas}")

    moda_grupo, moda_geral = {}, {}
    for c in MORFOLOGIA:
        cols = tabela.filter(regex=f"^{c}_")
        prop = cols.groupby(grupos).mean()
        codigo = prop.idxmax(axis=1).str.replace(c + "_", "", regex=False)
        pct = (prop.max(axis=1) * 100).round().astype(int).astype(str)
        moda_grupo[c] = codigo + " (" + pct + "%)"

        geral = cols.mean()
        moda_geral[c] = (f"{geral.idxmax().replace(c + '_', '')} "
                         f"({geral.max() * 100:.0f}%)")

    print("\nModa da morfologia no conjunto inteiro:")
    print("  " + " | ".join(f"{c}: {v}" for c, v in moda_geral.items()))
    print("\nModa da morfologia por grupo:")
    print(pd.DataFrame(moda_grupo).rename_axis("grupo").to_string())


# =============================================================================
# PRINCIPAL
# =============================================================================

def main():
    SAIDA.mkdir(exist_ok=True)

    with open(SAIDA / "kmeans_resultados.txt", "w", encoding="utf-8") as arquivo:
        original = sys.stdout
        sys.stdout = Tee(original, arquivo)
        try:
            tabela, X, gram = carregar_dados()
            print(f"{X.shape[0]} isolados, {X.shape[1]} atividades "
                  f"(semente {SEMENTE}, n_init {N_INIT})")

            k = escolher_k(X)

            titulo(f"K-MEANS COM k = {k}")
            grupos = ajustar(X, k).labels_ + 1  
            pd.Series(grupos, index=X.index, name="grupo").to_csv(
                SAIDA / "kmeans_grupos.csv")

            print("\nTamanho dos grupos:")
            print(pd.Series(grupos).value_counts().sort_index().to_string())

            tem_gram = gram.notna().values
            ari_gram = adjusted_rand_score(gram[tem_gram], grupos[tem_gram])
            estab_geral, estab_grupo = estabilidade(X, grupos, k)

            print("\nMétricas globais:")
            print("  Silhueta:       ", round(silhouette_score(X, grupos), 3))
            print("  Davies-Bouldin: ", round(davies_bouldin_score(X, grupos), 3))
            print("  ARI com o GRAM: ", round(ari_gram, 3),
                  f"({tem_gram.sum()} isolados com GRAM conhecido)")
            print(f"  Estabilidade:    {estab_geral:.3f} "
                  f"(ARI médio em {N_SORTEIOS} sorteios de {FRACAO:.0%})")

            print("\nTabela por grupo:")
            tab = tabela_por_grupo(X, grupos, gram, estab_grupo)
            print(tab.to_string())
            tab.to_csv(SAIDA / "kmeans_tabela_grupos.csv")

            caracterizar(tabela, X, grupos)

            print("\nConcluído. Arquivos gerados em saida/:")
            print("  kmeans_resultados.txt, kmeans_escolha_k.csv/.png,")
            print("  kmeans_grupos.csv, kmeans_tabela_grupos.csv")
        finally:
            sys.stdout = original


if __name__ == "__main__":
    main()

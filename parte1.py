# PARTE 1 - PRE-PROCESSAMENTO
# Rodar dentro da pasta do projeto: python parte1.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

Path("saida").mkdir(exist_ok=True)
atividades = [f"Atividade {i:02d}" for i in range(1, 24)]
morfologia = ["Cor", "Cores", "Borda", "Textura", "Forma", "Brilho", "Muco", "Relevo"]

# BLOCO 1 - EXPLORACAO: olhar os dados sem mudar nada
dados = pd.read_csv("Dados_REG.csv", sep="\t", dtype=str, keep_default_na=False)        # 1.1 ler a planilha
numeros = dados[atividades].replace({"Positivo": "1", "Negativo": "0"}).apply(pd.to_numeric, errors="coerce")

numeros.hist(bins=30, figsize=(20, 14))                                                  # 1.2 Grafico 1: histograma de cada atividade
plt.savefig("saida/grafico1_histogramas.png")
plt.figure(figsize=(14, 11))                                                             # 1.3 Grafico 2: quais atividades andam juntas
sns.heatmap(numeros.corr(method="spearman"), cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".1f")
plt.savefig("saida/grafico2_correlacao.png")

print(dados.nunique())                                                                   # 1.4 quantos valores cada coluna tem (tipo da coluna)
for coluna in ["GRAM"] + morfologia:
    print(dados[coluna].value_counts())                                                  # 1.5 contagem: GRAM errado e codigos estranhos
print(dados.isin(["NA", "", "#NÚM!"]).sum())                                             # 1.6 valores faltando
print(dados[dados["n°"].duplicated(keep=False)])                                         # 1.7 codigo repetido (1551)
print(pd.crosstab(dados["Cor"], dados["Cores"].str.contains("&")))                       # 1.8 Cor repete Cores
print(numeros.agg(["min", "median", "max"]).T)                                           # 1.9 valores negativos e muito altos
validos = dados[dados["GRAM"].isin(["Negativo", "Positivo", "Levedura"])]
for coluna in morfologia:
    print(pd.crosstab(validos["GRAM"], validos[coluna], normalize="index"))              # 1.10 morfologia x GRAM

# BLOCO 2 - LIMPEZA: corrigir os erros
dados = dados.apply(lambda c: c.str.strip()).replace({"NA": np.nan, "": np.nan, "#NÚM!": np.nan})   # 2.1 vazio de um jeito so
dados["GRAM"] = dados["GRAM"].replace({"POSITIVO": "Positivo", "Positiva": "Positivo"})            # 2.2 GRAM corrigido
dados.loc[~dados["GRAM"].isin(["Negativo", "Positivo", "Levedura"]), "GRAM"] = np.nan
repetido = dados["n°"].duplicated(keep=False)
dados.loc[repetido, "n°"] = dados.loc[repetido, "n°"] + ["_a", "_b"]                                 # 2.3 1551 vira 1551_a e 1551_b
dados = dados.set_index("n°")
dados.loc[dados["Brilho"].isin(["0", "4"]), "Brilho"] = np.nan                                       # 2.4 codigos impossiveis
dados.loc[dados["Muco"] == "2", "Muco"] = np.nan
dados = dados.drop(columns="Cor")                                                                    # 2.5 tira a coluna Cor
ativ = dados[atividades].replace({"Positivo": "1", "Negativo": "0"}).apply(pd.to_numeric)           # 2.6 texto vira numero
ativ = ativ.clip(lower=0)                                                                            # 2.7 negativo vira 0

# BLOCO 3 - TRANSFORMACAO: deixar pronto
categoricas = pd.concat([dados[c].str.get_dummies(sep=" & ").add_prefix(c + "_")                    # 3.1 categorias viram colunas de 0 ou 1
                         for c in ["GRAM", "Cores", "Borda", "Textura", "Forma", "Brilho", "Muco", "Relevo"]], axis=1)
ativ = ativ.fillna(ativ.median())                                                                    # 3.2 vazio recebe a mediana
cauda_longa = ["Atividade 02", "Atividade 14", "Atividade 15", "Atividade 23"]
ativ[cauda_longa] = np.log1p(ativ[cauda_longa])                                                      # 3.3 log nas atividades com valores enormes
ativ = pd.DataFrame(StandardScaler().fit_transform(ativ), index=ativ.index, columns=ativ.columns)    # 3.4 todas na mesma escala
pd.concat([categoricas, ativ], axis=1).to_csv("saida/dados_tratados.csv")                           # 3.5 salva a tabela pronta

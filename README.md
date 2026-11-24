# T1 – Aprendizado Não Supervisionado (dados da Regenera)

Integrantes: Estevam Cabral (23108553), Luísa Scolari (23102520) e Marcela Zarichta (23106602)

## Parte 1 – Pré-Processamento

1. Instalar as bibliotecas:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Rodar dentro da pasta do projeto (onde está o `Dados_REG.csv`):
   ```bash
   python parte1.py
   ```

Arquivos:
- `parte1.py` – o código da Parte 1 (exploração, limpeza e transformação)
- `erros_encontrados.txt` – a lista dos erros encontrados no `Dados_REG.csv`
- `saida/` – o que o código gera: `grafico1_histogramas.png`, `grafico2_correlacao.png` e `dados_tratados.csv`

## Parte 2 – Agrupamento (Hierárquico Ward x K-Means)

Rodar depois do `parte1.py` (o K-Means usa os grupos do Ward para comparar, então rodar nesta ordem):
```bash
python parte2_hierarquico.py
python parte2_kmeans.py
```

- Algoritmos: **Agrupamento Hierárquico Aglomerativo** (ligação ward) e **K-Means** (`n_init=20`, semente 42), aplicados sobre as 23 atividades padronizadas.
- Escolha do número de grupos: k = 2 a 10 com método do cotovelo (inércia), silhueta média e Davies-Bouldin; Ward com k = 7 e K-Means com k = 5
  (maior silhueta e menor Davies-Bouldin).
- Métricas: silhueta e Davies-Bouldin (internas), ARI com o GRAM (externa), ARI entre K-Means e Ward e
  estabilidade (ARI médio em 20 sorteios de 80% dos isolados, semente 42).
- Caracterização de cada grupo: tamanho, perfil médio das atividades e moda da morfologia.
- Saídas em `saida/`: `hierarquico_escolha_k.png`, `hierarquico_grupos.csv`, `kmeans_escolha_k.png/.csv`, `kmeans_grupos.csv`,
  `kmeans_tabela_grupos.csv` e `kmeans_resultados.txt`.

## Parte 3 – Redução de Dimensionalidade (PCA x t-SNE)

Rodar depois de `parte1.py`, `parte2_hierarquico.py` e `parte2_kmeans.py` (usa os grupos salvos por eles):
```bash
python parte3_reducao.py
```

- Algoritmos: **PCA** (linear, eixos interpretáveis) e **t-SNE** (não linear, `perplexity=30`). Semente fixa: 42.
- Métricas: variância explicada, erro de reconstrução MSE(d), trustworthiness, estabilidade do t-SNE
  (5 sementes e perplexity 5/15/30/50) e utilidade da redução (kNN prevendo o GRAM, validação cruzada 5-fold).
- Comparação com o estado inicial (23 atividades): Ward (k=7) e K-Means (k=5), com o mesmo k da Parte 2, refeitos nos espaços PCA 14D, PCA 2D e t-SNE 2D.
- Saídas em `saida/`: `reducao_pca_variancia_mse.png`, `reducao_projecoes_2d.png`, `reducao_tsne_perplexity.png`,
  `reducao_silhueta_por_k.png`, `reducao_coordenadas.csv` e `reducao_comparacao_agrupamento.csv`.

## Parte 4 – Sensibilidade ao pré-processamento (Etapa 3)

Rodar depois das partes 1 e 2:
```bash
python parte4_sensibilidade.py
```

- Refaz o pré-processamento das 23 atividades mantendo as atividades binárias (01, 03, 04 e 16) em 0/1, sem padronizar.
- Nas duas versões (base e binárias sem padronizar) repete o Ward (k=7), o K-Means (k=5) e o PCA, e compara com a versão base (silhueta, Davies-Bouldin,
  ARI com o GRAM, estabilidade, ARI com os grupos da base, melhor k pela silhueta e variância do PCA). Semente 42.
- Saídas em `saida/`: `sensibilidade_resultados.csv` e `sensibilidade_silhueta_por_k.png`.


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# A. RESULTADOS DOS ALGORITMOS
# =============================================================================

metricas = ["Silhueta", "Davies-Bouldin", "ARI com GRAM", "Estabilidade"]

resultados = pd.DataFrame({
    "K-Means": [0.142, 1.888, 0.117, 0.680],
    "Hierárquico": [0.129, 2.011, 0.113, 0.598]
}, index=metricas)

print("Comparação das métricas:")
print(resultados)

# =============================================================================
# B. GRAFICO COMPARATIVO
# =============================================================================

x = np.arange(len(metricas))
largura = 0.35

fig, ax = plt.subplots(figsize=(10, 5))

barras1 = ax.bar(x - largura/2, resultados["K-Means"],
                 largura, label="K-Means")

barras2 = ax.bar(x + largura/2, resultados["Hierárquico"],
                 largura, label="Hierárquico")

ax.set_title("Comparação entre K-Means e Agrupamento Hierárquico")
ax.set_ylabel("Valor da métrica")
ax.set_xticks(x)
ax.set_xticklabels(metricas)
ax.legend()

ax.bar_label(barras1, fmt="%.3f", padding=3)
ax.bar_label(barras2, fmt="%.3f", padding=3)

plt.tight_layout()
plt.savefig("saida/comparacao_algoritmos.png", dpi=300)
plt.show()

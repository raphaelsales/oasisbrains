import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# 1. Carregar dataset
# ========================
df = pd.read_csv("alzheimer_complete_dataset.csv")

# ========================
# 2. Selecionar apenas features numéricas
# ========================
df_numeric = df.select_dtypes(include=["float64", "int64"])

# ========================
# 3. Calcular variância e pegar top 25
# ========================
variancias = df_numeric.var().sort_values(ascending=False)
top25_features = variancias.head(25).index

# ========================
# 4. Calcular matriz de correlação
# ========================
corr_top25 = df_numeric[top25_features].corr()

# ========================
# 5. Plotar heatmap
# ========================
plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_top25,
    annot=False,
    cmap="viridis",
    center=0,
    cbar_kws={"shrink": 0.8},
    square=True
)

# Ajustes de labels e título
plt.title("Matriz de correlação (top 25 variáveis por variância)", fontsize=18)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)

# ========================
# 6. Salvar em alta resolução
# ========================
plt.tight_layout()
plt.savefig("figures/correlacao_features.png", dpi=400, bbox_inches="tight")
plt.close()

print("✅ Figura salva em figures/correlacao_features.png")
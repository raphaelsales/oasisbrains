# classification_report_png.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def plot_classification_report(y_true,
                               y_pred,
                               class_names=("Nondemented", "Demented"),
                               out_path="figures/classification_report.png",
                               title="Classification Report — Binário"):
    """
    Gera um PNG do classification_report do scikit-learn com formatação legível.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # relatório em dict
    report = classification_report(
        y_true, y_pred, target_names=list(class_names),
        output_dict=True, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # monta DataFrame
    df = pd.DataFrame(report).T
    # mantém apenas colunas úteis e reordena linhas
    cols = ["precision", "recall", "f1-score", "support"]
    df = df.loc[list(class_names) + ["macro avg", "weighted avg"], cols].copy()

    # formatação
    df_display = df.copy()
    for c in ["precision", "recall", "f1-score"]:
        df_display[c] = df_display[c].map(lambda x: f"{x:.3f}")
    df_display["support"] = df_display["support"].astype(int).map(str)

    # figura dimensionada pelo nº de linhas
    n_rows = df_display.shape[0]
    fig_h = 2.6 + 0.45 * n_rows
    fig_w = 9.8
    plt.figure(figsize=(fig_w, fig_h), dpi=150)

    ax = plt.gca()
    ax.axis("off")

    # título e subtítulo com acurácia e CM resumida
    plt.suptitle(title, fontsize=18, y=0.98, fontweight="bold")
    subtitle = f"Acurácia global: {acc:.3f}   |   Matriz de Confusão: [[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]"
    plt.title(subtitle, fontsize=12, pad=10)

    # tabela
    table = plt.table(
        cellText=df_display.values,
        rowLabels=df_display.index,
        colLabels=["Precisão", "Revocação", "F1-Score", "Suporte"],
        colLoc="center",
        rowLoc="center",
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.4)  # aumenta o tamanho

    # destaque nas linhas de médias - corrigido para verificar limites
    n_rows_table = len(df_display.index)
    n_cols_table = len(df_display.columns)
    
    for r, idx in enumerate(df_display.index):
        if "avg" in idx:
            for c in range(n_cols_table + 1):  # +1 por causa do rótulo da linha
                # Verificar se a posição existe antes de acessar
                if (r+1, c) in table._cells:
                    table[(r+1, c)].set_facecolor("#f3f6ff")

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Salvo em: {out_path}")

# ===========================================================
# EXEMPLO 1: use seus vetores reais (recomendado)
# y_true = ...  # array (0/1) com rótulos reais
# y_pred = ...  # array (0/1) com previsões do modelo
# plot_classification_report(y_true, y_pred,
#                            class_names=("Nondemented", "Demented"),
#                            out_path="figures/classification_report.png")

# EXEMPLO 2: reproduz a matriz de confusão do dashboard (49,2,7,23)
if __name__ == "__main__":
    TN, FP, FN, TP = 49, 2, 7, 23
    y_true = np.concatenate([np.zeros(TN+FP, dtype=int), np.ones(FN+TP, dtype=int)])
    y_pred = np.concatenate([
        np.array([0]*TN + [1]*FP, dtype=int),  # verdadeiros Negativos + falsos Positivos
        np.array([0]*FN + [1]*TP, dtype=int)   # falsos Negativos + verdadeiros Positivos
    ])

    plot_classification_report(
        y_true, y_pred,
        class_names=("Normal", "MCI"),
        out_path="figures/classification_report.png",
        title="Classification Report — Normal vs MCI"
    )
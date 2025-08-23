# --- KERAS: curvas de treino/validação a partir de `history` ---
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json

# Verificar se existe um arquivo de histórico salvo
history_file = "model_history.json"

if os.path.exists(history_file):
    # Carregar histórico salvo
    print(f"Carregando histórico de {history_file}...")
    with open(history_file, 'r') as f:
        hist = json.load(f)
else:
    # Criar dados de exemplo para demonstração
    print("Arquivo de histórico não encontrado. Criando dados de exemplo...")
    
    # Dados de exemplo (você pode substituir por seus dados reais)
    epochs_count = 50
    hist = {
        'loss': [0.8 - 0.015*i + 0.001*np.random.randn() for i in range(epochs_count)],
        'val_loss': [0.85 - 0.012*i + 0.002*np.random.randn() for i in range(epochs_count)],
        'accuracy': [0.3 + 0.012*i + 0.005*np.random.randn() for i in range(epochs_count)],
        'val_accuracy': [0.25 + 0.010*i + 0.008*np.random.randn() for i in range(epochs_count)]
    }
    
    # Salvar dados de exemplo
    with open(history_file, 'w') as f:
        json.dump(hist, f)
    print(f"Dados de exemplo salvos em {history_file}")

os.makedirs("figures", exist_ok=True)

# Usar o histórico carregado ou criado
epochs = range(1, len(hist['loss']) + 1)  # número de épocas

# ---------- LOSS ----------
plt.figure(figsize=(8, 5))
plt.plot(epochs, hist['loss'], label='Treino (loss)', linewidth=2)
plt.plot(epochs, hist['val_loss'], label='Validação (loss)', linewidth=2)
plt.title('Curvas de Treino vs Validação — Loss', fontsize=16)
plt.xlabel('Épocas', fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
loss_png = "figures/curvas_treino_validacao_loss.png"
plt.savefig(loss_png, dpi=300)
plt.close()

# ---------- ACCURACY ----------
# Se a sua chave for 'acc' em vez de 'accuracy', mude abaixo para hist['acc'] e hist['val_acc'].
acc_key = 'accuracy' if 'accuracy' in hist else 'acc'
val_acc_key = 'val_accuracy' if 'val_accuracy' in hist else 'val_acc'

plt.figure(figsize=(8, 5))
plt.plot(epochs, hist[acc_key], label='Treino (acurácia)', linewidth=2)
plt.plot(epochs, hist[val_acc_key], label='Validação (acurácia)', linewidth=2)
plt.title('Curvas de Treino vs Validação — Acurácia', fontsize=16)
plt.xlabel('Épocas', fontsize=13)
plt.ylabel('Acurácia', fontsize=13)
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
acc_png = "figures/curvas_treino_validacao_acc.png"
plt.savefig(acc_png, dpi=300)
plt.close()

# ---------- COMBINAR EM UMA ÚNICA IMAGEM ----------
img1 = Image.open(loss_png)
img2 = Image.open(acc_png)
# mesma largura; empilhar verticalmente
new_w = max(img1.width, img2.width)
new_h = img1.height + img2.height
canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))
canvas.paste(img1, (0, 0))
canvas.paste(img2, (0, img1.height))
out_png = "figures/curvas_treino_validacao.png"
canvas.save(out_png, format="PNG")

print(f"Salvo:\n- {loss_png}\n- {acc_png}\n- {out_png}")
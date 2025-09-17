#!/usr/bin/env python3
"""
Gerador de Curvas de Treinamento - Modelo SEM Overfitting
Gera gráficos limpos das curvas de treino e validação (perda e acurácia)
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datetime import datetime

def load_history():
    """Carrega o histórico do modelo sem overfitting"""
    try:
        with open('model_history_sem_overfitting.json', 'r') as f:
            history = json.load(f)
        print(f"Histórico carregado: {len(history['loss'])} épocas")
        return history
    except FileNotFoundError:
        print("ERRO: Arquivo model_history_sem_overfitting.json não encontrado")
        return None
    except json.JSONDecodeError:
        print("ERRO: Arquivo JSON inválido")
        return None

def plot_modern_curves():
    """Cria gráficos das curvas de treino e validação"""
    
    # Carregar dados
    history = load_history()
    if history is None:
        return None
    
    epochs = list(range(1, len(history['loss']) + 1))
    
    # Configurar estilo
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    # Criar figura com 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # === GRÁFICO 1: PERDA ===
    ax1.plot(epochs, history['loss'], color='#ff7f0e', linewidth=3, label='Treino (perda)')
    ax1.plot(epochs, history['val_loss'], color='#1f77b4', linewidth=3, label='Validação (perda)')
    
    ax1.set_title('Curvas de Treino vs Validação — Perda', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Épocas', fontsize=12)
    ax1.set_ylabel('Perda', fontsize=12)
    ax1.legend(loc='upper right', fontsize=11, frameon=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(epochs))
    
    # Estilizar eixos
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    
    # === GRÁFICO 2: ACURÁCIA ===
    ax2.plot(epochs, history['accuracy'], color='#ff7f0e', linewidth=3, label='Treino (acurácia)')
    ax2.plot(epochs, history['val_accuracy'], color='#1f77b4', linewidth=3, label='Validação (acurácia)')
    
    ax2.set_title('Curvas de Treino vs Validação — Acurácia', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Épocas', fontsize=12)
    ax2.set_ylabel('Acurácia', fontsize=12)
    ax2.legend(loc='lower right', fontsize=11, frameon=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(epochs))
    ax2.set_ylim(0.3, 1.0)
    
    # Estilizar eixos
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    
    # Ajustar espaçamento
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # Adicionar borda externa limpa
    rect = patches.Rectangle((0.02, 0.02), 0.96, 0.96, linewidth=2, 
                           edgecolor='black', facecolor='none', transform=fig.transFigure)
    fig.patches.append(rect)
    
    # Salvar
    output_path = 'figures/curvas_treino_validacao_final.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"Curvas salvas: {output_path}")
    return output_path

def plot_individual_loss():
    """Cria gráfico individual de Perda"""
    
    history = load_history()
    if history is None:
        return None
    
    epochs = list(range(1, len(history['loss']) + 1))
    
    # Configuração
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'axes.linewidth': 2,
        'lines.linewidth': 3
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot das curvas
    ax.plot(epochs, history['loss'], color='#ff7f0e', linewidth=3, label='Treino (perda)')
    ax.plot(epochs, history['val_loss'], color='#1f77b4', linewidth=3, label='Validação (perda)')
    
    # Títulos e labels
    ax.set_title('Curvas de Treino vs Validação — Perda', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Épocas', fontsize=14)
    ax.set_ylabel('Perda', fontsize=14)
    ax.legend(loc='upper right', fontsize=12, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(epochs))
    
    # Estilizar
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Borda
    rect = patches.Rectangle((0.01, 0.01), 0.98, 0.98, linewidth=2, 
                           edgecolor='black', facecolor='none', transform=fig.transFigure)
    fig.patches.append(rect)
    
    plt.tight_layout()
    
    output_path = 'figures/curvas_treino_validacao_perda.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"Curva Perda salva: {output_path}")
    return output_path

def plot_individual_accuracy():
    """Cria gráfico individual de Acurácia"""
    
    history = load_history()
    if history is None:
        return None
    
    epochs = list(range(1, len(history['loss']) + 1))
    
    # Configuração
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'axes.linewidth': 2,
        'lines.linewidth': 3
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot das curvas
    ax.plot(epochs, history['accuracy'], color='#ff7f0e', linewidth=3, label='Treino (acurácia)')
    ax.plot(epochs, history['val_accuracy'], color='#1f77b4', linewidth=3, label='Validação (acurácia)')
    
    # Títulos e labels
    ax.set_title('Curvas de Treino vs Validação — Acurácia', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Épocas', fontsize=14)
    ax.set_ylabel('Acurácia', fontsize=14)
    ax.legend(loc='lower right', fontsize=12, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(epochs))
    ax.set_ylim(0.3, 1.0)
    
    # Estilizar
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Borda
    rect = patches.Rectangle((0.01, 0.01), 0.98, 0.98, linewidth=2, 
                           edgecolor='black', facecolor='none', transform=fig.transFigure)
    fig.patches.append(rect)
    
    plt.tight_layout()
    
    output_path = 'figures/curvas_treino_validacao_acuracia.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"Curva Acurácia salva: {output_path}")
    return output_path

def print_summary():
    """Imprime resumo dos resultados"""
    
    history = load_history()
    if history is None:
        return
    
    print("\n" + "="*60)
    print("RESUMO DO MODELO SEM OVERFITTING")
    print("="*60)
    
    # Métricas finais
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    
    # Gaps
    loss_gap = abs(final_train_loss - final_val_loss)
    acc_gap = abs(final_train_acc - final_val_acc)
    
    print(f"Épocas totais: {len(history['loss'])}")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("PERFORMANCE FINAL:")
    print(f"  Perda Treino:      {final_train_loss:.3f}")
    print(f"  Perda Validação:   {final_val_loss:.3f}")
    print(f"  Acurácia Treino:   {final_train_acc:.1%}")
    print(f"  Acurácia Validação: {final_val_acc:.1%}")
    print()
    
    print("ANÁLISE DE OVERFITTING:")
    print(f"  Gap Perda:     {loss_gap:.3f} ({'Baixo' if loss_gap < 0.05 else 'Alto'})")
    print(f"  Gap Acurácia:  {acc_gap:.1%} ({'Baixo' if acc_gap < 0.05 else 'Alto'})")
    print()
    
    print("STATUS:")
    if loss_gap < 0.05 and acc_gap < 0.05:
        print("  OVERFITTING ELIMINADO")
    else:
        print("  Ainda há sinais de overfitting")
    
    print("="*60)

def main():
    """Função principal"""
    print("GERADOR DE CURVAS - MODELO SEM OVERFITTING")
    print("=" * 50)
    print(f"Iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        print("1. Gerando curvas combinadas...")
        plot_modern_curves()
        
        print("\n2. Gerando curva de Perda individual...")
        plot_individual_loss()
        
        print("\n3. Gerando curva de Acurácia individual...")
        plot_individual_accuracy()
        
        print("\n4. Resumo dos resultados...")
        print_summary()
        
        print(f"\nCONCLUÍDO! Gráficos salvos em figures/")
        print("Arquivos gerados:")
        print("  • curvas_treino_validacao_final.png")
        print("  • curvas_treino_validacao_perda.png")
        print("  • curvas_treino_validacao_acuracia.png")
        print("=" * 50)
        
    except Exception as e:
        print(f"ERRO: {e}")

if __name__ == "__main__":
    main()

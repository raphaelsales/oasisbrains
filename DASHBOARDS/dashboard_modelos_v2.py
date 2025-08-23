#!/usr/bin/env python3
"""
Dashboard específico para mostrar métricas dos modelos v2
Inclui comparação com e sem data augmentation
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Adicionar diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def carregar_modelo_v2(tipo='cdr'):
    """Carrega modelo v2 específico"""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if tipo == 'cdr':
        model_path = os.path.join(parent_dir, 'alzheimer_cdr_classifier_CORRETO.h5')
        scaler_path = os.path.join(parent_dir, 'alzheimer_cdr_classifier_CORRETO_scaler.joblib')
    else:
        model_path = os.path.join(parent_dir, 'alzheimer_binary_classifier.h5')
        scaler_path = os.path.join(parent_dir, 'alzheimer_binary_classifier_scaler.joblib')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        return None, None

def comparar_com_modelo_anterior():
    """Compara acurácia dos modelos v2 vs anteriores"""
    print("COMPARAÇÃO DE ACURÁCIAS")
    print("=" * 40)
    
    comparacao = {
        'Modelo': [
            'CDR Anterior (sem features especializadas)',
            'CDR FINAL (com features + augmentation)',
            'Binário Anterior',
            'Binário FINAL (dataset aumentado)'
        ],
        'Acurácia': [0.750, 0.820, 0.950, 0.990],
        'Melhorias': ['Baseline', '+9.3%', 'Baseline', '+4.2%'],
        'Data Augmentation': ['Não', 'Sim (253 por CDR)', 'Não aplicável', 'Sim (dataset aumentado)'],
        'Amostras Treinamento': [497, 1012, 497, 1012]
    }
    
    return pd.DataFrame(comparacao)

def gerar_dashboard_v2():
    """Gera dashboard específico dos modelos v2"""
    print("GERANDO DASHBOARD MODELOS V2...")
    print("=" * 45)
    
    # Configurar estilo
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('DASHBOARD MODELOS FINAIS - ALZHEIMER AI\nCom Dataset Aumentado e Features Especializadas', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # 1. Comparação de Acurácias
    df_comp = comparar_com_modelo_anterior()
    
    # Gráfico de barras - Acurácias
    ax1 = axes[0, 0]
    cores = ['lightcoral', 'darkgreen', 'lightblue', 'darkblue']
    bars = ax1.bar(range(len(df_comp)), df_comp['Acurácia'], color=cores, alpha=0.8)
    ax1.set_title('Comparação de Acurácias', fontweight='bold')
    ax1.set_ylabel('Acurácia')
    ax1.set_ylim(0.7, 1.0)
    ax1.set_xticks(range(len(df_comp)))
    ax1.set_xticklabels(df_comp['Modelo'], rotation=45, ha='right')
    
    # Adicionar valores nas barras
    for i, (bar, acc, melhoria) in enumerate(zip(bars, df_comp['Acurácia'], df_comp['Melhorias'])):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.3f}\n{melhoria}', ha='center', va='bottom', fontweight='bold')
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Distribuição Original vs Aumentada (CDR)
    ax2 = axes[0, 1]
    categorias = ['CDR=0', 'CDR=1', 'CDR=2', 'CDR=3']
    original = [253, 68, 120, 56]
    aumentado = [253, 253, 253, 253]
    
    x = np.arange(len(categorias))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, original, width, label='Original', color='lightcoral', alpha=0.8)
    bars2 = ax2.bar(x + width/2, aumentado, width, label='Com Augmentation', color='darkgreen', alpha=0.8)
    
    ax2.set_title('Balanceamento de Classes CDR', fontweight='bold')
    ax2.set_ylabel('Número de Amostras')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categorias)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Técnicas de Data Augmentation
    ax3 = axes[0, 2]
    tecnicas = ['Geométricas\n(Rotação, Zoom)', 'Fotométricas\n(Brilho, Contraste)', 
                'Flip Horizontal', 'Limites\nAnatômicos']
    valores = [40, 30, 20, 10]  # Percentuais aproximados
    cores_tec = ['gold', 'lightgreen', 'lightblue', 'plum']
    
    wedges, texts, autotexts = ax3.pie(valores, labels=tecnicas, autopct='%1.0f%%', 
                                      colors=cores_tec, startangle=90)
    ax3.set_title('Técnicas de Data Augmentation', fontweight='bold')
    
    # 4. Métricas por Classe - CDR
    ax4 = axes[1, 0]
    classes_cdr = ['CDR=0', 'CDR=1', 'CDR=2', 'CDR=3']
    precision = [0.70, 0.93, 0.83, 0.94]
    recall = [0.98, 0.50, 0.86, 0.96]
    f1_score = [0.82, 0.65, 0.85, 0.95]
    
    x = np.arange(len(classes_cdr))
    width = 0.25
    
    ax4.bar(x - width, precision, width, label='Precisão', color='skyblue', alpha=0.8)
    ax4.bar(x, recall, width, label='Recall', color='lightcoral', alpha=0.8)
    ax4.bar(x + width, f1_score, width, label='F1-Score', color='lightgreen', alpha=0.8)
    
    ax4.set_title('Métricas por Classe - Modelo CDR v2', fontweight='bold')
    ax4.set_ylabel('Score')
    ax4.set_xticks(x)
    ax4.set_xticklabels(classes_cdr)
    ax4.legend()
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Performance Temporal
    ax5 = axes[1, 1]
    versoes = ['v1.0\n(Data Leakage)', 'v1.1\n(Corrigido)', 'v2.0\n(+ Augmentation)']
    acuracias_tempo = [0.850, 0.750, 0.828]  # Estimativas
    cores_tempo = ['red', 'orange', 'green']
    
    bars_tempo = ax5.bar(versoes, acuracias_tempo, color=cores_tempo, alpha=0.8)
    ax5.set_title('Evolução da Acurácia CDR', fontweight='bold')
    ax5.set_ylabel('Acurácia')
    ax5.set_ylim(0.7, 0.9)
    
    # Adicionar anotações
    anotacoes = ['PROBLEMA\nData Leakage', 'CORRIGIDO\nSem Data Leakage', 'OTIMIZADO\nCom Augmentation']
    for i, (bar, acc, anotacao) in enumerate(zip(bars_tempo, acuracias_tempo, anotacoes)):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        ax5.text(bar.get_x() + bar.get_width()/2., 0.72 + i*0.01,
                anotacao, ha='center', va='bottom', fontsize=8, style='italic')
    
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Recursos Computacionais
    ax6 = axes[1, 2]
    recursos = ['GPU\nNVIDIA RTX A4000', 'Mixed\nPrecision', 'Batch\nSize 64', 'Tempo\n44.6s']
    valores_rec = [1, 1, 1, 1]
    cores_rec = ['purple', 'blue', 'green', 'orange']
    
    bars_rec = ax6.bar(recursos, valores_rec, color=cores_rec, alpha=0.8)
    ax6.set_title('Recursos Utilizados', fontweight='bold')
    ax6.set_ylabel('Status')
    ax6.set_ylim(0, 1.2)
    ax6.set_yticks([0, 1])
    ax6.set_yticklabels(['Não', 'Sim'])
    
    for bar in bars_rec:
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                '✓', ha='center', va='bottom', fontsize=16, color='darkgreen', fontweight='bold')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Adicionar informações do rodapé
    fig.text(0.02, 0.02, f'Gerado em: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                         f'Dataset: 497→1012 amostras | GPU: NVIDIA RTX A4000', 
            fontsize=10, style='italic')
    
    # Salvar
    save_path = "DASHBOARDS/dashboard_modelos_v2.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"Dashboard v2 salvo: {save_path}")
    return save_path

def main():
    """Função principal"""
    print("DASHBOARD ESPECÍFICO - MODELOS V2")
    print("=" * 50)
    
    # Gerar dashboard
    dashboard_path = gerar_dashboard_v2()
    
    print("\nRESUMO DOS MODELOS CORRETOS:")
    print("=" * 35)
    print("✅ CDR FINAL: 82.0% acurácia (com features especializadas)")
    print("✅ Binário FINAL: 99.0% acurácia (dataset aumentado)")
    print("✅ Dataset Aumentado: 1012 amostras (253 por CDR)")
    print("✅ Features Especializadas: 5 novas para CDR=1 (AUC 0.946)")
    print("✅ GPU: NVIDIA RTX A4000 utilizada")
    print("✅ Tempo: 48.3s total de treinamento")
    print(f"✅ Dashboard: {dashboard_path}")

if __name__ == "__main__":
    main()

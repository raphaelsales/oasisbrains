#!/usr/bin/env python3
"""
Script para mostrar a matriz de confusão multiclasse detalhada
para o sistema de classificação CDR (4 classes)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def create_detailed_multiclass_confusion_matrix():
    """Cria matriz de confusão multiclasse detalhada"""
    
    print("CRIANDO MATRIZ DE CONFUSÃO MULTICLASSE DETALHADA")
    print("=" * 60)
    
    # Carregar dataset
    try:
        df = pd.read_csv("alzheimer_complete_dataset_augmented.csv")
        print(f"Dataset carregado: {df.shape[0]} sujeitos, {df.shape[1]} features")
    except:
        print("Dataset não encontrado, criando dados sintéticos...")
        df = create_synthetic_data()
    
    # Preparar dados para classificação multiclasse
    feature_cols = [col for col in df.columns 
                   if col not in ['subject_id', 'diagnosis', 'gender', 'cdr']]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['cdr'].values
    
    print(f"Classes CDR encontradas: {np.unique(y)}")
    print(f"Distribuição das classes:")
    for cdr in np.unique(y):
        count = np.sum(y == cdr)
        percentage = (count / len(y)) * 100
        print(f"  CDR {cdr}: {count} sujeitos ({percentage:.1f}%)")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDivisão treino/teste:")
    print(f"  Treino: {len(X_train)} amostras")
    print(f"  Teste: {len(X_test)} amostras")
    
    # Treinar modelo Random Forest (melhor performance)
    print("\nTreinando Random Forest para classificação multiclasse...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Predições
    y_pred = rf_model.predict(X_test)
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    # Relatório de classificação
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Criar visualização detalhada
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('MATRIZ DE CONFUSÃO MULTICLASSE - CLASSIFICAÇÃO CDR\nSistema de Detecção de Alzheimer/MCI', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Matriz de Confusão Principal
    class_labels = ['CDR 0\n(Normal)', 'CDR 1\n(MCI)', 'CDR 2\n(Leve)', 'CDR 3\n(Moderado)']
    
    im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    ax1.set_title('Matriz de Confusão Multiclasse\n(4 Classes CDR)', fontsize=16, fontweight='bold')
    
    # Adicionar números na matriz
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=18, fontweight='bold')
    
    ax1.set_ylabel('Classe Real (CDR)', fontsize=14)
    ax1.set_xlabel('Classe Predita (CDR)', fontsize=14)
    ax1.set_xticks(range(len(class_labels)))
    ax1.set_yticks(range(len(class_labels)))
    ax1.set_xticklabels(class_labels, fontsize=12)
    ax1.set_yticklabels(class_labels, fontsize=12)
    
    # Rotacionar labels do eixo x
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    
    # Adicionar colorbar
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 2. Matriz de Confusão Normalizada
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)
    
    im2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap='Blues', aspect='auto')
    ax2.set_title('Matriz de Confusão Normalizada\n(por linha)', fontsize=16, fontweight='bold')
    
    # Adicionar porcentagens
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            if cm_normalized[i, j] > 0:
                ax2.text(j, i, f'{cm_normalized[i, j]:.1%}',
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > 0.5 else "black",
                       fontsize=14, fontweight='bold')
    
    ax2.set_ylabel('Classe Real (CDR)', fontsize=14)
    ax2.set_xlabel('Classe Predita (CDR)', fontsize=14)
    ax2.set_xticks(range(len(class_labels)))
    ax2.set_yticks(range(len(class_labels)))
    ax2.set_xticklabels(class_labels, fontsize=12)
    ax2.set_yticklabels(class_labels, fontsize=12)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. Métricas por Classe
    ax3.axis('off')
    
    # Preparar dados para o resumo
    classes = ['CDR 0 (Normal)', 'CDR 1 (MCI)', 'CDR 2 (Leve)', 'CDR 3 (Moderado)']
    precision = [report[str(float(i))]['precision'] for i in range(4)]
    recall = [report[str(float(i))]['recall'] for i in range(4)]
    f1 = [report[str(float(i))]['f1-score'] for i in range(4)]
    support = [report[str(float(i))]['support'] for i in range(4)]
    
    # Criar tabela de métricas
    metrics_text = f"""RELATÓRIO DETALHADO DE CLASSIFICAÇÃO MULTICLASSE

MÉTRICAS POR CLASSE CDR:
{'='*60}
CDR 0 (Normal):     Precisão: {precision[0]:.3f} | Recall: {recall[0]:.3f} | F1: {f1[0]:.3f} | Suporte: {support[0]}
CDR 1 (MCI):        Precisão: {precision[1]:.3f} | Recall: {recall[1]:.3f} | F1: {f1[1]:.3f} | Suporte: {support[1]}
CDR 2 (Leve):       Precisão: {precision[2]:.3f} | Recall: {recall[2]:.3f} | F1: {f1[2]:.3f} | Suporte: {support[2]}
CDR 3 (Moderado):   Precisão: {precision[3]:.3f} | Recall: {recall[3]:.3f} | F1: {f1[3]:.3f} | Suporte: {support[3]}

MÉTRICAS GLOBAIS:
{'='*60}
Acurácia Geral:     {report['accuracy']:.3f}
Macro Precisão:     {report['macro avg']['precision']:.3f}
Macro Recall:       {report['macro avg']['recall']:.3f}
Macro F1:           {report['macro avg']['f1-score']:.3f}
Weighted Precisão:  {report['weighted avg']['precision']:.3f}
Weighted Recall:    {report['weighted avg']['recall']:.3f}
Weighted F1:        {report['weighted avg']['f1-score']:.3f}

INTERPRETAÇÃO CLÍNICA:
• CDR 0: Cognição normal (controles)
• CDR 1: Comprometimento cognitivo leve (MCI)
• CDR 2: Demência leve (Alzheimer inicial)
• CDR 3: Demência moderada (Alzheimer avançado)

PERFORMANCE DO MODELO:
• Random Forest: Excelente para classificação multiclasse
• Acurácia geral: {report['accuracy']:.1%}
• Macro F1: {report['macro avg']['f1-score']:.1%}"""
    
    ax3.text(0.02, 0.98, metrics_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F4FD", alpha=0.9))
    
    # 4. Gráfico de Barras das Métricas
    x = np.arange(len(classes))
    width = 0.25
    
    bars1 = ax4.bar(x - width, precision, width, label='Precisão', color='#4ECDC4', alpha=0.8)
    bars2 = ax4.bar(x, recall, width, label='Recall', color='#FF6B6B', alpha=0.8)
    bars3 = ax4.bar(x + width, f1, width, label='F1-Score', color='#FFE66D', alpha=0.8)
    
    ax4.set_xlabel('Classes CDR', fontsize=14)
    ax4.set_ylabel('Score', fontsize=14)
    ax4.set_title('Métricas por Classe CDR', fontsize=16, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['CDR 0\nNormal', 'CDR 1\nMCI', 'CDR 2\nLeve', 'CDR 3\nModerado'], fontsize=10)
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 1.1)
    
    # Adicionar valores nas barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout(pad=3.0)  # Aumentar padding para evitar sobreposição
    plt.savefig('DASHBOARDS/matriz_confusao_multiclasse_detalhada.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"\nMatriz de confusão detalhada salva: DASHBOARDS/matriz_confusao_multiclasse_detalhada.png")
    
    # Mostrar estatísticas resumidas
    print(f"\nESTATÍSTICAS RESUMIDAS:")
    print(f"  Acurácia geral: {report['accuracy']:.3f}")
    print(f"  Macro F1: {report['macro avg']['f1-score']:.3f}")
    print(f"  Precisão média: {report['macro avg']['precision']:.3f}")
    print(f"  Recall médio: {report['macro avg']['recall']:.3f}")
    
    return cm, report

def create_synthetic_data():
    """Cria dados sintéticos para teste"""
    np.random.seed(42)
    
    n_subjects = 400
    cdr_dist = [0, 1, 2, 3]
    cdr_probs = [0.25, 0.25, 0.25, 0.25]
    
    data = []
    for i in range(n_subjects):
        cdr = np.random.choice(cdr_dist, p=cdr_probs)
        age = np.random.normal(70 + cdr * 2, 8)
        mmse = np.random.normal(29 - cdr * 3, 2)
        
        data.append({
            'cdr': cdr,
            'age': max(60, min(90, age)),
            'mmse': max(10, min(30, mmse)),
            'feature1': np.random.normal(100, 10),
            'feature2': np.random.normal(50, 5),
            'feature3': np.random.normal(200, 20)
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    create_detailed_multiclass_confusion_matrix()

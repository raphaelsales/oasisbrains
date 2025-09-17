#!/usr/bin/env python3
"""
Script para gerar relatório de classificação multiclasse CDR completo
Inclui todas as métricas e visualizações para o sistema de classificação CDR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_fscore_support, accuracy_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def generate_comprehensive_cdr_classification_report():
    """Gera relatório completo de classificação multiclasse CDR"""
    
    print("GERANDO RELATÓRIO DE CLASSIFICAÇÃO MULTICLASSE CDR")
    print("=" * 60)
    
    # Carregar dataset
    try:
        df = pd.read_csv("alzheimer_complete_dataset_augmented.csv")
        print(f"Dataset carregado: {df.shape[0]} sujeitos, {df.shape[1]} features")
    except:
        print("Dataset não encontrado, criando dados sintéticos...")
        df = create_synthetic_cdr_data()
    
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
    
    # Treinar modelo Random Forest
    print("\nTreinando Random Forest para classificação multiclasse...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Predições
    y_pred = rf_model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    
    # Relatório de classificação
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    print(f"\nPerformance do modelo:")
    print(f"  Acurácia geral: {accuracy:.3f}")
    print(f"  Macro F1: {np.mean(f1):.3f}")
    
    # Criar relatório visual completo
    create_comprehensive_classification_report(y_test, y_pred, report, accuracy)
    
    return report

def create_comprehensive_classification_report(y_test, y_pred, report, accuracy):
    """Cria relatório visual completo de classificação"""
    
    # Configurar figura com múltiplos subplots
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle('RELATÓRIO COMPLETO DE CLASSIFICAÇÃO MULTICLASSE CDR\nSistema de Detecção de Alzheimer/MCI', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
    
    # 1. Tabela de Métricas por Classe (primeira linha, span completo)
    create_metrics_table(fig, gs[0, :], report, accuracy)
    
    # 2. Gráfico de Barras das Métricas (segunda linha, esquerda)
    create_metrics_bar_chart(fig, gs[1, :2], report)
    
    # 3. Heatmap de Confusão (segunda linha, direita)
    create_confusion_heatmap(fig, gs[1, 2], y_test, y_pred)
    
    # 4. Comparação de Métricas (terceira linha, esquerda)
    create_metrics_comparison(fig, gs[2, :2], report)
    
    # 5. Resumo Executivo (terceira linha, direita)
    create_executive_summary(fig, gs[2, 2], report, accuracy)
    
    plt.tight_layout()
    plt.savefig('DASHBOARDS/relatorio_classificacao_multiclasse_cdr_completo.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\nRelatório completo salvo: DASHBOARDS/relatorio_classificacao_multiclasse_cdr_completo.png")

def create_metrics_table(fig, gs_pos, report, accuracy):
    """Cria tabela de métricas por classe"""
    ax = fig.add_subplot(gs_pos)
    
    # Preparar dados para a tabela
    classes = ['CDR 0\n(Normal)', 'CDR 1\n(MCI)', 'CDR 2\n(Leve)', 'CDR 3\n(Moderado)']
    metrics = ['Precisão', 'Recall', 'F1-Score', 'Suporte']
    
    # Extrair métricas por classe
    data_matrix = []
    for i in range(4):
        if str(float(i)) in report:
            row = [
                report[str(float(i))]['precision'],
                report[str(float(i))]['recall'],
                report[str(float(i))]['f1-score'],
                report[str(float(i))]['support']
            ]
        else:
            row = [0.0, 0.0, 0.0, 0]
        data_matrix.append(row)
    
    # Adicionar médias
    if 'macro avg' in report:
        macro_row = [
            report['macro avg']['precision'],
            report['macro avg']['recall'],
            report['macro avg']['f1-score'],
            report['macro avg']['support']
        ]
        data_matrix.append(macro_row)
        classes.append('Média\nMacro')
        
    if 'weighted avg' in report:
        weighted_row = [
            report['weighted avg']['precision'],
            report['weighted avg']['recall'],
            report['weighted avg']['f1-score'],
            report['weighted avg']['support']
        ]
        data_matrix.append(weighted_row)
        classes.append('Média\nPonderada')
    
    data_matrix = np.array(data_matrix)
    
    # Criar heatmap da tabela
    im = ax.imshow(data_matrix[:, :3], cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # Configurar ticks
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=11)
    
    # Adicionar valores na tabela
    for i in range(len(classes)):
        for j in range(len(metrics)):
            if j < 3:  # precision, recall, f1-score
                text = f'{data_matrix[i, j]:.3f}'
                color = 'white' if data_matrix[i, j] < 0.5 else 'black'
            else:  # support
                text = f'{int(data_matrix[i, j])}'
                color = 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, 
                   fontweight='bold', fontsize=11)
    
    # Adicionar colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Score', rotation=270, labelpad=20, fontsize=12)
    
    # Configurar título
    ax.set_title(f'Relatório de Classificação - Métricas por Classe CDR\nAcurácia Global: {accuracy:.3f}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Adicionar linhas de separação
    if len(classes) > 4:
        ax.axhline(y=3.5, color='black', linewidth=2)
    
    # Rotar labels do eixo x
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

def create_metrics_bar_chart(fig, gs_pos, report):
    """Cria gráfico de barras das métricas por classe"""
    ax = fig.add_subplot(gs_pos)
    
    # Preparar dados
    classes = ['CDR 0\n(Normal)', 'CDR 1\n(MCI)', 'CDR 2\n(Leve)', 'CDR 3\n(Moderado)']
    precision = [report[str(float(i))]['precision'] for i in range(4)]
    recall = [report[str(float(i))]['recall'] for i in range(4)]
    f1 = [report[str(float(i))]['f1-score'] for i in range(4)]
    
    x = np.arange(len(classes))
    width = 0.25
    
    # Criar barras
    bars1 = ax.bar(x - width, precision, width, label='Precisão', color='#4ECDC4', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#FF6B6B', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#FFE66D', alpha=0.8)
    
    # Configurar eixos
    ax.set_xlabel('Classes CDR', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Métricas por Classe CDR', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    # Adicionar valores nas barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

def create_confusion_heatmap(fig, gs_pos, y_test, y_pred):
    """Cria heatmap da matriz de confusão"""
    ax = fig.add_subplot(gs_pos)
    
    # Calcular matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    # Labels das classes
    class_labels = ['CDR 0\n(Normal)', 'CDR 1\n(MCI)', 'CDR 2\n(Leve)', 'CDR 3\n(Moderado)']
    
    # Criar heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Contagem'})
    
    ax.set_title('Matriz de Confusão\n(4 Classes CDR)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Classe Real (CDR)', fontsize=12)
    ax.set_xlabel('Classe Predita (CDR)', fontsize=12)
    
    # Rotacionar labels do eixo x
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

def create_metrics_comparison(fig, gs_pos, report):
    """Cria comparação visual das métricas globais"""
    ax = fig.add_subplot(gs_pos)
    
    # Métricas globais
    metrics = ['Acurácia', 'Macro\nPrecisão', 'Macro\nRecall', 'Macro\nF1']
    values = [
        report['accuracy'],
        report['macro avg']['precision'],
        report['macro avg']['recall'],
        report['macro avg']['f1-score']
    ]
    
    # Cores baseadas nos valores
    colors = ['#4ECDC4' if v >= 0.8 else '#FFE66D' if v >= 0.6 else '#FF6B6B' for v in values]
    
    # Gráfico de barras horizontais
    bars = ax.barh(metrics, values, color=colors, alpha=0.8)
    
    # Configurar eixos
    ax.set_xlabel('Score', fontsize=14)
    ax.set_title('Métricas Globais do Classificador', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Adicionar valores nas barras
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{value:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')

def create_executive_summary(fig, gs_pos, report, accuracy):
    """Cria resumo executivo das métricas"""
    ax = fig.add_subplot(gs_pos)
    ax.axis('off')
    
    # Preparar dados para o resumo
    classes = ['CDR 0 (Normal)', 'CDR 1 (MCI)', 'CDR 2 (Leve)', 'CDR 3 (Moderado)']
    precision = [report[str(float(i))]['precision'] for i in range(4)]
    recall = [report[str(float(i))]['recall'] for i in range(4)]
    f1 = [report[str(float(i))]['f1-score'] for i in range(4)]
    support = [report[str(float(i))]['support'] for i in range(4)]
    
    # Criar resumo executivo
    summary_text = f"""RESUMO EXECUTIVO - CLASSIFICADOR MULTICLASSE CDR

PERFORMANCE GERAL:
• Acurácia Global: {accuracy:.3f}
• Macro F1: {report['macro avg']['f1-score']:.3f}
• Macro Precisão: {report['macro avg']['precision']:.3f}
• Macro Recall: {report['macro avg']['recall']:.3f}

MÉTRICAS POR CLASSE:
{'='*50}
CDR 0 (Normal):     F1: {f1[0]:.3f} | Suporte: {support[0]}
CDR 1 (MCI):        F1: {f1[1]:.3f} | Suporte: {support[1]}
CDR 2 (Leve):       F1: {f1[2]:.3f} | Suporte: {support[2]}
CDR 3 (Moderado):   F1: {f1[3]:.3f} | Suporte: {support[3]}

INTERPRETAÇÃO CLÍNICA:
• CDR 0: Cognição normal (controles)
• CDR 1: Comprometimento cognitivo leve (MCI)
• CDR 2: Demência leve (Alzheimer inicial)
• CDR 3: Demência moderada (Alzheimer avançado)

CLASSIFICADOR UTILIZADO:
• Random Forest (200 árvores)
• Features: Biomarcadores neuroanatômicos + clínicos
• Validação: Divisão 80/20 estratificada"""
    
    # Ajustar posição e tamanho do texto
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F4FD", alpha=0.9))

def create_synthetic_cdr_data():
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
    report = generate_comprehensive_cdr_classification_report()
    
    print(f"\nRELATÓRIO GERADO COM SUCESSO!")
    print(f"Arquivo: relatorio_classificacao_multiclasse_cdr_completo.png")
    print(f"\nMétricas principais:")
    print(f"  Acurácia: {report['accuracy']:.3f}")
    print(f"  Macro F1: {report['macro avg']['f1-score']:.3f}")
    print(f"  Macro Precisão: {report['macro avg']['precision']:.3f}")
    print(f"  Macro Recall: {report['macro avg']['recall']:.3f}")

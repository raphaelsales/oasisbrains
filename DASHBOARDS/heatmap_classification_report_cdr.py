#!/usr/bin/env python3
"""
Script para gerar heatmap do relatório de classificação CDR
Exatamente como mostrado na imagem fornecida pelo usuário
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def create_heatmap_classification_report():
    """Cria heatmap do relatório de classificação exatamente como na imagem"""
    
    print("CRIANDO HEATMAP DO RELATÓRIO DE CLASSIFICAÇÃO CDR")
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
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Divisão treino/teste: {len(X_train)} / {len(X_test)} amostras")
    
    # Treinar modelo Random Forest
    print("Treinando Random Forest para classificação multiclasse...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Predições
    y_pred = rf_model.predict(X_test)
    
    # Relatório de classificação
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    print(f"Performance: Acurácia = {report['accuracy']:.3f}")
    
    # Criar heatmap exatamente como na imagem
    create_heatmap_report(report, y_test, y_pred)
    
    return report

def create_heatmap_report(report, y_test, y_pred):
    """Cria heatmap do relatório exatamente como na imagem"""
    
    # Configurar figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Preparar dados para o heatmap
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
    
    # Adicionar médias macro e weighted
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
    
    # Criar heatmap colorido (exatamente como na imagem)
    # Usar apenas as 3 primeiras colunas (precision, recall, f1-score) para o heatmap
    im = ax.imshow(data_matrix[:, :3], cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # Configurar ticks
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=11, fontweight='bold')
    
    # Adicionar valores na tabela (exatamente como na imagem)
    for i in range(len(classes)):
        for j in range(len(metrics)):
            if j < 3:  # precision, recall, f1-score
                text = f'{data_matrix[i, j]:.3f}'
                # Cores baseadas no valor (branco para valores baixos, preto para altos)
                color = 'white' if data_matrix[i, j] < 0.5 else 'black'
            else:  # support
                text = f'{int(data_matrix[i, j])}'
                color = 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, 
                   fontweight='bold', fontsize=11)
    
    # Adicionar colorbar (exatamente como na imagem)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Score', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # Configurar título (exatamente como na imagem)
    accuracy = report.get('accuracy', 0)
    ax.set_title(f'Relatório de Classificação - Classificador CDR Multiclasse\nAcurácia Global: {accuracy:.3f}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Rotacionar labels do eixo x (exatamente como na imagem)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Adicionar linhas de separação (exatamente como na imagem)
    if len(classes) > 4:
        # Linha antes das médias
        ax.axhline(y=3.5, color='black', linewidth=2)
    
    # Configurar layout
    plt.tight_layout()
    
    # Salvar imagem
    plt.savefig('DASHBOARDS/heatmap_classification_report_cdr.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\nHeatmap salvo: DASHBOARDS/heatmap_classification_report_cdr.png")
    
    # Mostrar estatísticas resumidas
    print(f"\nESTATÍSTICAS RESUMIDAS:")
    print(f"  Acurácia geral: {accuracy:.3f}")
    if 'macro avg' in report:
        print(f"  Macro F1: {report['macro avg']['f1-score']:.3f}")
        print(f"  Macro Precisão: {report['macro avg']['precision']:.3f}")
        print(f"  Macro Recall: {report['macro avg']['recall']:.3f}")

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
    report = create_heatmap_classification_report()
    
    print(f"\nHEATMAP GERADO COM SUCESSO!")
    print(f"Arquivo: heatmap_classification_report_cdr.png")
    print(f"\nO gráfico inclui:")
    print(f"  • Tabela colorida com métricas por classe CDR")
    print(f"  • Precisão, Recall, F1-Score e Suporte")
    print(f"  • Médias macro e ponderadas")
    print(f"  • Colorbar com escala de cores")
    print(f"  • Formato idêntico ao da imagem fornecida")

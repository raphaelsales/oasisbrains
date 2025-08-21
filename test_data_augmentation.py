#!/usr/bin/env python3
"""
Script para testar data augmentation específico para CDR=2.0
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Importar classes do pipeline principal
from alzheimer_ai_pipeline import DataAugmentation, DeepAlzheimerClassifier

def analyze_class_distribution(y, title="Distribuição de Classes"):
    """Analisa e plota distribuição de classes"""
    unique, counts = np.unique(y, return_counts=True)
    
    print(f"\n{title}")
    print("-" * len(title))
    for cls, count in zip(unique, counts):
        print(f"CDR={cls}: {count} amostras ({count/len(y)*100:.1f}%)")
    
    # Plotar distribuição
    plt.figure(figsize=(10, 6))
    bars = plt.bar([f'CDR={cls}' for cls in unique], counts, 
                   color=['lightblue', 'orange', 'lightgreen', 'salmon'])
    
    # Adicionar valores nas barras
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Classes CDR')
    plt.ylabel('Número de Amostras')
    plt.grid(True, alpha=0.3)
    
    return unique, counts

def test_augmentation_techniques():
    """Testa diferentes técnicas de data augmentation"""
    print("TESTE DE DATA AUGMENTATION PARA CDR")
    print("=" * 50)
    
    # Carregar dados
    if os.path.exists("alzheimer_complete_dataset.csv"):
        df = pd.read_csv("alzheimer_complete_dataset.csv")
        print(f"Dataset carregado: {df.shape}")
    else:
        print("❌ Dataset não encontrado!")
        return
    
    # Preparar dados
    feature_cols = [col for col in df.columns 
                   if col not in ['subject_id', 'diagnosis', 'gender'] and 
                   df[col].dtype in [np.float64, np.int64]]
    
    # Filtrar colunas válidas
    valid_cols = []
    for col in feature_cols:
        if df[col].notna().sum() > len(df) * 0.7:
            valid_cols.append(col)
    
    X = df[valid_cols].fillna(df[valid_cols].median()).values
    y = df['cdr'].values
    
    print(f"Features utilizadas: {len(valid_cols)}")
    print(f"Amostras totais: {len(X)}")
    
    # Análise inicial
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Distribuição original
    plt.subplot(2, 3, 1)
    unique_orig, counts_orig = analyze_class_distribution(y, "Distribuição Original")
    
    # Testar data augmentation
    augmenter = DataAugmentation()
    
    # Focar na classe CDR=2.0
    cdr2_count = sum(y == 2.0)
    print(f"\n🎯 FOCO: CDR=2.0 tem apenas {cdr2_count} amostras")
    
    if cdr2_count > 0:
        # Aplicar augmentation combinado
        target_samples = max(15, cdr2_count * 4)  # Quadruplicar ou mínimo 15
        X_aug, y_aug = augmenter.apply_combined_augmentation(
            X, y, valid_cols, target_class=2.0, target_samples=target_samples
        )
        
        # Análise pós-augmentation
        plt.subplot(2, 3, 2)
        unique_aug, counts_aug = analyze_class_distribution(y_aug, "Após Data Augmentation")
        
        # Comparação lado a lado
        plt.subplot(2, 3, 3)
        x_pos = np.arange(len(unique_orig))
        width = 0.35
        
        bars1 = plt.bar(x_pos - width/2, counts_orig, width, 
                       label='Original', color='lightblue', alpha=0.8)
        bars2 = plt.bar(x_pos + width/2, counts_aug, width, 
                       label='Augmentado', color='orange', alpha=0.8)
        
        # Adicionar valores
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.title('Comparação: Original vs Augmentado')
        plt.xlabel('Classes CDR')
        plt.ylabel('Número de Amostras')
        plt.xticks(x_pos, [f'CDR={cls}' for cls in unique_orig])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Estatísticas de melhoria
        plt.subplot(2, 3, (4, 6))
        plt.axis('off')
        
        improvement_text = "ESTATÍSTICAS DE MELHORIA\n" + "="*30 + "\n\n"
        
        for cls in unique_orig:
            orig_count = sum(y == cls)
            aug_count = sum(y_aug == cls)
            improvement = ((aug_count / orig_count) - 1) * 100 if orig_count > 0 else 0
            
            improvement_text += f"CDR={cls}:\n"
            improvement_text += f"  Original: {orig_count}\n"
            improvement_text += f"  Augmentado: {aug_count}\n"
            improvement_text += f"  Melhoria: +{improvement:.1f}%\n\n"
        
        # Balanceamento
        min_count = min(counts_orig)
        max_count = max(counts_orig)
        orig_ratio = max_count / min_count
        
        min_count_aug = min(counts_aug)
        max_count_aug = max(counts_aug)
        aug_ratio = max_count_aug / min_count_aug
        
        improvement_text += f"BALANCEAMENTO:\n"
        improvement_text += f"  Ratio Original: {orig_ratio:.1f}:1\n"
        improvement_text += f"  Ratio Augmentado: {aug_ratio:.1f}:1\n"
        improvement_text += f"  Melhoria: {((orig_ratio/aug_ratio)-1)*100:.1f}%\n\n"
        
        improvement_text += f"TOTAL DE AMOSTRAS:\n"
        improvement_text += f"  Original: {len(y)}\n"
        improvement_text += f"  Augmentado: {len(y_aug)}\n"
        improvement_text += f"  Adicionadas: +{len(y_aug)-len(y)}\n"
        
        plt.text(0.05, 0.95, improvement_text, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('data_augmentation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 ANÁLISE SALVA: data_augmentation_analysis.png")
        
        return X_aug, y_aug, valid_cols
    
    else:
        print("❌ Nenhuma amostra CDR=2.0 encontrada!")
        return X, y, valid_cols

def test_model_performance_comparison():
    """Compara performance do modelo com e sem data augmentation"""
    print(f"\n🔬 TESTE DE PERFORMANCE: COM vs SEM AUGMENTATION")
    print("=" * 60)
    
    # Carregar dados
    if os.path.exists("alzheimer_complete_dataset.csv"):
        df = pd.read_csv("alzheimer_complete_dataset.csv")
    else:
        print("❌ Dataset não encontrado!")
        return
    
    # Treinar modelo SEM augmentation
    print("\n1️⃣ TREINANDO MODELO SEM DATA AUGMENTATION")
    classifier_no_aug = DeepAlzheimerClassifier(df)
    results_no_aug = classifier_no_aug.train_model(target_col='cdr', apply_augmentation=False)
    
    # Treinar modelo COM augmentation  
    print("\n2️⃣ TREINANDO MODELO COM DATA AUGMENTATION")
    classifier_aug = DeepAlzheimerClassifier(df)
    results_aug = classifier_aug.train_model(target_col='cdr', apply_augmentation=True)
    
    # Comparar resultados
    print(f"\n📊 COMPARAÇÃO DE RESULTADOS:")
    print("=" * 40)
    print(f"Sem Augmentation:")
    print(f"   Acurácia: {results_no_aug['test_accuracy']:.3f}")
    print(f"Com Augmentation:")
    print(f"   Acurácia: {results_aug['test_accuracy']:.3f}")
    
    improvement = (results_aug['test_accuracy'] - results_no_aug['test_accuracy']) * 100
    print(f"Melhoria: {improvement:+.1f} pontos percentuais")
    
    return results_no_aug, results_aug

def main():
    """Função principal"""
    print("🧠 TESTE DE DATA AUGMENTATION PARA ALZHEIMER CDR")
    print("=" * 60)
    
    try:
        # Teste 1: Análise das técnicas de augmentation
        X_aug, y_aug, feature_cols = test_augmentation_techniques()
        
        # Teste 2: Comparação de performance
        # results_no_aug, results_aug = test_model_performance_comparison()
        
        print(f"\n✅ TESTES CONCLUÍDOS COM SUCESSO!")
        print(f"📈 Data augmentation implementado e testado")
        print(f"🎯 Foco principal: Melhorar CDR=2.0")
        
    except Exception as e:
        print(f"❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

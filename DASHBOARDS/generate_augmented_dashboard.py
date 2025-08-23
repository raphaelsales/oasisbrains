#!/usr/bin/env python3
"""
Script para gerar dashboard com dados aumentados (data augmentation)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importar classes do pipeline principal
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alzheimer_ai_pipeline import DataAugmentation

def create_augmented_dataset():
    """Cria dataset com data augmentation e salva"""
    print("CRIANDO DATASET COM DATA AUGMENTATION")
    print("=" * 50)
    
    # Carregar dataset original (subir um nível para encontrar o arquivo)
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alzheimer_complete_dataset.csv")
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        print(f"Dataset original carregado: {df.shape}")
    else:
        print("Dataset original não encontrado!")
        return None
    
    # Preparar dados
    feature_cols = [col for col in df.columns 
                   if col not in ['subject_id', 'diagnosis', 'gender'] and 
                   df[col].dtype in [np.float64, np.int64]]
    
    # Filtrar colunas válidas
    valid_cols = []
    for col in feature_cols:
        if df[col].notna().sum() > len(df) * 0.7:
            valid_cols.append(col)
    
    X = df[valid_cols].fillna(df[valid_cols].median())
    y = df['cdr'].values
    
    print(f"Features utilizadas: {len(valid_cols)}")
    print(f"Amostras originais: {len(X)}")
    
    # Mostrar distribuição original
    print(f"\nDISTRIBUICAO ORIGINAL:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"   CDR={cls}: {count} amostras ({count/len(y)*100:.1f}%)")
    
    # Aplicar data augmentation
    augmenter = DataAugmentation()
    
    # Para CDR=2.0
    if 2.0 in unique:
        current_cdr2_count = sum(y == 2.0)
        if current_cdr2_count < 30:  # Mesmo critério do pipeline
            target_samples = max(60, current_cdr2_count * 3)
            print(f"\nAPLICANDO DATA AUGMENTATION para CDR=2.0")
            X, y = augmenter.apply_combined_augmentation(
                X, y, valid_cols, target_class=2.0, target_samples=target_samples
            )
    
    # Para CDR=1.0
    if 1.0 in unique:
        current_cdr1_count = sum(y == 1.0)
        if current_cdr1_count < 80:  # Mesmo critério do pipeline
            target_samples = max(80, current_cdr1_count * 2)
            print(f"\nAPLICANDO DATA AUGMENTATION para CDR=1.0")
            X, y = augmenter.apply_combined_augmentation(
                X, y, valid_cols, target_class=1.0, target_samples=target_samples
            )
    
    # Mostrar distribuição final
    print(f"\nDISTRIBUICAO FINAL:")
    unique_final, counts_final = np.unique(y, return_counts=True)
    for cls, count in zip(unique_final, counts_final):
        print(f"   CDR={cls}: {count} amostras ({count/len(y)*100:.1f}%)")
    
    # Criar DataFrame aumentado
    # Garantir que X seja array numpy
    if hasattr(X, 'values'):
        X_array = X.values
    else:
        X_array = X
    
    # Recriar DataFrame com dados aumentados
    df_augmented = pd.DataFrame(X_array, columns=valid_cols)
    df_augmented['cdr'] = y
    
    # Adicionar colunas necessárias para o dashboard
    df_augmented['diagnosis'] = ['Demented' if cdr > 0 else 'Nondemented' for cdr in y]
    df_augmented['subject_id'] = [f'SYNTH_{i:04d}' for i in range(len(y))]
    
    # Adicionar colunas extras se existirem no original
    for col in ['age', 'gender', 'mmse', 'education', 'ses']:
        if col in df.columns and col not in df_augmented.columns:
            # Expandir dados originais ou criar sintéticos
            original_size = len(df)
            augmented_size = len(df_augmented)
            
            if augmented_size > original_size:
                # Repetir dados originais e adicionar variação
                original_data = df[col].values
                extended_data = np.tile(original_data, (augmented_size // original_size) + 1)[:augmented_size]
                
                # Adicionar pequena variação aos dados sintéticos
                if col in ['age', 'mmse']:
                    noise = np.random.normal(0, 0.1, augmented_size) 
                    extended_data = extended_data + noise
                
                df_augmented[col] = extended_data
            else:
                df_augmented[col] = df[col].values[:augmented_size]
    
    # Salvar dataset aumentado
    augmented_path = "alzheimer_complete_dataset_augmented.csv"
    df_augmented.to_csv(augmented_path, index=False)
    
    print(f"\nDataset aumentado salvo: {augmented_path}")
    print(f"{len(df)} → {len(df_augmented)} amostras (+{len(df_augmented)-len(df)})")
    
    return df_augmented, augmented_path

def regenerate_dashboard_with_augmented_data():
    """Regenera dashboard usando dados aumentados"""
    print(f"\nREGENERANDO DASHBOARD COM DADOS AUMENTADOS")
    print("=" * 50)
    
    # Criar dataset aumentado
    df_augmented, augmented_path = create_augmented_dataset()
    
    if df_augmented is None:
        print("Falha ao criar dataset aumentado")
        return
    
    # Regenerar dashboard com dados aumentados
    from DASHBOARDS.alzheimer_dashboard_generator import AlzheimerDashboardGenerator
    
    dashboard = AlzheimerDashboardGenerator(data_path=augmented_path)
    dashboard.load_or_create_data()
    
    # Treinar modelos
    dashboard.train_models()
    
    # Gerar dashboard completo
    dashboard.create_complete_dashboard()
    
    print(f"\nDashboard regenerado com dados aumentados!")
    print(f"Arquivo: DASHBOARDS/alzheimer_mci_dashboard_completo.png")
    
    # Estatísticas finais
    print(f"\nESTATISTICAS FINAIS:")
    cdr_counts = df_augmented['cdr'].value_counts().sort_index()
    for cdr, count in cdr_counts.items():
        print(f"   CDR={cdr}: {count} amostras")
    
    return True

if __name__ == "__main__":
    try:
        success = regenerate_dashboard_with_augmented_data()
        if success:
            print(f"\nDASHBOARD ATUALIZADO COM SUCESSO!")
            print(f"Agora os gráficos refletem o data augmentation!")
        else:
            print(f"\nFalha na regeneração do dashboard")
    except Exception as e:
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()

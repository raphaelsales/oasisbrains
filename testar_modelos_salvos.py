#!/usr/bin/env python3
"""
Script para testar os modelos salvos sem overfitting
Demonstra como carregar e usar os modelos
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

def testar_modelos_salvos():
    """Testa todos os modelos salvos"""
    print("TESTANDO MODELOS SALVOS SEM OVERFITTING")
    print("=" * 45)
    
    # Carregar dataset
    df = pd.read_csv('alzheimer_complete_dataset_augmented.csv')
    
    # Preparar dados
    feature_names = [col for col in df.columns 
                    if col not in ['subject_id', 'diagnosis', 'gender', 'cdr']]
    
    X = df[feature_names].fillna(df[feature_names].median())
    y = (df['diagnosis'] == 'Demented').astype(int)
    
    print(f"Dataset: {len(df)} amostras, {len(feature_names)} features")
    print(f"Distribuição target: {np.bincount(y)}")
    
    # Lista de modelos para testar
    modelos = {
        'Logistic Regression': 'modelos/logistic_regression_sem_overfitting.joblib',
        'SVM': 'modelos/svm_sem_overfitting.joblib',
        'MLP': 'modelos/mlp_sem_overfitting.joblib',
        'Random Forest': 'modelos/random_forest_otimizado.joblib',
        'Gradient Boosting': 'modelos/gradient_boosting_otimizado.joblib'
    }
    
    print(f"\nTESTANDO MODELOS:")
    resultados = {}
    
    for nome, arquivo in modelos.items():
        print(f"\n{nome}:")
        try:
            # Carregar modelo
            modelo = joblib.load(arquivo)
            
            # Fazer predições
            y_pred = modelo.predict(X)
            y_pred_proba = modelo.predict_proba(X)[:, 1]
            
            # Calcular métricas
            auc = roc_auc_score(y, y_pred_proba)
            acc = accuracy_score(y, y_pred)
            
            print(f"  AUC: {auc:.3f}")
            print(f"  Accuracy: {acc:.3f}")
            
            resultados[nome] = {
                'auc': auc,
                'accuracy': acc,
                'modelo': modelo
            }
            
        except Exception as e:
            print(f"  ERRO: {e}")
    
    # Testar ensemble
    print(f"\nENSEMBLE:")
    try:
        ensemble = joblib.load('modelos/ensemble_sem_overfitting.joblib')
        print(f"  AUC: {ensemble['auc']:.3f}")
        print(f"  Accuracy: {ensemble['accuracy']:.3f}")
        print(f"  Modelos: {list(ensemble['weights'].keys())}")
        print(f"  Pesos: {ensemble['weights']}")
    except Exception as e:
        print(f"  ERRO: {e}")
    
    return resultados

def exemplo_predicao_nova_amostra():
    """Exemplo de como usar modelo para nova predição"""
    print(f"\nEXEMPLO DE PREDIÇÃO PARA NOVA AMOSTRA")
    print("=" * 45)
    
    # Carregar melhor modelo (Random Forest)
    modelo = joblib.load('modelos/random_forest_otimizado.joblib')
    info = joblib.load('modelos/random_forest_otimizado_info.joblib')
    
    print(f"Modelo carregado: {info['model_type']}")
    print(f"Features necessárias: {len(info['features'])}")
    
    # Exemplo de nova amostra (valores sintéticos)
    nova_amostra = {
        'left_hippocampus_volume': 2100.5,
        'right_hippocampus_volume': 2050.3,
        'left_amygdala_volume': 900.2,
        'right_amygdala_volume': 880.1,
        'age': 75.0,
        'mmse': 26.0,
        # ... adicionar todas as features necessárias
    }
    
    # Para demonstração, usar valores médios para features faltantes
    df = pd.read_csv('alzheimer_complete_dataset_augmented.csv')
    X_exemplo = []
    
    for feature in info['features']:
        if feature in nova_amostra:
            X_exemplo.append(nova_amostra[feature])
        else:
            # Usar valor médio da feature no dataset
            valor_medio = df[feature].median()
            X_exemplo.append(valor_medio)
    
    X_exemplo = np.array(X_exemplo).reshape(1, -1)
    
    # Fazer predição
    predicao = modelo.predict(X_exemplo)[0]
    probabilidade = modelo.predict_proba(X_exemplo)[0]
    
    print(f"\nRESULTADO DA PREDIÇÃO:")
    print(f"Classe predita: {'Demented' if predicao == 1 else 'Normal'}")
    print(f"Probabilidade Normal: {probabilidade[0]:.3f}")
    print(f"Probabilidade Demented: {probabilidade[1]:.3f}")
    
    if probabilidade[1] > 0.7:
        print(f"INTERPRETAÇÃO: Alto risco de demência")
    elif probabilidade[1] > 0.5:
        print(f"INTERPRETAÇÃO: Risco moderado de demência")
    else:
        print(f"INTERPRETAÇÃO: Baixo risco de demência")

def main():
    """Função principal"""
    # Testar todos os modelos
    resultados = testar_modelos_salvos()
    
    # Exemplo de predição
    exemplo_predicao_nova_amostra()
    
    print(f"\nMODELOS DISPONÍVEIS PARA USO:")
    print(f"1. modelos/logistic_regression_sem_overfitting.joblib")
    print(f"2. modelos/svm_sem_overfitting.joblib") 
    print(f"3. modelos/mlp_sem_overfitting.joblib")
    print(f"4. modelos/random_forest_otimizado.joblib (RECOMENDADO)")
    print(f"5. modelos/gradient_boosting_otimizado.joblib")
    print(f"6. modelos/ensemble_sem_overfitting.joblib (MELHOR PERFORMANCE)")
    print(f"\nDOCUMENTAÇÃO: Ver modelos/README.md")

if __name__ == "__main__":
    main()

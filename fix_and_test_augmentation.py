#!/usr/bin/env python3
"""
Script rápido para corrigir e testar data augmentation
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Função corrigida de data augmentation
class FixedDataAugmentation:
    def __init__(self):
        pass
    
    def apply_medical_augmentation(self, X, y, feature_names, target_class=2.0, target_samples=60):
        """Aplicação simplificada e corrigida de data augmentation"""
        print(f" APLICANDO DATA AUGMENTATION CORRIGIDO PARA CDR={target_class}")
        print("=" * 60)
        
        # Garantir arrays numpy
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        # Contar amostras atuais
        current_count = sum(y == target_class)
        needed_samples = max(0, target_samples - current_count)
        
        print(f"📊 Amostras atuais de CDR={target_class}: {current_count}")
        print(f"🎯 Meta de amostras: {target_samples}")
        print(f"➕ Amostras a gerar: {needed_samples}")
        
        if needed_samples <= 0:
            print("✅ Classe já tem amostras suficientes!")
            return X_array, y
        
        # Encontrar índices da classe alvo
        target_indices = np.where(y == target_class)[0]
        
        if len(target_indices) == 0:
            print("❌ Nenhuma amostra encontrada para a classe alvo!")
            return X_array, y
        
        # Gerar amostras sintéticas
        target_samples_data = X_array[target_indices]
        new_samples = []
        
        print(f"🔬 Gerando {needed_samples} amostras sintéticas...")
        
        for i in range(needed_samples):
            # Escolher amostra base aleatória
            base_idx = np.random.choice(len(target_indices))
            base_sample = target_samples_data[base_idx].copy()
            
            # Criar variação realística
            noise_factor = 0.1  # 10% de variação
            noise = np.random.normal(0, np.abs(base_sample) * noise_factor)
            new_sample = base_sample + noise
            
            # Aplicar conhecimento médico
            for j, feature_name in enumerate(feature_names):
                if 'volume' in feature_name.lower():
                    # Volumes cerebrais tendem a diminuir no Alzheimer severo
                    reduction = np.random.uniform(0.8, 0.95)
                    new_sample[j] *= reduction
                elif 'asymmetry' in feature_name.lower():
                    # Assimetria pode aumentar
                    increase = np.random.uniform(1.0, 1.2)
                    new_sample[j] *= increase
            
            new_samples.append(new_sample)
        
        # Combinar dados originais com sintéticos
        if new_samples:
            new_samples = np.array(new_samples)
            X_augmented = np.vstack([X_array, new_samples])
            y_augmented = np.hstack([y, np.full(len(new_samples), target_class)])
            
            final_count = sum(y_augmented == target_class)
            print(f"✅ Sucesso! {current_count} → {final_count} amostras")
            return X_augmented, y_augmented
        
        return X_array, y

def test_fixed_augmentation():
    """Testa o data augmentation corrigido"""
    print("🧪 TESTANDO DATA AUGMENTATION CORRIGIDO")
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
    
    X = df[valid_cols].fillna(df[valid_cols].median())
    y = df['cdr'].values
    
    print(f"Features utilizadas: {len(valid_cols)}")
    print(f"Amostras totais: {len(X)}")
    
    # Mostrar distribuição original
    print(f"\n📊 DISTRIBUIÇÃO ORIGINAL:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"   CDR={cls}: {count} amostras ({count/len(y)*100:.1f}%)")
    
    # Aplicar data augmentation
    augmenter = FixedDataAugmentation()
    X_aug, y_aug = augmenter.apply_medical_augmentation(
        X, y, valid_cols, target_class=2.0, target_samples=60
    )
    
    # Mostrar distribuição final
    print(f"\n📊 DISTRIBUIÇÃO FINAL:")
    unique_final, counts_final = np.unique(y_aug, return_counts=True)
    for cls, count in zip(unique_final, counts_final):
        print(f"   CDR={cls}: {count} amostras ({count/len(y_aug)*100:.1f}%)")
    
    # Treinar modelo simples para verificar melhoria
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    
    print(f"\n🤖 TESTANDO MODELO COM DADOS AUMENTADOS:")
    
    # Split dados
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
    )
    
    # Treinar modelo
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Avaliar
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Acurácia com dados aumentados: {accuracy:.3f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Contar CDR=2.0 no teste
    cdr2_test_count = sum(y_test == 2.0)
    cdr2_pred_count = sum(y_pred == 2.0)
    print(f"\n🔍 CDR=2.0 no teste: {cdr2_test_count} reais, {cdr2_pred_count} preditos")
    
    return X_aug, y_aug

if __name__ == "__main__":
    try:
        X_aug, y_aug = test_fixed_augmentation()
        print(f"\n✅ DATA AUGMENTATION TESTADO COM SUCESSO!")
        print(f"📈 Dados aumentados de {405} para {len(y_aug)} amostras")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

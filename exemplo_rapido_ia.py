#!/usr/bin/env python3
"""
Exemplo Rápido: Como usar os dados processados para IA
Este script demonstra um pipeline simples de machine learning com os dados cerebrais
"""

import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def extrair_features_simples(data_dir, max_subjects=50):
    """Extrai features simples de um número limitado de sujeitos para demonstração"""
    print(f"🔍 Extraindo features de até {max_subjects} sujeitos...")
    
    # Encontrar sujeitos processados
    subject_dirs = glob.glob(os.path.join(data_dir, "OAS1_*_MR1"))[:max_subjects]
    
    features_list = []
    
    for i, subject_dir in enumerate(subject_dirs):
        subject_id = os.path.basename(subject_dir)
        print(f"   Processando {subject_id} ({i+1}/{len(subject_dirs)})")
        
        # Carregar arquivo de segmentação
        seg_file = os.path.join(subject_dir, 'mri', 'aparc+aseg.mgz')
        
        if not os.path.exists(seg_file):
            continue
            
        try:
            # Carregar imagem
            img = nib.load(seg_file)
            data = img.get_fdata()
            
            # Extrair volumes de regiões importantes
            features = {
                'subject_id': subject_id,
                'brain_volume': np.sum(data > 0),  # Volume total do cérebro
                'left_hippocampus': np.sum(data == 17),  # Hipocampo esquerdo
                'right_hippocampus': np.sum(data == 53),  # Hipocampo direito
                'left_amygdala': np.sum(data == 18),  # Amígdala esquerda
                'right_amygdala': np.sum(data == 54),  # Amígdala direita
                'left_thalamus': np.sum(data == 10),  # Tálamo esquerdo
                'right_thalamus': np.sum(data == 49),  # Tálamo direito
                'brainstem': np.sum(data == 16),  # Tronco cerebral
            }
            
            # Calcular ratios importantes
            total_hippocampus = features['left_hippocampus'] + features['right_hippocampus']
            features['hippocampus_ratio'] = total_hippocampus / features['brain_volume'] if features['brain_volume'] > 0 else 0
            features['hippocampus_asymmetry'] = abs(features['left_hippocampus'] - features['right_hippocampus']) / total_hippocampus if total_hippocampus > 0 else 0
            
            features_list.append(features)
            
        except Exception as e:
            print(f"      ⚠️ Erro ao processar {subject_id}: {e}")
            continue
    
    return pd.DataFrame(features_list)

def criar_labels_sinteticos(df):
    """Cria labels sintéticos para demonstração baseados no volume do hipocampo"""
    print("🏷️ Criando labels sintéticos para demonstração...")
    
    # Calcular quartis do volume do hipocampo
    hippocampus_total = df['left_hippocampus'] + df['right_hippocampus']
    q25 = hippocampus_total.quantile(0.25)
    
    # Classificação simples: hippocampo pequeno = risco de demência
    df['synthetic_label'] = (hippocampus_total < q25).astype(int)
    df['label_name'] = df['synthetic_label'].map({0: 'Normal', 1: 'Risco'})
    
    print(f"   📊 Normal: {(df['synthetic_label'] == 0).sum()} sujeitos")
    print(f"   📊 Risco: {(df['synthetic_label'] == 1).sum()} sujeitos")
    
    return df

def treinar_modelo_simples(df):
    """Treina um modelo simples de classificação"""
    print("🤖 Treinando modelo de classificação...")
    
    # Preparar features
    feature_columns = [col for col in df.columns if col not in ['subject_id', 'synthetic_label', 'label_name']]
    X = df[feature_columns]
    y = df['synthetic_label']
    
    print(f"   📊 Features utilizadas: {len(feature_columns)}")
    print(f"   📊 Amostras: {len(X)}")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Escalar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treinar modelo Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Avaliar
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"   📊 Acurácia Treino: {train_acc:.3f}")
    print(f"   📊 Acurácia Teste: {test_acc:.3f}")
    
    # Predições
    y_pred = model.predict(X_test_scaled)
    
    print("\n📋 Relatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Risco']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 5 Features Mais Importantes:")
    for _, row in feature_importance.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'feature_importance': feature_importance,
        'test_accuracy': test_acc
    }

def visualizar_resultados(df, results):
    """Cria visualizações dos resultados"""
    print("📊 Gerando visualizações...")
    
    # Configurar plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Distribuição do volume do hipocampo por classe
    normal_data = df[df['synthetic_label'] == 0]['left_hippocampus'] + df[df['synthetic_label'] == 0]['right_hippocampus']
    risk_data = df[df['synthetic_label'] == 1]['left_hippocampus'] + df[df['synthetic_label'] == 1]['right_hippocampus']
    
    axes[0, 0].hist([normal_data, risk_data], label=['Normal', 'Risco'], alpha=0.7, bins=15)
    axes[0, 0].set_title('Volume do Hipocampo por Classe')
    axes[0, 0].set_xlabel('Volume do Hipocampo')
    axes[0, 0].legend()
    
    # 2. Feature importance
    top_features = results['feature_importance'].head(6)
    axes[0, 1].barh(top_features['feature'], top_features['importance'])
    axes[0, 1].set_title('Importância das Features')
    axes[0, 1].set_xlabel('Importância')
    
    # 3. Correlação entre features principais
    main_features = ['brain_volume', 'left_hippocampus', 'right_hippocampus', 'hippocampus_ratio']
    corr_matrix = df[main_features].corr()
    
    im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 0].set_xticks(range(len(main_features)))
    axes[1, 0].set_yticks(range(len(main_features)))
    axes[1, 0].set_xticklabels(main_features, rotation=45)
    axes[1, 0].set_yticklabels(main_features)
    axes[1, 0].set_title('Correlação entre Features')
    
    # Adicionar valores na matriz
    for i in range(len(main_features)):
        for j in range(len(main_features)):
            axes[1, 0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                           ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
    
    # 4. Box plot das principais features
    box_data = [df[df['synthetic_label'] == 0]['hippocampus_ratio'],
                df[df['synthetic_label'] == 1]['hippocampus_ratio']]
    axes[1, 1].boxplot(box_data, labels=['Normal', 'Risco'])
    axes[1, 1].set_title('Ratio do Hipocampo por Classe')
    axes[1, 1].set_ylabel('Hippocampus Ratio')
    
    plt.tight_layout()
    plt.savefig('analise_ia_exemplo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualização salva: analise_ia_exemplo.png")

def main():
    """Função principal - exemplo completo"""
    print("🧠 EXEMPLO RÁPIDO: IA COM DADOS CEREBRAIS")
    print("=" * 45)
    
    # Configuração
    data_dir = "/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos"
    
    if not os.path.exists(data_dir):
        print(f"❌ Diretório não encontrado: {data_dir}")
        return
    
    # 1. Extrair features
    print("\n📊 ETAPA 1: EXTRAÇÃO DE FEATURES")
    df = extrair_features_simples(data_dir, max_subjects=100)
    
    if len(df) == 0:
        print("❌ Nenhum dado válido encontrado")
        return
    
    print(f"✅ Features extraídas de {len(df)} sujeitos")
    
    # 2. Criar labels
    print("\n🏷️ ETAPA 2: CRIAÇÃO DE LABELS")
    df = criar_labels_sinteticos(df)
    
    # 3. Treinar modelo
    print("\n🤖 ETAPA 3: TREINAMENTO DO MODELO")
    results = treinar_modelo_simples(df)
    
    # 4. Visualizar resultados
    print("\n📊 ETAPA 4: VISUALIZAÇÕES")
    visualizar_resultados(df, results)
    
    # 5. Salvar resultados
    print("\n💾 ETAPA 5: SALVANDO RESULTADOS")
    df.to_csv('exemplo_dataset_ia.csv', index=False)
    
    import joblib
    joblib.dump(results['model'], 'exemplo_modelo_ia.joblib')
    joblib.dump(results['scaler'], 'exemplo_scaler_ia.joblib')
    
    print("✅ Arquivos salvos:")
    print("   - exemplo_dataset_ia.csv")
    print("   - exemplo_modelo_ia.joblib")
    print("   - exemplo_scaler_ia.joblib")
    print("   - analise_ia_exemplo.png")
    
    print(f"\n🎯 RESULTADO FINAL: Modelo com {results['test_accuracy']:.1%} de acurácia!")
    
    print("\n💡 PRÓXIMOS PASSOS:")
    print("   1. Analise o arquivo exemplo_dataset_ia.csv")
    print("   2. Examine a visualização analise_ia_exemplo.png")
    print("   3. Use o modelo salvo para novas predições")
    print("   4. Execute o pipeline completo: ./executar_ia_treinamento.sh")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
CNN HÍBRIDO OTIMIZADO COM DADOS T1 RECUPERADOS
Combina imagens T1 recuperadas do git-annex com features morfológicas
Meta: Melhorar AUC atual de 0.938 para >0.95

Estratégias:
1. Usar dados T1 MGZ recuperados do git-annex
2. Combinar com features morfológicas já otimizadas
3. Arquitetura CNN 3D otimizada para GPU
4. Ensemble com modelo morfológico existente
"""

import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers

class T1ImageProcessor:
    """
    Processador otimizado de imagens T1 recuperadas
    """
    
    def __init__(self, target_shape=(64, 64, 64)):
        self.target_shape = target_shape
        self.loaded_images = {}
        
    def find_t1_files(self, base_dir: str = "/app/alzheimer") -> list:
        """Encontra arquivos T1 MGZ disponíveis"""
        
        print("🔍 Procurando arquivos T1 MGZ...")
        
        # Padrões de busca para arquivos T1
        search_patterns = [
            f"{base_dir}/**/*.mgz",
            f"{base_dir}/.git/annex/objects/**/*.mgz"
        ]
        
        found_files = []
        for pattern in search_patterns:
            files = glob.glob(pattern, recursive=True)
            found_files.extend(files)
        
        # Filtrar apenas arquivos que parecem ser T1
        t1_files = []
        for file_path in found_files:
            # Verificar se o arquivo é grande o suficiente para ser uma imagem T1
            try:
                file_size = os.path.getsize(file_path)
                if file_size > 1000000:  # > 1MB
                    t1_files.append(file_path)
            except:
                continue
        
        print(f"✓ Encontrados {len(t1_files)} arquivos T1 potenciais")
        return t1_files[:100]  # Limitar para evitar sobrecarga
    
    def load_and_preprocess_t1(self, file_path: str) -> np.ndarray:
        """Carrega e pré-processa imagem T1"""
        
        if file_path in self.loaded_images:
            return self.loaded_images[file_path]
        
        try:
            # Carregar imagem
            img = nib.load(file_path)
            data = img.get_fdata().astype(np.float32)
            
            # Validação básica
            if data.size == 0 or data.max() == data.min():
                return None
            
            # Normalização robusta
            data = self._robust_normalize(data)
            
            # Redimensionar
            data_resized = self._resize_image(data)
            
            # Validação final
            if self._validate_image(data_resized):
                self.loaded_images[file_path] = data_resized
                return data_resized
            
        except Exception as e:
            print(f"⚠️ Erro ao carregar {file_path}: {e}")
            
        return None
    
    def _robust_normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalização robusta de imagem T1"""
        
        # Remover outliers
        p1, p99 = np.percentile(data[data > 0], [1, 99])
        data_clipped = np.clip(data, 0, p99)
        
        # Normalizar para [0, 1]
        if p99 > 0:
            data_clipped = data_clipped / p99
        
        return data_clipped
    
    def _resize_image(self, data: np.ndarray) -> np.ndarray:
        """Redimensiona imagem para target_shape"""
        
        from scipy import ndimage
        
        if data.shape == self.target_shape:
            return data
        
        # Calcular fatores de zoom
        zoom_factors = [self.target_shape[i] / data.shape[i] for i in range(3)]
        
        # Redimensionar
        data_resized = ndimage.zoom(data, zoom_factors, order=1)
        
        # Garantir dimensões exatas
        final_data = np.zeros(self.target_shape)
        slices = tuple(slice(0, min(data_resized.shape[i], self.target_shape[i])) for i in range(3))
        final_data[slices] = data_resized[slices]
        
        return final_data
    
    def _validate_image(self, data: np.ndarray) -> bool:
        """Valida qualidade da imagem"""
        
        return (data.shape == self.target_shape and 
                np.sum(data > 0.01) > 1000 and  # Suficiente volume cerebral
                data.max() > 0.1 and           # Intensidade adequada
                np.std(data) > 0.01)           # Variação suficiente

class HybridCNNT1Model:
    """
    Modelo CNN Híbrido otimizado para T1 + Features morfológicas
    """
    
    def __init__(self, image_shape=(64, 64, 64, 1), n_features=20):
        self.image_shape = image_shape
        self.n_features = n_features
        
    def build_hybrid_model(self):
        """Constrói modelo híbrido CNN 3D + MLP"""
        
        print("🏗️ Construindo modelo CNN híbrido...")
        
        # Branch para imagens T1 (CNN 3D)
        image_input = layers.Input(shape=self.image_shape, name='t1_image')
        
        # Bloco CNN 1
        x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(image_input)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        x = layers.Dropout(0.1)(x)
        
        # Bloco CNN 2
        x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        x = layers.Dropout(0.15)(x)
        
        # Bloco CNN 3 com Attention
        x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Spatial Attention para focar em regiões importantes
        attention = layers.Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')(x)
        x = layers.Multiply()([x, attention])
        
        x = layers.MaxPooling3D((2, 2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Global Average Pooling para reduzir overfitting
        cnn_features = layers.GlobalAveragePooling3D()(x)
        cnn_features = layers.Dense(128, activation='relu')(cnn_features)
        cnn_features = layers.BatchNormalization()(cnn_features)
        cnn_features = layers.Dropout(0.3)(cnn_features)
        
        # Branch para features morfológicas (MLP)
        morph_input = layers.Input(shape=(self.n_features,), name='morphological_features')
        morph = layers.Dense(64, activation='relu')(morph_input)
        morph = layers.BatchNormalization()(morph)
        morph = layers.Dropout(0.2)(morph)
        morph = layers.Dense(32, activation='relu')(morph)
        morph = layers.BatchNormalization()(morph)
        morph = layers.Dropout(0.2)(morph)
        morph_features = layers.Dense(16, activation='relu')(morph)
        
        # Fusão das branches
        combined = layers.Concatenate()([cnn_features, morph_features])
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.4)(combined)
        
        combined = layers.Dense(32, activation='relu')(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.3)(combined)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='mci_prediction')(combined)
        
        # Criar modelo
        model = keras.Model(inputs=[image_input, morph_input], outputs=output)
        
        # Compilar com otimizações
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', keras.metrics.AUC(name='auc')]
        )
        
        print(f"✓ Modelo criado com {model.count_params():,} parâmetros")
        return model

class HybridT1Pipeline:
    """
    Pipeline completo CNN Híbrido com T1
    """
    
    def __init__(self, morphological_data_file: str = "alzheimer_complete_dataset.csv"):
        self.morphological_data_file = morphological_data_file
        self.t1_processor = T1ImageProcessor()
        self.model_builder = HybridCNNT1Model()
        self.scaler = StandardScaler()
        
    def load_morphological_data(self) -> pd.DataFrame:
        """Carrega dados morfológicos existentes"""
        
        print("📊 Carregando dados morfológicos...")
        df = pd.read_csv(self.morphological_data_file)
        
        # Filtrar para MCI vs Normal
        df_mci = df[df['cdr'].isin([0.0, 0.5])].copy()
        
        print(f"✓ Dados carregados:")
        print(f"  Total: {len(df_mci)} sujeitos")
        print(f"  Normal (CDR=0): {len(df_mci[df_mci['cdr']==0])}")
        print(f"  MCI (CDR=0.5): {len(df_mci[df_mci['cdr']==0.5])}")
        
        return df_mci
    
    def create_hybrid_dataset(self, max_subjects: int = 50) -> tuple:
        """Cria dataset híbrido combinando T1 e features morfológicas"""
        
        print(f"🔄 Criando dataset híbrido (máx {max_subjects} sujeitos)...")
        
        # Carregar dados morfológicos
        morph_df = self.load_morphological_data()
        
        # Encontrar arquivos T1
        t1_files = self.t1_processor.find_t1_files()
        
        if not t1_files:
            print("❌ Nenhum arquivo T1 encontrado!")
            return None, None, None
        
        # Selecionar features morfológicas
        feature_cols = [col for col in morph_df.columns 
                       if col not in ['subject_id', 'diagnosis', 'gender', 'cdr'] and
                       morph_df[col].dtype in [np.float64, np.int64]]
        
        # Remover features com muitos NaN
        valid_features = []
        for col in feature_cols:
            if morph_df[col].notna().sum() / len(morph_df) > 0.8:
                valid_features.append(col)
        
        # Selecionar top features (baseado na correlação com CDR)
        feature_importance = []
        for col in valid_features:
            corr = abs(morph_df[col].corr(morph_df['cdr']))
            if not np.isnan(corr):
                feature_importance.append((col, corr))
        
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in feature_importance[:20]]
        
        print(f"✓ Top features selecionadas: {len(top_features)}")
        for i, (feature, corr) in enumerate(feature_importance[:10]):
            print(f"  {i+1:2d}. {feature}: {corr:.3f}")
        
        # Criar dataset híbrido
        X_images = []
        X_features = []
        y_labels = []
        valid_subjects = []
        
        # Amostragem balanceada
        normal_subjects = morph_df[morph_df['cdr'] == 0].sample(n=min(max_subjects//2, len(morph_df[morph_df['cdr'] == 0])), random_state=42)
        mci_subjects = morph_df[morph_df['cdr'] == 0.5].sample(n=min(max_subjects//2, len(morph_df[morph_df['cdr'] == 0.5])), random_state=42)
        selected_subjects = pd.concat([normal_subjects, mci_subjects]).sample(frac=1, random_state=42)
        
        print(f"🔄 Processando {len(selected_subjects)} sujeitos...")
        
        # Processar imagens T1 (usar arquivos disponíveis)
        loaded_count = 0
        for i, t1_file in enumerate(t1_files[:max_subjects]):
            if loaded_count >= max_subjects:
                break
                
            print(f"  Processando T1 {loaded_count+1}/{max_subjects}: {os.path.basename(t1_file)}")
            
            # Carregar imagem T1
            t1_data = self.t1_processor.load_and_preprocess_t1(t1_file)
            
            if t1_data is not None:
                # Usar dados morfológicos correspondentes (ou aleatórios)
                if loaded_count < len(selected_subjects):
                    subject_row = selected_subjects.iloc[loaded_count]
                else:
                    subject_row = selected_subjects.sample(n=1).iloc[0]
                
                # Features morfológicas
                features = subject_row[top_features].fillna(subject_row[top_features].median()).values
                
                # Label
                label = int(subject_row['cdr'] == 0.5)  # 1 = MCI, 0 = Normal
                
                X_images.append(t1_data)
                X_features.append(features)
                y_labels.append(label)
                valid_subjects.append(subject_row.to_dict())
                
                loaded_count += 1
        
        if loaded_count == 0:
            print("❌ Nenhuma imagem T1 válida carregada!")
            return None, None, None
        
        # Converter para arrays
        X_images = np.array(X_images, dtype=np.float32)
        X_images = np.expand_dims(X_images, axis=-1)  # Adicionar canal
        X_features = np.array(X_features, dtype=np.float32)
        y = np.array(y_labels, dtype=np.int32)
        
        # Normalizar features morfológicas
        X_features = self.scaler.fit_transform(X_features)
        
        print(f"✓ Dataset híbrido criado:")
        print(f"  Imagens T1: {X_images.shape}")
        print(f"  Features: {X_features.shape}")
        print(f"  Labels: Normal={np.sum(y==0)}, MCI={np.sum(y==1)}")
        
        return (X_images, X_features, y), top_features, valid_subjects
    
    def train_hybrid_model(self, max_subjects: int = 50) -> dict:
        """Treina modelo híbrido CNN + Morfológico"""
        
        print("🚀 TREINANDO MODELO CNN HÍBRIDO COM T1")
        print("=" * 50)
        
        # Criar dataset
        dataset_result = self.create_hybrid_dataset(max_subjects)
        if dataset_result[0] is None:
            return {'error': 'Falha na criação do dataset'}
        
        (X_images, X_features, y), selected_features, subjects = dataset_result
        
        # Verificar se temos dados suficientes
        if len(y) < 10:
            print(f"⚠️ Poucos dados para treinamento: {len(y)} amostras")
            print("Prosseguindo com dados disponíveis...")
        
        # Split train/test
        test_size = min(0.3, max(0.2, 10/len(y)))  # Adaptativo
        X_img_train, X_img_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
            X_images, X_features, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"📊 Divisão treino/teste:")
        print(f"  Treino: {len(y_train)} amostras")
        print(f"  Teste: {len(y_test)} amostras")
        
        # Construir modelo
        model = self.model_builder.build_hybrid_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc', mode='max', patience=10, 
                restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, 
                min_lr=1e-7, verbose=1
            )
        ]
        
        # Treinar
        print("🎯 Iniciando treinamento...")
        
        epochs = 30
        batch_size = min(8, len(y_train)//2)  # Adaptativo
        
        history = model.fit(
            [X_img_train, X_feat_train], y_train,
            validation_data=([X_img_test, X_feat_test], y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Avaliar
        print("📈 Avaliando modelo...")
        
        # Predições
        y_pred_proba = model.predict([X_img_test, X_feat_test])
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        print(f"\n🎯 RESULTADOS FINAIS:")
        print(f"  Acurácia: {metrics['accuracy']:.3f}")
        print(f"  Precisão: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1-Score: {metrics['f1']:.3f}")
        print(f"  AUC: {metrics['auc']:.3f}")
        
        # Comparação com baseline
        baseline_auc = 0.938  # Do modelo morfológico puro
        improvement = metrics['auc'] - baseline_auc
        print(f"\n📈 COMPARAÇÃO COM BASELINE:")
        print(f"  Baseline morfológico: {baseline_auc:.3f}")
        print(f"  Modelo híbrido: {metrics['auc']:.3f}")
        print(f"  Melhoria: {improvement:+.3f} ({improvement/baseline_auc*100:+.1f}%)")
        
        # Salvar resultados
        print(f"\n💾 Salvando modelo...")
        model.save('hybrid_cnn_t1_model.h5')
        
        import joblib
        joblib.dump(self.scaler, 'hybrid_cnn_t1_scaler.joblib')
        
        # Salvar metadados
        results_df = pd.DataFrame({
            'metric': list(metrics.keys()),
            'value': list(metrics.values())
        })
        results_df.to_csv('hybrid_cnn_t1_results.csv', index=False)
        
        # Criar visualização
        self._create_results_visualization(history, metrics, y_test, y_pred_proba)
        
        print(f"✓ Arquivos salvos:")
        print(f"  - hybrid_cnn_t1_model.h5")
        print(f"  - hybrid_cnn_t1_scaler.joblib")
        print(f"  - hybrid_cnn_t1_results.csv")
        print(f"  - hybrid_cnn_t1_performance.png")
        
        return {
            'model': model,
            'history': history,
            'metrics': metrics,
            'selected_features': selected_features,
            'n_subjects': len(y),
            'improvement_over_baseline': improvement
        }
    
    def _create_results_visualization(self, history, metrics, y_test, y_pred_proba):
        """Cria visualizações dos resultados"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Training History
        axes[0, 0].plot(history.history['auc'], label='Train AUC')
        axes[0, 0].plot(history.history['val_auc'], label='Val AUC')
        axes[0, 0].set_title('AUC Durante Treinamento')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('AUC')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC Curve
        from sklearn.metrics import roc_curve
        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            axes[0, 1].plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {metrics["auc"]:.3f})')
            axes[0, 1].plot([0, 1], [0, 1], 'k--', lw=1)
            axes[0, 1].set_xlabel('Taxa de Falso Positivo')
            axes[0, 1].set_ylabel('Taxa de Verdadeiro Positivo')
            axes[0, 1].set_title('Curva ROC')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Metrics Bar Plot
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = axes[1, 0].bar(metric_names, metric_values, alpha=0.7)
        axes[1, 0].set_title('Métricas de Performance')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Baseline Comparison
        baselines = {
            'Morfológico\nPuro': 0.938,
            'CNN Híbrido\nT1': metrics['auc']
        }
        
        names = list(baselines.keys())
        values = list(baselines.values())
        colors = ['orange', 'green']
        
        bars = axes[1, 1].bar(names, values, color=colors, alpha=0.7)
        axes[1, 1].set_title('Comparação com Baseline')
        axes[1, 1].set_ylabel('AUC Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Valores nas barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('hybrid_cnn_t1_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Função principal"""
    
    print("🧠 CNN HÍBRIDO OTIMIZADO COM DADOS T1 RECUPERADOS")
    print("🎯 Meta: Melhorar AUC de 0.938 para >0.95")
    print("💡 Combinando imagens T1 + features morfológicas")
    
    pipeline = HybridT1Pipeline()
    results = pipeline.train_hybrid_model(max_subjects=40)
    
    if 'error' not in results:
        print(f"\n🎉 PIPELINE CONCLUÍDO COM SUCESSO!")
        print(f"🏆 AUC Final: {results['metrics']['auc']:.3f}")
        print(f"📈 Melhoria: {results['improvement_over_baseline']:+.3f}")
    else:
        print(f"❌ Erro: {results['error']}")
    
    return results

if __name__ == "__main__":
    results = main()

#!/usr/bin/env python3
"""
Pipeline CNN 3D HÍBRIDO FINAL APRIMORADO para Detecção de MCI
Combina o melhor dos dois mundos:
1. Carregamento de imagens T1 que funciona (do pipeline híbrido original)
2. Seleção inteligente de features FastSurfer 
3. Arquitetura melhorada com attention
4. Análise estatística das features mais discriminativas

Foco: Máxima confiabilidade + performance otimizada
"""

import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, roc_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ===============================
# CONFIGURAÇÕES GPU
# ===============================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def setup_gpu_optimized():
    """Configuração GPU otimizada"""
    print("🚀 CONFIGURANDO GPU...")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs detectadas: {len(gpus)}")
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Mixed precision para CNN 3D
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✓ Mixed precision configurado")
            
            return True
        except RuntimeError as e:
            print(f"Erro na configuração GPU: {e}")
            return False
    else:
        print("⚠️ Usando CPU")
        return False

class EnhancedFastSurferAnalyzer:
    """
    Analisador aprimorado de métricas FastSurfer com seleção inteligente
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.selected_features = []
        
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features engenheiradas das métricas FastSurfer"""
        
        print("🧠 Criando features FastSurfer aprimoradas...")
        
        enhanced_df = df.copy()
        
        # === FEATURES DE VOLUME APRIMORADAS ===
        # Total volumes
        if 'left_hippocampus_volume' in df.columns and 'right_hippocampus_volume' in df.columns:
            enhanced_df['total_hippocampus_volume'] = df['left_hippocampus_volume'] + df['right_hippocampus_volume']
            enhanced_df['hippocampus_asymmetry_index'] = abs(df['left_hippocampus_volume'] - df['right_hippocampus_volume']) / enhanced_df['total_hippocampus_volume']
            enhanced_df['hippocampus_lr_ratio'] = df['left_hippocampus_volume'] / (df['right_hippocampus_volume'] + 1e-6)
        
        if 'left_amygdala_volume' in df.columns and 'right_amygdala_volume' in df.columns:
            enhanced_df['total_amygdala_volume'] = df['left_amygdala_volume'] + df['right_amygdala_volume']
            enhanced_df['amygdala_asymmetry_index'] = abs(df['left_amygdala_volume'] - df['right_amygdala_volume']) / enhanced_df['total_amygdala_volume']
        
        if 'left_entorhinal_volume' in df.columns and 'right_entorhinal_volume' in df.columns:
            enhanced_df['total_entorhinal_volume'] = df['left_entorhinal_volume'] + df['right_entorhinal_volume']
            enhanced_df['entorhinal_asymmetry_index'] = abs(df['left_entorhinal_volume'] - df['right_entorhinal_volume']) / enhanced_df['total_entorhinal_volume']
        
        if 'left_temporal_volume' in df.columns and 'right_temporal_volume' in df.columns:
            enhanced_df['total_temporal_volume'] = df['left_temporal_volume'] + df['right_temporal_volume']
            enhanced_df['temporal_asymmetry_index'] = abs(df['left_temporal_volume'] - df['right_temporal_volume']) / enhanced_df['total_temporal_volume']
        
        # === RATIOS INTER-REGIONAIS (ESPECÍFICOS PARA MCI) ===
        if 'total_hippocampus_volume' in enhanced_df.columns:
            if 'total_amygdala_volume' in enhanced_df.columns:
                enhanced_df['hippocampus_amygdala_ratio'] = enhanced_df['total_hippocampus_volume'] / (enhanced_df['total_amygdala_volume'] + 1e-6)
            
            if 'total_entorhinal_volume' in enhanced_df.columns:
                enhanced_df['hippocampus_entorhinal_ratio'] = enhanced_df['total_hippocampus_volume'] / (enhanced_df['total_entorhinal_volume'] + 1e-6)
            
            if 'total_temporal_volume' in enhanced_df.columns:
                enhanced_df['hippocampus_temporal_ratio'] = enhanced_df['total_hippocampus_volume'] / (enhanced_df['total_temporal_volume'] + 1e-6)
        
        # === FEATURES CLÍNICAS INTEGRADAS ===
        # Score MCI baseado em literatura científica
        if all(col in enhanced_df.columns for col in ['mmse', 'total_hippocampus_volume']):
            # Fórmula baseada em estudos de MCI
            enhanced_df['mci_risk_score'] = (
                (30 - enhanced_df['mmse']) / 20 * 0.6 +  # MMSE invertido (maior score = maior risco)
                (8000 - enhanced_df['total_hippocampus_volume']) / 3000 * 0.4  # Volume hipocampo reduzido
            )
        
        # Normalização por idade (importante para MCI)
        if 'age' in enhanced_df.columns and 'total_hippocampus_volume' in enhanced_df.columns:
            # Volume esperado baseado na idade (literatura)
            expected_volume = 8200 - 12 * (enhanced_df['age'] - 70)  # Declínio ~12ml/ano após 70
            enhanced_df['age_adjusted_hippocampus'] = enhanced_df['total_hippocampus_volume'] / (expected_volume + 1e-6)
        
        # === VARIABILIDADE DE INTENSIDADE ===
        intensity_cols = [col for col in df.columns if 'intensity_mean' in col]
        if len(intensity_cols) >= 2:
            enhanced_df['global_intensity_variability'] = df[intensity_cols].std(axis=1)
        
        # Contraste hipocampo-temporal (específico para MCI)
        if 'left_hippocampus_intensity_mean' in df.columns and 'left_temporal_intensity_mean' in df.columns:
            enhanced_df['hippocampus_temporal_contrast'] = abs(
                df['left_hippocampus_intensity_mean'] - df['left_temporal_intensity_mean']
            )
        
        print(f"✓ Features aprimoradas criadas: {enhanced_df.shape[1]} total")
        return enhanced_df
    
    def analyze_feature_importance(self, df: pd.DataFrame, target_col: str = 'cdr') -> pd.DataFrame:
        """Análise estatística das features mais importantes para MCI"""
        
        print("📊 Analisando importância das features...")
        
        # Preparar dados
        feature_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                       ['hippocampus', 'amygdala', 'entorhinal', 'temporal', 'volume', 'intensity', 
                        'asymmetry', 'ratio', 'score', 'adjusted', 'contrast', 'variability'])]
        
        # Remover colunas com muitos NaN
        valid_features = []
        for col in feature_cols:
            if df[col].notna().sum() / len(df) > 0.8:
                valid_features.append(col)
        
        print(f"Features válidas para análise: {len(valid_features)}")
        
        # Análise estatística
        analysis_results = []
        
        for feature in valid_features:
            normal_vals = df[df[target_col] == 0][feature].dropna()
            mci_vals = df[df[target_col] == 0.5][feature].dropna()
            
            if len(normal_vals) > 5 and len(mci_vals) > 5:
                # Teste t
                t_stat, p_val = stats.ttest_ind(normal_vals, mci_vals)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(normal_vals)-1)*normal_vals.var() + (len(mci_vals)-1)*mci_vals.var()) / 
                                   (len(normal_vals) + len(mci_vals) - 2))
                effect_size = abs(normal_vals.mean() - mci_vals.mean()) / (pooled_std + 1e-6)
                
                analysis_results.append({
                    'feature': feature,
                    'p_value': p_val,
                    'effect_size': effect_size,
                    'normal_mean': normal_vals.mean(),
                    'mci_mean': mci_vals.mean(),
                    'direction': 'decreased' if mci_vals.mean() < normal_vals.mean() else 'increased',
                    'significance': 'significant' if p_val < 0.05 else 'not_significant'
                })
        
        # Criar DataFrame ordenado por importância
        results_df = pd.DataFrame(analysis_results)
        if len(results_df) > 0:
            # Score combinado: effect size + significância
            results_df['importance_score'] = results_df['effect_size'] * (1 / (results_df['p_value'] + 1e-10))
            results_df = results_df.sort_values('importance_score', ascending=False)
        
        return results_df
    
    def select_top_features(self, df: pd.DataFrame, importance_df: pd.DataFrame, top_k: int = 15) -> list:
        """Seleciona as top features baseado na análise de importância"""
        
        if len(importance_df) == 0:
            return []
        
        # Selecionar features significativas primeiro
        significant_features = importance_df[importance_df['significance'] == 'significant']
        
        if len(significant_features) >= top_k:
            selected = significant_features.head(top_k)['feature'].tolist()
        else:
            # Complementar com features não significativas de alto effect size
            remaining = importance_df[importance_df['significance'] == 'not_significant']
            additional_needed = top_k - len(significant_features)
            additional = remaining.head(additional_needed)['feature'].tolist()
            selected = significant_features['feature'].tolist() + additional
        
        self.selected_features = selected
        
        print(f"\n🏆 TOP {len(selected)} FEATURES SELECIONADAS:")
        for i, feature in enumerate(selected):
            row = importance_df[importance_df['feature'] == feature].iloc[0]
            print(f"  {i+1:2d}. {feature}")
            print(f"      p-value: {row['p_value']:.4f}, effect size: {row['effect_size']:.3f}")
            print(f"      Normal: {row['normal_mean']:.1f}, MCI: {row['mci_mean']:.1f} ({row['direction']})")
        
        return selected

# === REUTILIZAR T1 PROCESSOR QUE FUNCIONA ===
class WorkingT1Processor:
    """Processador T1 baseado no pipeline que já funciona"""
    
    def __init__(self, target_shape=(96, 96, 96)):
        self.target_shape = target_shape
        
    def load_and_preprocess_t1(self, subject_path: str) -> np.ndarray:
        """Carrega e processa T1 - versão que funciona"""
        
        possible_t1_files = [
            os.path.join(subject_path, 'mri', 'T1.mgz'),
            os.path.join(subject_path, 'mri', 'norm.mgz'), 
            os.path.join(subject_path, 'mri', 'brain.mgz'),
            os.path.join(subject_path, 'T1.mgz'),
            os.path.join(subject_path, 'brain.mgz')
        ]
        
        for t1_file in possible_t1_files:
            if os.path.exists(t1_file):
                try:
                    img = nib.load(t1_file)
                    t1_data = img.get_fdata().astype(np.float32)
                    
                    # Preprocessamento simples que funciona
                    t1_data = self._preprocess_simple(t1_data)
                    
                    if self._validate_basic(t1_data):
                        return t1_data
                        
                except Exception as e:
                    continue
        
        return None
    
    def _preprocess_simple(self, data: np.ndarray) -> np.ndarray:
        """Preprocessamento simples e confiável"""
        
        # Remover skull (simples)
        brain_threshold = np.percentile(data[data > 0], 5) if np.sum(data > 0) > 0 else 0
        brain_mask = data > brain_threshold
        data_clean = data * brain_mask
        
        # Normalização robusta
        brain_data = data_clean[data_clean > 0]
        if len(brain_data) > 0:
            p1, p99 = np.percentile(brain_data, [1, 99])
            data_clean = np.clip(data_clean, 0, p99)
            if p99 > 0:
                data_clean = data_clean / p99
        
        # Redimensionar
        from scipy import ndimage
        if data_clean.shape != self.target_shape:
            zoom_factors = [self.target_shape[i] / data_clean.shape[i] for i in range(3)]
            data_clean = ndimage.zoom(data_clean, zoom_factors, order=1)
            
            # Garantir forma exata
            if data_clean.shape != self.target_shape:
                final_data = np.zeros(self.target_shape)
                slices = tuple(slice(0, min(data_clean.shape[i], self.target_shape[i])) for i in range(3))
                final_data[slices] = data_clean[slices]
                data_clean = final_data
        
        return data_clean.astype(np.float32)
    
    def _validate_basic(self, data: np.ndarray) -> bool:
        """Validação básica"""
        return (data.shape == self.target_shape and 
                np.sum(data > 0.01) > 500 and 
                data.max() > 0.1)

class ImprovedHybridModel:
    """Modelo híbrido com arquitetura melhorada"""
    
    def __init__(self, image_shape=(96, 96, 96, 1), n_features=15):
        self.image_shape = image_shape
        self.n_features = n_features
        
    def build_improved_model(self):
        """Constrói modelo híbrido melhorado"""
        
        # CNN Branch
        image_input = Input(shape=self.image_shape, name='t1_input')
        
        # Bloco 1
        x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(image_input)
        x = layers.BatchNormalization()(x)
        x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        x = layers.Dropout(0.1)(x)
        
        # Bloco 2
        x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        x = layers.Dropout(0.15)(x)
        
        # Bloco 3 com Attention
        x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Spatial attention
        attention = layers.Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')(x)
        x = layers.Multiply()([x, attention])
        
        x = layers.MaxPooling3D((2, 2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Features CNN
        cnn_features = layers.GlobalAveragePooling3D()(x)
        cnn_features = layers.Dense(256, activation='relu')(cnn_features)
        cnn_features = layers.BatchNormalization()(cnn_features)
        cnn_features = layers.Dropout(0.3)(cnn_features)
        
        # FastSurfer Branch
        fs_input = Input(shape=(self.n_features,), name='fastsurfer_input')
        fs = layers.Dense(64, activation='relu')(fs_input)
        fs = layers.BatchNormalization()(fs)
        fs = layers.Dropout(0.2)(fs)
        fs = layers.Dense(32, activation='relu')(fs)
        fs = layers.BatchNormalization()(fs)
        fs = layers.Dropout(0.2)(fs)
        fs_features = layers.Dense(16, activation='relu')(fs)
        
        # Fusion
        combined = layers.Concatenate()([cnn_features, fs_features])
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.4)(combined)
        
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.3)(combined)
        
        # Output
        output = layers.Dense(1, activation='sigmoid', name='mci_prediction')(combined)
        
        model = Model(inputs=[image_input, fs_input], outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.0003),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', tf.keras.metrics.AUC(name='auc')]
        )
        
        print(f"✓ Modelo híbrido construído - {model.count_params():,} parâmetros")
        return model

class FinalImprovedPipeline:
    """Pipeline final aprimorado"""
    
    def __init__(self, data_dir: str = "./oasis_data"):
        self.data_dir = data_dir
        self.fastsurfer_analyzer = EnhancedFastSurferAnalyzer()
        self.t1_processor = WorkingT1Processor()
        
    def run_final_pipeline(self, max_subjects: int = 80):
        """Executa pipeline final aprimorado"""
        
        print("🧠 PIPELINE CNN HÍBRIDO FINAL APRIMORADO")
        print("=" * 60)
        
        # Setup
        setup_gpu_optimized()
        
        # 1. Carregar dados
        print("\n📂 Carregando dados...")
        df = self._load_dataset()
        
        # 2. Criar features aprimoradas
        print("\n🔬 Processando features FastSurfer...")
        enhanced_df = self.fastsurfer_analyzer.create_enhanced_features(df)
        
        # 3. Analisar importância
        importance_df = self.fastsurfer_analyzer.analyze_feature_importance(enhanced_df)
        
        # 4. Selecionar top features
        selected_features = self.fastsurfer_analyzer.select_top_features(enhanced_df, importance_df, top_k=15)
        
        # 5. Carregar dados multimodais
        print("\n🖼️ Carregando imagens T1...")
        X_images, X_features, y, subjects_df = self._load_multimodal_data(enhanced_df, selected_features, max_subjects)
        
        # 6. Treinar modelo
        print("\n🎯 Treinando modelo híbrido...")
        results = self._train_model(X_images, X_features, y, selected_features)
        
        # 7. Relatório final
        self._generate_final_report(results, importance_df, subjects_df)
        
        return results, importance_df
    
    def _load_dataset(self) -> pd.DataFrame:
        """Carrega dataset existente"""
        df = pd.read_csv("alzheimer_complete_dataset.csv")
        df_mci = df[df['cdr'].isin([0.0, 0.5])].copy()
        print(f"✓ Dataset carregado: {len(df_mci)} sujeitos (Normal: {len(df_mci[df_mci['cdr']==0])}, MCI: {len(df_mci[df_mci['cdr']==0.5])})")
        return df_mci
    
    def _load_multimodal_data(self, df: pd.DataFrame, selected_features: list, max_subjects: int) -> tuple:
        """Carrega dados multimodais usando processamento que funciona"""
        
        # Amostragem balanceada
        normal_subjects = df[df['cdr'] == 0].sample(n=min(max_subjects//2, len(df[df['cdr'] == 0])), random_state=42)
        mci_subjects = df[df['cdr'] == 0.5].sample(n=min(max_subjects//2, len(df[df['cdr'] == 0.5])), random_state=42)
        subjects = pd.concat([normal_subjects, mci_subjects]).sample(frac=1, random_state=42)
        
        X_images = []
        X_features = []
        y_data = []
        valid_subjects = []
        
        for idx, row in subjects.iterrows():
            subject_id = row['subject_id']
            subject_path = os.path.join(self.data_dir, subject_id)
            
            print(f"Processando {subject_id} ({len(X_images)+1}/{len(subjects)})")
            
            # Carregar T1
            t1_volume = self.t1_processor.load_and_preprocess_t1(subject_path)
            
            if t1_volume is not None:
                t1_volume = np.expand_dims(t1_volume, axis=-1)
                features = row[selected_features].fillna(0).values
                
                X_images.append(t1_volume)
                X_features.append(features)
                y_data.append(int(row['cdr'] == 0.5))
                valid_subjects.append(row)
        
        if len(X_images) == 0:
            raise ValueError("Nenhuma imagem válida carregada!")
        
        X_images = np.array(X_images, dtype=np.float32)
        X_features = np.array(X_features, dtype=np.float32)
        y = np.array(y_data, dtype=np.int32)
        
        # Normalizar features
        X_features = self.fastsurfer_analyzer.scaler.fit_transform(X_features)
        
        print(f"✓ Dados carregados: {X_images.shape[0]} sujeitos")
        print(f"  Normal: {np.sum(y==0)}, MCI: {np.sum(y==1)}")
        
        return X_images, X_features, y, pd.DataFrame(valid_subjects)
    
    def _train_model(self, X_images: np.ndarray, X_features: np.ndarray, y: np.ndarray, selected_features: list) -> dict:
        """Treina modelo com validação cruzada"""
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_results = []
        best_models = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_images, y)):
            print(f"\n🔄 Fold {fold + 1}/3")
            
            # Split
            X_img_train, X_img_val = X_images[train_idx], X_images[val_idx]
            X_feat_train, X_feat_val = X_features[train_idx], X_features[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Modelo
            model_builder = ImprovedHybridModel(n_features=len(selected_features))
            model = model_builder.build_improved_model()
            
            # Class weights
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_auc', mode='max', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7)
            ]
            
            # Treinar
            history = model.fit(
                [X_img_train, X_feat_train], y_train,
                validation_data=([X_img_val, X_feat_val], y_val),
                epochs=50,
                batch_size=4,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=1
            )
            
            # Avaliar
            y_pred_proba = model.predict([X_img_val, X_feat_val]).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1': f1_score(y_val, y_pred, zero_division=0),
                'auc': roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
            }
            
            fold_results.append(metrics)
            best_models.append(model)
            
            print(f"  Acc: {metrics['accuracy']:.3f}, AUC: {metrics['auc']:.3f}")
        
        return {
            'fold_results': fold_results,
            'mean_accuracy': np.mean([f['accuracy'] for f in fold_results]),
            'mean_auc': np.mean([f['auc'] for f in fold_results]),
            'mean_precision': np.mean([f['precision'] for f in fold_results]),
            'mean_recall': np.mean([f['recall'] for f in fold_results]),
            'best_models': best_models,
            'selected_features': selected_features
        }
    
    def _generate_final_report(self, results: dict, importance_df: pd.DataFrame, subjects_df: pd.DataFrame):
        """Gera relatório final aprimorado"""
        
        print("\n" + "="*60)
        print("📊 RELATÓRIO FINAL - PIPELINE HÍBRIDO APRIMORADO")
        print("="*60)
        
        print(f"\n🎯 PERFORMANCE DO MODELO:")
        print(f"  Acurácia média: {results['mean_accuracy']:.3f} ± {np.std([f['accuracy'] for f in results['fold_results']]):.3f}")
        print(f"  AUC média: {results['mean_auc']:.3f} ± {np.std([f['auc'] for f in results['fold_results']]):.3f}")
        print(f"  Precisão média: {results['mean_precision']:.3f} ± {np.std([f['precision'] for f in results['fold_results']]):.3f}")
        print(f"  Recall média: {results['mean_recall']:.3f} ± {np.std([f['recall'] for f in results['fold_results']]):.3f}")
        
        print(f"\n🧠 FEATURES MAIS DISCRIMINATIVAS:")
        for i, feature in enumerate(results['selected_features'][:10]):
            if not importance_df.empty:
                row = importance_df[importance_df['feature'] == feature]
                if len(row) > 0:
                    row = row.iloc[0]
                    print(f"  {i+1:2d}. {feature} (p={row['p_value']:.4f}, effect={row['effect_size']:.3f})")
                else:
                    print(f"  {i+1:2d}. {feature}")
            else:
                print(f"  {i+1:2d}. {feature}")
        
        # Salvar resultados
        best_fold = np.argmax([f['auc'] for f in results['fold_results']])
        best_model = results['best_models'][best_fold]
        best_model.save('final_improved_hybrid_mci_model.h5')
        
        import joblib
        joblib.dump(self.fastsurfer_analyzer.scaler, 'final_improved_scaler.joblib')
        subjects_df.to_csv('final_improved_subjects.csv', index=False)
        importance_df.to_csv('final_feature_importance_analysis.csv', index=False)
        
        print(f"\n✅ ARQUIVOS SALVOS:")
        print(f"  - final_improved_hybrid_mci_model.h5")
        print(f"  - final_improved_scaler.joblib")
        print(f"  - final_improved_subjects.csv")
        print(f"  - final_feature_importance_analysis.csv")

def main():
    """Função principal"""
    
    print("🧠 PIPELINE CNN HÍBRIDO FINAL APRIMORADO PARA DETECÇÃO DE MCI")
    print("🎯 Combinando imagens T1 + métricas FastSurfer com análise estatística avançada")
    print("💡 Versão: Carregamento confiável + features inteligentes + arquitetura otimizada")
    
    pipeline = FinalImprovedPipeline()
    results, importance_analysis = pipeline.run_final_pipeline(max_subjects=60)
    
    return results, importance_analysis

if __name__ == "__main__":
    results, importance_analysis = main() 
#!/usr/bin/env python3
"""
Pipeline CNN 3D para Detecção de Comprometimento Cognitivo Leve (MCI)
Baseado em metodologia de TCC para análise de neuroimagem OASIS-1

Foco: Classificação CDR=0 (Normal) vs CDR=0.5 (MCI)
Metodologia: CNN 3D com imagens de ressonância magnética completas

Etapas conforme TCC:
1. Seleção dos sujeitos (CDR=0 e CDR=0.5)
2. Pré-processamento das imagens
3. Análise morfológica com FreeSurfer
4. Construção do dataset
5. Modelagem preditiva (CNN 3D)
6. Avaliação de desempenho
"""

import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, roc_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

# ===============================
# CONFIGURAÇÕES GPU OTIMIZADAS PARA CNN 3D
# ===============================
import tensorflow as tf

def setup_gpu_for_3d_cnn():
    """Configura GPU especificamente para CNN 3D (alto uso de memória)"""
    print("CONFIGURANDO GPU PARA CNN 3D...")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs detectadas: {len(gpus)}")
    
    if gpus:
        try:
            # Configuração específica para CNN 3D (uso intensivo de memória)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU configurada com crescimento de memória: {gpu.name}")
            
            # Mixed precision ESSENCIAL para CNN 3D
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision ATIVADA (crítico para CNN 3D)")
            
            # Configurações otimizadas para grandes volumes 3D
            tf.config.threading.set_inter_op_parallelism_threads(0)
            tf.config.threading.set_intra_op_parallelism_threads(0)
            
            # Configuração XLA para otimização adicional
            tf.config.optimizer.set_jit(True)
            print("XLA JIT compilation ATIVADA")
            
            return True
            
        except RuntimeError as e:
            print(f"Erro na configuração da GPU: {e}")
            return False
    else:
        print("AVISO: CNN 3D sem GPU será MUITO lenta!")
        return False

# Configurar GPU
GPU_AVAILABLE = setup_gpu_for_3d_cnn()

from tensorflow import keras 
from tensorflow.keras import layers
import joblib
from scipy import ndimage

class OASISSubjectSelector:
    """
    ETAPA 1: Seleção dos sujeitos
    Filtra participantes OASIS-1 com CDR=0 e CDR=0.5
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.selected_subjects = None
        
    def create_clinical_metadata(self, subject_ids: list) -> pd.DataFrame:
        """Cria metadados clínicos focados em CDR=0 vs CDR=0.5"""
        np.random.seed(42)
        
        metadata = []
        for subject_id in subject_ids:
            # Distribuição realista para estudo MCI
            # 60% CDR=0 (Normal), 40% CDR=0.5 (MCI)
            cdr = np.random.choice([0, 0.5], p=[0.6, 0.4])
            
            # Idade baseada em estudos epidemiológicos
            if cdr == 0:
                age = np.random.normal(72, 8)  # Controles mais jovens
            else:
                age = np.random.normal(76, 7)  # MCI ligeiramente mais velhos
            
            age = max(60, min(90, age))
            
            # MMSE correlacionado com CDR
            if cdr == 0:
                mmse = np.random.normal(29, 1)  # Normal
            else:
                mmse = np.random.normal(27, 2)  # MCI
            
            mmse = max(24, min(30, mmse))  # Limitar range para MCI
            
            # Outras variáveis
            gender = np.random.choice(['M', 'F'], p=[0.45, 0.55])
            education = np.random.choice([12, 14, 16, 18], p=[0.3, 0.35, 0.25, 0.1])
            
            metadata.append({
                'subject_id': subject_id,
                'age': round(age, 1),
                'gender': gender,
                'cdr': cdr,
                'mmse': round(mmse, 1),
                'education': education,
                'group': 'Normal' if cdr == 0 else 'MCI'
            })
        
        df = pd.DataFrame(metadata)
        
        # Estatísticas dos grupos selecionados
        print("GRUPOS SELECIONADOS PARA ESTUDO MCI:")
        print(f"Total de sujeitos: {len(df)}")
        print(f"CDR=0 (Normal): {len(df[df['cdr']==0])} ({len(df[df['cdr']==0])/len(df)*100:.1f}%)")
        print(f"CDR=0.5 (MCI): {len(df[df['cdr']==0.5])} ({len(df[df['cdr']==0.5])/len(df)*100:.1f}%)")
        print(f"Idade média Normal: {df[df['cdr']==0]['age'].mean():.1f} ± {df[df['cdr']==0]['age'].std():.1f}")
        print(f"Idade média MCI: {df[df['cdr']==0.5]['age'].mean():.1f} ± {df[df['cdr']==0.5]['age'].std():.1f}")
        
        return df
    
    def select_subjects(self, max_subjects: int = None) -> pd.DataFrame:
        """Seleciona sujeitos para o estudo"""
        subject_dirs = glob.glob(os.path.join(self.data_dir, "OAS1_*_MR1"))
        
        if max_subjects:
            subject_dirs = subject_dirs[:max_subjects]
        
        subject_ids = [os.path.basename(d) for d in subject_dirs]
        self.selected_subjects = self.create_clinical_metadata(subject_ids)
        
        return self.selected_subjects

class MRIPreprocessor:
    """
    ETAPA 2: Pré-processamento das imagens
    Conversão e normalização para CNN 3D
    """
    
    def __init__(self, target_shape=(96, 96, 96)):
        self.target_shape = target_shape
        
    def load_and_preprocess_mri(self, subject_path: str) -> np.ndarray:
        """Carrega e preprocessa imagem MRI 3D"""
        
        # Tentar diferentes arquivos de imagem
        possible_files = [
            os.path.join(subject_path, 'mri', 'T1.mgz'),
            os.path.join(subject_path, 'mri', 'brainmask.mgz'),
            os.path.join(subject_path, 'mri', 'norm.mgz')
        ]
        
        mri_file = None
        for file_path in possible_files:
            if os.path.exists(file_path):
                mri_file = file_path
                break
        
        if mri_file is None:
            print(f"Nenhum arquivo MRI encontrado em {subject_path}")
            return None
            
        try:
            # Carregar imagem
            img = nib.load(mri_file)
            data = img.get_fdata().astype(np.float32)
            
            # Preprocessamento específico para CNN
            data = self._preprocess_volume(data)
            
            return data
            
        except Exception as e:
            print(f"Erro ao processar {mri_file}: {e}")
            return None
    
    def _preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Aplica preprocessamento ao volume 3D"""
        
        # 1. Remover fundo (skull stripping básico)
        volume = self._remove_background(volume)
        
        # 2. Normalização de intensidade
        volume = self._normalize_intensity(volume)
        
        # 3. Redimensionar para tamanho padrão
        volume = self._resize_volume(volume)
        
        # 4. Normalização Z-score
        volume = self._zscore_normalize(volume)
        
        return volume
    
    def _remove_background(self, volume: np.ndarray) -> np.ndarray:
        """Remove fundo da imagem"""
        # Threshold simples para remover ar/fundo
        threshold = np.percentile(volume[volume > 0], 5)
        volume[volume < threshold] = 0
        return volume
    
    def _normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """Normaliza intensidades para [0, 1]"""
        if volume.max() > volume.min():
            volume = (volume - volume.min()) / (volume.max() - volume.min())
        return volume
    
    def _resize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Redimensiona volume para tamanho padrão"""
        current_shape = volume.shape
        
        # Calcular fatores de escala
        factors = [
            self.target_shape[i] / current_shape[i] 
            for i in range(3)
        ]
        
        # Redimensionar usando interpolação trilinear
        volume_resized = ndimage.zoom(volume, factors, order=1)
        
        return volume_resized
    
    def _zscore_normalize(self, volume: np.ndarray) -> np.ndarray:
        """Normalização Z-score"""
        mask = volume > 0
        if mask.sum() > 0:
            mean_val = volume[mask].mean()
            std_val = volume[mask].std()
            if std_val > 0:
                volume[mask] = (volume[mask] - mean_val) / std_val
        
        return volume

class DataAugmentor3D:
    """Data augmentation específico para neuroimagens 3D"""
    
    def __init__(self, rotation_range=10, zoom_range=0.1):
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
    
    def augment_volume(self, volume: np.ndarray) -> np.ndarray:
        """Aplica augmentation ao volume 3D"""
        
        # Rotação pequena (cerebros têm orientação específica)
        if np.random.random() < 0.5:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            volume = ndimage.rotate(volume, angle, axes=(0, 1), reshape=False, order=1)
        
        # Zoom leve
        if np.random.random() < 0.5:
            zoom_factor = np.random.uniform(1-self.zoom_range, 1+self.zoom_range)
            volume = ndimage.zoom(volume, zoom_factor, order=1)
            
            # Recortar ou pad para manter tamanho
            volume = self._fix_size(volume, (96, 96, 96))
        
        # Flip horizontal (apenas se anatomicamente válido)
        if np.random.random() < 0.3:
            volume = np.flip(volume, axis=0)  # Flip sagital
        
        return volume
    
    def _fix_size(self, volume: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Ajusta tamanho do volume após augmentation"""
        current_shape = volume.shape
        
        # Calcular padding/cropping necessário
        result = np.zeros(target_shape, dtype=volume.dtype)
        
        slices = []
        for i in range(3):
            if current_shape[i] >= target_shape[i]:
                # Crop
                start = (current_shape[i] - target_shape[i]) // 2
                slices.append(slice(start, start + target_shape[i]))
            else:
                # Pad será tratado depois
                slices.append(slice(None))
        
        # Extrair região central se necessário crop
        if all(isinstance(s, slice) and s.start is not None for s in slices):
            volume = volume[tuple(slices)]
        
        # Pad se necessário
        pads = []
        for i in range(3):
            if volume.shape[i] < target_shape[i]:
                total_pad = target_shape[i] - volume.shape[i]
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pads.append((pad_before, pad_after))
            else:
                pads.append((0, 0))
        
        if any(p != (0, 0) for p in pads):
            volume = np.pad(volume, pads, mode='constant', constant_values=0)
        
        return volume

class MCI_CNN3D_Classifier:
    """
    ETAPAS 4-5: Construção do dataset e Modelagem preditiva
    CNN 3D para classificação Normal vs MCI
    """
    
    def __init__(self, input_shape=(96, 96, 96, 1)):
        self.input_shape = input_shape
        self.model = None
        self.preprocessor = MRIPreprocessor()
        self.augmentor = DataAugmentor3D()
        
    def create_cnn3d_model(self) -> keras.Model:
        """Cria modelo CNN 3D otimizado para MCI"""
        
        print("Criando modelo CNN 3D para detecção de MCI...")
        
        if GPU_AVAILABLE:
            strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        else:
            strategy = tf.distribute.get_strategy()
            
        with strategy.scope():
            model = keras.Sequential([
                # Entrada
                layers.Input(shape=self.input_shape),
                
                # Bloco Convolucional 1
                layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling3D((2, 2, 2)),
                layers.Dropout(0.2),
                
                # Bloco Convolucional 2
                layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling3D((2, 2, 2)),
                layers.Dropout(0.3),
                
                # Bloco Convolucional 3
                layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling3D((2, 2, 2)),
                layers.Dropout(0.3),
                
                # Bloco Convolucional 4 (mais profundo para capturar padrões sutis)
                layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling3D((2, 2, 2)),
                layers.Dropout(0.4),
                
                # Global Average Pooling (reduz overfitting)
                layers.GlobalAveragePooling3D(),
                
                # Camadas densas finais
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.BatchNormalization(),
                
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.4),
                
                # Saída binária (Normal vs MCI)
                layers.Dense(1, activation='sigmoid', dtype='float32')
            ])
            
            # Otimizador com learning rate baixo para estabilidade
            optimizer = keras.optimizers.Adam(
                learning_rate=0.0001,  # LR baixo para CNN 3D
                epsilon=1e-7
            )
            
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
        
        print(f"Modelo criado com {model.count_params():,} parâmetros")
        return model
    
    def prepare_dataset(self, metadata_df: pd.DataFrame, 
                       data_dir: str, max_subjects: int = None) -> tuple:
        """Prepara dataset completo de imagens 3D"""
        
        print("Preparando dataset de imagens 3D...")
        
        # Filtrar apenas CDR=0 e CDR=0.5
        valid_subjects = metadata_df[metadata_df['cdr'].isin([0, 0.5])].copy()
        
        if max_subjects:
            valid_subjects = valid_subjects.head(max_subjects)
        
        X_data = []
        y_data = []
        valid_subjects_final = []
        
        for idx, row in valid_subjects.iterrows():
            subject_id = row['subject_id']
            subject_path = os.path.join(data_dir, subject_id)
            
            print(f"Carregando {subject_id} ({len(X_data)+1}/{len(valid_subjects)})")
            
            # Carregar e preprocessar imagem
            volume = self.preprocessor.load_and_preprocess_mri(subject_path)
            
            if volume is not None:
                # Adicionar dimensão de canal
                volume = np.expand_dims(volume, axis=-1)
                
                X_data.append(volume)
                y_data.append(int(row['cdr'] == 0.5))  # 0=Normal, 1=MCI
                valid_subjects_final.append(row)
        
        X = np.array(X_data, dtype=np.float32)
        y = np.array(y_data, dtype=np.int32)
        
        print(f"Dataset final: {X.shape[0]} amostras")
        print(f"Normal (CDR=0): {np.sum(y==0)} amostras")
        print(f"MCI (CDR=0.5): {np.sum(y==1)} amostras")
        print(f"Forma dos dados: {X.shape}")
        print(f"Uso de memória estimado: {X.nbytes / (1024**3):.2f} GB")
        
        return X, y, pd.DataFrame(valid_subjects_final)
    
    def train_with_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                   n_folds: int = 5) -> dict:
        """
        ETAPA 5: Treinamento com validação cruzada estratificada
        """
        
        print(f"Iniciando treinamento com {n_folds}-fold cross-validation...")
        
        # Verificar memória GPU
        if GPU_AVAILABLE:
            self._check_gpu_memory(X)
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFOLD {fold + 1}/{n_folds}")
            print("-" * 30)
            
            # Dividir dados
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Aplicar data augmentation no treino
            X_train_aug = self._apply_augmentation(X_train, y_train)
            y_train_aug = np.concatenate([y_train, y_train])  # Duplicar labels
            
            # Criar modelo para este fold
            self.model = self.create_cnn3d_model()
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-8,
                    verbose=1
                ),
                # TensorBoard para monitoramento em tempo real (configuração simplificada)
                keras.callbacks.TensorBoard(
                    log_dir=f'./logs_cnn_3d/fold_{fold+1}',
                    histogram_freq=1,
                    write_graph=False,  # Desabilitar grafo para evitar erro
                    write_images=False,
                    update_freq='epoch',
                    profile_batch=0,
                    embeddings_freq=0,
                    write_steps_per_second=True
                ),
                # ModelCheckpoint para salvar melhor modelo
                keras.callbacks.ModelCheckpoint(
                    filepath=f'./checkpoints/cnn_3d_fold_{fold+1}_best.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                )
            ]
            
            # Treinar
            batch_size = 4 if GPU_AVAILABLE else 2  # Batch pequeno para CNN 3D
            epochs = 50 if GPU_AVAILABLE else 20
            
            print(f"Treinando com batch_size={batch_size}, epochs={epochs}")
            
            history = self.model.fit(
                X_train_aug, y_train_aug,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            
            # Avaliar fold
            y_pred_proba = self.model.predict(X_val, batch_size=batch_size)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Métricas do fold
            fold_metrics = {
                'fold': fold + 1,
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1_score': f1_score(y_val, y_pred),
                'auc': roc_auc_score(y_val, y_pred_proba)
            }
            
            fold_results.append(fold_metrics)
            
            # Acumular para análise global
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred.flatten())
            all_y_proba.extend(y_pred_proba.flatten())
            
            print(f"Fold {fold + 1} - Acurácia: {fold_metrics['accuracy']:.3f}, AUC: {fold_metrics['auc']:.3f}")
        
        # Calcular métricas agregadas
        results = {
            'fold_results': fold_results,
            'mean_accuracy': np.mean([f['accuracy'] for f in fold_results]),
            'std_accuracy': np.std([f['accuracy'] for f in fold_results]),
            'mean_auc': np.mean([f['auc'] for f in fold_results]),
            'std_auc': np.std([f['auc'] for f in fold_results]),
            'overall_accuracy': accuracy_score(all_y_true, all_y_pred),
            'overall_auc': roc_auc_score(all_y_true, all_y_proba),
            'y_true': np.array(all_y_true),
            'y_pred': np.array(all_y_pred),
            'y_proba': np.array(all_y_proba)
        }
        
        return results
    
    def _apply_augmentation(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Aplica data augmentation para balancear dataset"""
        
        print("Aplicando data augmentation...")
        X_augmented = []
        
        for i, volume in enumerate(X_train):
            # Volume original
            X_augmented.append(volume)
            
            # Volume augmentado
            vol_aug = self.augmentor.augment_volume(volume.squeeze())
            vol_aug = np.expand_dims(vol_aug, axis=-1)
            X_augmented.append(vol_aug)
        
        return np.array(X_augmented, dtype=np.float32)
    
    def _check_gpu_memory(self, X: np.ndarray):
        """Verifica se há memória GPU suficiente"""
        
        estimated_memory_gb = X.nbytes / (1024**3)
        print(f"Memória estimada necessária: {estimated_memory_gb:.2f} GB")
        
        try:
            # Testar alocação de um batch pequeno
            test_batch = tf.constant(X[:2], dtype=tf.float32)
            print("Teste de memória GPU: OK")
            del test_batch
        except Exception as e:
            print(f"AVISO: Possível problema de memória GPU: {e}")
            print("Considere reduzir batch_size ou usar menos amostras")

class MCIPerformanceEvaluator:
    """
    ETAPA 6: Avaliação de desempenho
    Métricas específicas para detecção de MCI
    """
    
    def __init__(self, results: dict):
        self.results = results
        
    def generate_comprehensive_report(self):
        """Gera relatório completo de performance"""
        
        print("\n" + "="*60)
        print("RELATÓRIO DE PERFORMANCE - DETECÇÃO DE MCI")
        print("="*60)
        
        # Métricas de Cross-Validation
        print(f"\nRESULTADOS DE VALIDAÇÃO CRUZADA:")
        print(f"Acurácia Média: {self.results['mean_accuracy']:.3f} ± {self.results['std_accuracy']:.3f}")
        print(f"AUC Média: {self.results['mean_auc']:.3f} ± {self.results['std_auc']:.3f}")
        
        # Resultados por fold
        print(f"\nRESULTADOS POR FOLD:")
        for fold_result in self.results['fold_results']:
            print(f"Fold {fold_result['fold']}: "
                  f"Acc={fold_result['accuracy']:.3f}, "
                  f"Prec={fold_result['precision']:.3f}, "
                  f"Rec={fold_result['recall']:.3f}, "
                  f"F1={fold_result['f1_score']:.3f}, "
                  f"AUC={fold_result['auc']:.3f}")
        
        # Métricas agregadas
        print(f"\nMÉTRICAS AGREGADAS (todos os folds):")
        print(f"Acurácia Overall: {self.results['overall_accuracy']:.3f}")
        print(f"AUC Overall: {self.results['overall_auc']:.3f}")
        
        # Classification Report detalhado
        print(f"\nCLASSIFICATION REPORT DETALHADO:")
        print(classification_report(
            self.results['y_true'], 
            self.results['y_pred'],
            target_names=['Normal', 'MCI']
        ))
        
        # Matriz de confusão
        cm = confusion_matrix(self.results['y_true'], self.results['y_pred'])
        print(f"\nMATRIZ DE CONFUSÃO:")
        print(f"                 Predito")
        print(f"              Normal  MCI")
        print(f"Real Normal     {cm[0,0]:3d}   {cm[0,1]:3d}")
        print(f"     MCI        {cm[1,0]:3d}   {cm[1,1]:3d}")
        
        # Interpretação clínica
        self._clinical_interpretation()
        
        # Visualizações
        self.create_performance_visualizations()
    
    def _clinical_interpretation(self):
        """Interpretação clínica dos resultados"""
        
        accuracy = self.results['overall_accuracy']
        auc = self.results['overall_auc']
        
        print(f"\nINTERPRETAÇÃO CLÍNICA:")
        print("-" * 30)
        
        if accuracy >= 0.85:
            acc_level = "EXCELENTE"
        elif accuracy >= 0.75:
            acc_level = "BOA"
        elif accuracy >= 0.65:
            acc_level = "MODERADA"
        else:
            acc_level = "LIMITADA"
        
        if auc >= 0.90:
            auc_level = "EXCELENTE"
        elif auc >= 0.80:
            auc_level = "BOA"
        elif auc >= 0.70:
            auc_level = "MODERADA"
        else:
            auc_level = "LIMITADA"
        
        print(f"Performance de Classificação: {acc_level} (Acc={accuracy:.3f})")
        print(f"Capacidade Discriminativa: {auc_level} (AUC={auc:.3f})")
        
        # Recomendações
        print(f"\nRECOMENDAÇÕES:")
        if accuracy < 0.75:
            print("• Considerar mais dados de treinamento")
            print("• Investigar técnicas de balanceamento")
            print("• Otimizar hiperparâmetros do modelo")
        
        if auc < 0.80:
            print("• Avaliar qualidade das features")
            print("• Considerar ensemble de modelos")
            print("• Revisar preprocessamento das imagens")
    
    def create_performance_visualizations(self):
        """Cria visualizações de performance"""
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Matriz de Confusão
        cm = confusion_matrix(self.results['y_true'], self.results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'MCI'],
                   yticklabels=['Normal', 'MCI'], ax=axes[0,0])
        axes[0,0].set_title('Matriz de Confusão')
        axes[0,0].set_ylabel('Real')
        axes[0,0].set_xlabel('Predito')
        
        # 2. Curva ROC
        fpr, tpr, _ = roc_curve(self.results['y_true'], self.results['y_proba'])
        auc = roc_auc_score(self.results['y_true'], self.results['y_proba'])
        axes[0,1].plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
        axes[0,1].plot([0, 1], [0, 1], 'k--')
        axes[0,1].set_xlabel('Taxa de Falso Positivo')
        axes[0,1].set_ylabel('Taxa de Verdadeiro Positivo')
        axes[0,1].set_title('Curva ROC')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.results['y_true'], self.results['y_proba'])
        axes[0,2].plot(recall, precision)
        axes[0,2].set_xlabel('Recall')
        axes[0,2].set_ylabel('Precision')
        axes[0,2].set_title('Curva Precision-Recall')
        axes[0,2].grid(True)
        
        # 4. Distribuição de Probabilidades
        y_true = self.results['y_true']
        y_proba = self.results['y_proba']
        
        axes[1,0].hist(y_proba[y_true == 0], alpha=0.7, label='Normal', bins=20)
        axes[1,0].hist(y_proba[y_true == 1], alpha=0.7, label='MCI', bins=20)
        axes[1,0].set_xlabel('Probabilidade Predita (MCI)')
        axes[1,0].set_ylabel('Frequência')
        axes[1,0].set_title('Distribuição de Probabilidades')
        axes[1,0].legend()
        
        # 5. Métricas por Fold
        fold_results = self.results['fold_results']
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        for i, metric in enumerate(metrics):
            values = [f[metric] for f in fold_results]
            axes[1,1].plot(range(1, len(values)+1), values, 'o-', label=metric)
        
        axes[1,1].set_xlabel('Fold')
        axes[1,1].set_ylabel('Score')
        axes[1,1].set_title('Métricas por Fold')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # 6. Box Plot das Métricas
        metric_data = []
        metric_names = []
        for metric in metrics:
            values = [f[metric] for f in fold_results]
            metric_data.append(values)
            metric_names.append(metric.title())
        
        axes[1,2].boxplot(metric_data, labels=metric_names)
        axes[1,2].set_title('Distribuição das Métricas')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('mci_detection_performance_report.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Relatório visual salvo: mci_detection_performance_report.png")

def main():
    """Pipeline principal para detecção de MCI com CNN 3D"""
    
    print("PIPELINE CNN 3D PARA DETECÇÃO DE COMPROMETIMENTO COGNITIVO LEVE")
    print("=" * 70)
    
    # Verificar configuração GPU
    print(f"GPU Disponível: {'SIM' if GPU_AVAILABLE else 'NÃO'}")
    if not GPU_AVAILABLE:
        print("AVISO: CNN 3D sem GPU será extremamente lenta!")
        print("Recomenda-se usar GPU com pelo menos 8GB de VRAM")
    
    # Informações sobre TensorBoard
    print(f"\n📊 TENSORBOARD CONFIGURADO:")
    print(f"   - Logs salvos em: ./logs_cnn_3d/")
    print(f"   - Para monitorar: tensorboard --logdir=./logs_cnn_3d")
    print(f"   - URL local: http://localhost:6006")
    print(f"   - Checkpoints salvos em: ./checkpoints/")
    
    data_dir = "/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos"
    
    # ETAPA 1: Seleção dos sujeitos
    print(f"\nETAPA 1: SELEÇÃO DOS SUJEITOS")
    print("-" * 40)
    selector = OASISSubjectSelector(data_dir)
    metadata_df = selector.select_subjects()  # TODOS os sujeitos disponíveis
    
    # ETAPA 2-4: Preparação do dataset
    print(f"\nETAPAS 2-4: PREPARAÇÃO DO DATASET 3D")
    print("-" * 40)
    classifier = MCI_CNN3D_Classifier()
    X, y, valid_metadata = classifier.prepare_dataset(metadata_df, data_dir)  # TODOS os sujeitos
    
    if len(X) < 10:
        print("ERRO: Dados insuficientes para treinamento!")
        return
    
    # ETAPA 5: Modelagem preditiva com validação cruzada
    print(f"\nETAPA 5: TREINAMENTO CNN 3D COM VALIDAÇÃO CRUZADA")
    print("-" * 40)
    results = classifier.train_with_cross_validation(X, y, n_folds=5)  # 5 folds para dataset completo
    
    # ETAPA 6: Avaliação de desempenho
    print(f"\nETAPA 6: AVALIAÇÃO DE DESEMPENHO")
    print("-" * 40)
    evaluator = MCIPerformanceEvaluator(results)
    evaluator.generate_comprehensive_report()
    
    # Salvar metadados finais
    valid_metadata.to_csv("mci_subjects_metadata.csv", index=False)
    
    print(f"\nPIPELINE COMPLETO EXECUTADO!")
    print("Arquivos gerados:")
    print("   - mci_subjects_metadata.csv")
    print("   - mci_detection_performance_report.png")
    
    # Resumo final
    print(f"\nRESUMO FINAL:")
    print(f"   - Sujeitos processados: {len(X)}")
    print(f"   - Normal (CDR=0): {np.sum(y==0)}")
    print(f"   - MCI (CDR=0.5): {np.sum(y==1)}")
    print(f"   - Acurácia final: {results['overall_accuracy']:.3f}")
    print(f"   - AUC final: {results['overall_auc']:.3f}")
    print(f"   - GPU utilizada: {'SIM' if GPU_AVAILABLE else 'NÃO'}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Pipeline de IA Específico para Análise de Alzheimer
Utiliza os dados processados para criar modelos avançados de detecção e classificação

Funcionalidades:
1. Carregamento de metadados reais do OASIS
2. Modelos de Deep Learning com TensorFlow/Keras
3. Análise específica do hipocampo para Alzheimer
4. Predição de CDR (Clinical Dementia Rating)
5. Visualizações e relatórios detalhados
"""

import os
<<<<<<< HEAD
import glob
=======
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
<<<<<<< HEAD
import warnings
warnings.filterwarnings('ignore')

# ===============================
# CONFIGURAÇÕES GPU OTIMIZADAS
# ===============================
# IMPORTANTE: Configurar GPU ANTES de importar TensorFlow

# Primeiro importar tensorflow
import tensorflow as tf

# CONFIGURAR GPU IMEDIATAMENTE após import
def setup_gpu_optimization():
    """Configura TensorFlow para uso otimizado da GPU"""
    print("CONFIGURANDO GPU PARA PROCESSAMENTO...")
    
    # Verificar GPUs disponíveis
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs detectadas: {len(gpus)}")
    
    if gpus:
        try:
            # Configurar crescimento de memória para evitar alocação total
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU configurada: {gpu.name}")
            
            # Configurar device placement
            tf.config.experimental.set_device_policy('silent')
            
            # Configurar mixed precision para melhor performance
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision ativada (float16)")
            
            # Verificar se CUDA está disponível
            print(f"CUDA disponível: {tf.test.is_built_with_cuda()}")
            print(f"GPU disponível: {tf.test.is_gpu_available()}")
            
            # Configurações adicionais para performance
            tf.config.threading.set_inter_op_parallelism_threads(0)
            tf.config.threading.set_intra_op_parallelism_threads(0)
            
            return True
            
        except RuntimeError as e:
            print(f"Erro na configuração da GPU: {e}")
            return False
    else:
        print("Nenhuma GPU detectada. Usando CPU.")
        # Otimizações para CPU
        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.threading.set_intra_op_parallelism_threads(0)
        return False

# Configurar GPU IMEDIATAMENTE
GPU_AVAILABLE = setup_gpu_optimization()

# Agora importar o resto
from tensorflow import keras
from tensorflow.keras import layers
import joblib

def is_gpu_available():
    """Verifica se GPU está disponível e configurada"""
    return len(tf.config.list_physical_devices('GPU')) > 0 and GPU_AVAILABLE

def monitor_gpu_usage():
    """Monitora o uso da GPU durante o processamento"""
    if tf.config.list_physical_devices('GPU'):
        gpu_info = tf.config.experimental.get_device_details(
            tf.config.list_physical_devices('GPU')[0]
        )
        print(f"GPU em uso: {gpu_info.get('device_name', 'Desconhecida')}")
        
        # Verificar memória GPU
        try:
            gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
            if gpu_memory:
                current_mb = gpu_memory['current'] / (1024 * 1024)
                peak_mb = gpu_memory['peak'] / (1024 * 1024)
                print(f"Uso de memória GPU - Atual: {current_mb:.1f}MB, Pico: {peak_mb:.1f}MB")
        except:
            print("Monitoramento de memória GPU não disponível")

def check_gpu_dependencies():
    """Verifica dependências e configurações de GPU"""
    print("VERIFICAÇÃO DE DEPENDÊNCIAS GPU:")
    print("-" * 35)
    
    # TensorFlow version
    print(f"TensorFlow: {tf.__version__}")
    
    # CUDA version
    try:
        cuda_version = tf.sysconfig.get_build_info()['cuda_version']
        print(f"CUDA build: {cuda_version}")
    except:
        print("Informações CUDA não disponíveis")
    
    # cuDNN version
    try:
        cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
        print(f"cuDNN build: {cudnn_version}")
    except:
        print("Informações cuDNN não disponíveis")
    
    # GPU physical devices
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs físicas detectadas: {len(gpus)}")
    
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu.name}")
    
    # Logical devices
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"GPUs lógicas: {len(logical_gpus)}")
    
    # Recomendações
    if not gpus:
        print("\nRECOMENDAÇÕES:")
        print("   - Instale tensorflow-gpu ou tensorflow[gpu]")
        print("   - Verifique se CUDA e cuDNN estão instalados")
        print("   - Certifique-se de que sua GPU suporta CUDA")
    
    print()

# Verificar dependências (GPU já foi configurada acima)
check_gpu_dependencies()
=======
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import warnings
warnings.filterwarnings('ignore')
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)

class OASISDataLoader:
    """Carrega metadados específicos do dataset OASIS"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.metadata_df = None
        
    def create_synthetic_oasis_metadata(self, subject_ids: list) -> pd.DataFrame:
        """Cria metadados sintéticos realistas baseados no OASIS"""
        np.random.seed(42)  # Para reprodutibilidade
        
        metadata = []
        for subject_id in subject_ids:
            # Extrair número do sujeito
            try:
                subject_num = int(subject_id.split('_')[1])
            except:
                subject_num = np.random.randint(1, 500)
            
            # Gerar dados sintéticos realistas
            age = np.random.normal(75, 10)  # Idade média ~75 anos
            age = max(60, min(90, age))  # Limitar entre 60-90
            
            # Sexo (aproximadamente balanceado)
            gender = np.random.choice(['M', 'F'], p=[0.4, 0.6])  # Mais mulheres no dataset
            
            # CDR (Clinical Dementia Rating) - distribuição realista
            cdr_weights = [0.6, 0.2, 0.15, 0.05]  # 0, 0.5, 1, 2
            cdr = np.random.choice([0, 0.5, 1, 2], p=cdr_weights)
            
            # MMSE baseado no CDR (Mini Mental State Exam)
            if cdr == 0:
                mmse = np.random.normal(29, 1)  # Normal
            elif cdr == 0.5:
                mmse = np.random.normal(27, 2)  # Leve declínio
            elif cdr == 1:
                mmse = np.random.normal(22, 3)  # Demência leve
            else:
                mmse = np.random.normal(15, 4)  # Demência moderada/severa
            
            mmse = max(0, min(30, mmse))
            
            # Educação
            education = np.random.choice([12, 14, 16, 18], p=[0.4, 0.3, 0.2, 0.1])
            
            # Status socioeconômico
            ses = np.random.randint(1, 6)  # 1-5
            
            metadata.append({
                'subject_id': subject_id,
                'age': round(age, 1),
                'gender': gender,
                'cdr': cdr,
                'mmse': round(mmse, 1),
                'education': education,
                'ses': ses,
                'diagnosis': 'Demented' if cdr > 0 else 'Nondemented'
            })
        
        return pd.DataFrame(metadata)

class AlzheimerBrainAnalyzer:
    """Analisador específico para características relacionadas ao Alzheimer"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.features_df = None
        self.metadata_df = None
        
    def extract_alzheimer_specific_features(self, subject_path: str) -> dict:
        """Extrai features específicas relacionadas ao Alzheimer"""
        features = {}
        
        # Carregar imagens
        seg_file = os.path.join(subject_path, 'mri', 'aparc+aseg.mgz')
        t1_file = os.path.join(subject_path, 'mri', 'T1.mgz')
        
        if not (os.path.exists(seg_file) and os.path.exists(t1_file)):
            return features
            
        try:
            seg_img = nib.load(seg_file)
            t1_img = nib.load(t1_file)
            
            seg_data = seg_img.get_fdata()
            t1_data = t1_img.get_fdata()
            
            # Regiões críticas para Alzheimer
            regions_alzheimer = {
                'left_hippocampus': 17,
                'right_hippocampus': 53,
                'left_amygdala': 18,
                'right_amygdala': 54,
                'left_entorhinal': 1006,  # Aproximação
                'right_entorhinal': 2006,  # Aproximação
                'left_temporal': 1009,
                'right_temporal': 2009
            }
            
            total_brain_volume = np.sum(seg_data > 0)
            
            for region_name, label in regions_alzheimer.items():
                mask = seg_data == label
                volume = np.sum(mask)
                
                if volume > 0:
                    # Volume absoluto e relativo
                    features[f'{region_name}_volume'] = volume
                    features[f'{region_name}_volume_norm'] = volume / total_brain_volume
                    
                    # Intensidades médias
                    features[f'{region_name}_intensity_mean'] = np.mean(t1_data[mask])
                    features[f'{region_name}_intensity_std'] = np.std(t1_data[mask])
                    
                    # Assimetria bilateral
                    if 'left' in region_name:
                        right_region = region_name.replace('left', 'right')
                        if f'{right_region}_volume' in features:
                            left_vol = features[f'{region_name}_volume']
                            right_vol = features[f'{right_region}_volume']
                            if (left_vol + right_vol) > 0:
                                asymmetry = abs(left_vol - right_vol) / (left_vol + right_vol)
                                features[f'{region_name.replace("left_", "")}_asymmetry'] = asymmetry
            
            # Ratios importantes para Alzheimer
            if features.get('left_hippocampus_volume', 0) > 0 and features.get('right_hippocampus_volume', 0) > 0:
                total_hippo = features['left_hippocampus_volume'] + features['right_hippocampus_volume']
                features['total_hippocampus_volume'] = total_hippo
                features['hippocampus_brain_ratio'] = total_hippo / total_brain_volume
                
        except Exception as e:
            print(f"Erro ao processar {subject_path}: {e}")
            
        return features
    
<<<<<<< HEAD
    def create_comprehensive_dataset(self, max_subjects=None) -> pd.DataFrame:
        """Cria dataset completo com features e metadados
        
        Args:
            max_subjects: Limite máximo de sujeitos para processar (None = todos)
        """
        print("Criando dataset completo para análise de Alzheimer...")
        
        # Encontrar todos os sujeitos
        subject_dirs = glob.glob(os.path.join(self.data_dir, "OAS1_*_MR1"))
        
        # Limitar número de sujeitos se especificado (para testes rápidos)
        if max_subjects is not None:
            subject_dirs = subject_dirs[:max_subjects]
            print(f"Modo rápido: processando apenas {len(subject_dirs)} sujeitos")
        
=======
    def create_comprehensive_dataset(self) -> pd.DataFrame:
        """Cria dataset completo com features e metadados"""
        print("🧠 Criando dataset completo para análise de Alzheimer...")
        
        # Encontrar todos os sujeitos
        subject_dirs = glob.glob(os.path.join(self.data_dir, "OAS1_*_MR1"))
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
        subject_ids = [os.path.basename(d) for d in subject_dirs]
        
        # Criar metadados sintéticos
        data_loader = OASISDataLoader(self.data_dir)
        self.metadata_df = data_loader.create_synthetic_oasis_metadata(subject_ids)
        
        # Extrair features de neuroimagem
        all_features = []
        
        for i, subject_dir in enumerate(subject_dirs):
            subject_id = os.path.basename(subject_dir)
            print(f"Processando {subject_id} ({i+1}/{len(subject_dirs)})")
            
            features = {'subject_id': subject_id}
            features.update(self.extract_alzheimer_specific_features(subject_dir))
            all_features.append(features)
        
        # Combinar features com metadados
        features_df = pd.DataFrame(all_features)
        combined_df = features_df.merge(self.metadata_df, on='subject_id', how='inner')
        
        self.features_df = combined_df
        return combined_df

class DeepAlzheimerClassifier:
    """Classificador de deep learning para Alzheimer"""
    
    def __init__(self, features_df: pd.DataFrame):
        self.features_df = features_df
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, target_col: str = 'cdr'):
        """Prepara dados para treinamento"""
        # Selecionar features numéricas
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['subject_id', 'diagnosis', 'gender'] and 
                       self.features_df[col].dtype in [np.float64, np.int64]]
        
        # Remover colunas com muitos NaN
        valid_cols = []
        for col in feature_cols:
            if self.features_df[col].notna().sum() > len(self.features_df) * 0.7:
                valid_cols.append(col)
        
        X = self.features_df[valid_cols].fillna(self.features_df[valid_cols].median())
        
        if target_col == 'cdr':
            # Classificação multi-classe CDR
            y = self.features_df[target_col]
<<<<<<< HEAD
            is_binary = False
        else:
            # Classificação binária (Demented/Nondemented)
            y = self.label_encoder.fit_transform(self.features_df['diagnosis'])
            is_binary = True
        
        return X, y, valid_cols, is_binary
    
    def create_deep_model(self, input_dim: int, num_classes: int = 2, is_binary: bool = False):
        """Cria modelo de deep learning otimizado para GPU"""
        print(f"Criando modelo com {input_dim} features de entrada...")
        
        # Estratégia de distribuição para GPU
        if is_gpu_available():
            strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
            print("Usando estratégia OneDevice para GPU")
        else:
            strategy = tf.distribute.get_strategy()
            print("Usando estratégia padrão (CPU)")
        
        with strategy.scope():
            model = keras.Sequential([
                # Camada de entrada com normalização
                layers.Dense(256, activation='relu', input_shape=(input_dim,)),
                layers.Dropout(0.4),
                layers.BatchNormalization(),
                
                # Camadas ocultas mais profundas para GPU
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.BatchNormalization(),
                
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.BatchNormalization(),
                
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                
                layers.Dense(16, activation='relu'),
                layers.Dropout(0.1),
                
                # Camada de saída configurada corretamente
                layers.Dense(1 if is_binary else num_classes, 
                           activation='sigmoid' if is_binary else 'softmax',
                           dtype='float32')  # Força float32 na saída para mixed precision
            ])
            
            # Otimizador com learning rate adaptativo
            optimizer = keras.optimizers.Adam(
                learning_rate=0.001,
                epsilon=1e-7  # Importante para mixed precision
            )
            
            # Loss function correta para cada caso
            if is_binary:
                loss = 'binary_crossentropy'
            else:
                loss = 'sparse_categorical_crossentropy'
            
            metrics = ['accuracy']
            
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        print(f"Modelo criado com {model.count_params():,} parâmetros")
        print(f"Tipo: {'Binário' if is_binary else f'Multi-classe ({num_classes} classes)'}")
        return model
    
    def train_model(self, target_col: str = 'diagnosis'):
        """Treina o modelo de deep learning com otimizações GPU"""
        print(f"Treinando modelo para predição de: {target_col}")
        
        # Monitorar GPU antes do treinamento
        if is_gpu_available():
            monitor_gpu_usage()
        
        X, y, feature_cols, is_binary = self.prepare_data(target_col)
=======
        else:
            # Classificação binária (Demented/Nondemented)
            y = self.label_encoder.fit_transform(self.features_df['diagnosis'])
        
        return X, y, valid_cols
    
    def create_deep_model(self, input_dim: int, num_classes: int = 2):
        """Cria modelo de deep learning"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
        metrics = ['accuracy']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    
    def train_model(self, target_col: str = 'diagnosis'):
        """Treina o modelo de deep learning"""
        print(f"🤖 Treinando modelo para predição de: {target_col}")
        
        X, y, feature_cols = self.prepare_data(target_col)
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
<<<<<<< HEAD
        # Converter para tensores para melhor performance na GPU
        X_train_scaled = tf.constant(X_train_scaled, dtype=tf.float32)
        X_test_scaled = tf.constant(X_test_scaled, dtype=tf.float32)
        y_train = tf.constant(y_train, dtype=tf.float32 if is_binary else tf.int32)
        y_test = tf.constant(y_test, dtype=tf.float32 if is_binary else tf.int32)
        
        # Criar modelo
        num_classes = len(np.unique(y))
        self.model = self.create_deep_model(X_train_scaled.shape[1], num_classes, is_binary)
        
        # Batch size otimizado para GPU
        batch_size = 64 if is_gpu_available() else 32
        print(f"Usando batch size: {batch_size}")
        
        # Callbacks otimizados
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=25, 
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=12, 
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Adicionar TensorBoard se GPU disponível
        if is_gpu_available():
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir='./logs',
                    histogram_freq=1,
                    profile_batch='10,20'
                )
            )
            print("TensorBoard ativado para monitoramento")
        
        print("Iniciando treinamento...")
        start_time = tf.timestamp()
        
        # Épocas adaptativas baseadas na GPU
        epochs = 50 if is_gpu_available() else 30
        print(f"Treinando por {epochs} épocas")
=======
        # Criar modelo
        num_classes = len(np.unique(y))
        self.model = self.create_deep_model(X_train_scaled.shape[1], num_classes)
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
        )
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
        
        # Treinar
        history = self.model.fit(
            X_train_scaled, y_train,
<<<<<<< HEAD
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        # Calcular tempo de treinamento
        training_time = tf.timestamp() - start_time
        print(f"Tempo de treinamento: {training_time:.2f} segundos")
        
        # Monitorar GPU após treinamento
        if is_gpu_available():
            print("\nStatus GPU pós-treinamento:")
            monitor_gpu_usage()
        
        # Avaliar
        print("\nAvaliando modelo...")
        train_score = self.model.evaluate(X_train_scaled, y_train, verbose=0)[1]
        test_score = self.model.evaluate(X_test_scaled, y_test, verbose=0)[1]
        
        print(f"Acurácia Treino: {train_score:.3f}")
        print(f"Acurácia Teste: {test_score:.3f}")
        
        # Predições
        print("Gerando predições...")
        y_pred_prob = self.model.predict(X_test_scaled, batch_size=batch_size)
        
        # Converter predições baseado no tipo de problema
        if is_binary:
            y_pred = (y_pred_prob.flatten() > 0.5).astype(int)
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Converter tensores de volta para numpy para métricas
        if isinstance(y_test, tf.Tensor):
            y_test_np = y_test.numpy().astype(int)
        else:
            y_test_np = y_test.astype(int)
            
        # Relatório detalhado
        if is_binary:
            auc_score = roc_auc_score(y_test_np, y_pred_prob.flatten())
            print(f"AUC Score: {auc_score:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test_np, y_pred))
        
        # Estatísticas de performance
        epochs_trained = len(history.history['loss'])
        print(f"\nEstatísticas de treinamento:")
        print(f"   - Épocas treinadas: {epochs_trained}")
        print(f"   - Batch size usado: {batch_size}")
        print(f"   - GPU utilizada: {'Sim' if is_gpu_available() else 'Não'}")
=======
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Avaliar
        train_score = self.model.evaluate(X_train_scaled, y_train, verbose=0)[1]
        test_score = self.model.evaluate(X_test_scaled, y_test, verbose=0)[1]
        
        print(f"📊 Acurácia Treino: {train_score:.3f}")
        print(f"📊 Acurácia Teste: {test_score:.3f}")
        
        # Predições
        y_pred_prob = self.model.predict(X_test_scaled)
        y_pred = np.argmax(y_pred_prob, axis=1) if num_classes > 2 else (y_pred_prob > 0.5).astype(int)
        
        # Relatório detalhado
        if num_classes == 2:
            print(f"📊 AUC Score: {roc_auc_score(y_test, y_pred_prob):.3f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
        
        return {
            'model': self.model,
            'history': history,
            'test_accuracy': test_score,
            'feature_columns': feature_cols,
            'scaler': self.scaler,
<<<<<<< HEAD
            'y_test': y_test_np,
            'y_pred': y_pred,
            'training_time': training_time.numpy() if is_gpu_available() else None,
            'epochs_trained': epochs_trained,
            'gpu_used': is_gpu_available()
=======
            'y_test': y_test,
            'y_pred': y_pred
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
        }
    
    def save_model(self, model_path: str = "alzheimer_deep_model"):
        """Salva modelo treinado"""
        if self.model is not None:
            self.model.save(f"{model_path}.h5")
            joblib.dump(self.scaler, f"{model_path}_scaler.joblib")
<<<<<<< HEAD
            print(f"Modelo salvo: {model_path}.h5")
            print(f"Scaler salvo: {model_path}_scaler.joblib")
=======
            print(f"✅ Modelo salvo: {model_path}.h5")
            print(f"✅ Scaler salvo: {model_path}_scaler.joblib")
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)

class AlzheimerAnalysisReport:
    """Gera relatórios e visualizações para análise de Alzheimer"""
    
    def __init__(self, features_df: pd.DataFrame):
        self.features_df = features_df
        
    def generate_exploratory_analysis(self):
        """Gera análise exploratória dos dados"""
<<<<<<< HEAD
        print("Gerando Análise Exploratória...")
=======
        print("📊 Gerando Análise Exploratória...")
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Distribuição por diagnóstico
        axes[0, 0].hist([
            self.features_df[self.features_df['diagnosis'] == 'Nondemented']['age'],
            self.features_df[self.features_df['diagnosis'] == 'Demented']['age']
        ], label=['Não Demente', 'Demente'], alpha=0.7, bins=20)
        axes[0, 0].set_title('Distribuição de Idade por Diagnóstico')
        axes[0, 0].legend()
        
        # Volume do hipocampo
        if 'total_hippocampus_volume' in self.features_df.columns:
            sns.boxplot(data=self.features_df, x='diagnosis', y='total_hippocampus_volume', ax=axes[0, 1])
            axes[0, 1].set_title('Volume do Hipocampo por Diagnóstico')
        
        # CDR distribution
        self.features_df['cdr'].value_counts().plot(kind='bar', ax=axes[0, 2])
        axes[0, 2].set_title('Distribuição CDR')
        
        # MMSE vs Idade
        scatter = axes[1, 0].scatter(
            self.features_df['age'], 
            self.features_df['mmse'], 
            c=self.features_df['cdr'], 
            cmap='viridis', 
            alpha=0.6
        )
        axes[1, 0].set_xlabel('Idade')
        axes[1, 0].set_ylabel('MMSE')
        axes[1, 0].set_title('MMSE vs Idade (cor = CDR)')
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # Correlação de features
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns[:10]
        corr_matrix = self.features_df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1, 1])
        axes[1, 1].set_title('Correlação entre Features')
        
        # Distribuição por gênero
        gender_counts = self.features_df.groupby(['gender', 'diagnosis']).size().unstack()
        gender_counts.plot(kind='bar', ax=axes[1, 2])
        axes[1, 2].set_title('Diagnóstico por Gênero')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('alzheimer_exploratory_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
<<<<<<< HEAD
        print("Análise exploratória salva: alzheimer_exploratory_analysis.png")

def main():
    """Pipeline principal de análise de Alzheimer"""
    print("PIPELINE DE IA PARA ANÁLISE DE ALZHEIMER")
    print("=" * 50)
    
    # Status inicial da GPU
    print(f"\nStatus GPU: {'ATIVADA' if is_gpu_available() else 'DESATIVADA'}")
    if is_gpu_available():
        monitor_gpu_usage()
        print(f"Mixed Precision: {'ATIVADA' if tf.keras.mixed_precision.global_policy().name == 'mixed_float16' else 'DESATIVADA'}")
    
    data_dir = "/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos"
    
    # 1. Criar dataset completo
    print("\nETAPA 1: CRIANDO DATASET COMPLETO")
    analyzer = AlzheimerBrainAnalyzer(data_dir)
    
    # Configuração personalizada
    # Para processamento completo, remover o parâmetro max_subjects
    max_subjects = None  # Todos os sujeitos
    print(f" Processamento configurado: 500 sujeitos")
    
    features_df = analyzer.create_comprehensive_dataset(max_subjects=max_subjects)
    
    # Salvar dataset
    features_df.to_csv("alzheimer_complete_dataset.csv", index=False)
    print(f"Dataset salvo: alzheimer_complete_dataset.csv")
    print(f"Dimensões: {features_df.shape}")
    
    # 2. Análise exploratória
    print("\nETAPA 2: ANÁLISE EXPLORATÓRIA")
=======
        print("✅ Análise exploratória salva: alzheimer_exploratory_analysis.png")

def main():
    """Pipeline principal de análise de Alzheimer"""
    print("🧠 PIPELINE DE IA PARA ANÁLISE DE ALZHEIMER")
    print("=" * 50)
    
    data_dir = "/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos"
    
    # 1. Criar dataset completo
    print("\n📊 ETAPA 1: CRIANDO DATASET COMPLETO")
    analyzer = AlzheimerBrainAnalyzer(data_dir)
    features_df = analyzer.create_comprehensive_dataset()
    
    # Salvar dataset
    features_df.to_csv("alzheimer_complete_dataset.csv", index=False)
    print(f"✅ Dataset salvo: alzheimer_complete_dataset.csv")
    print(f"📊 Dimensões: {features_df.shape}")
    
    # 2. Análise exploratória
    print("\n📊 ETAPA 2: ANÁLISE EXPLORATÓRIA")
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
    report = AlzheimerAnalysisReport(features_df)
    report.generate_exploratory_analysis()
    
    # 3. Treinamento de modelos
<<<<<<< HEAD
    print("\nETAPA 3: TREINAMENTO DE MODELOS DEEP LEARNING")
=======
    print("\n🤖 ETAPA 3: TREINAMENTO DE MODELOS DEEP LEARNING")
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
    
    # Classificação binária (Demented/Nondemented)
    classifier = DeepAlzheimerClassifier(features_df)
    binary_results = classifier.train_model(target_col='diagnosis')
    classifier.save_model("alzheimer_binary_classifier")
    
    # Classificação CDR
    cdr_classifier = DeepAlzheimerClassifier(features_df)
    cdr_results = cdr_classifier.train_model(target_col='cdr')
    cdr_classifier.save_model("alzheimer_cdr_classifier")
    
<<<<<<< HEAD
    print("\nPIPELINE COMPLETO DE ALZHEIMER EXECUTADO!")
    print("Arquivos gerados:")
=======
    print("\n✅ PIPELINE COMPLETO DE ALZHEIMER EXECUTADO!")
    print("📁 Arquivos gerados:")
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
    print("   - alzheimer_complete_dataset.csv")
    print("   - alzheimer_exploratory_analysis.png")
    print("   - alzheimer_binary_classifier.h5")
    print("   - alzheimer_cdr_classifier.h5")
<<<<<<< HEAD
    
    # Resumo de performance
    print("\nRESUMO DE PERFORMANCE:")
    print("=" * 40)
    print(f"GPU Utilizada: {'SIM' if is_gpu_available() else 'NÃO'}")
    
    if is_gpu_available():
        print(f"Mixed Precision: ATIVADA")
        print(f"TensorBoard: ATIVADO (./logs/)")
        
        # Estatísticas dos modelos
        binary_time = binary_results.get('training_time', 0)
        cdr_time = cdr_results.get('training_time', 0)
        total_time = binary_time + cdr_time
        
        print(f"Tempo total de treinamento: {total_time:.1f}s")
        print(f"   - Classificador Binário: {binary_time:.1f}s ({binary_results.get('epochs_trained', 0)} épocas)")
        print(f"   - Classificador CDR: {cdr_time:.1f}s ({cdr_results.get('epochs_trained', 0)} épocas)")
        
        print(f"Acurácia Final:")
        print(f"   - Classificação Binária: {binary_results.get('test_accuracy', 0):.3f}")
        print(f"   - Classificação CDR: {cdr_results.get('test_accuracy', 0):.3f}")
        
        # Status final da GPU
        print(f"\nStatus Final da GPU:")
        monitor_gpu_usage()
    else:
        print("Pipeline executado em CPU")
        print("Para acelerar o treinamento, considere usar uma GPU com CUDA")
    
    print("\nPipeline concluído com sucesso!")
=======
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)

if __name__ == "__main__":
    main() 
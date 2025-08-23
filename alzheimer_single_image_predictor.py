#!/usr/bin/env python3
"""
Preditor de Alzheimer para Uma Única Imagem de Ressonância Magnética
Permite testar o diagnóstico usando os modelos treinados com uma única imagem

Funcionalidades:
1. Carregamento de uma única imagem de ressonância (T1.mgz + aparc+aseg.mgz)
2. Extração das mesmas features usadas no treinamento
3. Pré-processamento com o scaler treinado
4. Predição usando modelos binário e multiclasse (CDR)
5. Interface simples para teste
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Importar o setup de GPU do pipeline principal
import sys
sys.path.append('/app/alzheimer')

try:
    from alzheimer_ai_pipeline import setup_gpu_optimization, is_gpu_available
    setup_gpu_optimization()
except ImportError:
    print("Aviso: Não foi possível importar configurações de GPU")
    def is_gpu_available():
        return False

class AlzheimerSingleImagePredictor:
    """Preditor de Alzheimer para uma única imagem de ressonância"""
    
    def __init__(self, models_dir="/app/alzheimer"):
        """
        Inicializa o preditor carregando os modelos treinados
        
        Args:
            models_dir: Diretório onde estão salvos os modelos treinados
        """
        self.models_dir = models_dir
        self.binary_model = None
        self.cdr_model = None
        self.binary_scaler = None
        self.cdr_scaler = None
        self.feature_columns = None
        
        # Configurações das regiões cerebrais (mesmo do pipeline original)
        self.regions_alzheimer = {
            'left_hippocampus': 17,
            'right_hippocampus': 53,
            'left_amygdala': 18,
            'right_amygdala': 54,
            'left_entorhinal': 1006,
            'right_entorhinal': 2006,
            'left_temporal': 1009,
            'right_temporal': 2009
        }
        
        self.load_trained_models()
    
    def load_trained_models(self):
        """Carrega os modelos treinados e scalers"""
        print("Carregando modelos treinados...")
        
        try:
            # Carregar modelo binário v3 (sem data leakage)
            binary_model_path = os.path.join(self.models_dir, "alzheimer_binary_classifier_v3_CORRETO.h5")
            binary_scaler_path = os.path.join(self.models_dir, "alzheimer_binary_classifier_v3_CORRETO_scaler.joblib")
            
            if os.path.exists(binary_model_path) and os.path.exists(binary_scaler_path):
                self.binary_model = tf.keras.models.load_model(binary_model_path)
                self.binary_scaler = joblib.load(binary_scaler_path)
                print("✅ Modelo binário v3 (sem data leakage) carregado com sucesso")
            else:
                print("⚠️ Modelo binário não encontrado")
            
            # Carregar modelo CDR v2 (multiclasse com data augmentation)
            cdr_model_path = os.path.join(self.models_dir, "alzheimer_cdr_classifier_CORRETO_v2.h5")
            cdr_scaler_path = os.path.join(self.models_dir, "alzheimer_cdr_classifier_CORRETO_v2_scaler.joblib")
            
            if os.path.exists(cdr_model_path) and os.path.exists(cdr_scaler_path):
                self.cdr_model = tf.keras.models.load_model(cdr_model_path)
                self.cdr_scaler = joblib.load(cdr_scaler_path)
                print("✅ Modelo CDR v2 (com data augmentation) carregado com sucesso")
            else:
                print("❌ Modelo CDR v2 não encontrado! Execute o treinamento primeiro.")
                self.cdr_model = None
                self.cdr_scaler = None
                
            # Obter colunas de features do scaler se disponível
            if hasattr(self.cdr_scaler, 'feature_names_in_') and self.cdr_scaler is not None:
                self.feature_columns = self.cdr_scaler.feature_names_in_.tolist()
                print(f"📊 Features esperadas: {len(self.feature_columns)}")
                
        except Exception as e:
            print(f"❌ Erro ao carregar modelos: {e}")
    
    def extract_features_from_single_image(self, subject_path):
        """
        Extrai features de uma única imagem (mesmo processo do pipeline original)
        
        Args:
            subject_path: Caminho para o diretório do sujeito contendo mri/T1.mgz e mri/aparc+aseg.mgz
            
        Returns:
            dict: Features extraídas da imagem
        """
        print(f"Extraindo features de: {subject_path}")
        
        features = {}
        
        # Caminhos dos arquivos necessários
        seg_file = os.path.join(subject_path, 'mri', 'aparc+aseg.mgz')
        t1_file = os.path.join(subject_path, 'mri', 'T1.mgz')
        
        # Verificar se os arquivos existem
        if not os.path.exists(seg_file):
            raise FileNotFoundError(f"Arquivo de segmentação não encontrado: {seg_file}")
        if not os.path.exists(t1_file):
            raise FileNotFoundError(f"Arquivo T1 não encontrado: {t1_file}")
        
        try:
            print("📖 Carregando imagens...")
            seg_img = nib.load(seg_file)
            t1_img = nib.load(t1_file)
            
            seg_data = seg_img.get_fdata()
            t1_data = t1_img.get_fdata()
            
            print(f"   Segmentação: {seg_data.shape}")
            print(f"   T1: {t1_data.shape}")
            
            # Calcular volume total do cérebro
            total_brain_volume = np.sum(seg_data > 0)
            print(f"   Volume total do cérebro: {total_brain_volume:,} voxels")
            
            # Extrair features para cada região
            print("🧠 Extraindo features das regiões cerebrais...")
            for region_name, label in self.regions_alzheimer.items():
                mask = seg_data == label
                volume = np.sum(mask)
                
                if volume > 0:
                    features[f'{region_name}_volume'] = volume
                    features[f'{region_name}_volume_norm'] = volume / total_brain_volume
                    features[f'{region_name}_intensity_mean'] = np.mean(t1_data[mask])
                    features[f'{region_name}_intensity_std'] = np.std(t1_data[mask])
                    
                    print(f"   {region_name}: {volume:,} voxels")
                else:
                    print(f"   ⚠️ {region_name}: região não encontrada")
                    # Adicionar valores padrão para features faltantes
                    features[f'{region_name}_volume'] = 0
                    features[f'{region_name}_volume_norm'] = 0
                    features[f'{region_name}_intensity_mean'] = 0
                    features[f'{region_name}_intensity_std'] = 0
            
            # NOTA: As assimetrias não foram incluídas no modelo treinado original
            # portanto não vamos calculá-las aqui para manter compatibilidade
            print("📏 Assimetrias não incluídas (compatibilidade com modelo treinado)")
            
            # Calcular features específicas do hipocampo
            if features.get('left_hippocampus_volume', 0) > 0 and features.get('right_hippocampus_volume', 0) > 0:
                total_hippo = features['left_hippocampus_volume'] + features['right_hippocampus_volume']
                features['total_hippocampus_volume'] = total_hippo
                features['hippocampus_brain_ratio'] = total_hippo / total_brain_volume
                print(f"   Hipocampo total: {total_hippo:,} voxels ({features['hippocampus_brain_ratio']:.6f} do cérebro)")
            
            print(f"✅ Features extraídas: {len(features)} características")
            
        except Exception as e:
            print(f"❌ Erro ao processar imagem: {e}")
            raise
        
        return features
    
    def add_synthetic_demographic_features(self, features, age=75, gender='F', mmse=27, education=16, ses=3):
        """
        Adiciona features demográficas sintéticas (necessárias para o modelo)
        
        Args:
            features: Dict com features já extraídas
            age: Idade do paciente
            gender: Gênero ('M' ou 'F')
            mmse: Score MMSE (0-30)
            education: Anos de educação
            ses: Status socioeconômico (1-5)
        """
        print(f"📋 Adicionando features demográficas:")
        print(f"   Idade: {age}")
        print(f"   Gênero: {gender}")
        print(f"   MMSE: {mmse}")
        print(f"   Educação: {education} anos")
        print(f"   SES: {ses}")
        
        # Estimar CDR baseado em múltiplas features (compatibilidade com modelo mal treinado)
        hippo_ratio = features.get('hippocampus_brain_ratio', 0.007)
        
        # Lógica melhorada combinando MMSE + hipocampo
        if mmse >= 28 and hippo_ratio > 0.0065:
            estimated_cdr = 0.0  # Normal
        elif mmse >= 24 and hippo_ratio > 0.006:
            estimated_cdr = 0.5  # MCI 
        elif mmse >= 18 and hippo_ratio > 0.0055:
            estimated_cdr = 1.0  # Demência leve
        else:
            estimated_cdr = 2.0  # Demência moderada
        
        features.update({
            'age': age,
            'cdr': estimated_cdr,  # Necessário para modelo binário
            'mmse': mmse,
            'education': education,
            'ses': ses
        })
        
        print(f"   CDR estimado (para compatibilidade): {estimated_cdr}")
        
        return features
    
    def prepare_features_for_prediction(self, features, target_col='cdr'):
        """
        Prepara features para predição organizando na ordem correta
        
        Args:
            features: Dict com features extraídas
            target_col: 'cdr' ou 'diagnosis' (target do modelo)
            
        Returns:
            np.array: Features organizadas para predição
        """
        # IMPORTANTE: Usar a ordem exata das features  do dataset de treinamento
        # Baseado no dataset original (excluindo subject_id, diagnosis, gender, e o target)
        # AMBOS os modelos agora usam as mesmas 38 features (SEM CDR)
        # Ambos excluem: subject_id, diagnosis, gender, cdr
        expected_features = [
            'left_hippocampus_volume', 'left_hippocampus_volume_norm', 
            'left_hippocampus_intensity_mean', 'left_hippocampus_intensity_std',
            'right_hippocampus_volume', 'right_hippocampus_volume_norm',
            'right_hippocampus_intensity_mean', 'right_hippocampus_intensity_std',
                'left_amygdala_volume', 'left_amygdala_volume_norm',
                'left_amygdala_intensity_mean', 'left_amygdala_intensity_std',
                'right_amygdala_volume', 'right_amygdala_volume_norm',
                'right_amygdala_intensity_mean', 'right_amygdala_intensity_std',
                'left_entorhinal_volume', 'left_entorhinal_volume_norm',
                'left_entorhinal_intensity_mean', 'left_entorhinal_intensity_std',
                'right_entorhinal_volume', 'right_entorhinal_volume_norm',
                'right_entorhinal_intensity_mean', 'right_entorhinal_intensity_std',
                'left_temporal_volume', 'left_temporal_volume_norm',
                'left_temporal_intensity_mean', 'left_temporal_intensity_std',
                'right_temporal_volume', 'right_temporal_volume_norm',
                'right_temporal_intensity_mean', 'right_temporal_intensity_std',
                'total_hippocampus_volume', 'hippocampus_brain_ratio',
            'age', 'mmse', 'education', 'ses'  # SEM 'cdr' - ambos os modelos corretos!
        ]
        
        # Organizar features na ordem esperada
        feature_array = []
        missing_features = []
        
        for feature_name in expected_features:
            if feature_name in features:
                feature_array.append(features[feature_name])
            else:
                feature_array.append(0.0)  # Valor padrão para features faltantes
                missing_features.append(feature_name)
        
        if missing_features:
            print(f"⚠️ Features faltantes (usando valor 0): {missing_features}")
        
        print(f"🔍 Debug: {len(feature_array)} features preparadas para modelo {target_col}")
        print(f"   Features esperadas: {len(expected_features)}")
        
        return np.array(feature_array).reshape(1, -1)
    
    def predict_single_image(self, subject_path, age=75, gender='F', mmse=27, education=16, ses=3, show_details=True):
        """
        Faz predição completa para uma única imagem
        
        Args:
            subject_path: Caminho para o diretório do sujeito
            age: Idade do paciente
            gender: Gênero
            mmse: Score MMSE
            education: Anos de educação
            ses: Status socioeconômico
            show_details: Mostrar detalhes da predição
            
        Returns:
            dict: Resultados das predições
        """
        print("\n" + "="*60)
        print("🔬 DIAGNÓSTICO DE ALZHEIMER - IMAGEM ÚNICA")
        print("="*60)
        
        results = {
            'subject_path': subject_path,
            'demographics': {'age': age, 'gender': gender, 'mmse': mmse, 'education': education, 'ses': ses},
            'features': {},
            'predictions': {}
        }
        
        try:
            # 1. Extrair features da imagem
            print("\n1️⃣ EXTRAÇÃO DE FEATURES")
            features = self.extract_features_from_single_image(subject_path)
            
            # 2. Adicionar features demográficas
            print("\n2️⃣ ADIÇÃO DE FEATURES DEMOGRÁFICAS")
            features = self.add_synthetic_demographic_features(features, age, gender, mmse, education, ses)
            results['features'] = features.copy()
            
            # 3. Predição binária (Demented vs Non-demented)
            if self.binary_model is not None and self.binary_scaler is not None:
                print("\n3️⃣ PREDIÇÃO BINÁRIA (Demented/Non-demented)")
                
                X_binary = self.prepare_features_for_prediction(features, 'diagnosis')
                X_binary_scaled = self.binary_scaler.transform(X_binary)
                
                binary_prob = self.binary_model.predict(X_binary_scaled, verbose=0)[0][0]
                binary_pred = 1 if binary_prob > 0.5 else 0
                binary_diagnosis = "Demented" if binary_pred == 1 else "Non-demented"
                
                results['predictions']['binary'] = {
                    'probability': float(binary_prob),
                    'prediction': int(binary_pred),
                    'diagnosis': binary_diagnosis,
                    'confidence': float(max(binary_prob, 1-binary_prob))
                }
                
                if show_details:
                    print(f"   📊 Probabilidade: {binary_prob:.3f}")
                    print(f"   🎯 Predição: {binary_diagnosis}")
                    print(f"   🎲 Confiança: {results['predictions']['binary']['confidence']:.3f}")
            
            # 4. Predição CDR (multiclasse)
            if self.cdr_model is not None and self.cdr_scaler is not None:
                print("\n4️⃣ PREDIÇÃO CDR (Clinical Dementia Rating)")
                print("✅ Usando modelo CDR corrigido (sem incluir 'cdr' como feature)")
                
                X_cdr = self.prepare_features_for_prediction(features, 'cdr')
                X_cdr_scaled = self.cdr_scaler.transform(X_cdr)
                
                cdr_probs = self.cdr_model.predict(X_cdr_scaled, verbose=0)[0]
                cdr_pred_idx = np.argmax(cdr_probs)
                
                # Mapear índices para valores CDR
                cdr_mapping = {0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0}
                cdr_pred = cdr_mapping.get(cdr_pred_idx, 0.0)
                
                # Interpretação do CDR
                cdr_interpretation = {
                    0.0: "Normal",
                    0.5: "Demência Questionável (MCI)",
                    1.0: "Demência Leve",
                    2.0: "Demência Moderada"
                }
                
                results['predictions']['cdr'] = {
                    'probabilities': cdr_probs.tolist(),
                    'prediction_index': int(cdr_pred_idx),
                    'cdr_score': float(cdr_pred),
                    'interpretation': cdr_interpretation.get(cdr_pred, "Desconhecido"),
                    'confidence': float(np.max(cdr_probs))
                }
                
                if show_details:
                    print(f"   📊 Probabilidades por classe:")
                    for i, prob in enumerate(cdr_probs):
                        cdr_val = cdr_mapping.get(i, i)
                        interpretation = cdr_interpretation.get(cdr_val, "Desconhecido")
                        print(f"      CDR {cdr_val} ({interpretation}): {prob:.3f}")
                    print(f"   🎯 Predição: CDR {cdr_pred} ({cdr_interpretation.get(cdr_pred, 'Desconhecido')})")
                    print(f"   🎲 Confiança: {results['predictions']['cdr']['confidence']:.3f}")
            
            # 5. Resumo final
            print("\n" + "="*60)
            print("📋 RESUMO DO DIAGNÓSTICO")
            print("="*60)
            
            if 'binary' in results['predictions']:
                print(f"🔍 Classificação Binária: {results['predictions']['binary']['diagnosis']}")
                print(f"   Confiança: {results['predictions']['binary']['confidence']:.1%}")
            
            if 'cdr' in results['predictions']:
                cdr_info = results['predictions']['cdr']
                print(f"🎯 Score CDR: {cdr_info['cdr_score']} ({cdr_info['interpretation']})")
                print(f"   Confiança: {cdr_info['confidence']:.1%}")
            
            print(f"\n👤 Dados do Paciente:")
            print(f"   Idade: {age} anos")
            print(f"   MMSE: {mmse}/30")
            print(f"   Educação: {education} anos")
            
            return results
            
        except Exception as e:
            print(f"❌ Erro durante a predição: {e}")
            results['error'] = str(e)
            return results
    
    def create_prediction_visualization(self, results, save_path="predicao_alzheimer.png"):
        """Cria visualização dos resultados da predição"""
        if 'error' in results:
            print(f"❌ Não é possível criar visualização devido ao erro: {results['error']}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Predição binária
        if 'binary' in results['predictions']:
            binary_data = results['predictions']['binary']
            
            # Gráfico de barras para classificação binária
            categories = ['Non-demented', 'Demented']
            probs = [1 - binary_data['probability'], binary_data['probability']]
            colors = ['green' if binary_data['diagnosis'] == 'Non-demented' else 'red', 
                     'red' if binary_data['diagnosis'] == 'Demented' else 'green']
            
            bars = axes[0, 0].bar(categories, probs, color=colors, alpha=0.7)
            axes[0, 0].set_title('Classificação Binária', fontweight='bold')
            axes[0, 0].set_ylabel('Probabilidade')
            axes[0, 0].set_ylim(0, 1)
            
            # Adicionar valores nas barras
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Predição CDR
        if 'cdr' in results['predictions']:
            cdr_data = results['predictions']['cdr']
            
            cdr_labels = ['CDR 0.0\n(Normal)', 'CDR 0.5\n(MCI)', 'CDR 1.0\n(Leve)', 'CDR 2.0\n(Moderada)']
            cdr_probs = cdr_data['probabilities']
            
            # Colorir barra predita de forma diferente
            colors = ['lightblue'] * len(cdr_probs)
            colors[cdr_data['prediction_index']] = 'orange'
            
            bars = axes[0, 1].bar(cdr_labels, cdr_probs, color=colors, alpha=0.7)
            axes[0, 1].set_title('Classificação CDR (Multiclasse)', fontweight='bold')
            axes[0, 1].set_ylabel('Probabilidade')
            axes[0, 1].set_ylim(0, 1)
            plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
            
            # Adicionar valores nas barras
            for bar, prob in zip(bars, cdr_probs):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 3. Features principais do hipocampo
        if 'features' in results:
            features = results['features']
            hippo_features = {
                'Volume Total': features.get('total_hippocampus_volume', 0),
                'Ratio Cérebro': features.get('hippocampus_brain_ratio', 0) * 1000,  # Escalar para visualização
                'Vol. Esquerdo': features.get('left_hippocampus_volume', 0),
                'Vol. Direito': features.get('right_hippocampus_volume', 0)
            }
            
            bars = axes[1, 0].bar(hippo_features.keys(), hippo_features.values(), 
                                 color='lightcoral', alpha=0.7)
            axes[1, 0].set_title('Features do Hipocampo', fontweight='bold')
            axes[1, 0].set_ylabel('Valores')
            plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
            
            # Adicionar valores nas barras
            for bar, (name, value) in zip(bars, hippo_features.items()):
                height = bar.get_height()
                if name == 'Ratio Cérebro':
                    text = f'{value/1000:.6f}'  # Mostrar valor real
                else:
                    text = f'{value:.0f}'
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               text, ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 4. Dados demográficos
        if 'demographics' in results:
            demo = results['demographics']
            demo_data = {
                'Idade': demo['age'],
                'MMSE': demo['mmse'],
                'Educação': demo['education'],
                'SES': demo['ses']
            }
            
            bars = axes[1, 1].bar(demo_data.keys(), demo_data.values(), 
                                 color='lightgreen', alpha=0.7)
            axes[1, 1].set_title('Dados Demográficos', fontweight='bold')
            axes[1, 1].set_ylabel('Valores')
            
            # Adicionar valores nas barras
            for bar, (name, value) in zip(bars, demo_data.items()):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{value}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualização salva: {save_path}")
        return save_path

def main():
    """Exemplo de uso do preditor"""
    print("🧠 TESTE DO PREDITOR DE ALZHEIMER PARA IMAGEM ÚNICA")
    print("="*60)
    
    # Inicializar preditor
    predictor = AlzheimerSingleImagePredictor()
    
    # Exemplo de uso (substituir pelo caminho real)
    example_subject_path = "/app/alzheimer/data/raw/oasis_data/outputs_fastsurfer_definitivo_todos/OAS1_0001_MR1"
    
    if os.path.exists(example_subject_path):
        print(f"🎯 Testando com: {example_subject_path}")
        
        # Fazer predição
        results = predictor.predict_single_image(
            subject_path=example_subject_path,
            age=75,
            gender='F',
            mmse=28,
            education=16,
            ses=3
        )
        
        # Criar visualização
        predictor.create_prediction_visualization(results)
        
    else:
        print(f"⚠️ Caminho de exemplo não encontrado: {example_subject_path}")
        print("\n📋 COMO USAR:")
        print("1. Tenha um diretório com estrutura:")
        print("   subject_folder/")
        print("   ├── mri/")
        print("   │   ├── T1.mgz")
        print("   │   └── aparc+aseg.mgz")
        print("\n2. Use o código:")
        print("   predictor = AlzheimerSingleImagePredictor()")
        print("   results = predictor.predict_single_image('caminho/para/sujeito', age=75, mmse=28)")
        print("   predictor.create_prediction_visualization(results)")

if __name__ == "__main__":
    main()

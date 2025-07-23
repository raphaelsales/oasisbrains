#!/usr/bin/env python3
"""
Sistema de Extração de Features para Treinamento de IA
Extrai características dos dados processados pelo FastSurfer para treinar modelos de machine learning

Funcionalidades:
1. Extrai features volumétricas de todas as regiões cerebrais
2. Extrai features morfométricas (espessura cortical, curvatura)
3. Cria features do hipocampo especificamente
4. Prepara dataset em formato CSV para ML
5. Implementa modelos de classificação/regressão
6. Avaliação e validação cruzada
"""

import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class FastSurferFeatureExtractor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.features_df = None
        self.scaler = StandardScaler()
        
    def extract_volumetric_features(self, subject_path: str) -> Dict[str, float]:
        """Extrai features volumétricas do arquivo aseg.stats"""
        stats_file = os.path.join(subject_path, 'stats', 'aseg.stats')
        features = {}
        
        if not os.path.exists(stats_file):
            return features
            
        with open(stats_file, 'r') as f:
            for line in f:
                if line.startswith('# Measure'):
                    parts = line.strip().split(', ')
                    if len(parts) >= 4:
                        measure_name = parts[1]
                        try:
                            value = float(parts[3])
                            features[f"vol_{measure_name}"] = value
                        except:
                            continue
                            
        return features
    
    def extract_cortical_features(self, subject_path: str) -> Dict[str, float]:
        """Extrai features corticais (espessura, área, curvatura)"""
        features = {}
        
        for hemisphere in ['lh', 'rh']:
            # Espessura cortical
            aparc_file = os.path.join(subject_path, 'stats', f'{hemisphere}.aparc.DKTatlas.mapped.stats')
            if os.path.exists(aparc_file):
                thickness_values = self._parse_cortical_stats(aparc_file, 'thickness')
                for region, value in thickness_values.items():
                    features[f"{hemisphere}_thickness_{region}"] = value
                    
            # Curvatura
            curv_file = os.path.join(subject_path, 'stats', f'{hemisphere}.curv.stats')
            if os.path.exists(curv_file):
                curv_values = self._parse_curvature_stats(curv_file)
                for metric, value in curv_values.items():
                    features[f"{hemisphere}_curv_{metric}"] = value
                    
        return features
    
    def _parse_cortical_stats(self, stats_file: str, metric: str) -> Dict[str, float]:
        """Parse arquivos de estatísticas corticais"""
        values = {}
        with open(stats_file, 'r') as f:
            for line in f:
                if not line.startswith('#') and line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        region = parts[0]
                        try:
                            if metric == 'thickness':
                                thickness = float(parts[4])  # Coluna ThickAvg
                                values[region] = thickness
                        except:
                            continue
        return values
    
    def _parse_curvature_stats(self, curv_file: str) -> Dict[str, float]:
        """Parse estatísticas de curvatura"""
        values = {}
        with open(curv_file, 'r') as f:
            for line in f:
                if 'total surface area' in line:
                    try:
                        values['surface_area'] = float(line.split()[-2])
                    except:
                        continue
                elif 'mean curvature' in line:
                    try:
                        values['mean_curvature'] = float(line.split()[-1])
                    except:
                        continue
        return values
    
    def extract_hippocampus_features(self, subject_path: str) -> Dict[str, float]:
        """Extrai features específicas do hipocampo"""
        features = {}
        
        # Carregar segmentação
        seg_file = os.path.join(subject_path, 'mri', 'aparc+aseg.mgz')
        t1_file = os.path.join(subject_path, 'mri', 'T1.mgz')
        
        if os.path.exists(seg_file) and os.path.exists(t1_file):
            try:
                seg_img = nib.load(seg_file)
                t1_img = nib.load(t1_file)
                
                seg_data = seg_img.get_fdata()
                t1_data = t1_img.get_fdata()
                
                # Hipocampo esquerdo (17) e direito (53)
                left_hippo_mask = seg_data == 17
                right_hippo_mask = seg_data == 53
                
                # Volume (número de voxels)
                features['left_hippo_volume'] = np.sum(left_hippo_mask)
                features['right_hippo_volume'] = np.sum(right_hippo_mask)
                features['total_hippo_volume'] = features['left_hippo_volume'] + features['right_hippo_volume']
                
                # Intensidades médias
                if np.any(left_hippo_mask):
                    features['left_hippo_intensity_mean'] = np.mean(t1_data[left_hippo_mask])
                    features['left_hippo_intensity_std'] = np.std(t1_data[left_hippo_mask])
                
                if np.any(right_hippo_mask):
                    features['right_hippo_intensity_mean'] = np.mean(t1_data[right_hippo_mask])
                    features['right_hippo_intensity_std'] = np.std(t1_data[right_hippo_mask])
                
                # Assimetria
                if features['left_hippo_volume'] + features['right_hippo_volume'] > 0:
                    features['hippo_asymmetry'] = abs(features['left_hippo_volume'] - features['right_hippo_volume']) / features['total_hippo_volume']
                
            except Exception as e:
                print(f"Erro ao processar hipocampo de {subject_path}: {e}")
                
        return features
    
    def extract_all_features(self) -> pd.DataFrame:
        """Extrai todas as features de todos os sujeitos"""
        all_features = []
        
        print("Extraindo features de todos os sujeitos...")
        
        # Encontrar todos os sujeitos processados
        subject_dirs = glob.glob(os.path.join(self.data_dir, "OAS1_*_MR1"))
        
        for i, subject_dir in enumerate(subject_dirs):
            subject_id = os.path.basename(subject_dir)
            print(f"Processando {subject_id} ({i+1}/{len(subject_dirs)})")
            
            features = {'subject_id': subject_id}
            
            # Extrair diferentes tipos de features
            features.update(self.extract_volumetric_features(subject_dir))
            features.update(self.extract_cortical_features(subject_dir))
            features.update(self.extract_hippocampus_features(subject_dir))
            
            # Adicionar metadados do OASIS se disponível
            features.update(self._extract_oasis_metadata(subject_id))
            
            all_features.append(features)
        
        self.features_df = pd.DataFrame(all_features)
        return self.features_df
    
    def _extract_oasis_metadata(self, subject_id: str) -> Dict[str, Any]:
        """Extrai metadados do OASIS (idade, sexo, diagnóstico) se disponível"""
        # Por enquanto, extrair informações básicas do ID
        features = {}
        
        # Extrair número do sujeito
        try:
            subject_num = int(subject_id.split('_')[1])
            features['subject_number'] = subject_num
        except:
            features['subject_number'] = 0
            
        # Aqui você pode adicionar lógica para carregar metadados reais do OASIS
        # como idade, sexo, CDR (Clinical Dementia Rating), etc.
        
        return features
    
    def save_features(self, output_file: str = "features_dataset.csv"):
        """Salva features em arquivo CSV"""
        if self.features_df is not None:
            # Remover colunas com muitos valores NaN
            threshold = len(self.features_df) * 0.5  # 50% dos dados
            cleaned_df = self.features_df.dropna(axis=1, thresh=threshold)
            
            # Preencher NaN restantes com mediana
            numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(cleaned_df[numeric_columns].median())
            
            cleaned_df.to_csv(output_file, index=False)
            print(f"✅ Features salvas em: {output_file}")
            print(f"📊 Dimensões: {cleaned_df.shape}")
            print(f"📋 Features disponíveis: {len(cleaned_df.columns)-1}")
            
            return cleaned_df
        else:
            print("❌ Nenhuma feature extraída ainda. Execute extract_all_features() primeiro.")
            return None

class AIModelTrainer:
    def __init__(self, features_df: pd.DataFrame):
        self.features_df = features_df
        self.models = {}
        self.scalers = {}
        
    def prepare_classification_data(self, target_column: str = None):
        """Prepara dados para classificação (ex: normal vs demência)"""
        if target_column is None:
            # Criar target sintético baseado no volume do hipocampo
            # (normalmente você teria labels reais do OASIS)
            median_hippo = self.features_df['total_hippo_volume'].median()
            self.features_df['synthetic_diagnosis'] = (self.features_df['total_hippo_volume'] < median_hippo * 0.8).astype(int)
            target_column = 'synthetic_diagnosis'
        
        # Separar features e target
        feature_columns = [col for col in self.features_df.columns if col not in ['subject_id', target_column]]
        X = self.features_df[feature_columns]
        y = self.features_df[target_column]
        
        return X, y, feature_columns
    
    def train_classification_models(self, target_column: str = None):
        """Treina modelos de classificação"""
        X, y, feature_columns = self.prepare_classification_data(target_column)
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Escalar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['classification'] = scaler
        
        # Treinar modelos
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🤖 Treinando {name}...")
            
            # Treinar
            model.fit(X_train_scaled, y_train)
            
            # Avaliar
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            # Predições
            y_pred = model.predict(X_test_scaled)
            
            results[name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'y_test': y_test
            }
            
            print(f"  📊 Acurácia treino: {train_score:.3f}")
            print(f"  📊 Acurácia teste: {test_score:.3f}")
            print(f"  📊 CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # Feature importance (para Random Forest)
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"  🔝 Top 5 features importantes:")
                for idx, row in feature_importance.head().iterrows():
                    print(f"     {row['feature']}: {row['importance']:.3f}")
        
        self.models['classification'] = results
        return results
    
    def train_regression_models(self, target_column: str = 'total_hippo_volume'):
        """Treina modelos de regressão para predizer volumes"""
        feature_columns = [col for col in self.features_df.columns 
                          if col not in ['subject_id', target_column] and col != target_column]
        
        X = self.features_df[feature_columns]
        y = self.features_df[target_column]
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Escalar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['regression'] = scaler
        
        # Treinar modelos
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🤖 Treinando {name} (Regressão)...")
            
            # Treinar
            model.fit(X_train_scaled, y_train)
            
            # Predições
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Métricas
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'predictions': y_pred_test,
                'y_test': y_test
            }
            
            print(f"  📊 R² treino: {train_r2:.3f}")
            print(f"  📊 R² teste: {test_r2:.3f}")
            print(f"  📊 RMSE treino: {train_rmse:.1f}")
            print(f"  📊 RMSE teste: {test_rmse:.1f}")
        
        self.models['regression'] = results
        return results
    
    def save_models(self, output_dir: str = "trained_models"):
        """Salva modelos treinados"""
        os.makedirs(output_dir, exist_ok=True)
        
        for task_type, task_models in self.models.items():
            for model_name, model_data in task_models.items():
                model_file = os.path.join(output_dir, f"{task_type}_{model_name}_model.joblib")
                joblib.dump(model_data['model'], model_file)
                print(f"✅ Modelo salvo: {model_file}")
        
        # Salvar scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_file = os.path.join(output_dir, f"{scaler_name}_scaler.joblib")
            joblib.dump(scaler, scaler_file)
            print(f"✅ Scaler salvo: {scaler_file}")

def main():
    """Função principal para executar o pipeline completo"""
    print("🧠 SISTEMA DE TREINAMENTO DE IA PARA DADOS CEREBRAIS")
    print("=" * 60)
    
    # Configurações
    data_dir = "/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos"
    
    # 1. Extrair features
    print("\n📊 ETAPA 1: EXTRAÇÃO DE FEATURES")
    extractor = FastSurferFeatureExtractor(data_dir)
    features_df = extractor.extract_all_features()
    cleaned_df = extractor.save_features("brain_features_dataset.csv")
    
    if cleaned_df is None:
        print("❌ Erro na extração de features")
        return
    
    # 2. Treinar modelos
    print("\n🤖 ETAPA 2: TREINAMENTO DE MODELOS")
    trainer = AIModelTrainer(cleaned_df)
    
    # Classificação
    print("\n🎯 TREINAMENTO DE CLASSIFICAÇÃO:")
    classification_results = trainer.train_classification_models()
    
    # Regressão
    print("\n📈 TREINAMENTO DE REGRESSÃO:")
    regression_results = trainer.train_regression_models()
    
    # 3. Salvar modelos
    print("\n💾 ETAPA 3: SALVANDO MODELOS")
    trainer.save_models()
    
    print("\n✅ PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
    print(f"📁 Dataset: brain_features_dataset.csv")
    print(f"📁 Modelos: trained_models/")

if __name__ == "__main__":
    main() 
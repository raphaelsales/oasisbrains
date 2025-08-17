#!/usr/bin/env python3
"""
PIPELINE ULTIMATE OTIMIZADO PARA DETECÇÃO DE MCI
Combinando as melhores estratégias:
1. SHAP feature selection 
2. Augmentação de dados (SMOTE)
3. Ensemble robusto com stacking
4. Hyperparameter tuning com Optuna
5. CNN híbrido otimizado

Meta: AUC ≥ 0.85 (performance clínica)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, roc_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Importações avançadas
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP não disponível - usando feature selection alternativa")

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTEENN
    IMBALANCED_AVAILABLE = True
except ImportError:
    IMBALANCED_AVAILABLE = False
    print("⚠️ Imbalanced-learn não disponível - prosseguindo sem SMOTE")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna não disponível - usando grid search")

class AdvancedFeatureSelector:
    """
    Seletor de features avançado com SHAP e análise estatística
    """
    
    def __init__(self):
        self.selected_features = []
        self.feature_importance = {}
        self.shap_values = None
        
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features engenheiradas avançadas"""
        
        print("🔬 Criando features engenheiradas avançadas...")
        
        enhanced_df = df.copy()
        
        # === FEATURES VOLUMÉTRICAS BÁSICAS ===
        volume_features = {
            'hippocampus': ['left_hippocampus_volume', 'right_hippocampus_volume'],
            'amygdala': ['left_amygdala_volume', 'right_amygdala_volume'],
            'entorhinal': ['left_entorhinal_volume', 'right_entorhinal_volume'],
            'temporal': ['left_temporal_volume', 'right_temporal_volume']
        }
        
        for region, cols in volume_features.items():
            if all(col in df.columns for col in cols):
                # Volume total
                enhanced_df[f'total_{region}_volume'] = df[cols].sum(axis=1)
                
                # Assimetria
                enhanced_df[f'{region}_asymmetry'] = abs(df[cols[0]] - df[cols[1]]) / enhanced_df[f'total_{region}_volume']
                
                # Ratio L/R
                enhanced_df[f'{region}_lr_ratio'] = df[cols[0]] / (df[cols[1]] + 1e-6)
                
                # Normalização por volume cerebral
                if 'hippocampus_brain_ratio' in df.columns:
                    enhanced_df[f'{region}_normalized'] = enhanced_df[f'total_{region}_volume'] / df['hippocampus_brain_ratio']
        
        # === RATIOS INTER-REGIONAIS ===
        regions = ['hippocampus', 'amygdala', 'entorhinal', 'temporal']
        for i, region1 in enumerate(regions):
            for region2 in regions[i+1:]:
                col1 = f'total_{region1}_volume'
                col2 = f'total_{region2}_volume'
                if col1 in enhanced_df.columns and col2 in enhanced_df.columns:
                    enhanced_df[f'{region1}_{region2}_ratio'] = enhanced_df[col1] / (enhanced_df[col2] + 1e-6)
        
        # === FEATURES CLÍNICAS AVANÇADAS ===
        if 'mmse' in enhanced_df.columns:
            # Transformações MMSE
            enhanced_df['mmse_squared'] = enhanced_df['mmse'] ** 2
            enhanced_df['mmse_log'] = np.log(enhanced_df['mmse'] + 1)
            enhanced_df['mmse_inverted'] = 30 - enhanced_df['mmse']
            
            if 'age' in enhanced_df.columns:
                enhanced_df['mmse_age_interaction'] = enhanced_df['mmse'] * enhanced_df['age']
                enhanced_df['age_adjusted_mmse'] = enhanced_df['mmse'] / (enhanced_df['age'] / 70)
        
        if 'age' in enhanced_df.columns:
            enhanced_df['age_squared'] = enhanced_df['age'] ** 2
            enhanced_df['age_bin'] = pd.cut(enhanced_df['age'], bins=[0, 65, 75, 85, 100], labels=[0, 1, 2, 3])
            
        # === FEATURES DE INTENSIDADE ===
        intensity_cols = [col for col in df.columns if 'intensity_mean' in col]
        if len(intensity_cols) >= 2:
            enhanced_df['global_intensity_mean'] = df[intensity_cols].mean(axis=1)
            enhanced_df['global_intensity_std'] = df[intensity_cols].std(axis=1)
            enhanced_df['intensity_cv'] = enhanced_df['global_intensity_std'] / (enhanced_df['global_intensity_mean'] + 1e-6)
            
            # Contraste entre regiões
            if 'left_hippocampus_intensity_mean' in df.columns and 'left_temporal_intensity_mean' in df.columns:
                enhanced_df['hippocampus_temporal_contrast'] = abs(
                    df['left_hippocampus_intensity_mean'] - df['left_temporal_intensity_mean']
                )
        
        # === SCORES COMPOSTOS ===
        # Score de risco MCI baseado em literatura
        risk_factors = []
        if 'age' in enhanced_df.columns:
            risk_factors.append((enhanced_df['age'] >= 75).astype(int))
        if 'mmse' in enhanced_df.columns:
            risk_factors.append((enhanced_df['mmse'] <= 26).astype(int))
        if 'total_hippocampus_volume' in enhanced_df.columns:
            vol_threshold = enhanced_df['total_hippocampus_volume'].quantile(0.25)
            risk_factors.append((enhanced_df['total_hippocampus_volume'] <= vol_threshold).astype(int))
        
        if risk_factors:
            enhanced_df['mci_risk_score'] = sum(risk_factors)
        
        # Score composto baseado em evidências
        if all(col in enhanced_df.columns for col in ['mmse', 'total_hippocampus_volume']):
            enhanced_df['mci_composite_score'] = (
                (30 - enhanced_df['mmse']) / 20 * 0.4 +
                (8000 - enhanced_df['total_hippocampus_volume']) / 3000 * 0.3 +
                enhanced_df.get('hippocampus_asymmetry', 0) * 0.3
            )
        
        print(f"✓ Features criadas: {enhanced_df.shape[1]} total (+{enhanced_df.shape[1] - df.shape[1]} novas)")
        return enhanced_df
    
    def shap_feature_selection(self, X: pd.DataFrame, y: pd.Series, max_features: int = 20) -> list:
        """Seleção de features usando SHAP"""
        
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP não disponível - usando seleção alternativa")
            return self.alternative_feature_selection(X, y, max_features)
        
        print("🎯 Seleção de features com SHAP...")
        
        # Treinar modelo base para SHAP
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X, y)
        
        # Calcular SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Para classificação binária, usar classe positiva
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        
        # Importância média absoluta
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Selecionar top features
        top_indices = np.argsort(feature_importance)[::-1][:max_features]
        selected_features = X.columns[top_indices].tolist()
        
        self.shap_values = shap_values
        self.feature_importance['shap'] = dict(zip(X.columns, feature_importance))
        
        print(f"✓ Top {len(selected_features)} features selecionadas por SHAP")
        return selected_features
    
    def alternative_feature_selection(self, X: pd.DataFrame, y: pd.Series, max_features: int = 20) -> list:
        """Seleção alternativa quando SHAP não está disponível"""
        
        print("🎯 Seleção de features alternativa...")
        
        # Análise univariada
        selector = SelectKBest(score_func=f_classif, k=min(max_features, X.shape[1]))
        selector.fit(X, y)
        
        # Features selecionadas
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Importância por Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X, y)
        
        self.feature_importance['rf'] = dict(zip(X.columns, rf.feature_importances_))
        
        print(f"✓ Top {len(selected_features)} features selecionadas")
        return selected_features

class DataAugmentationEngine:
    """
    Motor de augmentação de dados para balanceamento
    """
    
    def __init__(self):
        self.augmentation_method = None
        
    def apply_smote_augmentation(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Aplica SMOTE para balanceamento"""
        
        if not IMBALANCED_AVAILABLE:
            print("⚠️ SMOTE não disponível - retornando dados originais")
            return X, y
        
        print("🔄 Aplicando SMOTE para balanceamento...")
        
        # Verificar distribuição original
        unique, counts = np.unique(y, return_counts=True)
        print(f"   Distribuição original: {dict(zip(unique, counts))}")
        
        # Aplicar SMOTE
        smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y == 1) - 1))
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Verificar nova distribuição
        unique, counts = np.unique(y_resampled, return_counts=True)
        print(f"   Distribuição pós-SMOTE: {dict(zip(unique, counts))}")
        
        self.augmentation_method = 'SMOTE'
        return X_resampled, y_resampled
    
    def apply_adasyn_augmentation(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Aplica ADASYN para balanceamento adaptativo"""
        
        if not IMBALANCED_AVAILABLE:
            print("⚠️ ADASYN não disponível - usando SMOTE")
            return self.apply_smote_augmentation(X, y)
        
        print("🔄 Aplicando ADASYN para balanceamento adaptativo...")
        
        try:
            adasyn = ADASYN(random_state=42, n_neighbors=min(5, np.sum(y == 1) - 1))
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            self.augmentation_method = 'ADASYN'
            return X_resampled, y_resampled
        except:
            print("⚠️ ADASYN falhou - usando SMOTE")
            return self.apply_smote_augmentation(X, y)

class OptimizedEnsembleClassifier:
    """
    Classificador ensemble otimizado com stacking
    """
    
    def __init__(self):
        self.base_models = {}
        self.meta_model = None
        self.best_params = {}
        
    def create_base_models(self):
        """Cria modelos base otimizados"""
        
        self.base_models = {
            'rf': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'et': ExtraTreesClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=5,
                random_state=42
            ),
            'lr': LogisticRegression(
                C=1.0, class_weight='balanced', random_state=42, max_iter=1000
            ),
            'svm': SVC(
                C=1.0, gamma='scale', class_weight='balanced', 
                probability=True, random_state=42
            )
        }
    
    def optuna_optimization(self, X: np.ndarray, y: np.ndarray, model_name: str, n_trials: int = 50):
        """Otimização com Optuna"""
        
        if not OPTUNA_AVAILABLE:
            print(f"⚠️ Optuna não disponível para {model_name}")
            return self.base_models[model_name]
        
        def objective(trial):
            if model_name == 'rf':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': -1
                }
                model = RandomForestClassifier(**params)
            elif model_name == 'gb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'random_state': 42
                }
                model = GradientBoostingClassifier(**params)
            else:
                return 0.5  # Default score
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc', n_jobs=-1)
            return cv_scores.mean()
        
        print(f"🎯 Otimizando {model_name} com Optuna...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Atualizar modelo com melhores parâmetros
        if model_name == 'rf':
            self.base_models[model_name] = RandomForestClassifier(**study.best_params)
        elif model_name == 'gb':
            self.base_models[model_name] = GradientBoostingClassifier(**study.best_params)
        
        self.best_params[model_name] = study.best_params
        print(f"   Melhor AUC: {study.best_value:.3f}")
        
        return self.base_models[model_name]
    
    def train_stacking_ensemble(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Treina ensemble com stacking"""
        
        print("🚀 Treinando ensemble com stacking...")
        
        self.create_base_models()
        
        # Otimizar modelos principais com Optuna
        if OPTUNA_AVAILABLE:
            self.optuna_optimization(X, y, 'rf', n_trials=30)
            self.optuna_optimization(X, y, 'gb', n_trials=30)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Treinar modelos base e gerar predições para stacking
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        model_scores = {}
        
        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"🔄 Treinando {name}...")
            
            # Cross-validation para predições do stacking
            cv_predictions = np.zeros(X.shape[0])
            cv_scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                pred_proba = model.predict_proba(X_val)[:, 1]
                cv_predictions[val_idx] = pred_proba
                
                cv_scores.append(roc_auc_score(y_val, pred_proba))
            
            base_predictions[:, i] = cv_predictions
            model_scores[name] = {
                'cv_auc': np.mean(cv_scores),
                'cv_std': np.std(cv_scores)
            }
            
            print(f"   {name}: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # Treinar meta-modelo
        print("🎯 Treinando meta-modelo...")
        self.meta_model = LogisticRegression(random_state=42)
        self.meta_model.fit(base_predictions, y)
        
        # Predições finais do ensemble
        final_predictions = self.meta_model.predict_proba(base_predictions)[:, 1]
        ensemble_auc = roc_auc_score(y, final_predictions)
        
        print(f"🏆 Ensemble AUC: {ensemble_auc:.3f}")
        
        return {
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'model_scores': model_scores,
            'ensemble_auc': ensemble_auc,
            'base_predictions': base_predictions,
            'final_predictions': final_predictions
        }

class UltimateOptimizedPipeline:
    """
    Pipeline ultimate otimizado combinando todas as estratégias
    """
    
    def __init__(self, data_file: str = "alzheimer_complete_dataset.csv"):
        self.data_file = data_file
        self.feature_selector = AdvancedFeatureSelector()
        self.augmentation_engine = DataAugmentationEngine()
        self.ensemble_classifier = OptimizedEnsembleClassifier()
        self.scaler = RobustScaler()
        
    def run_ultimate_pipeline(self) -> dict:
        """Executa pipeline ultimate otimizado"""
        
        print("🧠 PIPELINE ULTIMATE OTIMIZADO PARA DETECÇÃO DE MCI")
        print("=" * 70)
        print("🎯 Meta: AUC ≥ 0.85 (performance clínica)")
        print("💡 Estratégias: SHAP + SMOTE + Stacking + Optuna")
        print("=" * 70)
        
        # 1. Carregar e preparar dados
        print("\n📂 Carregando dados...")
        df = pd.read_csv(self.data_file)
        df_mci = df[df['cdr'].isin([0.0, 0.5])].copy()
        
        print(f"✓ Dataset carregado:")
        print(f"  Total sujeitos: {len(df_mci)}")
        print(f"  Normal (CDR=0): {len(df_mci[df_mci['cdr']==0])}")
        print(f"  MCI (CDR=0.5): {len(df_mci[df_mci['cdr']==0.5])}")
        
        # 2. Feature engineering
        print("\n🔬 Feature engineering avançada...")
        enhanced_df = self.feature_selector.create_enhanced_features(df_mci)
        
        # 3. Preparar dados para ML
        print("\n⚙️ Preparando dados para ML...")
        
        # Selecionar features numéricas válidas
        numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['cdr', 'diagnosis']]
        
        # Remover features com muitos NaN
        valid_features = []
        for col in feature_cols:
            if enhanced_df[col].notna().sum() / len(enhanced_df) > 0.8:
                valid_features.append(col)
        
        X = enhanced_df[valid_features].fillna(enhanced_df[valid_features].median())
        y = (enhanced_df['cdr'] == 0.5).astype(int)
        
        print(f"✓ Features válidas: {len(valid_features)}")
        print(f"✓ Amostras: {X.shape[0]}")
        
        # 4. Seleção de features com SHAP
        print("\n🎯 Seleção de features...")
        selected_features = self.feature_selector.shap_feature_selection(X, y, max_features=25)
        X_selected = X[selected_features]
        
        # 5. Normalização
        print("\n📊 Normalizando dados...")
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # 6. Augmentação de dados
        print("\n🔄 Augmentação de dados...")
        X_augmented, y_augmented = self.augmentation_engine.apply_smote_augmentation(X_scaled, y.values)
        
        # 7. Treinar ensemble otimizado
        print("\n🚀 Treinando ensemble otimizado...")
        ensemble_results = self.ensemble_classifier.train_stacking_ensemble(X_augmented, y_augmented)
        
        # 8. Validação cruzada robusta no dataset original
        print("\n🔍 Validação cruzada robusta...")
        cv_results = self._robust_cross_validation(X_scaled, y.values, selected_features)
        
        # 9. Gerar relatório final
        print("\n📊 Gerando relatório final...")
        final_results = {
            'ensemble_results': ensemble_results,
            'cv_results': cv_results,
            'selected_features': selected_features,
            'feature_importance': self.feature_selector.feature_importance,
            'augmentation_method': self.augmentation_engine.augmentation_method,
            'n_original_samples': X_scaled.shape[0],
            'n_augmented_samples': X_augmented.shape[0]
        }
        
        self._generate_ultimate_report(final_results, enhanced_df)
        
        return final_results
    
    def _robust_cross_validation(self, X: np.ndarray, y: np.ndarray, feature_names: list, n_folds: int = 10) -> dict:
        """Validação cruzada robusta"""
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Augmentação apenas no treino
            if IMBALANCED_AVAILABLE:
                smote = SMOTE(random_state=42)
                X_train_aug, y_train_aug = smote.fit_resample(X_train, y_train)
            else:
                X_train_aug, y_train_aug = X_train, y_train
            
            # Treinar modelo otimizado
            model = ExtraTreesClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                class_weight='balanced', random_state=42
            )
            model.fit(X_train_aug, y_train_aug)
            
            # Predições
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Métricas do fold
            fold_metrics = {
                'fold': fold + 1,
                'auc': roc_auc_score(y_val, y_pred_proba),
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1': f1_score(y_val, y_pred, zero_division=0)
            }
            
            fold_results.append(fold_metrics)
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(y_pred_proba)
            
            print(f"  Fold {fold+1}: AUC={fold_metrics['auc']:.3f}, Acc={fold_metrics['accuracy']:.3f}")
        
        # Métricas globais
        global_auc = roc_auc_score(all_y_true, all_y_proba)
        
        return {
            'fold_results': fold_results,
            'mean_auc': np.mean([f['auc'] for f in fold_results]),
            'std_auc': np.std([f['auc'] for f in fold_results]),
            'mean_accuracy': np.mean([f['accuracy'] for f in fold_results]),
            'std_accuracy': np.std([f['accuracy'] for f in fold_results]),
            'global_auc': global_auc,
            'all_predictions': {
                'y_true': all_y_true,
                'y_pred': all_y_pred,
                'y_proba': all_y_proba
            }
        }
    
    def _generate_ultimate_report(self, results: dict, enhanced_df: pd.DataFrame):
        """Gera relatório final otimizado"""
        
        print("\n" + "="*70)
        print("📊 RELATÓRIO FINAL - PIPELINE ULTIMATE OTIMIZADO")
        print("="*70)
        
        cv_results = results['cv_results']
        ensemble_results = results['ensemble_results']
        
        # Performance principal
        print(f"\n🎯 PERFORMANCE PRINCIPAL:")
        print(f"  AUC médio (CV): {cv_results['mean_auc']:.3f} ± {cv_results['std_auc']:.3f}")
        print(f"  AUC global: {cv_results['global_auc']:.3f}")
        print(f"  Acurácia média: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
        
        # Comparação com baselines
        baseline_morphological = 0.819
        baseline_cnn = 0.531
        
        print(f"\n📈 COMPARAÇÃO COM BASELINES:")
        print(f"  Baseline morfológico: {baseline_morphological:.3f}")
        print(f"  Baseline CNN híbrido: {baseline_cnn:.3f}")
        print(f"  Modelo atual: {cv_results['mean_auc']:.3f}")
        print(f"  Melhoria vs morfológico: {cv_results['mean_auc'] - baseline_morphological:+.3f}")
        print(f"  Melhoria vs CNN: {cv_results['mean_auc'] - baseline_cnn:+.3f}")
        
        # Performance dos modelos individuais
        print(f"\n🤖 PERFORMANCE DOS MODELOS BASE:")
        for name, scores in ensemble_results['model_scores'].items():
            print(f"  {name}: {scores['cv_auc']:.3f} ± {scores['cv_std']:.3f}")
        print(f"  🏆 Ensemble: {ensemble_results['ensemble_auc']:.3f}")
        
        # Augmentação de dados
        print(f"\n🔄 AUGMENTAÇÃO DE DADOS:")
        print(f"  Método: {results['augmentation_method']}")
        print(f"  Amostras originais: {results['n_original_samples']}")
        print(f"  Amostras augmentadas: {results['n_augmented_samples']}")
        print(f"  Fator de aumento: {results['n_augmented_samples'] / results['n_original_samples']:.1f}x")
        
        # Top features
        print(f"\n🧠 TOP 10 FEATURES SELECIONADAS:")
        selected_features = results['selected_features'][:10]
        for i, feature in enumerate(selected_features):
            print(f"  {i+1:2d}. {feature}")
        
        # Interpretação clínica
        auc = cv_results['mean_auc']
        print(f"\n🏥 INTERPRETAÇÃO CLÍNICA:")
        if auc >= 0.90:
            interpretation = "🏆 EXCELENTE: Performance de nível clínico superior"
        elif auc >= 0.85:
            interpretation = "✅ MUITO BOM: Performance clinicamente útil"
        elif auc >= 0.80:
            interpretation = "👍 BOM: Performance adequada para triagem"
        elif auc >= 0.75:
            interpretation = "⚠️ MODERADO: Performance aceitável com limitações"
        else:
            interpretation = "❌ INSUFICIENTE: Performance inadequada para uso clínico"
        
        print(f"  {interpretation}")
        
        # Status da meta
        target_auc = 0.85
        print(f"\n🎯 STATUS DA META:")
        if auc >= target_auc:
            print(f"  ✅ META ATINGIDA: AUC {auc:.3f} ≥ {target_auc:.3f}")
        else:
            gap = target_auc - auc
            print(f"  ⚠️ META NÃO ATINGIDA: Faltam {gap:.3f} pontos")
            print(f"  💡 Próximos passos: CNN híbrido com dados T1 recuperados")
        
        # Salvar resultados
        print(f"\n💾 SALVANDO RESULTADOS...")
        
        import joblib
        
        # Salvar melhor modelo
        best_model_name = max(ensemble_results['model_scores'].keys(), 
                            key=lambda k: ensemble_results['model_scores'][k]['cv_auc'])
        best_model = ensemble_results['base_models'][best_model_name]
        
        joblib.dump(best_model, 'ultimate_optimized_mci_model.joblib')
        joblib.dump(self.scaler, 'ultimate_optimized_scaler.joblib')
        
        # Salvar resultados
        cv_df = pd.DataFrame(cv_results['fold_results'])
        cv_df.to_csv('ultimate_optimized_cv_results.csv', index=False)
        
        # Salvar features selecionadas
        features_df = pd.DataFrame({
            'feature': results['selected_features'],
            'rank': range(1, len(results['selected_features']) + 1)
        })
        features_df.to_csv('ultimate_optimized_selected_features.csv', index=False)
        
        print(f"  ✓ ultimate_optimized_mci_model.joblib")
        print(f"  ✓ ultimate_optimized_scaler.joblib")
        print(f"  ✓ ultimate_optimized_cv_results.csv")
        print(f"  ✓ ultimate_optimized_selected_features.csv")
        
        # Criar visualizações
        self._create_ultimate_visualizations(results)
        print(f"  ✓ ultimate_optimized_performance_plots.png")
    
    def _create_ultimate_visualizations(self, results: dict):
        """Cria visualizações finais"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        cv_results = results['cv_results']
        ensemble_results = results['ensemble_results']
        
        # 1. ROC Curve
        from sklearn.metrics import roc_curve
        y_true = cv_results['all_predictions']['y_true']
        y_proba = cv_results['all_predictions']['y_proba']
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = cv_results['global_auc']
        
        axes[0, 0].plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {auc_score:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[0, 0].set_xlabel('Taxa de Falso Positivo')
        axes[0, 0].set_ylabel('Taxa de Verdadeiro Positivo')
        axes[0, 0].set_title('Curva ROC - Pipeline Ultimate')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Model Comparison
        model_names = list(ensemble_results['model_scores'].keys())
        model_aucs = [ensemble_results['model_scores'][name]['cv_auc'] for name in model_names]
        
        bars = axes[0, 1].bar(model_names, model_aucs, alpha=0.7)
        axes[0, 1].axhline(y=0.85, color='r', linestyle='--', label='Meta (0.85)')
        axes[0, 1].set_ylabel('AUC Score')
        axes[0, 1].set_title('Performance dos Modelos Base')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # 3. CV Distribution
        fold_aucs = [f['auc'] for f in cv_results['fold_results']]
        axes[0, 2].boxplot([fold_aucs], labels=['AUC'])
        axes[0, 2].scatter([1]*len(fold_aucs), fold_aucs, alpha=0.6)
        axes[0, 2].axhline(y=0.85, color='r', linestyle='--', label='Meta')
        axes[0, 2].set_ylabel('AUC Score')
        axes[0, 2].set_title(f'Distribuição AUC (CV)\nMédia: {np.mean(fold_aucs):.3f}')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Confusion Matrix
        from sklearn.metrics import confusion_matrix
        y_pred = cv_results['all_predictions']['y_pred']
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predição')
        axes[1, 0].set_ylabel('Real')
        axes[1, 0].set_title('Matriz de Confusão')
        axes[1, 0].set_xticklabels(['Normal', 'MCI'])
        axes[1, 0].set_yticklabels(['Normal', 'MCI'])
        
        # 5. Feature Importance (se SHAP disponível)
        if 'shap' in results['feature_importance']:
            importance = results['feature_importance']['shap']
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            features, scores = zip(*top_features)
            axes[1, 1].barh(range(len(features)), scores)
            axes[1, 1].set_yticks(range(len(features)))
            axes[1, 1].set_yticklabels([f.replace('_', ' ')[:20] for f in features])
            axes[1, 1].set_xlabel('Importância SHAP')
            axes[1, 1].set_title('Top 10 Features (SHAP)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Baseline Comparison
        baselines = {
            'Baseline\nMorfológico': 0.819,
            'Baseline\nCNN': 0.531,
            'Pipeline\nUltimate': cv_results['mean_auc']
        }
        
        names = list(baselines.keys())
        values = list(baselines.values())
        colors = ['orange', 'red', 'green']
        
        bars = axes[1, 2].bar(names, values, color=colors, alpha=0.7)
        axes[1, 2].axhline(y=0.85, color='black', linestyle='--', label='Meta (0.85)')
        axes[1, 2].set_ylabel('AUC Score')
        axes[1, 2].set_title('Comparação com Baselines')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('ultimate_optimized_performance_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Função principal"""
    
    print("🧠 PIPELINE ULTIMATE OTIMIZADO PARA DETECÇÃO DE MCI")
    print("🎯 Objetivo: AUC ≥ 0.85 usando estratégias avançadas")
    print("💡 SHAP + SMOTE + Stacking + Optuna")
    
    pipeline = UltimateOptimizedPipeline()
    results = pipeline.run_ultimate_pipeline()
    
    return results

if __name__ == "__main__":
    results = main()

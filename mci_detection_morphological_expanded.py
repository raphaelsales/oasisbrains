#!/usr/bin/env python3
"""
PIPELINE MORFOLÓGICO EXPANDIDO PARA DETECÇÃO DE MCI
Versão otimizada usando TODOS os 405 sujeitos disponíveis
Meta: Superar AUC = 0.819 com features avançadas e ensemble

Estratégias:
1. Usar todos os 405 sujeitos (vs 312 anterior)
2. Feature engineering avançada
3. Ensemble de múltiplos algoritmos
4. Validação cruzada estratificada robusta
5. Otimização de hiperparâmetros
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFECV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, roc_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
from scipy.stats import yeojohnson
import warnings
warnings.filterwarnings('ignore')

class AdvancedMorphologicalFeatureEngineer:
    """
    Engenharia de features morfológicas avançada para MCI
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        self.selected_features = []
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features morfológicas avançadas"""
        
        print("🔬 Criando features morfológicas avançadas...")
        
        enhanced_df = df.copy()
        
        # === FEATURES VOLUMÉTRICAS BÁSICAS ===
        if 'left_hippocampus_volume' in df.columns and 'right_hippocampus_volume' in df.columns:
            enhanced_df['total_hippocampus_volume'] = df['left_hippocampus_volume'] + df['right_hippocampus_volume']
            enhanced_df['hippocampus_asymmetry'] = abs(df['left_hippocampus_volume'] - df['right_hippocampus_volume']) / enhanced_df['total_hippocampus_volume']
            enhanced_df['hippocampus_lr_ratio'] = df['left_hippocampus_volume'] / (df['right_hippocampus_volume'] + 1e-6)
        
        if 'left_amygdala_volume' in df.columns and 'right_amygdala_volume' in df.columns:
            enhanced_df['total_amygdala_volume'] = df['left_amygdala_volume'] + df['right_amygdala_volume']
            enhanced_df['amygdala_asymmetry'] = abs(df['left_amygdala_volume'] - df['right_amygdala_volume']) / enhanced_df['total_amygdala_volume']
        
        if 'left_entorhinal_volume' in df.columns and 'right_entorhinal_volume' in df.columns:
            enhanced_df['total_entorhinal_volume'] = df['left_entorhinal_volume'] + df['right_entorhinal_volume']
            enhanced_df['entorhinal_asymmetry'] = abs(df['left_entorhinal_volume'] - df['right_entorhinal_volume']) / enhanced_df['total_entorhinal_volume']
        
        if 'left_temporal_volume' in df.columns and 'right_temporal_volume' in df.columns:
            enhanced_df['total_temporal_volume'] = df['left_temporal_volume'] + df['right_temporal_volume']
            enhanced_df['temporal_asymmetry'] = abs(df['left_temporal_volume'] - df['right_temporal_volume']) / enhanced_df['total_temporal_volume']
        
        # === RATIOS INTER-REGIONAIS AVANÇADOS ===
        volume_regions = ['hippocampus', 'amygdala', 'entorhinal', 'temporal']
        for i, region1 in enumerate(volume_regions):
            for region2 in volume_regions[i+1:]:
                col1 = f'total_{region1}_volume'
                col2 = f'total_{region2}_volume'
                if col1 in enhanced_df.columns and col2 in enhanced_df.columns:
                    enhanced_df[f'{region1}_{region2}_ratio'] = enhanced_df[col1] / (enhanced_df[col2] + 1e-6)
        
        # === FEATURES CLÍNICAS INTEGRADAS ===
        if 'mmse' in enhanced_df.columns and 'total_hippocampus_volume' in enhanced_df.columns:
            # Score MCI composto (baseado em literatura)
            enhanced_df['mci_composite_score'] = (
                (30 - enhanced_df['mmse']) / 20 * 0.4 +  # MMSE invertido
                (8000 - enhanced_df['total_hippocampus_volume']) / 3000 * 0.3 +  # Volume hipocampo
                (enhanced_df['hippocampus_asymmetry'] * 100) * 0.3  # Assimetria
            )
        
        # === AJUSTES POR IDADE E GÊNERO ===
        if 'age' in enhanced_df.columns:
            # Normalização por idade para volumes
            for col in ['total_hippocampus_volume', 'total_amygdala_volume', 'total_entorhinal_volume']:
                if col in enhanced_df.columns:
                    # Volume esperado baseado na idade (declínio linear)
                    expected_vol = enhanced_df[col].mean() - (enhanced_df['age'] - 70) * enhanced_df[col].std() * 0.01
                    enhanced_df[f'{col}_age_adjusted'] = enhanced_df[col] / (expected_vol + 1e-6)
        
        if 'gender' in enhanced_df.columns:
            # Ajuste por gênero (diferenças conhecidas)
            gender_encoded = enhanced_df['gender'].map({'M': 1, 'F': 0})
            for col in ['total_hippocampus_volume', 'total_amygdala_volume']:
                if col in enhanced_df.columns:
                    enhanced_df[f'{col}_gender_adjusted'] = enhanced_df[col] * (1 + gender_encoded * 0.05)
        
        # === FEATURES DE INTENSIDADE AVANÇADAS ===
        intensity_cols = [col for col in df.columns if 'intensity_mean' in col]
        if len(intensity_cols) >= 2:
            enhanced_df['global_intensity_mean'] = df[intensity_cols].mean(axis=1)
            enhanced_df['global_intensity_std'] = df[intensity_cols].std(axis=1)
            enhanced_df['intensity_cv'] = enhanced_df['global_intensity_std'] / (enhanced_df['global_intensity_mean'] + 1e-6)
        
        # Contraste entre regiões específicas
        contrast_pairs = [
            ('left_hippocampus_intensity_mean', 'left_temporal_intensity_mean'),
            ('right_hippocampus_intensity_mean', 'right_temporal_intensity_mean'),
            ('left_amygdala_intensity_mean', 'left_entorhinal_intensity_mean'),
            ('right_amygdala_intensity_mean', 'right_entorhinal_intensity_mean')
        ]
        
        for col1, col2 in contrast_pairs:
            if col1 in df.columns and col2 in df.columns:
                contrast_name = f'{col1.split("_")[1]}_{col2.split("_")[1]}_contrast'
                enhanced_df[contrast_name] = abs(df[col1] - df[col2])
        
        # === FEATURES POLINOMIAIS (INTERAÇÕES) ===
        if 'mmse' in enhanced_df.columns and 'age' in enhanced_df.columns:
            enhanced_df['mmse_age_interaction'] = enhanced_df['mmse'] * enhanced_df['age']
            enhanced_df['mmse_squared'] = enhanced_df['mmse'] ** 2
            enhanced_df['age_squared'] = enhanced_df['age'] ** 2
        
        # === FEATURES DE RISCO COMBINADAS ===
        risk_factors = []
        if 'age' in enhanced_df.columns:
            risk_factors.append((enhanced_df['age'] >= 75).astype(int))
        if 'mmse' in enhanced_df.columns:
            risk_factors.append((enhanced_df['mmse'] <= 26).astype(int))
        if 'total_hippocampus_volume' in enhanced_df.columns:
            hippo_threshold = enhanced_df['total_hippocampus_volume'].quantile(0.25)
            risk_factors.append((enhanced_df['total_hippocampus_volume'] <= hippo_threshold).astype(int))
        
        if risk_factors:
            enhanced_df['combined_risk_score'] = sum(risk_factors)
        
        print(f"✓ Features criadas: {enhanced_df.shape[1]} total (+{enhanced_df.shape[1] - df.shape[1]} novas)")
        return enhanced_df
    
    def analyze_feature_discriminability(self, df: pd.DataFrame, target_col: str = 'cdr') -> pd.DataFrame:
        """Análise avançada de discriminabilidade das features"""
        
        print("📊 Analisando discriminabilidade das features...")
        
        # Identificar features numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['cdr', 'diagnosis']]
        
        analysis_results = []
        
        # Separar grupos
        normal_group = df[df[target_col] == 0]
        mci_group = df[df[target_col] == 0.5]
        
        for feature in feature_cols:
            if df[feature].notna().sum() / len(df) < 0.8:  # Skip features com muitos NaN
                continue
                
            normal_vals = normal_group[feature].dropna()
            mci_vals = mci_group[feature].dropna()
            
            if len(normal_vals) < 10 or len(mci_vals) < 10:
                continue
            
            # Testes estatísticos
            t_stat, p_val = stats.ttest_ind(normal_vals, mci_vals)
            u_stat, p_val_mann = stats.mannwhitneyu(normal_vals, mci_vals, alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(normal_vals)-1)*normal_vals.var() + 
                                (len(mci_vals)-1)*mci_vals.var()) / 
                               (len(normal_vals) + len(mci_vals) - 2))
            cohens_d = abs(normal_vals.mean() - mci_vals.mean()) / (pooled_std + 1e-6)
            
            # AUC para feature individual
            try:
                combined_vals = np.concatenate([normal_vals, mci_vals])
                combined_labels = np.concatenate([np.zeros(len(normal_vals)), np.ones(len(mci_vals))])
                feature_auc = roc_auc_score(combined_labels, combined_vals)
                if feature_auc < 0.5:
                    feature_auc = 1 - feature_auc  # Inverter se necessário
            except:
                feature_auc = 0.5
            
            # Mutual information
            try:
                mi_score = mutual_info_classif(df[[feature]].fillna(0), df[target_col] == 0.5, random_state=42)[0]
            except:
                mi_score = 0
            
            analysis_results.append({
                'feature': feature,
                'normal_mean': normal_vals.mean(),
                'normal_std': normal_vals.std(),
                'mci_mean': mci_vals.mean(),
                'mci_std': mci_vals.std(),
                'p_value_ttest': p_val,
                'p_value_mann': p_val_mann,
                'cohens_d': cohens_d,
                'individual_auc': feature_auc,
                'mutual_info': mi_score,
                'direction': 'decreased' if mci_vals.mean() < normal_vals.mean() else 'increased'
            })
        
        results_df = pd.DataFrame(analysis_results)
        if len(results_df) > 0:
            # Score composto de importância
            results_df['importance_score'] = (
                results_df['cohens_d'] * 0.3 +
                results_df['individual_auc'] * 0.3 +
                results_df['mutual_info'] * 0.2 +
                (1 / (results_df['p_value_ttest'] + 1e-10)) * 0.2
            )
            results_df = results_df.sort_values('importance_score', ascending=False)
        
        return results_df
    
    def select_optimal_features(self, df: pd.DataFrame, importance_df: pd.DataFrame, max_features: int = 25) -> list:
        """Seleção otimizada de features"""
        
        if len(importance_df) == 0:
            return []
        
        # Estratégia multi-critério
        # 1. Features estatisticamente significativas
        significant_features = importance_df[
            (importance_df['p_value_ttest'] < 0.05) & 
            (importance_df['cohens_d'] > 0.2)
        ].head(15)['feature'].tolist()
        
        # 2. Features com alto AUC individual
        high_auc_features = importance_df[
            importance_df['individual_auc'] > 0.6
        ].head(10)['feature'].tolist()
        
        # 3. Features com alta mutual information
        high_mi_features = importance_df.nlargest(10, 'mutual_info')['feature'].tolist()
        
        # Combinar sem duplicatas
        selected = list(set(significant_features + high_auc_features + high_mi_features))
        
        # Limitar ao máximo especificado
        if len(selected) > max_features:
            selected = importance_df[importance_df['feature'].isin(selected)].head(max_features)['feature'].tolist()
        
        self.selected_features = selected
        
        print(f"\n🎯 TOP {len(selected)} FEATURES SELECIONADAS:")
        for i, feature in enumerate(selected[:15]):  # Mostrar top 15
            if feature in importance_df['feature'].values:
                row = importance_df[importance_df['feature'] == feature].iloc[0]
                print(f"  {i+1:2d}. {feature}")
                print(f"      p-value: {row['p_value_ttest']:.4f}, Cohen's d: {row['cohens_d']:.3f}, AUC: {row['individual_auc']:.3f}")
        
        return selected

class EnsembleMCIClassifier:
    """
    Classificador ensemble otimizado para MCI
    """
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.feature_importance = {}
        
    def create_optimized_models(self):
        """Cria conjunto de modelos otimizados"""
        
        self.models = {
            'random_forest': RandomForestClassifier(
                random_state=42, n_jobs=-1, class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=42
            ),
            'extra_trees': ExtraTreesClassifier(
                random_state=42, n_jobs=-1, class_weight='balanced'
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, class_weight='balanced', max_iter=1000
            ),
            'svm_rbf': SVC(
                random_state=42, class_weight='balanced', probability=True
            ),
            'mlp': MLPClassifier(
                random_state=42, max_iter=500
            )
        }
        
        # Parâmetros para otimização
        self.param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'extra_trees': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm_rbf': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
        }
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> dict:
        """Treina ensemble com validação cruzada e otimização de hiperparâmetros"""
        
        print("🚀 Treinando ensemble de modelos...")
        
        self.create_optimized_models()
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        results = {}
        best_models = {}
        
        for name, model in self.models.items():
            print(f"\n🔄 Otimizando {name}...")
            
            # Grid search com validação cruzada
            if name in self.param_grids:
                grid_search = RandomizedSearchCV(
                    model, self.param_grids[name],
                    cv=cv, scoring='roc_auc', n_iter=20,
                    random_state=42, n_jobs=-1
                )
                grid_search.fit(X, y)
                best_model = grid_search.best_estimator_
                self.best_params[name] = grid_search.best_params_
                best_score = grid_search.best_score_
            else:
                # Validação cruzada simples
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
                best_model = model.fit(X, y)
                best_score = scores.mean()
                self.best_params[name] = "default"
            
            # Avaliar modelo final
            best_model.fit(X, y)
            y_pred_proba = best_model.predict_proba(X)[:, 1]
            y_pred = best_model.predict(X)
            
            # Feature importance (se disponível)
            if hasattr(best_model, 'feature_importances_'):
                self.feature_importance[name] = best_model.feature_importances_
            elif hasattr(best_model, 'coef_'):
                self.feature_importance[name] = np.abs(best_model.coef_[0])
            
            results[name] = {
                'model': best_model,
                'cv_auc': best_score,
                'train_auc': roc_auc_score(y, y_pred_proba),
                'train_accuracy': accuracy_score(y, y_pred),
                'best_params': self.best_params[name]
            }
            
            print(f"  ✓ {name}: CV AUC = {best_score:.3f}")
        
        # Ensemble voting
        print(f"\n🎯 Criando ensemble final...")
        ensemble_pred = np.zeros(len(y))
        
        for name, result in results.items():
            model = result['model']
            pred_proba = model.predict_proba(X)[:, 1]
            ensemble_pred += pred_proba * result['cv_auc']  # Peso por performance
        
        ensemble_pred /= sum([r['cv_auc'] for r in results.values()])
        ensemble_auc = roc_auc_score(y, ensemble_pred)
        
        results['ensemble'] = {
            'train_auc': ensemble_auc,
            'predictions': ensemble_pred
        }
        
        print(f"  🏆 Ensemble AUC: {ensemble_auc:.3f}")
        
        return results

class ExpandedMorphologicalPipeline:
    """
    Pipeline morfológico expandido para todos os 405 sujeitos
    """
    
    def __init__(self, data_file: str = "alzheimer_complete_dataset.csv"):
        self.data_file = data_file
        self.feature_engineer = AdvancedMorphologicalFeatureEngineer()
        self.classifier = EnsembleMCIClassifier()
        
    def run_expanded_pipeline(self):
        """Executa pipeline completo expandido"""
        
        print("🧠 PIPELINE MORFOLÓGICO EXPANDIDO PARA DETECÇÃO DE MCI")
        print("=" * 70)
        print("🎯 Meta: Superar AUC = 0.819 usando todos os 405 sujeitos")
        print("=" * 70)
        
        # 1. Carregar dados
        print("\n📂 Carregando dataset completo...")
        df = pd.read_csv(self.data_file)
        df_mci = df[df['cdr'].isin([0.0, 0.5])].copy()
        
        print(f"✓ Dataset carregado:")
        print(f"  Total sujeitos: {len(df_mci)}")
        print(f"  Normal (CDR=0): {len(df_mci[df_mci['cdr']==0])}")
        print(f"  MCI (CDR=0.5): {len(df_mci[df_mci['cdr']==0.5])}")
        print(f"  Features originais: {df_mci.shape[1]}")
        
        # 2. Feature engineering
        print("\n🔬 Aplicando feature engineering avançada...")
        enhanced_df = self.feature_engineer.create_advanced_features(df_mci)
        
        # 3. Análise de discriminabilidade
        print("\n📊 Analisando discriminabilidade das features...")
        importance_df = self.feature_engineer.analyze_feature_discriminability(enhanced_df)
        
        # 4. Seleção de features
        print("\n🎯 Selecionando features otimizadas...")
        selected_features = self.feature_engineer.select_optimal_features(enhanced_df, importance_df, max_features=25)
        
        # 5. Preparar dados para treinamento
        print("\n⚙️ Preparando dados para treinamento...")
        X = enhanced_df[selected_features].fillna(enhanced_df[selected_features].median())
        y = (enhanced_df['cdr'] == 0.5).astype(int)  # 1 = MCI, 0 = Normal
        
        # Normalização
        scaler = self.feature_engineer.scaler
        X_scaled = scaler.fit_transform(X)
        
        # Converter para numpy arrays para validação cruzada
        y_array = y.values
        
        print(f"✓ Dados preparados:")
        print(f"  Amostras: {X_scaled.shape[0]}")
        print(f"  Features: {X_scaled.shape[1]}")
        print(f"  Classe 0 (Normal): {np.sum(y==0)}")
        print(f"  Classe 1 (MCI): {np.sum(y==1)}")
        
        # 6. Treinar ensemble
        print("\n🚀 Treinando ensemble de modelos...")
        results = self.classifier.train_ensemble(X_scaled, y_array, cv_folds=5)
        
        # 7. Validação cruzada robusta
        print("\n🔍 Validação cruzada robusta...")
        cv_results = self._robust_cross_validation(X_scaled, y_array, selected_features)
        
        # 8. Gerar relatório final
        print("\n📊 Gerando relatório final...")
        self._generate_comprehensive_report(results, cv_results, importance_df, enhanced_df)
        
        return results, cv_results, importance_df
    
    def _robust_cross_validation(self, X: np.ndarray, y: np.ndarray, feature_names: list, n_folds: int = 10) -> dict:
        """Validação cruzada robusta com múltiplas métricas"""
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Treinar modelo simples para validação
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=200, max_depth=15, 
                class_weight='balanced', random_state=42
            )
            model.fit(X_train, y_train)
            
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
        global_accuracy = accuracy_score(all_y_true, all_y_pred)
        
        return {
            'fold_results': fold_results,
            'mean_auc': np.mean([f['auc'] for f in fold_results]),
            'std_auc': np.std([f['auc'] for f in fold_results]),
            'mean_accuracy': np.mean([f['accuracy'] for f in fold_results]),
            'std_accuracy': np.std([f['accuracy'] for f in fold_results]),
            'global_auc': global_auc,
            'global_accuracy': global_accuracy,
            'all_predictions': {
                'y_true': all_y_true,
                'y_pred': all_y_pred,
                'y_proba': all_y_proba
            }
        }
    
    def _generate_comprehensive_report(self, train_results: dict, cv_results: dict, 
                                     importance_df: pd.DataFrame, enhanced_df: pd.DataFrame):
        """Gera relatório abrangente dos resultados"""
        
        print("\n" + "="*70)
        print("📊 RELATÓRIO FINAL - PIPELINE MORFOLÓGICO EXPANDIDO")
        print("="*70)
        
        # Performance do modelo
        print(f"\n🎯 PERFORMANCE DO MODELO:")
        print(f"  AUC média (CV): {cv_results['mean_auc']:.3f} ± {cv_results['std_auc']:.3f}")
        print(f"  AUC global: {cv_results['global_auc']:.3f}")
        print(f"  Acurácia média: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
        
        # Comparação com baseline
        baseline_auc = 0.819
        improvement = cv_results['mean_auc'] - baseline_auc
        print(f"\n📈 COMPARAÇÃO COM BASELINE:")
        print(f"  Baseline anterior: {baseline_auc:.3f}")
        print(f"  Modelo atual: {cv_results['mean_auc']:.3f}")
        print(f"  Melhoria: {improvement:+.3f} ({improvement/baseline_auc*100:+.1f}%)")
        
        # Modelos individuais
        print(f"\n🤖 PERFORMANCE DOS MODELOS INDIVIDUAIS:")
        for name, result in train_results.items():
            if name != 'ensemble':
                print(f"  {name}: CV AUC = {result['cv_auc']:.3f}")
        
        if 'ensemble' in train_results:
            print(f"  🏆 Ensemble: AUC = {train_results['ensemble']['train_auc']:.3f}")
        
        # Top features
        print(f"\n🧠 TOP 10 FEATURES MAIS DISCRIMINATIVAS:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']}")
            print(f"      AUC: {row['individual_auc']:.3f}, Cohen's d: {row['cohens_d']:.3f}, p: {row['p_value_ttest']:.4f}")
        
        # Salvar resultados
        print(f"\n💾 SALVANDO RESULTADOS...")
        
        # Salvar modelo ensemble
        import joblib
        best_model = train_results['random_forest']['model']  # Ou o melhor modelo
        joblib.dump(best_model, 'morphological_expanded_mci_model.joblib')
        joblib.dump(self.feature_engineer.scaler, 'morphological_expanded_scaler.joblib')
        
        # Salvar análises
        importance_df.to_csv('morphological_expanded_feature_importance.csv', index=False)
        
        # Salvar métricas
        cv_df = pd.DataFrame(cv_results['fold_results'])
        cv_df.to_csv('morphological_expanded_cv_results.csv', index=False)
        
        # Criar visualizações
        self._create_visualizations(cv_results, importance_df)
        
        print(f"  ✓ morphological_expanded_mci_model.joblib")
        print(f"  ✓ morphological_expanded_scaler.joblib") 
        print(f"  ✓ morphological_expanded_feature_importance.csv")
        print(f"  ✓ morphological_expanded_cv_results.csv")
        print(f"  ✓ morphological_expanded_performance_plots.png")
        
        # Interpretação clínica
        print(f"\n🏥 INTERPRETAÇÃO CLÍNICA:")
        if cv_results['mean_auc'] >= 0.85:
            print(f"  🏆 EXCELENTE: Modelo clinicamente útil (AUC ≥ 0.85)")
        elif cv_results['mean_auc'] >= 0.80:
            print(f"  ✅ BOM: Modelo adequado para triagem (AUC ≥ 0.80)")
        elif cv_results['mean_auc'] >= 0.75:
            print(f"  ⚠️ MODERADO: Modelo necessita refinamento (AUC ≥ 0.75)")
        else:
            print(f"  ❌ INSUFICIENTE: Performance inadequada para uso clínico")
        
        print(f"\n🎯 RECOMENDAÇÕES:")
        if improvement > 0:
            print(f"  ✅ Pipeline expandido melhorou a performance")
            print(f"  🚀 Continuar refinando features e algoritmos")
        else:
            print(f"  ⚠️ Performance não melhorou significativamente")
            print(f"  🔍 Investigar outras estratégias (CNN, dados externos)")
    
    def _create_visualizations(self, cv_results: dict, importance_df: pd.DataFrame):
        """Cria visualizações dos resultados"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
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
        axes[0, 0].set_title('Curva ROC - Modelo Expandido')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Feature Importance
        top_features = importance_df.head(15)
        axes[0, 1].barh(range(len(top_features)), top_features['individual_auc'])
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels([f.replace('_', ' ').title()[:20] for f in top_features['feature']])
        axes[0, 1].set_xlabel('AUC Individual')
        axes[0, 1].set_title('Top 15 Features por AUC')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. CV Results Distribution
        fold_aucs = [f['auc'] for f in cv_results['fold_results']]
        axes[1, 0].boxplot([fold_aucs], labels=['AUC'])
        axes[1, 0].scatter([1]*len(fold_aucs), fold_aucs, alpha=0.6)
        axes[1, 0].set_ylabel('AUC Score')
        axes[1, 0].set_title(f'Distribuição AUC (CV)\nMédia: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Confusion Matrix
        from sklearn.metrics import confusion_matrix
        y_pred = cv_results['all_predictions']['y_pred']
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Predição')
        axes[1, 1].set_ylabel('Real')
        axes[1, 1].set_title('Matriz de Confusão')
        axes[1, 1].set_xticklabels(['Normal', 'MCI'])
        axes[1, 1].set_yticklabels(['Normal', 'MCI'])
        
        plt.tight_layout()
        plt.savefig('morphological_expanded_performance_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Função principal"""
    
    print("🧠 PIPELINE MORFOLÓGICO EXPANDIDO PARA DETECÇÃO DE MCI")
    print("🎯 Objetivo: Superar AUC = 0.819 usando todos os 405 sujeitos")
    print("💡 Estratégia: Feature engineering avançada + Ensemble otimizado")
    
    pipeline = ExpandedMorphologicalPipeline()
    results, cv_results, importance_analysis = pipeline.run_expanded_pipeline()
    
    return results, cv_results, importance_analysis

if __name__ == "__main__":
    results, cv_results, importance_analysis = main()

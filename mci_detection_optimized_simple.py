#!/usr/bin/env python3
"""
PIPELINE OTIMIZADO SIMPLIFICADO PARA DETECÇÃO DE MCI
Versão que funciona com dependências básicas
Estratégias implementadas:
1. Feature engineering avançada
2. Ensemble robusto
3. Validação cruzada otimizada
4. Seleção de features inteligente

Meta: AUC ≥ 0.85
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, roc_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Engenheiro de features avançado
    """
    
    def __init__(self):
        self.created_features = []
        
    def create_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features abrangentes e otimizadas"""
        
        print("🔬 Criando features abrangentes...")
        
        enhanced_df = df.copy()
        original_cols = set(df.columns)
        
        # === FEATURES VOLUMÉTRICAS AVANÇADAS ===
        volume_pairs = {
            'hippocampus': ['left_hippocampus_volume', 'right_hippocampus_volume'],
            'amygdala': ['left_amygdala_volume', 'right_amygdala_volume'],
            'entorhinal': ['left_entorhinal_volume', 'right_entorhinal_volume'],
            'temporal': ['left_temporal_volume', 'right_temporal_volume']
        }
        
        for region, (left, right) in volume_pairs.items():
            if left in df.columns and right in df.columns:
                # Volume total
                total_col = f'total_{region}_volume'
                enhanced_df[total_col] = df[left] + df[right]
                
                # Assimetria absoluta e relativa
                enhanced_df[f'{region}_asymmetry_abs'] = abs(df[left] - df[right])
                enhanced_df[f'{region}_asymmetry_rel'] = enhanced_df[f'{region}_asymmetry_abs'] / enhanced_df[total_col]
                
                # Ratio L/R com tratamento de zeros
                enhanced_df[f'{region}_lr_ratio'] = df[left] / (df[right] + 1e-6)
                enhanced_df[f'{region}_rl_ratio'] = df[right] / (df[left] + 1e-6)
                
                # Diferença normalizada
                enhanced_df[f'{region}_diff_norm'] = (df[left] - df[right]) / enhanced_df[total_col]
        
        # === RATIOS INTER-REGIONAIS OTIMIZADOS ===
        regions = ['hippocampus', 'amygdala', 'entorhinal', 'temporal']
        for i, region1 in enumerate(regions):
            for region2 in regions[i+1:]:
                col1 = f'total_{region1}_volume'
                col2 = f'total_{region2}_volume'
                if col1 in enhanced_df.columns and col2 in enhanced_df.columns:
                    # Ratio bidirecional
                    enhanced_df[f'{region1}_{region2}_ratio'] = enhanced_df[col1] / (enhanced_df[col2] + 1e-6)
                    enhanced_df[f'{region2}_{region1}_ratio'] = enhanced_df[col2] / (enhanced_df[col1] + 1e-6)
                    
                    # Diferença relativa
                    total_both = enhanced_df[col1] + enhanced_df[col2]
                    enhanced_df[f'{region1}_{region2}_rel_diff'] = (enhanced_df[col1] - enhanced_df[col2]) / total_both
        
        # === FEATURES CLÍNICAS OTIMIZADAS ===
        if 'mmse' in enhanced_df.columns:
            # Transformações MMSE
            enhanced_df['mmse_squared'] = enhanced_df['mmse'] ** 2
            enhanced_df['mmse_cubed'] = enhanced_df['mmse'] ** 3
            enhanced_df['mmse_sqrt'] = np.sqrt(enhanced_df['mmse'])
            enhanced_df['mmse_log'] = np.log(enhanced_df['mmse'] + 1)
            enhanced_df['mmse_inverted'] = 30 - enhanced_df['mmse']
            enhanced_df['mmse_normalized'] = enhanced_df['mmse'] / 30
            
            # Categorias MMSE
            enhanced_df['mmse_severe'] = (enhanced_df['mmse'] < 24).astype(int)
            enhanced_df['mmse_moderate'] = ((enhanced_df['mmse'] >= 24) & (enhanced_df['mmse'] < 27)).astype(int)
            enhanced_df['mmse_mild'] = (enhanced_df['mmse'] >= 27).astype(int)
            
            if 'age' in enhanced_df.columns:
                # Interações MMSE-idade
                enhanced_df['mmse_age_product'] = enhanced_df['mmse'] * enhanced_df['age']
                enhanced_df['mmse_age_ratio'] = enhanced_df['mmse'] / (enhanced_df['age'] + 1e-6)
                enhanced_df['age_adjusted_mmse'] = enhanced_df['mmse'] / (enhanced_df['age'] / 70)
                
                # MMSE esperado por idade (baseado em literatura)
                expected_mmse = 30 - 0.05 * (enhanced_df['age'] - 65)  # Declínio ~0.05/ano
                enhanced_df['mmse_age_residual'] = enhanced_df['mmse'] - expected_mmse
        
        if 'age' in enhanced_df.columns:
            # Transformações idade
            enhanced_df['age_squared'] = enhanced_df['age'] ** 2
            enhanced_df['age_cubed'] = enhanced_df['age'] ** 3
            enhanced_df['age_sqrt'] = np.sqrt(enhanced_df['age'])
            enhanced_df['age_normalized'] = (enhanced_df['age'] - 65) / 20  # Normalizar em torno de 65
            
            # Categorias de idade
            enhanced_df['age_young_old'] = (enhanced_df['age'] < 75).astype(int)
            enhanced_df['age_old_old'] = (enhanced_df['age'] >= 85).astype(int)
            enhanced_df['age_risk_high'] = (enhanced_df['age'] >= 80).astype(int)
        
        if 'education' in enhanced_df.columns:
            # Features educação
            enhanced_df['education_squared'] = enhanced_df['education'] ** 2
            enhanced_df['education_high'] = (enhanced_df['education'] >= 16).astype(int)
            enhanced_df['education_low'] = (enhanced_df['education'] <= 12).astype(int)
            
            if 'age' in enhanced_df.columns:
                enhanced_df['education_age_product'] = enhanced_df['education'] * enhanced_df['age']
        
        # === FEATURES DE INTENSIDADE AVANÇADAS ===
        intensity_mean_cols = [col for col in df.columns if 'intensity_mean' in col]
        intensity_std_cols = [col for col in df.columns if 'intensity_std' in col]
        
        if len(intensity_mean_cols) >= 2:
            # Estatísticas globais de intensidade
            enhanced_df['global_intensity_mean'] = df[intensity_mean_cols].mean(axis=1)
            enhanced_df['global_intensity_std'] = df[intensity_mean_cols].std(axis=1)
            enhanced_df['global_intensity_max'] = df[intensity_mean_cols].max(axis=1)
            enhanced_df['global_intensity_min'] = df[intensity_mean_cols].min(axis=1)
            enhanced_df['global_intensity_range'] = enhanced_df['global_intensity_max'] - enhanced_df['global_intensity_min']
            enhanced_df['global_intensity_cv'] = enhanced_df['global_intensity_std'] / (enhanced_df['global_intensity_mean'] + 1e-6)
            
            # Contraste entre regiões específicas
            contrast_pairs = [
                ('left_hippocampus_intensity_mean', 'left_temporal_intensity_mean', 'hippo_temp_left'),
                ('right_hippocampus_intensity_mean', 'right_temporal_intensity_mean', 'hippo_temp_right'),
                ('left_amygdala_intensity_mean', 'left_entorhinal_intensity_mean', 'amyg_entorh_left'),
                ('right_amygdala_intensity_mean', 'right_entorhinal_intensity_mean', 'amyg_entorh_right')
            ]
            
            for col1, col2, name in contrast_pairs:
                if col1 in df.columns and col2 in df.columns:
                    enhanced_df[f'{name}_contrast'] = abs(df[col1] - df[col2])
                    enhanced_df[f'{name}_ratio'] = df[col1] / (df[col2] + 1e-6)
        
        if len(intensity_std_cols) >= 2:
            # Variabilidade de intensidade
            enhanced_df['global_intensity_variability'] = df[intensity_std_cols].mean(axis=1)
            enhanced_df['intensity_variability_std'] = df[intensity_std_cols].std(axis=1)
        
        # === SCORES COMPOSTOS BASEADOS EM EVIDÊNCIAS ===
        # Score MCI baseado na literatura neurológica
        risk_components = []
        
        if 'age' in enhanced_df.columns:
            # Idade ≥ 75 anos (fator de risco)
            age_risk = ((enhanced_df['age'] >= 75).astype(float) * 0.25)
            risk_components.append(age_risk)
        
        if 'mmse' in enhanced_df.columns:
            # MMSE ≤ 26 (indicativo de comprometimento)
            mmse_risk = ((enhanced_df['mmse'] <= 26).astype(float) * 0.35)
            risk_components.append(mmse_risk)
        
        if 'total_hippocampus_volume' in enhanced_df.columns:
            # Volume hipocampal baixo (abaixo do Q25)
            hippo_threshold = enhanced_df['total_hippocampus_volume'].quantile(0.25)
            hippo_risk = ((enhanced_df['total_hippocampus_volume'] <= hippo_threshold).astype(float) * 0.30)
            risk_components.append(hippo_risk)
        
        if 'education' in enhanced_df.columns:
            # Baixa escolaridade (fator de risco)
            edu_risk = ((enhanced_df['education'] <= 12).astype(float) * 0.10)
            risk_components.append(edu_risk)
        
        if risk_components:
            enhanced_df['mci_risk_score'] = sum(risk_components)
        
        # Score de atrofia cerebral
        atrophy_components = []
        
        for region in ['hippocampus', 'amygdala', 'entorhinal', 'temporal']:
            vol_col = f'total_{region}_volume'
            asym_col = f'{region}_asymmetry_rel'
            
            if vol_col in enhanced_df.columns:
                # Volume baixo
                vol_norm = enhanced_df[vol_col] / enhanced_df[vol_col].median()
                atrophy_components.append(1 - vol_norm)
            
            if asym_col in enhanced_df.columns:
                # Alta assimetria
                asym_norm = enhanced_df[asym_col] / enhanced_df[asym_col].median()
                atrophy_components.append(asym_norm * 0.5)
        
        if atrophy_components:
            enhanced_df['atrophy_score'] = np.mean(atrophy_components, axis=0)
        
        # === FEATURES POLINOMIAIS SELECIONADAS ===
        # Apenas as interações mais relevantes para evitar explosão de features
        key_features = ['mmse', 'age', 'total_hippocampus_volume']
        
        for feat in key_features:
            if feat in enhanced_df.columns:
                # Interações com MMSE (mais discriminativo)
                if feat != 'mmse' and 'mmse' in enhanced_df.columns:
                    enhanced_df[f'mmse_{feat}_interaction'] = enhanced_df['mmse'] * enhanced_df[feat]
        
        # Contar novas features
        new_features = set(enhanced_df.columns) - original_cols
        self.created_features = list(new_features)
        
        print(f"✓ Features criadas: {len(new_features)} novas ({enhanced_df.shape[1]} total)")
        
        return enhanced_df
    
    def analyze_feature_quality(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Analisa qualidade das features para seleção"""
        
        print("📊 Analisando qualidade das features...")
        
        analysis_results = []
        
        for feature in X.columns:
            feature_data = X[feature].dropna()
            
            if len(feature_data) < len(X) * 0.8:  # Muitos NaN
                continue
            
            # Separar por classe
            normal_vals = X[y == 0][feature].dropna()
            mci_vals = X[y == 1][feature].dropna()
            
            if len(normal_vals) < 10 or len(mci_vals) < 10:
                continue
            
            # Testes estatísticos
            try:
                t_stat, p_val_ttest = stats.ttest_ind(normal_vals, mci_vals)
                u_stat, p_val_mann = stats.mannwhitneyu(normal_vals, mci_vals, alternative='two-sided')
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(normal_vals)-1)*normal_vals.var() + 
                                    (len(mci_vals)-1)*mci_vals.var()) / 
                                   (len(normal_vals) + len(mci_vals) - 2))
                cohens_d = abs(normal_vals.mean() - mci_vals.mean()) / (pooled_std + 1e-6)
                
                # AUC individual
                combined_vals = np.concatenate([normal_vals, mci_vals])
                combined_labels = np.concatenate([np.zeros(len(normal_vals)), np.ones(len(mci_vals))])
                individual_auc = roc_auc_score(combined_labels, combined_vals)
                if individual_auc < 0.5:
                    individual_auc = 1 - individual_auc
                
                # Variabilidade (menor é melhor para estabilidade)
                cv_normal = normal_vals.std() / (normal_vals.mean() + 1e-6)
                cv_mci = mci_vals.std() / (mci_vals.mean() + 1e-6)
                avg_cv = (cv_normal + cv_mci) / 2
                
                analysis_results.append({
                    'feature': feature,
                    'p_value_ttest': p_val_ttest,
                    'p_value_mann': p_val_mann,
                    'cohens_d': cohens_d,
                    'individual_auc': individual_auc,
                    'avg_cv': avg_cv,
                    'normal_mean': normal_vals.mean(),
                    'mci_mean': mci_vals.mean(),
                    'direction': 'decreased' if mci_vals.mean() < normal_vals.mean() else 'increased',
                    'missing_rate': 1 - (len(feature_data) / len(X))
                })
                
            except Exception as e:
                continue
        
        if not analysis_results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(analysis_results)
        
        # Score composto de qualidade
        results_df['quality_score'] = (
            results_df['cohens_d'] * 0.3 +
            results_df['individual_auc'] * 0.3 +
            (1 / (results_df['p_value_ttest'] + 1e-10)) * 0.2 +
            (1 / (results_df['avg_cv'] + 1e-6)) * 0.1 +
            (1 - results_df['missing_rate']) * 0.1
        )
        
        results_df = results_df.sort_values('quality_score', ascending=False)
        
        return results_df
    
    def select_optimal_features(self, X: pd.DataFrame, y: pd.Series, max_features: int = 25) -> list:
        """Seleciona features ótimas baseado em análise de qualidade"""
        
        quality_analysis = self.analyze_feature_quality(X, y)
        
        if len(quality_analysis) == 0:
            print("⚠️ Nenhuma feature válida encontrada")
            return list(X.columns[:max_features])
        
        # Estratégia multi-critério
        # 1. Features estatisticamente significativas com alto effect size
        tier1 = quality_analysis[
            (quality_analysis['p_value_ttest'] < 0.01) & 
            (quality_analysis['cohens_d'] > 0.5)
        ]['feature'].tolist()
        
        # 2. Features com AUC individual alto
        tier2 = quality_analysis[
            (quality_analysis['individual_auc'] > 0.65) &
            (~quality_analysis['feature'].isin(tier1))
        ]['feature'].tolist()
        
        # 3. Features com bom score geral
        tier3 = quality_analysis[
            (~quality_analysis['feature'].isin(tier1 + tier2))
        ].head(max_features)['feature'].tolist()
        
        # Combinar tiers
        selected = (tier1 + tier2 + tier3)[:max_features]
        
        print(f"\n🎯 FEATURES SELECIONADAS ({len(selected)}):")
        print(f"  Tier 1 (sig + effect): {len(tier1)}")
        print(f"  Tier 2 (alto AUC): {len(tier2)}")
        print(f"  Tier 3 (score geral): {len(tier3[:max_features-len(tier1)-len(tier2)])}")
        
        # Mostrar top features
        for i, feature in enumerate(selected[:15]):
            if feature in quality_analysis['feature'].values:
                row = quality_analysis[quality_analysis['feature'] == feature].iloc[0]
                print(f"  {i+1:2d}. {feature[:30]:<30} (AUC: {row['individual_auc']:.3f}, p: {row['p_value_ttest']:.4f})")
        
        return selected

class OptimizedEnsemble:
    """
    Ensemble otimizado com votação inteligente
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        
    def create_models(self):
        """Cria modelos otimizados"""
        
        self.models = {
            'rf_balanced': RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_split=5, min_samples_leaf=2,
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'et_balanced': ExtraTreesClassifier(
                n_estimators=300, max_depth=15, min_samples_split=5, min_samples_leaf=2,
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'gb_optimized': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.08, max_depth=6, subsample=0.8,
                random_state=42
            ),
            'lr_balanced': LogisticRegression(
                C=0.5, class_weight='balanced', random_state=42, max_iter=1000,
                solver='liblinear'
            ),
            'svm_rbf': SVC(
                C=1.0, gamma='scale', class_weight='balanced', probability=True,
                random_state=42
            )
        }
    
    def train_weighted_ensemble(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Treina ensemble com pesos baseados em performance"""
        
        print("🚀 Treinando ensemble otimizado...")
        
        self.create_models()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        model_performances = {}
        
        # Avaliar cada modelo
        for name, model in self.models.items():
            print(f"  🔄 Avaliando {name}...")
            
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
            model_performances[name] = {
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"     AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Calcular pesos baseados em performance
        aucs = [perf['cv_auc_mean'] for perf in model_performances.values()]
        min_auc = min(aucs)
        
        for name, perf in model_performances.items():
            # Peso baseado em performance relativa
            relative_performance = perf['cv_auc_mean'] - min_auc
            self.weights[name] = max(0.1, relative_performance)  # Peso mínimo de 0.1
        
        # Normalizar pesos
        total_weight = sum(self.weights.values())
        self.weights = {name: weight/total_weight for name, weight in self.weights.items()}
        
        print(f"\n🎯 Pesos do ensemble:")
        for name, weight in self.weights.items():
            print(f"     {name}: {weight:.3f}")
        
        # Criar voting classifier com pesos
        estimators = [(name, model) for name, model in self.models.items()]
        ensemble = VotingClassifier(
            estimators=estimators, 
            voting='soft',
            weights=list(self.weights.values())
        )
        
        # Treinar ensemble final
        ensemble.fit(X, y)
        
        # Avaliar ensemble
        ensemble_scores = cross_val_score(ensemble, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        ensemble_performance = {
            'cv_auc_mean': ensemble_scores.mean(),
            'cv_auc_std': ensemble_scores.std(),
            'cv_scores': ensemble_scores
        }
        
        print(f"\n🏆 Ensemble final: {ensemble_performance['cv_auc_mean']:.3f} ± {ensemble_performance['cv_auc_std']:.3f}")
        
        return {
            'ensemble': ensemble,
            'individual_performances': model_performances,
            'ensemble_performance': ensemble_performance,
            'weights': self.weights
        }

class OptimizedPipeline:
    """
    Pipeline otimizado final
    """
    
    def __init__(self, data_file: str = "alzheimer_complete_dataset.csv"):
        self.data_file = data_file
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ensemble = OptimizedEnsemble()
        self.scaler = RobustScaler()
        
    def run_optimized_pipeline(self) -> dict:
        """Executa pipeline otimizado"""
        
        print("🧠 PIPELINE OTIMIZADO PARA DETECÇÃO DE MCI")
        print("=" * 60)
        print("🎯 Meta: AUC ≥ 0.85")
        print("💡 Feature Engineering + Ensemble Inteligente")
        print("=" * 60)
        
        # 1. Carregar dados
        print("\n📂 Carregando dados...")
        df = pd.read_csv(self.data_file)
        df_mci = df[df['cdr'].isin([0.0, 0.5])].copy()
        
        print(f"✓ Dataset:")
        print(f"  Total: {len(df_mci)} sujeitos")
        print(f"  Normal: {len(df_mci[df_mci['cdr']==0])}")
        print(f"  MCI: {len(df_mci[df_mci['cdr']==0.5])}")
        
        # 2. Feature engineering
        print("\n🔬 Feature engineering...")
        enhanced_df = self.feature_engineer.create_comprehensive_features(df_mci)
        
        # 3. Preparar dados
        print("\n⚙️ Preparando dados...")
        numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['cdr', 'diagnosis']]
        
        X = enhanced_df[feature_cols].fillna(enhanced_df[feature_cols].median())
        y = (enhanced_df['cdr'] == 0.5).astype(int)
        
        print(f"✓ Features candidatas: {len(feature_cols)}")
        
        # 4. Seleção de features
        print("\n🎯 Seleção de features...")
        selected_features = self.feature_engineer.select_optimal_features(X, y, max_features=25)
        X_selected = X[selected_features]
        
        # 5. Normalização
        print("\n📊 Normalização...")
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # 6. Treinar ensemble
        print("\n🚀 Treinando ensemble...")
        ensemble_results = self.ensemble.train_weighted_ensemble(X_scaled, y.values)
        
        # 7. Validação final
        print("\n🔍 Validação final...")
        final_cv_results = self._final_validation(X_scaled, y.values)
        
        # 8. Resultados
        results = {
            'ensemble_results': ensemble_results,
            'final_cv_results': final_cv_results,
            'selected_features': selected_features,
            'created_features': self.feature_engineer.created_features,
            'n_total_features': len(feature_cols),
            'n_selected_features': len(selected_features)
        }
        
        self._generate_final_report(results, enhanced_df)
        
        return results
    
    def _final_validation(self, X: np.ndarray, y: np.ndarray, n_folds: int = 10) -> dict:
        """Validação final robusta"""
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        # Usar melhor modelo individual (Extra Trees geralmente)
        best_model = ExtraTreesClassifier(
            n_estimators=300, max_depth=15, min_samples_split=5,
            class_weight='balanced', random_state=42
        )
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            best_model.fit(X_train, y_train)
            
            y_pred_proba = best_model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
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
            
            print(f"  Fold {fold+1}: AUC={fold_metrics['auc']:.3f}")
        
        return {
            'fold_results': fold_results,
            'mean_auc': np.mean([f['auc'] for f in fold_results]),
            'std_auc': np.std([f['auc'] for f in fold_results]),
            'mean_accuracy': np.mean([f['accuracy'] for f in fold_results]),
            'global_auc': roc_auc_score(all_y_true, all_y_proba),
            'all_predictions': {
                'y_true': all_y_true,
                'y_pred': all_y_pred,
                'y_proba': all_y_proba
            }
        }
    
    def _generate_final_report(self, results: dict, enhanced_df: pd.DataFrame):
        """Gera relatório final"""
        
        print("\n" + "="*60)
        print("📊 RELATÓRIO FINAL - PIPELINE OTIMIZADO")
        print("="*60)
        
        cv_results = results['final_cv_results']
        ensemble_results = results['ensemble_results']
        
        # Performance
        print(f"\n🎯 PERFORMANCE:")
        print(f"  AUC médio: {cv_results['mean_auc']:.3f} ± {cv_results['std_auc']:.3f}")
        print(f"  AUC global: {cv_results['global_auc']:.3f}")
        print(f"  Acurácia: {cv_results['mean_accuracy']:.3f}")
        
        # Comparação
        baselines = {'Morfológico original': 0.819, 'CNN híbrido': 0.531}
        current_auc = cv_results['mean_auc']
        
        print(f"\n📈 COMPARAÇÃO:")
        for name, baseline in baselines.items():
            improvement = current_auc - baseline
            print(f"  vs {name}: {improvement:+.3f} ({improvement/baseline*100:+.1f}%)")
        
        # Features
        print(f"\n🔬 FEATURES:")
        print(f"  Originais: {results['n_total_features'] - len(results['created_features'])}")
        print(f"  Criadas: {len(results['created_features'])}")
        print(f"  Selecionadas: {results['n_selected_features']}")
        
        # Ensemble
        print(f"\n🤖 ENSEMBLE:")
        for name, perf in ensemble_results['individual_performances'].items():
            weight = ensemble_results['weights'][name]
            print(f"  {name}: {perf['cv_auc_mean']:.3f} (peso: {weight:.2f})")
        
        # Status da meta
        target = 0.85
        print(f"\n🎯 META (AUC ≥ {target}):")
        if current_auc >= target:
            print(f"  ✅ ATINGIDA! ({current_auc:.3f} ≥ {target})")
            status = "SUCESSO"
        else:
            gap = target - current_auc
            print(f"  ⚠️ Faltam {gap:.3f} pontos")
            status = "EM PROGRESSO"
        
        # Interpretação clínica
        print(f"\n🏥 INTERPRETAÇÃO CLÍNICA:")
        if current_auc >= 0.90:
            interpretation = "🏆 EXCELENTE: Nível clínico superior"
        elif current_auc >= 0.85:
            interpretation = "✅ MUITO BOM: Clinicamente útil"
        elif current_auc >= 0.80:
            interpretation = "👍 BOM: Adequado para triagem"
        else:
            interpretation = "⚠️ MODERADO: Necessita melhorias"
        
        print(f"  {interpretation}")
        
        # Próximos passos
        print(f"\n🚀 PRÓXIMOS PASSOS:")
        if status == "SUCESSO":
            print(f"  • Implementar CNN híbrido com dados T1 recuperados")
            print(f"  • Validação externa em outros datasets")
            print(f"  • Análise de interpretabilidade (SHAP)")
        else:
            print(f"  • Implementar CNN híbrido para ganho adicional")
            print(f"  • Explorar dados externos (ADNI)")
            print(f"  • Otimizar hiperparâmetros avançados")
        
        # Salvar
        print(f"\n💾 SALVANDO RESULTADOS...")
        
        import joblib
        best_model = ensemble_results['ensemble']
        joblib.dump(best_model, 'optimized_mci_model.joblib')
        joblib.dump(self.scaler, 'optimized_scaler.joblib')
        
        # CSVs
        cv_df = pd.DataFrame(cv_results['fold_results'])
        cv_df.to_csv('optimized_cv_results.csv', index=False)
        
        features_df = pd.DataFrame({
            'feature': results['selected_features'],
            'rank': range(1, len(results['selected_features']) + 1)
        })
        features_df.to_csv('optimized_selected_features.csv', index=False)
        
        print(f"  ✓ optimized_mci_model.joblib")
        print(f"  ✓ optimized_scaler.joblib")  
        print(f"  ✓ optimized_cv_results.csv")
        print(f"  ✓ optimized_selected_features.csv")
        
        # Gráficos
        self._create_visualizations(results)
        print(f"  ✓ optimized_performance_plots.png")
    
    def _create_visualizations(self, results: dict):
        """Cria visualizações"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        cv_results = results['final_cv_results']
        
        # 1. ROC Curve
        from sklearn.metrics import roc_curve
        y_true = cv_results['all_predictions']['y_true']
        y_proba = cv_results['all_predictions']['y_proba']
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = cv_results['global_auc']
        
        axes[0, 0].plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {auc_score:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[0, 0].axhline(y=0.85, color='r', linestyle=':', alpha=0.7, label='Meta (0.85)')
        axes[0, 0].set_xlabel('Taxa de Falso Positivo')
        axes[0, 0].set_ylabel('Taxa de Verdadeiro Positivo')
        axes[0, 0].set_title('Curva ROC - Pipeline Otimizado')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. CV Distribution
        fold_aucs = [f['auc'] for f in cv_results['fold_results']]
        axes[0, 1].boxplot([fold_aucs], labels=['AUC'])
        axes[0, 1].scatter([1]*len(fold_aucs), fold_aucs, alpha=0.6, color='blue')
        axes[0, 1].axhline(y=0.85, color='r', linestyle='--', label='Meta')
        axes[0, 1].set_ylabel('AUC Score')
        axes[0, 1].set_title(f'Distribuição AUC\n{np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Baseline Comparison
        baselines = {
            'Morfológico\nOriginal': 0.819,
            'CNN\nHíbrido': 0.531,
            'Pipeline\nOtimizado': cv_results['mean_auc']
        }
        
        names = list(baselines.keys())
        values = list(baselines.values())
        colors = ['orange', 'red', 'green']
        
        bars = axes[1, 0].bar(names, values, color=colors, alpha=0.7)
        axes[1, 0].axhline(y=0.85, color='black', linestyle='--', label='Meta')
        axes[1, 0].set_ylabel('AUC Score')
        axes[1, 0].set_title('Comparação com Baselines')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Valores nas barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Ensemble Performance
        ensemble_results = results['ensemble_results']
        model_names = list(ensemble_results['individual_performances'].keys())
        model_aucs = [ensemble_results['individual_performances'][name]['cv_auc_mean'] 
                     for name in model_names]
        
        axes[1, 1].bar(model_names, model_aucs, alpha=0.7)
        axes[1, 1].axhline(y=ensemble_results['ensemble_performance']['cv_auc_mean'], 
                          color='red', linestyle='-', label='Ensemble')
        axes[1, 1].set_ylabel('AUC Score')
        axes[1, 1].set_title('Performance Individual vs Ensemble')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('optimized_performance_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Função principal"""
    
    print("🧠 PIPELINE OTIMIZADO PARA DETECÇÃO DE MCI")
    print("🎯 Meta: AUC ≥ 0.85 com estratégias avançadas")
    
    pipeline = OptimizedPipeline()
    results = pipeline.run_optimized_pipeline()
    
    return results

if __name__ == "__main__":
    results = main()

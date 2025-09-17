#!/usr/bin/env python3
"""
Correção de Overfitting nos Modelos Afetados
Aplica técnicas de regularização específicas para cada tipo de modelo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class OverfittingCorrector:
    """Corretor de overfitting para modelos específicos"""
    
    def __init__(self, dataset_path='alzheimer_complete_dataset_augmented.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.corrected_models = {}
        
    def load_data(self):
        """Carrega dados balanceados"""
        print("CARREGANDO DATASET PARA CORREÇÃO DE OVERFITTING")
        print("=" * 55)
        
        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset carregado: {self.df.shape[0]} sujeitos, {self.df.shape[1]} features")
        
        # Preparar features
        self.feature_names = [col for col in self.df.columns 
                             if col not in ['subject_id', 'diagnosis', 'gender', 'cdr']]
        
        self.X = self.df[self.feature_names].fillna(self.df[self.feature_names].median())
        
        # Target binário
        if 'diagnosis' in self.df.columns:
            self.y = (self.df['diagnosis'] == 'Demented').astype(int)
        else:
            self.y = (self.df['cdr'] > 0).astype(int)
        
        print(f"Features utilizadas: {len(self.feature_names)}")
        print(f"Distribuição target: {np.bincount(self.y)}")
        
        return self.X, self.y
    
    def correct_logistic_regression(self):
        """Corrige overfitting na Regressão Logística com regularização"""
        print("\nCORRIGINDO LOGISTIC REGRESSION - REGULARIZAÇÃO L1/L2")
        print("=" * 60)
        
        # Parâmetros para testar regularização
        param_grid = {
            'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'classifier__penalty': ['l1', 'l2', 'elasticnet'],
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__max_iter': [1000, 2000]
        }
        
        # Pipeline com normalização
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # Grid search com validação cruzada
        print("Executando Grid Search para otimização...")
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X, self.y)
        
        # Melhor modelo
        best_lr = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Melhores parâmetros: {best_params}")
        print(f"Melhor AUC CV: {best_score:.3f}")
        
        # Testar overfitting no melhor modelo
        overfitting_results = self._test_overfitting(best_lr, "Logistic Regression Corrigida")
        
        self.corrected_models['Logistic Regression'] = {
            'model': best_lr,
            'params': best_params,
            'cv_score': best_score,
            'overfitting_results': overfitting_results
        }
        
        return best_lr, overfitting_results
    
    def correct_svm(self):
        """Corrige overfitting no SVM com regularização C e kernel"""
        print("\nCORRIGINDO SVM - REGULARIZAÇÃO C E KERNEL")
        print("=" * 45)
        
        # Parâmetros para regularização do SVM
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0, 100.0],
            'classifier__kernel': ['rbf', 'linear'],
            'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
        }
        
        # Pipeline com normalização
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(probability=True, random_state=42))
        ])
        
        print("Executando Grid Search para SVM...")
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X, self.y)
        
        best_svm = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Melhores parâmetros: {best_params}")
        print(f"Melhor AUC CV: {best_score:.3f}")
        
        # Testar overfitting
        overfitting_results = self._test_overfitting(best_svm, "SVM Corrigido")
        
        self.corrected_models['SVM'] = {
            'model': best_svm,
            'params': best_params,
            'cv_score': best_score,
            'overfitting_results': overfitting_results
        }
        
        return best_svm, overfitting_results
    
    def correct_mlp(self):
        """Corrige overfitting no MLP com regularização e early stopping"""
        print("\nCORRIGINDO MLP - REGULARIZAÇÃO ALPHA E EARLY STOPPING")
        print("=" * 55)
        
        # Parâmetros para regularização do MLP
        param_grid = {
            'classifier__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
            'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'classifier__learning_rate': ['constant', 'adaptive'],
            'classifier__early_stopping': [True],
            'classifier__validation_fraction': [0.2],
            'classifier__n_iter_no_change': [10, 20]
        }
        
        # Pipeline com normalização
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(random_state=42, max_iter=1000))
        ])
        
        print("Executando Grid Search para MLP...")
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X, self.y)
        
        best_mlp = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Melhores parâmetros: {best_params}")
        print(f"Melhor AUC CV: {best_score:.3f}")
        
        # Testar overfitting
        overfitting_results = self._test_overfitting(best_mlp, "MLP Corrigido")
        
        self.corrected_models['MLP'] = {
            'model': best_mlp,
            'params': best_params,
            'cv_score': best_score,
            'overfitting_results': overfitting_results
        }
        
        return best_mlp, overfitting_results
    
    def optimize_random_forest(self):
        """Otimiza Random Forest para reduzir qualquer overfitting residual"""
        print("\nOTIMIZANDO RANDOM FOREST - CONTROLE DE COMPLEXIDADE")
        print("=" * 55)
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        print("Executando Grid Search para Random Forest...")
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42), param_grid, 
            cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X, self.y)
        
        best_rf = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Melhores parâmetros: {best_params}")
        print(f"Melhor AUC CV: {best_score:.3f}")
        
        # Testar overfitting
        overfitting_results = self._test_overfitting(best_rf, "Random Forest Otimizado")
        
        self.corrected_models['Random Forest'] = {
            'model': best_rf,
            'params': best_params,
            'cv_score': best_score,
            'overfitting_results': overfitting_results
        }
        
        return best_rf, overfitting_results
    
    def optimize_gradient_boosting(self):
        """Otimiza Gradient Boosting com regularização"""
        print("\nOTIMIZANDO GRADIENT BOOSTING - REGULARIZAÇÃO")
        print("=" * 45)
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        print("Executando Grid Search para Gradient Boosting...")
        grid_search = GridSearchCV(
            GradientBoostingClassifier(random_state=42), param_grid,
            cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X, self.y)
        
        best_gb = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Melhores parâmetros: {best_params}")
        print(f"Melhor AUC CV: {best_score:.3f}")
        
        # Testar overfitting
        overfitting_results = self._test_overfitting(best_gb, "Gradient Boosting Otimizado")
        
        self.corrected_models['Gradient Boosting'] = {
            'model': best_gb,
            'params': best_params,
            'cv_score': best_score,
            'overfitting_results': overfitting_results
        }
        
        return best_gb, overfitting_results
    
    def _test_overfitting(self, model, model_name):
        """Testa overfitting em um modelo específico"""
        print(f"\nTestando overfitting em {model_name}...")
        
        # Dividir dados para teste
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Treinar modelo
        model.fit(X_train, y_train)
        
        # Predições
        if hasattr(model, 'predict_proba'):
            train_pred = model.predict_proba(X_train)[:, 1]
            test_pred = model.predict_proba(X_test)[:, 1]
        else:
            train_pred = model.decision_function(X_train)
            test_pred = model.decision_function(X_test)
        
        # Métricas
        train_auc = roc_auc_score(y_train, train_pred)
        test_auc = roc_auc_score(y_test, test_pred)
        gap = train_auc - test_auc
        
        # Validação cruzada
        cv_scores = cross_val_score(model, self.X, self.y, cv=10, scoring='roc_auc')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        results = {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'gap': gap,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'is_overfitting': gap > 0.05 or cv_std > 0.05
        }
        
        print(f"   Train AUC: {train_auc:.3f}")
        print(f"   Test AUC: {test_auc:.3f}")
        print(f"   Gap: {gap:.3f}")
        print(f"   CV AUC: {cv_mean:.3f} ± {cv_std:.3f}")
        
        if results['is_overfitting']:
            print(f"   Status: AINDA COM OVERFITTING")
        else:
            print(f"   Status: OVERFITTING CORRIGIDO")
        
        return results
    
    def compare_before_after(self):
        """Compara modelos antes e depois da correção"""
        print("\nCOMPARAÇÃO ANTES vs DEPOIS DA CORREÇÃO")
        print("=" * 45)
        
        # Modelos originais (sem otimização)
        original_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        comparison_results = {}
        
        for name, original_model in original_models.items():
            print(f"\nComparando {name}...")
            
            # Teste modelo original
            original_results = self._test_overfitting(original_model, f"{name} Original")
            
            # Resultados do modelo corrigido
            if name in self.corrected_models:
                corrected_results = self.corrected_models[name]['overfitting_results']
                
                comparison_results[name] = {
                    'original': original_results,
                    'corrected': corrected_results,
                    'improvement': {
                        'gap_reduction': original_results['gap'] - corrected_results['gap'],
                        'cv_std_reduction': original_results['cv_std'] - corrected_results['cv_std'],
                        'auc_improvement': corrected_results['cv_mean'] - original_results['cv_mean']
                    }
                }
                
                print(f"   MELHORIA:")
                print(f"     Gap: {original_results['gap']:.3f} -> {corrected_results['gap']:.3f}")
                print(f"     CV Std: {original_results['cv_std']:.3f} -> {corrected_results['cv_std']:.3f}")
                print(f"     CV AUC: {original_results['cv_mean']:.3f} -> {corrected_results['cv_mean']:.3f}")
        
        return comparison_results
    
    def create_correction_visualization(self, comparison_results):
        """Cria visualização da correção de overfitting"""
        print("\nGerando visualização das correções...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(comparison_results.keys())
        
        # 1. Gap Train-Test
        original_gaps = [comparison_results[m]['original']['gap'] for m in models]
        corrected_gaps = [comparison_results[m]['corrected']['gap'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, original_gaps, width, label='Original', alpha=0.7, color='red')
        ax1.bar(x + width/2, corrected_gaps, width, label='Corrigido', alpha=0.7, color='green')
        ax1.set_title('Gap Train-Test AUC', fontweight='bold')
        ax1.set_ylabel('Gap AUC')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.05, color='orange', linestyle='--', label='Threshold Overfitting')
        
        # 2. Variância CV
        original_stds = [comparison_results[m]['original']['cv_std'] for m in models]
        corrected_stds = [comparison_results[m]['corrected']['cv_std'] for m in models]
        
        ax2.bar(x - width/2, original_stds, width, label='Original', alpha=0.7, color='red')
        ax2.bar(x + width/2, corrected_stds, width, label='Corrigido', alpha=0.7, color='green')
        ax2.set_title('Variância CV (Std)', fontweight='bold')
        ax2.set_ylabel('Std AUC')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.05, color='orange', linestyle='--', label='Threshold Variância')
        
        # 3. AUC Médio CV
        original_aucs = [comparison_results[m]['original']['cv_mean'] for m in models]
        corrected_aucs = [comparison_results[m]['corrected']['cv_mean'] for m in models]
        
        ax3.bar(x - width/2, original_aucs, width, label='Original', alpha=0.7, color='red')
        ax3.bar(x + width/2, corrected_aucs, width, label='Corrigido', alpha=0.7, color='green')
        ax3.set_title('AUC Médio CV', fontweight='bold')
        ax3.set_ylabel('AUC')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Resumo das melhorias
        gap_reductions = [comparison_results[m]['improvement']['gap_reduction'] for m in models]
        
        colors = ['green' if gr > 0 else 'red' for gr in gap_reductions]
        ax4.bar(x, gap_reductions, color=colors, alpha=0.7)
        ax4.set_title('Redução do Gap (Melhoria)', fontweight='bold')
        ax4.set_ylabel('Redução do Gap')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('correcao_overfitting_modelos.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_correction_report(self, comparison_results):
        """Gera relatório da correção de overfitting"""
        print("\nRELATÓRIO FINAL - CORREÇÃO DE OVERFITTING")
        print("=" * 50)
        
        total_models = len(comparison_results)
        improved_models = 0
        
        for model_name, results in comparison_results.items():
            print(f"\n{model_name}:")
            
            original = results['original']
            corrected = results['corrected']
            improvement = results['improvement']
            
            print(f"   Gap Original: {original['gap']:.3f}")
            print(f"   Gap Corrigido: {corrected['gap']:.3f}")
            print(f"   Redução Gap: {improvement['gap_reduction']:.3f}")
            
            print(f"   CV Std Original: {original['cv_std']:.3f}")
            print(f"   CV Std Corrigido: {corrected['cv_std']:.3f}")
            print(f"   Redução Std: {improvement['cv_std_reduction']:.3f}")
            
            print(f"   AUC Original: {original['cv_mean']:.3f}")
            print(f"   AUC Corrigido: {corrected['cv_mean']:.3f}")
            print(f"   Melhoria AUC: {improvement['auc_improvement']:.3f}")
            
            # Status da correção
            gap_improved = improvement['gap_reduction'] > 0
            std_improved = improvement['cv_std_reduction'] > 0
            no_overfitting = not corrected['is_overfitting']
            
            if no_overfitting and gap_improved:
                print(f"   Status: CORRIGIDO COM SUCESSO")
                improved_models += 1
            elif gap_improved or std_improved:
                print(f"   Status: MELHORADO")
                improved_models += 0.5
            else:
                print(f"   Status: SEM MELHORIA SIGNIFICATIVA")
        
        print(f"\nRESUMO GERAL:")
        print(f"   Total de modelos: {total_models}")
        print(f"   Modelos melhorados: {improved_models}")
        print(f"   Taxa de sucesso: {improved_models/total_models*100:.1f}%")
        
        if improved_models == total_models:
            print(f"   EXCELENTE: Todos os modelos foram corrigidos!")
        elif improved_models >= total_models * 0.8:
            print(f"   BOM: Maioria dos modelos corrigidos")
        else:
            print(f"   MODERADO: Alguns modelos ainda precisam de ajustes")

def main():
    """Função principal para correção de overfitting"""
    print("CORREÇÃO DE OVERFITTING - MODELOS BALANCEADOS")
    print("=" * 55)
    
    # Inicializar corretor
    corrector = OverfittingCorrector()
    
    # Carregar dados
    X, y = corrector.load_data()
    
    print("\nINICIANDO CORREÇÕES...")
    
    # Corrigir cada modelo
    corrector.correct_logistic_regression()
    corrector.correct_svm()
    corrector.correct_mlp()
    corrector.optimize_random_forest()
    corrector.optimize_gradient_boosting()
    
    # Comparar antes vs depois
    comparison_results = corrector.compare_before_after()
    
    # Criar visualização
    corrector.create_correction_visualization(comparison_results)
    
    # Relatório final
    corrector.generate_correction_report(comparison_results)
    
    print(f"\nCORREÇÃO FINALIZADA!")
    print(f"Arquivo gerado: correcao_overfitting_modelos.png")

if __name__ == "__main__":
    main()

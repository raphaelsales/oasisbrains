#!/usr/bin/env python3
"""
Verificação de Overfitting no Dataset Augmentado e Balanceado
Análise abrangente para detectar overfitting usando múltiplas métricas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class OverfittingAnalyzer:
    """Analisador completo de overfitting para dataset balanceado"""
    
    def __init__(self, dataset_path='alzheimer_complete_dataset_augmented.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        
    def load_data(self):
        """Carrega e prepara dados balanceados"""
        print("CARREGANDO DATASET AUGMENTADO BALANCEADO")
        print("=" * 55)
        
        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset carregado: {self.df.shape[0]} sujeitos, {self.df.shape[1]} features")
        
        # Verificar balanceamento
        if 'cdr' in self.df.columns:
            cdr_dist = self.df['cdr'].value_counts().sort_index()
            print(f"\nDistribuição CDR:")
            for cdr, count in cdr_dist.items():
                print(f"   CDR = {cdr}: {count} amostras")
        
        # Preparar features
        self.feature_names = [col for col in self.df.columns 
                             if col not in ['subject_id', 'diagnosis', 'gender', 'cdr']]
        
        self.X = self.df[self.feature_names].fillna(self.df[self.feature_names].median())
        
        # Target binário
        if 'diagnosis' in self.df.columns:
            self.y = (self.df['diagnosis'] == 'Demented').astype(int)
        else:
            self.y = (self.df['cdr'] > 0).astype(int)
        
        print(f"Features selecionadas: {len(self.feature_names)}")
        print(f"Distribuição target: {np.bincount(self.y)}")
        
        return self.X, self.y
    
    def learning_curves_analysis(self):
        """Análise de curvas de aprendizado para detectar overfitting"""
        print("\n📈 ANÁLISE DE CURVAS DE APRENDIZADO")
        print("=" * 45)
        
        # Modelos para testar
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        overfitting_results = {}
        
        for i, (name, model) in enumerate(models.items()):
            print(f"\nAnalisando {name}...")
            
            # Calcular curvas de aprendizado
            train_sizes, train_scores, val_scores = learning_curve(
                model, self.X, self.y, cv=5, 
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='roc_auc', random_state=42
            )
            
            # Calcular médias e desvios
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Plotar
            ax = axes[i]
            ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Treino')
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.3, color='blue')
            
            ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validação')
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.3, color='red')
            
            ax.set_title(f'{name}\nCurvas de Aprendizado', fontweight='bold')
            ax.set_xlabel('Tamanho do Dataset de Treino')
            ax.set_ylabel('AUC Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Calcular gap de overfitting
            final_train_score = train_mean[-1]
            final_val_score = val_mean[-1]
            overfitting_gap = final_train_score - final_val_score
            
            overfitting_results[name] = {
                'train_score': final_train_score,
                'val_score': final_val_score,
                'gap': overfitting_gap,
                'is_overfitting': overfitting_gap > 0.05  # Threshold de 5%
            }
            
            # Adicionar texto com gap
            gap_text = f'Gap: {overfitting_gap:.3f}'
            status = '❌ Overfitting' if overfitting_gap > 0.05 else '✅ OK'
            ax.text(0.02, 0.98, f'{gap_text}\n{status}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
            
            print(f"   Train AUC: {final_train_score:.3f}")
            print(f"   Val AUC: {final_val_score:.3f}")
            print(f"   Gap: {overfitting_gap:.3f} {'❌ Overfitting' if overfitting_gap > 0.05 else '✅ OK'}")
        
        plt.tight_layout()
        plt.savefig('overfitting_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return overfitting_results
    
    def validation_curves_analysis(self):
        """Análise de curvas de validação para diferentes hiperparâmetros"""
        print("\n🎛️ ANÁLISE DE CURVAS DE VALIDAÇÃO")
        print("=" * 40)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Random Forest - n_estimators
        print("Analisando Random Forest (n_estimators)...")
        param_range = [10, 50, 100, 200, 300, 500]
        train_scores, val_scores = validation_curve(
            RandomForestClassifier(random_state=42), self.X, self.y,
            param_name='n_estimators', param_range=param_range,
            cv=5, scoring='roc_auc'
        )
        
        self._plot_validation_curve(axes[0, 0], param_range, train_scores, val_scores,
                                   'Random Forest', 'n_estimators')
        
        # 2. Random Forest - max_depth
        print("Analisando Random Forest (max_depth)...")
        param_range = [3, 5, 10, 15, 20, 25, None]
        train_scores, val_scores = validation_curve(
            RandomForestClassifier(n_estimators=100, random_state=42), self.X, self.y,
            param_name='max_depth', param_range=param_range,
            cv=5, scoring='roc_auc'
        )
        
        # Converter None para string para plotagem
        param_range_str = [str(p) if p is not None else 'None' for p in param_range]
        self._plot_validation_curve(axes[0, 1], param_range_str, train_scores, val_scores,
                                   'Random Forest', 'max_depth')
        
        # 3. Gradient Boosting - n_estimators
        print("Analisando Gradient Boosting (n_estimators)...")
        param_range = [50, 100, 150, 200, 300]
        train_scores, val_scores = validation_curve(
            GradientBoostingClassifier(random_state=42), self.X, self.y,
            param_name='n_estimators', param_range=param_range,
            cv=5, scoring='roc_auc'
        )
        
        self._plot_validation_curve(axes[1, 0], param_range, train_scores, val_scores,
                                   'Gradient Boosting', 'n_estimators')
        
        # 4. Gradient Boosting - learning_rate
        print("Analisando Gradient Boosting (learning_rate)...")
        param_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        train_scores, val_scores = validation_curve(
            GradientBoostingClassifier(n_estimators=100, random_state=42), self.X, self.y,
            param_name='learning_rate', param_range=param_range,
            cv=5, scoring='roc_auc'
        )
        
        self._plot_validation_curve(axes[1, 1], param_range, train_scores, val_scores,
                                   'Gradient Boosting', 'learning_rate')
        
        plt.tight_layout()
        plt.savefig('overfitting_validation_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_validation_curve(self, ax, param_range, train_scores, val_scores, model_name, param_name):
        """Helper para plotar curva de validação"""
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(range(len(param_range)), train_mean, 'o-', color='blue', label='Treino')
        ax.fill_between(range(len(param_range)), train_mean - train_std, train_mean + train_std, 
                       alpha=0.3, color='blue')
        
        ax.plot(range(len(param_range)), val_mean, 'o-', color='red', label='Validação')
        ax.fill_between(range(len(param_range)), val_mean - val_std, val_mean + val_std, 
                       alpha=0.3, color='red')
        
        ax.set_title(f'{model_name}\n{param_name}', fontweight='bold')
        ax.set_xlabel(param_name)
        ax.set_ylabel('AUC Score')
        ax.set_xticks(range(len(param_range)))
        ax.set_xticklabels(param_range, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Identificar overfitting
        max_gap = np.max(train_mean - val_mean)
        if max_gap > 0.05:
            ax.text(0.02, 0.02, '❌ Overfitting detectado', 
                   transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
        else:
            ax.text(0.02, 0.02, '✅ Sem overfitting', 
                   transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7))
    
    def cross_validation_analysis(self):
        """Análise de validação cruzada robusta"""
        print("\n🔄 ANÁLISE DE VALIDAÇÃO CRUZADA ROBUSTA")
        print("=" * 45)
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        cv_results = {}
        
        print("Executando validação cruzada 10-fold...")
        
        for name, model in models.items():
            print(f"\nTestando {name}...")
            
            # Validação cruzada para múltiplas métricas
            accuracy_scores = cross_val_score(model, self.X, self.y, cv=10, scoring='accuracy')
            auc_scores = cross_val_score(model, self.X, self.y, cv=10, scoring='roc_auc')
            
            cv_results[name] = {
                'accuracy_mean': np.mean(accuracy_scores),
                'accuracy_std': np.std(accuracy_scores),
                'auc_mean': np.mean(auc_scores),
                'auc_std': np.std(auc_scores),
                'accuracy_scores': accuracy_scores,
                'auc_scores': auc_scores
            }
            
            print(f"   Accuracy: {np.mean(accuracy_scores):.3f} ± {np.std(accuracy_scores):.3f}")
            print(f"   AUC: {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
            
            # Detectar alta variância (sinal de overfitting)
            if np.std(auc_scores) > 0.05:
                print(f"   ⚠️ Alta variância detectada (std={np.std(auc_scores):.3f})")
            else:
                print(f"   ✅ Variância aceitável (std={np.std(auc_scores):.3f})")
        
        # Plotar resultados
        self._plot_cv_results(cv_results)
        
        return cv_results
    
    def _plot_cv_results(self, cv_results):
        """Plotar resultados da validação cruzada"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        models = list(cv_results.keys())
        
        # Accuracy
        acc_means = [cv_results[model]['accuracy_mean'] for model in models]
        acc_stds = [cv_results[model]['accuracy_std'] for model in models]
        
        ax1.bar(range(len(models)), acc_means, yerr=acc_stds, capsize=5, alpha=0.7)
        ax1.set_title('Accuracy - Validação Cruzada 10-fold', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # AUC
        auc_means = [cv_results[model]['auc_mean'] for model in models]
        auc_stds = [cv_results[model]['auc_std'] for model in models]
        
        ax2.bar(range(len(models)), auc_means, yerr=auc_stds, capsize=5, alpha=0.7, color='orange')
        ax2.set_title('AUC - Validação Cruzada 10-fold', fontweight='bold')
        ax2.set_ylabel('AUC')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('overfitting_cross_validation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_test_holdout_analysis(self):
        """Análise com holdout set separado para verificação final"""
        print("\n🎯 ANÁLISE TRAIN/VALIDATION/TEST HOLDOUT")
        print("=" * 45)
        
        # Dividir em 60% treino, 20% validação, 20% teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 de 0.8 = 0.2 do total
        )
        
        print(f"Train: {len(X_train)} amostras")
        print(f"Validation: {len(X_val)} amostras")
        print(f"Test: {len(X_test)} amostras")
        
        # Normalizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        holdout_results = {}
        
        for name, model in models.items():
            print(f"\nTestando {name}...")
            
            # Treinar modelo
            if name in ['SVM', 'Logistic Regression', 'MLP']:
                model.fit(X_train_scaled, y_train)
                train_pred = model.predict_proba(X_train_scaled)[:, 1]
                val_pred = model.predict_proba(X_val_scaled)[:, 1]
                test_pred = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                train_pred = model.predict_proba(X_train)[:, 1]
                val_pred = model.predict_proba(X_val)[:, 1]
                test_pred = model.predict_proba(X_test)[:, 1]
            
            # Calcular métricas
            train_auc = roc_auc_score(y_train, train_pred)
            val_auc = roc_auc_score(y_val, val_pred)
            test_auc = roc_auc_score(y_test, test_pred)
            
            holdout_results[name] = {
                'train_auc': train_auc,
                'val_auc': val_auc,
                'test_auc': test_auc,
                'train_val_gap': train_auc - val_auc,
                'val_test_gap': val_auc - test_auc
            }
            
            print(f"   Train AUC: {train_auc:.3f}")
            print(f"   Val AUC: {val_auc:.3f}")
            print(f"   Test AUC: {test_auc:.3f}")
            print(f"   Train-Val Gap: {train_auc - val_auc:.3f}")
            print(f"   Val-Test Gap: {val_auc - test_auc:.3f}")
            
            # Verificar overfitting
            if train_auc - val_auc > 0.05:
                print(f"   ❌ Overfitting detectado (gap > 0.05)")
            elif abs(val_auc - test_auc) > 0.05:
                print(f"   ⚠️ Possível problema de generalização")
            else:
                print(f"   ✅ Sem overfitting detectado")
        
        return holdout_results
    
    def generate_overfitting_report(self, learning_results, cv_results, holdout_results):
        """Gerar relatório final sobre overfitting"""
        print("\n📋 RELATÓRIO FINAL - DETECÇÃO DE OVERFITTING")
        print("=" * 55)
        
        report = {
            'dataset_info': {
                'total_samples': len(self.df),
                'features': len(self.feature_names),
                'class_distribution': dict(self.df['cdr'].value_counts().sort_index()) if 'cdr' in self.df.columns else None
            },
            'overfitting_analysis': {}
        }
        
        print(f"📊 INFORMAÇÕES DO DATASET:")
        print(f"   Total de amostras: {report['dataset_info']['total_samples']}")
        print(f"   Número de features: {report['dataset_info']['features']}")
        
        if report['dataset_info']['class_distribution']:
            print(f"   Distribuição CDR: {report['dataset_info']['class_distribution']}")
        
        print(f"\n🔍 ANÁLISE DE OVERFITTING POR MODELO:")
        
        for model_name in learning_results.keys():
            print(f"\n   {model_name}:")
            
            # Learning curves
            learning_gap = learning_results[model_name]['gap']
            is_overfitting_lc = learning_results[model_name]['is_overfitting']
            
            # Cross validation
            cv_std = cv_results[model_name]['auc_std']
            high_variance = cv_std > 0.05
            
            # Holdout
            holdout_gap = holdout_results[model_name]['train_val_gap']
            generalization_gap = abs(holdout_results[model_name]['val_test_gap'])
            
            print(f"      Learning Curves Gap: {learning_gap:.3f} {'❌' if is_overfitting_lc else '✅'}")
            print(f"      CV Variância (std): {cv_std:.3f} {'⚠️' if high_variance else '✅'}")
            print(f"      Holdout Train-Val Gap: {holdout_gap:.3f} {'❌' if holdout_gap > 0.05 else '✅'}")
            print(f"      Generalização Gap: {generalization_gap:.3f} {'⚠️' if generalization_gap > 0.05 else '✅'}")
            
            # Conclusão por modelo
            overfitting_indicators = sum([
                is_overfitting_lc,
                high_variance, 
                holdout_gap > 0.05,
                generalization_gap > 0.05
            ])
            
            if overfitting_indicators == 0:
                status = "✅ SEM OVERFITTING"
            elif overfitting_indicators <= 1:
                status = "⚠️ OVERFITTING LEVE"
            elif overfitting_indicators <= 2:
                status = "❌ OVERFITTING MODERADO"
            else:
                status = "🚨 OVERFITTING SEVERO"
            
            print(f"      STATUS: {status}")
            
            report['overfitting_analysis'][model_name] = {
                'learning_gap': learning_gap,
                'cv_std': cv_std,
                'holdout_gap': holdout_gap,
                'generalization_gap': generalization_gap,
                'overfitting_indicators': overfitting_indicators,
                'status': status
            }
        
        print(f"\n🎯 CONCLUSÕES GERAIS:")
        
        # Análise geral
        total_models = len(learning_results)
        clean_models = sum(1 for analysis in report['overfitting_analysis'].values() 
                          if analysis['overfitting_indicators'] == 0)
        
        print(f"   Total de modelos testados: {total_models}")
        print(f"   Modelos sem overfitting: {clean_models}/{total_models}")
        print(f"   Taxa de sucesso: {clean_models/total_models*100:.1f}%")
        
        if clean_models == total_models:
            print(f"   🎉 EXCELENTE: Todos os modelos estão livres de overfitting!")
        elif clean_models >= total_models * 0.8:
            print(f"   ✅ BOM: Maioria dos modelos sem overfitting")
        elif clean_models >= total_models * 0.5:
            print(f"   ⚠️ MODERADO: Alguns modelos com overfitting")
        else:
            print(f"   ❌ CRÍTICO: Maioria dos modelos com overfitting")
        
        print(f"\n💡 RECOMENDAÇÕES:")
        if clean_models == total_models:
            print(f"   • Dataset balanceado está funcionando bem")
            print(f"   • Continuar com os hiperparâmetros atuais")
            print(f"   • Considerar usar os modelos em produção")
        else:
            print(f"   • Considerar mais regularização nos modelos problemáticos")
            print(f"   • Implementar early stopping")
            print(f"   • Reduzir complexidade dos modelos")
            print(f"   • Verificar se data augmentation está introduzindo ruído")
        
        return report

def main():
    """Função principal para análise completa de overfitting"""
    print("🔍 ANÁLISE COMPLETA DE OVERFITTING - DATASET BALANCEADO")
    print("=" * 65)
    
    # Inicializar analisador
    analyzer = OverfittingAnalyzer()
    
    # Carregar dados
    X, y = analyzer.load_data()
    
    # Executar análises
    print("\n🚀 INICIANDO ANÁLISES...")
    
    # 1. Learning curves
    learning_results = analyzer.learning_curves_analysis()
    
    # 2. Validation curves
    analyzer.validation_curves_analysis()
    
    # 3. Cross validation
    cv_results = analyzer.cross_validation_analysis()
    
    # 4. Holdout analysis
    holdout_results = analyzer.train_test_holdout_analysis()
    
    # 5. Relatório final
    report = analyzer.generate_overfitting_report(learning_results, cv_results, holdout_results)
    
    print(f"\n✅ ANÁLISE COMPLETA FINALIZADA!")
    print(f"📁 Arquivos gerados:")
    print(f"   • overfitting_learning_curves.png")
    print(f"   • overfitting_validation_curves.png") 
    print(f"   • overfitting_cross_validation.png")

if __name__ == "__main__":
    main()

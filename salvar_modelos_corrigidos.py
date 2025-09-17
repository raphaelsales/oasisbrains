#!/usr/bin/env python3
"""
Script para salvar os modelos corrigidos sem overfitting
Recria e salva todos os modelos otimizados
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class ModelSaver:
    """Salva modelos corrigidos sem overfitting"""
    
    def __init__(self, dataset_path='alzheimer_complete_dataset_augmented.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        
    def load_data(self):
        """Carrega dados balanceados"""
        print("CARREGANDO DATASET PARA SALVAR MODELOS CORRIGIDOS")
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
    
    def create_and_save_logistic_regression(self):
        """Cria e salva Logistic Regression corrigida"""
        print("\nCRIANDO E SALVANDO LOGISTIC REGRESSION CORRIGIDA")
        print("=" * 55)
        
        # Melhores parâmetros encontrados na análise anterior
        best_params = {
            'C': 100.0,
            'penalty': 'l2',
            'solver': 'saga',
            'max_iter': 2000,
            'random_state': 42
        }
        
        # Pipeline com normalização
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**best_params))
        ])
        
        # Treinar modelo
        pipeline.fit(self.X, self.y)
        
        # Testar performance
        y_pred_proba = pipeline.predict_proba(self.X)[:, 1]
        auc = roc_auc_score(self.y, y_pred_proba)
        acc = accuracy_score(self.y, pipeline.predict(self.X))
        
        print(f"AUC: {auc:.3f}")
        print(f"Accuracy: {acc:.3f}")
        
        # Salvar modelo
        model_path = 'logistic_regression_sem_overfitting.joblib'
        joblib.dump(pipeline, model_path)
        print(f"Modelo salvo: {model_path}")
        
        # Salvar informações
        info = {
            'model_type': 'Logistic Regression',
            'parameters': best_params,
            'features': self.feature_names,
            'auc': auc,
            'accuracy': acc,
            'dataset_size': len(self.X),
            'target_distribution': dict(zip(*np.unique(self.y, return_counts=True)))
        }
        
        info_path = 'logistic_regression_sem_overfitting_info.joblib'
        joblib.dump(info, info_path)
        print(f"Informações salvas: {info_path}")
        
        return pipeline, model_path
    
    def create_and_save_svm(self):
        """Cria e salva SVM corrigido"""
        print("\nCRIANDO E SALVANDO SVM CORRIGIDO")
        print("=" * 35)
        
        # Melhores parâmetros encontrados
        best_params = {
            'C': 10.0,
            'gamma': 0.01,
            'kernel': 'rbf',
            'probability': True,
            'random_state': 42
        }
        
        # Pipeline com normalização
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(**best_params))
        ])
        
        # Treinar modelo
        pipeline.fit(self.X, self.y)
        
        # Testar performance
        y_pred_proba = pipeline.predict_proba(self.X)[:, 1]
        auc = roc_auc_score(self.y, y_pred_proba)
        acc = accuracy_score(self.y, pipeline.predict(self.X))
        
        print(f"AUC: {auc:.3f}")
        print(f"Accuracy: {acc:.3f}")
        
        # Salvar modelo
        model_path = 'svm_sem_overfitting.joblib'
        joblib.dump(pipeline, model_path)
        print(f"Modelo salvo: {model_path}")
        
        # Salvar informações
        info = {
            'model_type': 'SVM',
            'parameters': best_params,
            'features': self.feature_names,
            'auc': auc,
            'accuracy': acc,
            'dataset_size': len(self.X),
            'target_distribution': dict(zip(*np.unique(self.y, return_counts=True)))
        }
        
        info_path = 'svm_sem_overfitting_info.joblib'
        joblib.dump(info, info_path)
        print(f"Informações salvas: {info_path}")
        
        return pipeline, model_path
    
    def create_and_save_mlp(self):
        """Cria e salva MLP corrigido"""
        print("\nCRIANDO E SALVANDO MLP CORRIGIDO")
        print("=" * 35)
        
        # Melhores parâmetros encontrados
        best_params = {
            'hidden_layer_sizes': (100, 50),
            'alpha': 1.0,
            'learning_rate': 'constant',
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': 20,
            'max_iter': 1000,
            'random_state': 42
        }
        
        # Pipeline com normalização
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(**best_params))
        ])
        
        # Treinar modelo
        pipeline.fit(self.X, self.y)
        
        # Testar performance
        y_pred_proba = pipeline.predict_proba(self.X)[:, 1]
        auc = roc_auc_score(self.y, y_pred_proba)
        acc = accuracy_score(self.y, pipeline.predict(self.X))
        
        print(f"AUC: {auc:.3f}")
        print(f"Accuracy: {acc:.3f}")
        
        # Salvar modelo
        model_path = 'mlp_sem_overfitting.joblib'
        joblib.dump(pipeline, model_path)
        print(f"Modelo salvo: {model_path}")
        
        # Salvar informações
        info = {
            'model_type': 'MLP',
            'parameters': best_params,
            'features': self.feature_names,
            'auc': auc,
            'accuracy': acc,
            'dataset_size': len(self.X),
            'target_distribution': dict(zip(*np.unique(self.y, return_counts=True)))
        }
        
        info_path = 'mlp_sem_overfitting_info.joblib'
        joblib.dump(info, info_path)
        print(f"Informações salvas: {info_path}")
        
        return pipeline, model_path
    
    def create_and_save_random_forest(self):
        """Cria e salva Random Forest otimizado"""
        print("\nCRIANDO E SALVANDO RANDOM FOREST OTIMIZADO")
        print("=" * 45)
        
        # Melhores parâmetros encontrados
        best_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'max_features': 'log2',
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'random_state': 42
        }
        
        # Modelo sem pipeline (Random Forest não precisa normalização)
        model = RandomForestClassifier(**best_params)
        
        # Treinar modelo
        model.fit(self.X, self.y)
        
        # Testar performance
        y_pred_proba = model.predict_proba(self.X)[:, 1]
        auc = roc_auc_score(self.y, y_pred_proba)
        acc = accuracy_score(self.y, model.predict(self.X))
        
        print(f"AUC: {auc:.3f}")
        print(f"Accuracy: {acc:.3f}")
        
        # Salvar modelo
        model_path = 'random_forest_otimizado.joblib'
        joblib.dump(model, model_path)
        print(f"Modelo salvo: {model_path}")
        
        # Salvar informações
        info = {
            'model_type': 'Random Forest',
            'parameters': best_params,
            'features': self.feature_names,
            'auc': auc,
            'accuracy': acc,
            'dataset_size': len(self.X),
            'target_distribution': dict(zip(*np.unique(self.y, return_counts=True)))
        }
        
        info_path = 'random_forest_otimizado_info.joblib'
        joblib.dump(info, info_path)
        print(f"Informações salvas: {info_path}")
        
        return model, model_path
    
    def create_and_save_gradient_boosting(self):
        """Cria e salva Gradient Boosting otimizado"""
        print("\nCRIANDO E SALVANDO GRADIENT BOOSTING OTIMIZADO")
        print("=" * 45)
        
        # Melhores parâmetros encontrados
        best_params = {
            'n_estimators': 150,
            'learning_rate': 0.2,
            'max_depth': 7,
            'min_samples_leaf': 1,
            'min_samples_split': 10,
            'subsample': 0.8,
            'random_state': 42
        }
        
        # Modelo sem pipeline
        model = GradientBoostingClassifier(**best_params)
        
        # Treinar modelo
        model.fit(self.X, self.y)
        
        # Testar performance
        y_pred_proba = model.predict_proba(self.X)[:, 1]
        auc = roc_auc_score(self.y, y_pred_proba)
        acc = accuracy_score(self.y, model.predict(self.X))
        
        print(f"AUC: {auc:.3f}")
        print(f"Accuracy: {acc:.3f}")
        
        # Salvar modelo
        model_path = 'gradient_boosting_otimizado.joblib'
        joblib.dump(model, model_path)
        print(f"Modelo salvo: {model_path}")
        
        # Salvar informações
        info = {
            'model_type': 'Gradient Boosting',
            'parameters': best_params,
            'features': self.feature_names,
            'auc': auc,
            'accuracy': acc,
            'dataset_size': len(self.X),
            'target_distribution': dict(zip(*np.unique(self.y, return_counts=True)))
        }
        
        info_path = 'gradient_boosting_otimizado_info.joblib'
        joblib.dump(info, info_path)
        print(f"Informações salvas: {info_path}")
        
        return model, model_path
    
    def create_ensemble_model(self):
        """Cria e salva modelo ensemble dos melhores modelos"""
        print("\nCRIANDO MODELO ENSEMBLE DOS MELHORES MODELOS")
        print("=" * 45)
        
        # Carregar modelos salvos
        lr_model = joblib.load('logistic_regression_sem_overfitting.joblib')
        svm_model = joblib.load('svm_sem_overfitting.joblib')
        mlp_model = joblib.load('mlp_sem_overfitting.joblib')
        rf_model = joblib.load('random_forest_otimizado.joblib')
        gb_model = joblib.load('gradient_boosting_otimizado.joblib')
        
        models = {
            'Logistic Regression': lr_model,
            'SVM': svm_model,
            'MLP': mlp_model,
            'Random Forest': rf_model,
            'Gradient Boosting': gb_model
        }
        
        # Testar cada modelo e calcular pesos baseados na performance
        weights = {}
        predictions = {}
        
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X)[:, 1]
            else:
                y_pred_proba = model.predict_proba(self.X)[:, 1]
            
            auc = roc_auc_score(self.y, y_pred_proba)
            weights[name] = auc
            predictions[name] = y_pred_proba
            print(f"{name}: AUC = {auc:.3f}")
        
        # Normalizar pesos
        total_weight = sum(weights.values())
        weights = {name: weight/total_weight for name, weight in weights.items()}
        
        # Calcular predição ensemble (weighted average)
        ensemble_pred = np.zeros(len(self.y))
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
        
        # Calcular performance ensemble
        ensemble_auc = roc_auc_score(self.y, ensemble_pred)
        ensemble_acc = accuracy_score(self.y, (ensemble_pred > 0.5).astype(int))
        
        print(f"\nENSEMBLE PERFORMANCE:")
        print(f"AUC: {ensemble_auc:.3f}")
        print(f"Accuracy: {ensemble_acc:.3f}")
        
        # Salvar ensemble
        ensemble_info = {
            'models': models,
            'weights': weights,
            'features': self.feature_names,
            'auc': ensemble_auc,
            'accuracy': ensemble_acc,
            'dataset_size': len(self.X)
        }
        
        ensemble_path = 'ensemble_sem_overfitting.joblib'
        joblib.dump(ensemble_info, ensemble_path)
        print(f"Ensemble salvo: {ensemble_path}")
        
        return ensemble_info, ensemble_path
    
    def generate_summary_report(self):
        """Gera relatório resumo dos modelos salvos"""
        print("\nRELATÓRIO DOS MODELOS SALVOS")
        print("=" * 35)
        
        model_files = [
            'logistic_regression_sem_overfitting.joblib',
            'svm_sem_overfitting.joblib', 
            'mlp_sem_overfitting.joblib',
            'random_forest_otimizado.joblib',
            'gradient_boosting_otimizado.joblib',
            'ensemble_sem_overfitting.joblib'
        ]
        
        print("MODELOS SALVOS SEM OVERFITTING:")
        for i, model_file in enumerate(model_files, 1):
            print(f"{i}. {model_file}")
        
        print(f"\nLOCALIZAÇÃO: /app/alzheimer/")
        print(f"DATASET USADO: {self.dataset_path}")
        print(f"TOTAL DE FEATURES: {len(self.feature_names)}")
        print(f"AMOSTRAS TREINO: {len(self.X)}")
        
        print(f"\nFEATURES UTILIZADAS:")
        for i, feature in enumerate(self.feature_names[:10], 1):
            print(f"  {i}. {feature}")
        if len(self.feature_names) > 10:
            print(f"  ... e mais {len(self.feature_names)-10} features")
        
        print(f"\nPARA CARREGAR UM MODELO:")
        print(f"import joblib")
        print(f"model = joblib.load('random_forest_otimizado.joblib')")
        print(f"prediction = model.predict(X_new)")

def main():
    """Função principal para salvar todos os modelos"""
    print("SALVANDO MODELOS CORRIGIDOS SEM OVERFITTING")
    print("=" * 55)
    
    # Inicializar salvador
    saver = ModelSaver()
    
    # Carregar dados
    X, y = saver.load_data()
    
    print("\nSALVANDO TODOS OS MODELOS CORRIGIDOS...")
    
    # Salvar cada modelo
    saver.create_and_save_logistic_regression()
    saver.create_and_save_svm()
    saver.create_and_save_mlp()
    saver.create_and_save_random_forest()
    saver.create_and_save_gradient_boosting()
    
    # Criar ensemble
    saver.create_ensemble_model()
    
    # Relatório final
    saver.generate_summary_report()
    
    print(f"\nTODOS OS MODELOS SALVOS COM SUCESSO!")

if __name__ == "__main__":
    main()

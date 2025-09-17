#!/usr/bin/env python3
"""
Gerador de Dashboard para Análise de Alzheimer/MCI
Cria visualizações completas baseadas no desempenho do modelo SEM OVERFITTING
Usa o modelo corrigido alzheimer_sem_overfitting.h5
INCLUI MATRIZ DE CONFUSÃO MULTICLASSE (4 classes CDR)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve, 
                           roc_auc_score, precision_score, recall_score, f1_score,
                           accuracy_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo visual
plt.style.use('default')
sns.set_palette("husl")

class AlzheimerDashboardGenerator:
    """Gerador de Dashboard completo para análise de Alzheimer/MCI com classificação multiclasse"""
    
    def __init__(self, data_path=None):
        # Usar dataset aumentado por padrão
        if data_path is None:
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_path = os.path.join(parent_dir, "alzheimer_complete_dataset_augmented.csv")
        else:
            self.data_path = data_path
        self.df = None
        self.models = {}
        self.results = {}
        self.corrected_model = None
        self.corrected_scaler = None
        self.label_encoder = LabelEncoder()
        self.multiclass_results = {}
        
    def load_or_create_data(self):
        """Carrega dados existentes ou cria dados sintéticos realistas"""
        
        # Tentar primeiro o dataset augmentado
        augmented_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     "alzheimer_complete_dataset_augmented.csv")
        
        if os.path.exists(augmented_path):
            print(f"Carregando dataset augmentado: {augmented_path}")
            self.df = pd.read_csv(augmented_path)
            print(f"Dataset augmentado carregado: {self.df.shape[0]} sujeitos, {self.df.shape[1]} features")
            
            # Verificar distribuição das classes
            if 'cdr' in self.df.columns:
                dist = self.df['cdr'].value_counts().sort_index()
                print(f"Distribuição CDR: {dict(dist)}")
        elif self.data_path and os.path.exists(self.data_path):
            print(f"Carregando dados: {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset carregado: {self.df.shape[0]} sujeitos, {self.df.shape[1]} features")
        else:
            print("Criando dataset sintético baseado em OASIS...")
            self.df = self.create_synthetic_alzheimer_data()
            print(f"Dataset sintético criado: {self.df.shape[0]} sujeitos, {self.df.shape[1]} features")
            
        return self.df
    
    def create_synthetic_alzheimer_data(self):
        """Cria dataset sintético realista baseado em características OASIS com 4 classes CDR"""
        np.random.seed(42)
        
        n_subjects = 405  # Número baseado no relatório
        
        # Distribuição CDR baseada no relatório
        cdr_dist = [0, 0.5, 1, 2]
        cdr_probs = [0.625, 0.168, 0.158, 0.049]  # Do relatório clínico
        
        data = []
        
        for i in range(n_subjects):
            # CDR baseado na distribuição real
            cdr = np.random.choice(cdr_dist, p=cdr_probs)
            
            # Idade correlacionada com CDR
            if cdr == 0:
                age = np.random.normal(70, 8)
            elif cdr == 0.5:
                age = np.random.normal(73.9, 8.6)  # Do relatório MCI
            elif cdr == 1:
                age = np.random.normal(76, 7)
            else:
                age = np.random.normal(78, 6)
            age = max(60, min(90, age))
            
            # MMSE correlacionado com CDR
            if cdr == 0:
                mmse = np.random.normal(29, 1)
            elif cdr == 0.5:
                mmse = np.random.normal(27.1, 1.8)  # Do relatório MCI
            elif cdr == 1:
                mmse = np.random.normal(23, 2)
            else:
                mmse = np.random.normal(18, 3)
            mmse = max(10, min(30, mmse))
            
            # Gênero com prevalência feminina em MCI
            if cdr == 0.5:
                gender = np.random.choice(['M', 'F'], p=[0.368, 0.632])  # 63.2% feminino em MCI
            else:
                gender = np.random.choice(['M', 'F'], p=[0.45, 0.55])
            
            # Biomarcadores cerebrais
            # Volume do hipocampo (mais afetado em Alzheimer)
            base_hippo = 4000
            if cdr == 0:
                hippo_factor = np.random.normal(1.0, 0.1)
            elif cdr == 0.5:
                hippo_factor = np.random.normal(0.993, 0.08)  # -0.7% do relatório
            elif cdr == 1:
                hippo_factor = np.random.normal(0.85, 0.1)
            else:
                hippo_factor = np.random.normal(0.75, 0.12)
            
            left_hippocampus_volume = base_hippo * hippo_factor * np.random.normal(0.5, 0.05)
            right_hippocampus_volume = base_hippo * hippo_factor * np.random.normal(0.5, 0.05)
            
            # Córtex entorrinal (mais discriminativo)
            base_entorrinal = 1200
            if cdr == 0:
                entorrinal_factor = np.random.normal(1.0, 0.08)
            elif cdr == 0.5:
                entorrinal_factor_left = np.random.normal(0.963, 0.1)  # -3.7% esquerdo
                entorrinal_factor_right = np.random.normal(0.986, 0.08)  # -1.4% direito
            elif cdr == 1:
                entorrinal_factor_left = np.random.normal(0.82, 0.12)
                entorrinal_factor_right = np.random.normal(0.85, 0.10)
            else:
                entorrinal_factor_left = np.random.normal(0.70, 0.15)
                entorrinal_factor_right = np.random.normal(0.72, 0.13)
            
            if cdr == 0:
                left_entorhinal_volume = base_entorrinal * entorrinal_factor * np.random.normal(0.5, 0.05)
                right_entorhinal_volume = base_entorrinal * entorrinal_factor * np.random.normal(0.5, 0.05)
            else:
                left_entorhinal_volume = base_entorrinal * entorrinal_factor_left
                right_entorhinal_volume = base_entorrinal * entorrinal_factor_right
            
            # Amígdala
            base_amygdala = 1800
            if cdr == 0:
                amygdala_factor = np.random.normal(1.0, 0.08)
            elif cdr == 0.5:
                amygdala_factor = np.random.normal(0.99, 0.08)  # -1.0% esquerda
            else:
                amygdala_factor = np.random.normal(0.85, 0.12)
                
            left_amygdala_volume = base_amygdala * amygdala_factor * np.random.normal(0.5, 0.05)
            right_amygdala_volume = base_amygdala * amygdala_factor * np.random.normal(0.5, 0.05)
            
            # Lobo temporal
            base_temporal = 15000
            if cdr == 0:
                temporal_factor = np.random.normal(1.0, 0.06)
            elif cdr == 0.5:
                temporal_factor = np.random.normal(0.978, 0.08)  # -2.2% esquerdo
            else:
                temporal_factor = np.random.normal(0.88, 0.10)
                
            left_temporal_volume = base_temporal * temporal_factor * np.random.normal(0.5, 0.03)
            right_temporal_volume = base_temporal * temporal_factor * np.random.normal(0.5, 0.03)
            
            # Intensidades médias (valores sintéticos)
            left_hippocampus_intensity_mean = np.random.normal(100, 10)
            right_hippocampus_intensity_mean = np.random.normal(100, 10)
            left_entorhinal_intensity_std = np.random.normal(15, 3)
            right_entorhinal_intensity_std = np.random.normal(15, 3)
            left_amygdala_intensity_mean = np.random.normal(95, 8)
            right_amygdala_intensity_std = np.random.normal(12, 2)
            left_temporal_intensity_std = np.random.normal(18, 4)
            
            # Features adicionais
            education = np.random.choice([12, 14, 16, 18], p=[0.4, 0.3, 0.2, 0.1])
            
            # Manter CDR original para classificação multiclasse
            diagnosis_cdr = cdr
            
            data.append({
                'subject_id': f'OAS1_{i:04d}_MR1',
                'age': round(age, 1),
                'gender': gender,
                'cdr': diagnosis_cdr,
                'mmse': round(mmse, 1),
                'education': education,
                'diagnosis': 'Nondemented' if cdr == 0 else 'Demented',  # Para compatibilidade
                'left_hippocampus_volume': round(left_hippocampus_volume, 2),
                'right_hippocampus_volume': round(right_hippocampus_volume, 2),
                'left_entorhinal_volume': round(left_entorhinal_volume, 2),
                'right_entorhinal_volume': round(right_entorhinal_volume, 2),
                'left_amygdala_volume': round(left_amygdala_volume, 2),
                'right_amygdala_volume': round(right_amygdala_volume, 2),
                'left_temporal_volume': round(left_temporal_volume, 2),
                'right_temporal_volume': round(right_temporal_volume, 2),
                'left_hippocampus_intensity_mean': round(left_hippocampus_intensity_mean, 2),
                'right_hippocampus_intensity_mean': round(right_hippocampus_intensity_mean, 2),
                'left_entorhinal_intensity_std': round(left_entorhinal_intensity_std, 2),
                'right_entorhinal_intensity_std': round(right_entorhinal_intensity_std, 2),
                'left_amygdala_intensity_mean': round(left_amygdala_intensity_mean, 2),
                'right_amygdala_intensity_std': round(right_amygdala_intensity_std, 2),
                'left_temporal_intensity_std': round(left_temporal_intensity_std, 2)
            })
        
        return pd.DataFrame(data)
    
    def train_multiclass_models(self):
        """Treina modelos multiclasse para as 4 classes CDR"""
        print("Treinando modelos multiclasse para classificação CDR...")
        
        # Preparar dados para classificação multiclasse
        feature_cols = [col for col in self.df.columns 
                       if col not in ['subject_id', 'diagnosis', 'gender', 'cdr']]
        
        X = self.df[feature_cols].fillna(self.df[feature_cols].median())
        
        # Usar CDR para classificação multiclasse (4 classes: 0, 0.5, 1, 2)
        y_multiclass = self.df['cdr'].values
        
        # Dividir dados para treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_multiclass, test_size=0.2, random_state=42, stratify=y_multiclass
        )
        
        print(f"Treinando modelos multiclasse em {len(X_train)} amostras...")
        print(f"Classes CDR: {np.unique(y_train)}")
        print(f"Distribuição treino: {np.bincount(y_train.astype(int))}")
        print(f"Distribuição teste: {np.bincount(y_test.astype(int))}")
        
        # Modelos multiclasse
        multiclass_models = {
            'Random Forest Multiclasse': RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting Multiclasse': GradientBoostingClassifier(
                n_estimators=200, max_depth=6, random_state=42
            ),
            'SVM Multiclasse': OneVsRestClassifier(SVC(
                kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42
            )),
            'MLP Multiclasse': MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
            )
        }
        
        self.multiclass_results = {}
        self.feature_names = feature_cols
        self.X_test_multiclass = X_test
        self.y_test_multiclass = y_test
        
        for name, model in multiclass_models.items():
            try:
                print(f"  Treinando {name}...")
                
                # Treinar modelo
                model.fit(X_train, y_train)
                
                # Fazer predições
                y_pred = model.predict(X_test)
                
                # Calcular métricas multiclasse
                accuracy = accuracy_score(y_test, y_pred)
                
                # Calcular métricas por classe
                from sklearn.metrics import precision_recall_fscore_support
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_test, y_pred, average=None, zero_division=0
                )
                
                # Calcular macro e weighted averages
                macro_precision = np.mean(precision)
                macro_recall = np.mean(recall)
                macro_f1 = np.mean(f1)
                
                weighted_precision = np.average(precision, weights=support)
                weighted_recall = np.average(recall, weights=support)
                weighted_f1 = np.average(f1, weights=support)
                
                # Matriz de confusão
                cm = confusion_matrix(y_test, y_pred)
                
                self.multiclass_results[name] = {
                    'model': model,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'accuracy': accuracy,
                    'precision_per_class': precision,
                    'recall_per_class': recall,
                    'f1_per_class': f1,
                    'support_per_class': support,
                    'macro_precision': macro_precision,
                    'macro_recall': macro_recall,
                    'macro_f1': macro_f1,
                    'weighted_precision': weighted_precision,
                    'weighted_recall': weighted_recall,
                    'weighted_f1': weighted_f1,
                    'confusion_matrix': cm
                }
                
                print(f"    {name}: Acc = {accuracy:.3f}, Macro F1 = {macro_f1:.3f}")
                
            except Exception as e:
                print(f"    {name}: Erro no treinamento - {e}")
        
        return self.multiclass_results
    
    def load_corrected_models(self):
        """Método legado - agora treina modelos multiclasse"""
        return self.train_multiclass_models()
    
    def plot_multiclass_confusion_matrix(self, fig, gs_pos):
        """Matriz de confusão multiclasse para as 4 classes CDR"""
        ax = fig.add_subplot(gs_pos)
        
        # Usar o melhor modelo (maior acurácia)
        best_model_name = max(self.multiclass_results.keys(), 
                             key=lambda k: self.multiclass_results[k]['accuracy'])
        best_results = self.multiclass_results[best_model_name]
        
        cm = best_results['confusion_matrix']
        
        # Labels das classes CDR
        class_labels = ['CDR 0\n(Normal)', 'CDR 0.5\n(MCI)', 'CDR 1\n(Leve)', 'CDR 2\n(Moderado)']
        
        # Plot da matriz de confusão
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title('Matriz de Confusão Multiclasse\n(4 Classes CDR)', fontsize=14, fontweight='bold')
        
        # Adicionar números na matriz
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=16, fontweight='bold')
        
        ax.set_ylabel('Classe Real (CDR)', fontsize=12)
        ax.set_xlabel('Classe Predita (CDR)', fontsize=12)
        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels, fontsize=10)
        ax.set_yticklabels(class_labels, fontsize=10)
        
        # Rotacionar labels do eixo x para melhor legibilidade
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Adicionar colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Adicionar métricas de performance (posicionamento ajustado para evitar sobreposição)
        accuracy = best_results['accuracy']
        macro_f1 = best_results['macro_f1']
        ax.text(0.5, -0.45, f'Acurácia: {accuracy:.3f} | Macro F1: {macro_f1:.3f}', 
               transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Adicionar estatísticas por classe (posicionamento ajustado)
        stats_text = "Precisão por classe: "
        for i, (prec, rec, f1_score) in enumerate(zip(
            best_results['precision_per_class'], 
            best_results['recall_per_class'], 
            best_results['f1_per_class']
        )):
            stats_text += f"CDR{i}: {f1_score:.2f} "
        
        ax.text(0.5, -0.70, stats_text, 
               transform=ax.transAxes, ha='center', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
    
    def plot_multiclass_metrics_summary(self, fig, gs_pos):
        """Resumo das métricas multiclasse por classe"""
        ax = fig.add_subplot(gs_pos)
        ax.axis('off')
        
        # Usar o melhor modelo
        best_model_name = max(self.multiclass_results.keys(), 
                             key=lambda k: self.multiclass_results[k]['accuracy'])
        best_results = self.multiclass_results[best_model_name]
        
        # Preparar dados para o resumo
        classes = ['CDR 0\n(Normal)', 'CDR 0.5\n(MCI)', 'CDR 1\n(Leve)', 'CDR 2\n(Moderado)']
        precision = best_results['precision_per_class']
        recall = best_results['recall_per_class']
        f1 = best_results['f1_per_class']
        support = best_results['support_per_class']
        
        # Criar tabela de métricas
        metrics_text = f"""RESUMO MULTICLASSE - MODELO: {best_model_name}

MÉTRICAS POR CLASSE CDR:
{'='*50}
CDR 0 (Normal):     Precisão: {precision[0]:.3f} | Recall: {recall[0]:.3f} | F1: {f1[0]:.3f} | Suporte: {support[0]}
CDR 0.5 (MCI):      Precisão: {precision[1]:.3f} | Recall: {recall[1]:.3f} | F1: {f1[1]:.3f} | Suporte: {support[1]}
CDR 1 (Leve):       Precisão: {precision[2]:.3f} | Recall: {recall[2]:.3f} | F1: {f1[2]:.3f} | Suporte: {support[2]}
CDR 2 (Moderado):   Precisão: {precision[3]:.3f} | Recall: {recall[3]:.3f} | F1: {f1[3]:.3f} | Suporte: {support[3]}

MÉTRICAS GLOBAIS:
{'='*50}
Acurácia Geral:     {best_results['accuracy']:.3f}
Macro Precisão:     {best_results['macro_precision']:.3f}
Macro Recall:       {best_results['macro_recall']:.3f}
Macro F1:           {best_results['macro_f1']:.3f}
Weighted Precisão:  {best_results['weighted_precision']:.3f}
Weighted Recall:    {best_results['weighted_recall']:.3f}
Weighted F1:        {best_results['weighted_f1']:.3f}

INTERPRETAÇÃO CLÍNICA:
• CDR 0: Cognição normal
• CDR 0.5: Comprometimento cognitivo leve (MCI)
• CDR 1: Demência leve
• CDR 2: Demência moderada"""
        
        # Ajustar posição e tamanho do texto
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F4FD", alpha=0.9))
    
    def create_multiclass_dashboard(self):
        """Cria dashboard focado na classificação multiclasse CDR"""
        
        # Configurar figura
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('DASHBOARD MULTICLASSE - CLASSIFICAÇÃO CDR (4 CLASSES)\nSistema de Detecção de Alzheimer/MCI', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Grid layout para o dashboard multiclasse (espaçamento aumentado para evitar sobreposição)
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.5], 
                             hspace=0.6, wspace=0.3)
        
        # 1. Matriz de Confusão Multiclasse (primeira linha, span completo)
        self.plot_multiclass_confusion_matrix(fig, gs[0, :])
        
        # 2. Resumo das Métricas Multiclasse (segunda linha, span completo)
        self.plot_multiclass_metrics_summary(fig, gs[1, :])
        
        # 3. Comparação de Modelos Multiclasse (terceira linha, esquerda)
        self.plot_multiclass_model_comparison(fig, gs[2, :2])
        
        # 4. Distribuição das Classes CDR (terceira linha, direita)
        self.plot_cdr_distribution(fig, gs[2, 2])
        
        # 5. Espaço final (quarta linha)
        ax_final = fig.add_subplot(gs[3, :])
        ax_final.axis('off')
        
        plt.tight_layout()
        plt.savefig('alzheimer_multiclass_cdr_dashboard.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print("Dashboard multiclasse salvo: DASHBOARDS/alzheimer_multiclass_cdr_dashboard.png")
    
    def plot_multiclass_model_comparison(self, fig, gs_pos):
        """Comparação de performance dos modelos multiclasse"""
        ax = fig.add_subplot(gs_pos)
        
        models = list(self.multiclass_results.keys())
        metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
        
        # Preparar dados
        data = []
        for model in models:
            for metric in metrics:
                data.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': self.multiclass_results[model][metric]
                })
        
        df_metrics = pd.DataFrame(data)
        
        # Pivot para heatmap
        pivot_df = df_metrics.pivot(index='Model', columns='Metric', values='Score')
        
        # Heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   ax=ax, cbar_kws={'label': 'Score'})
        
        ax.set_title('Comparação de Modelos Multiclasse\n(4 Classes CDR)', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Rotacionar labels
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    def plot_cdr_distribution(self, fig, gs_pos):
        """Distribuição das classes CDR no dataset"""
        ax = fig.add_subplot(gs_pos)
        
        # Contar ocorrências de cada CDR
        cdr_counts = self.df['cdr'].value_counts().sort_index()
        cdr_labels = ['CDR 0\n(Normal)', 'CDR 0.5\n(MCI)', 'CDR 1\n(Leve)', 'CDR 2\n(Moderado)']
        
        # Cores para cada classe
        colors = ['#4ECDC4', '#FF6B6B', '#FFE66D', '#FF8E8E']
        
        # Gráfico de barras
        bars = ax.bar(range(len(cdr_counts)), cdr_counts.values, color=colors, alpha=0.8)
        
        # Configurar eixo x
        ax.set_xticks(range(len(cdr_counts)))
        ax.set_xticklabels(cdr_labels, fontsize=10, rotation=45, ha='right')
        
        # Configurar eixo y
        ax.set_ylabel('Número de Sujeitos', fontsize=12)
        ax.set_title('Distribuição das Classes CDR\nno Dataset', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        for i, (bar, count) in enumerate(zip(bars, cdr_counts.values)):
            percentage = (count / len(self.df)) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{count}\n({percentage:.1f}%)', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Adicionar estatísticas gerais
        total_subjects = len(self.df)
        ax.text(0.02, 0.98, f'Total: {total_subjects} sujeitos', 
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

def main():
    """Função principal para gerar o dashboard"""
    print("GERADOR DE DASHBOARD - ANÁLISE DE ALZHEIMER/MCI")
    print("=" * 60)
    
    # Verificar se existe dataset
    dataset_path = "alzheimer_complete_dataset.csv"
    
    # Criar gerador
    dashboard = AlzheimerDashboardGenerator(dataset_path)
    
    # Carregar ou criar dados
    dashboard.load_or_create_data()
    
    # Treinar modelos
    dashboard.train_multiclass_models()
    
    # Gerar dashboard completo
    print("\nGerando dashboard completo...")
    dashboard.create_multiclass_dashboard()
    
    print("\nDASHBOARD GERADO COM SUCESSO!")
    print("Arquivo: alzheimer_multiclass_cdr_dashboard.png")
    print("\nO dashboard inclui:")
    print("   • Matriz de confusão multiclasse (4 classes CDR)")
    print("   • Resumo das métricas multiclasse por classe")
    print("   • Comparação de modelos multiclasse")
    print("   • Distribuição das classes CDR")
    print("\nMODELOS UTILIZADOS:")
    print("   • Random Forest Multiclasse")
    print("   • Gradient Boosting Multiclasse")
    print("   • SVM Multiclasse")
    print("   • MLP Multiclasse")

if __name__ == "__main__":
    main()

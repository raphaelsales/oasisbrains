#!/usr/bin/env python3
"""
GERADOR DE DASHBOARD ALTERNATIVO - ANÁLISE DE ALZHEIMER/MCI SEM MMSE
============================================================

Este script é uma versão alternativa que NÃO utiliza o MMSE (Mini-Mental State Examination)
como métrica de entrada. Em vez disso, foca exclusivamente em:

1. Biomarcadores neuroanatômicos (volumes cerebrais)
2. Características demográficas (idade, gênero, educação)
3. Status socioeconômico (SES)
4. Classificação CDR (Clinical Dementia Rating)

VANTAGENS DA ABORDAGEM SEM MMSE:
- Evita viés de avaliação cognitiva subjetiva
- Foca em marcadores objetivos (estruturais)
- Útil para casos onde MMSE não está disponível
- Permite análise puramente baseada em neuroimagem

AUTOR: Sistema de IA para Análise de Alzheimer
VERSÃO: 2.0 (Sem MMSE)
DATA: Setembro 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

class AlzheimerDashboardGeneratorSemMMSE:
    """Gerador de dashboard para análise de Alzheimer sem usar MMSE"""
    
    def __init__(self):
        self.data = None
        self.label_encoder = None
        self.multiclass_results = {}
        self.models = {}
        self.scaler = StandardScaler()
        
        # Configurações de estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_or_create_data(self):
        """Carrega ou cria dataset sem MMSE"""
        print("Carregando dataset sem MMSE...")
        
        # Tentar carregar dataset existente
        if os.path.exists("alzheimer_complete_dataset_augmented.csv"):
            print("Dataset aumentado encontrado, carregando...")
            self.data = pd.read_csv("alzheimer_complete_dataset_augmented.csv")
            
            # Remover coluna MMSE se existir
            if 'mmse' in self.data.columns:
                print("Removendo coluna MMSE do dataset...")
                self.data = self.data.drop('mmse', axis=1)
                print("Coluna MMSE removida com sucesso")
            
            print(f"Dataset carregado: {len(self.data)} sujeitos, {len(self.data.columns)} features")
            
            # Verificar distribuição CDR
            cdr_dist = self.data['cdr'].value_counts().sort_index()
            print(f"Distribuição CDR: {dict(cdr_dist)}")
            
        else:
            print("Dataset não encontrado. Criando dados sintéticos sem MMSE...")
            self.create_synthetic_data_sem_mmse()
    
    def create_synthetic_data_sem_mmse(self):
        """Cria dados sintéticos sem MMSE, focando em biomarcadores estruturais"""
        print("Criando dataset sintético sem MMSE...")
        
        np.random.seed(42)
        n_subjects = 1000
        
        # Gerar dados baseados em distribuições realistas
        data = []
        
        for i in range(n_subjects):
            # Características demográficas
            age = np.random.normal(75, 10)
            age = max(60, min(90, age))
            
            gender = np.random.choice(['M', 'F'], p=[0.4, 0.6])
            education = np.random.choice([12, 14, 16, 18], p=[0.4, 0.3, 0.2, 0.1])
            ses = np.random.randint(1, 6)
            
            # CDR com distribuição realista (sem MMSE)
            cdr_weights = [0.6, 0.2, 0.15, 0.05]  # 0, 0.5, 1, 2
            cdr = np.random.choice([0, 0.5, 1, 2], p=cdr_weights)
            
            # Biomarcadores estruturais baseados no CDR (sem MMSE)
            if cdr == 0:  # Normal
                hippo_vol = np.random.normal(7000, 500)
                amygdala_vol = np.random.normal(3000, 300)
                entorhinal_vol = np.random.normal(800, 100)
                temporal_vol = np.random.normal(25000, 2000)
            elif cdr == 0.5:  # MCI
                hippo_vol = np.random.normal(6500, 600)
                amygdala_vol = np.random.normal(2800, 350)
                entorhinal_vol = np.random.normal(700, 120)
                temporal_vol = np.random.normal(23000, 2500)
            elif cdr == 1:  # Leve
                hippo_vol = np.random.normal(5800, 700)
                amygdala_vol = np.random.normal(2500, 400)
                entorhinal_vol = np.random.normal(600, 150)
                temporal_vol = np.random.normal(21000, 3000)
            else:  # Moderado
                hippo_vol = np.random.normal(4800, 800)
                amygdala_vol = np.random.normal(2200, 450)
                entorhinal_vol = np.random.normal(500, 180)
                temporal_vol = np.random.normal(19000, 3500)
            
            # Garantir valores positivos
            hippo_vol = max(3000, hippo_vol)
            amygdala_vol = max(1500, amygdala_vol)
            entorhinal_vol = max(200, entorhinal_vol)
            temporal_vol = max(15000, temporal_vol)
            
            # Features derivadas
            total_brain_vol = np.random.normal(1200000, 100000)
            hippo_ratio = hippo_vol / total_brain_vol
            asymmetry = np.random.normal(0.05, 0.03)
            
            # Intensidades T1 (sem MMSE)
            if cdr == 0:
                t1_intensity = np.random.normal(120, 10)
            elif cdr == 0.5:
                t1_intensity = np.random.normal(115, 12)
            elif cdr == 1:
                t1_intensity = np.random.normal(110, 15)
            else:
                t1_intensity = np.random.normal(105, 18)
            
            # Diagnóstico baseado no CDR
            diagnosis = 'Demented' if cdr > 0 else 'Nondemented'
            
            subject_data = {
                'subject_id': f'OAS1_{i:04d}_MR1',
                'age': round(age, 1),
                'gender': gender,
                'cdr': cdr,
                'education': education,
                'ses': ses,
                'diagnosis': diagnosis,
                
                # Biomarcadores estruturais principais
                'left_hippocampus_volume': hippo_vol * (1 - asymmetry),
                'right_hippocampus_volume': hippo_vol * (1 + asymmetry),
                'total_hippocampus_volume': hippo_vol,
                'hippocampus_brain_ratio': hippo_ratio,
                
                'left_amygdala_volume': amygdala_vol * (1 - asymmetry * 0.5),
                'right_amygdala_volume': amygdala_vol * (1 + asymmetry * 0.5),
                'total_amygdala_volume': amygdala_vol,
                
                'left_entorhinal_volume': entorhinal_vol * (1 - asymmetry * 0.3),
                'right_entorhinal_volume': entorhinal_vol * (1 + asymmetry * 0.3),
                'total_entorhinal_volume': entorhinal_vol,
                
                'left_temporal_volume': temporal_vol * (1 - asymmetry * 0.2),
                'right_temporal_volume': temporal_vol * (1 + asymmetry * 0.2),
                'total_temporal_volume': temporal_vol,
                
                # Features derivadas
                'hippocampus_asymmetry': asymmetry,
                'temporal_asymmetry': asymmetry * 0.8,
                'amygdala_asymmetry': asymmetry * 0.6,
                
                # Intensidades T1
                'hippocampus_intensity_mean': t1_intensity,
                'amygdala_intensity_mean': t1_intensity * 0.95,
                'entorhinal_intensity_mean': t1_intensity * 0.9,
                'temporal_intensity_mean': t1_intensity * 0.85,
                
                # Volumes normalizados
                'left_hippocampus_volume_norm': (hippo_vol * (1 - asymmetry)) / total_brain_vol,
                'right_hippocampus_volume_norm': (hippo_vol * (1 + asymmetry)) / total_brain_vol,
                'left_amygdala_volume_norm': (amygdala_vol * (1 - asymmetry * 0.5)) / total_brain_vol,
                'right_amygdala_volume_norm': (amygdala_vol * (1 + asymmetry * 0.5)) / total_brain_vol,
                
                # Ratios importantes
                'hippo_amygdala_ratio': hippo_vol / amygdala_vol,
                'hippo_entorhinal_ratio': hippo_vol / entorhinal_vol,
                'temporal_hippo_ratio': temporal_vol / hippo_vol
            }
            
            data.append(subject_data)
        
        self.data = pd.DataFrame(data)
        print(f"Dataset sintético criado: {len(self.data)} sujeitos, {len(self.data.columns)} features")
        print("FEATURES PRINCIPAIS (sem MMSE):")
        print("   • Biomarcadores estruturais (volumes)")
        print("   • Características demográficas")
        print("   • Features derivadas e ratios")
        print("   • Intensidades T1")
        
        # Salvar dataset
        self.data.to_csv("alzheimer_dataset_sem_mmse.csv", index=False)
        print("Dataset salvo: alzheimer_dataset_sem_mmse.csv")
    
    def train_multiclass_models(self):
        """Treina modelos multiclasse sem MMSE"""
        print("\nTreinando modelos multiclasse para classificação CDR (sem MMSE)...")
        
        # Preparar dados
        exclude_cols = ['subject_id', 'diagnosis', 'gender', 'cdr']  # Excluir CDR das features
        feature_cols = [col for col in self.data.columns 
                       if col not in exclude_cols and 
                       self.data[col].dtype in [np.float64, np.int64]]
        
        X = self.data[feature_cols].fillna(self.data[feature_cols].median())
        y = self.data['cdr']
        
        # Converter CDR para inteiros para classificação
        cdr_mapping = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
        y = y.map(cdr_mapping)
        
        print(f"Features utilizadas ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols):
            print(f"   {i+1:2d}. {col}")
        
        print(f"\nTarget: CDR (classes: {sorted(y.unique())})")
        print(f"Dataset: {X.shape[0]} amostras, {X.shape[1]} features")
        
        # Split estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Divisão treino/teste: {len(X_train)}/{len(X_test)} amostras")
        print(f"Distribuição treino: {dict(pd.Series(y_train).value_counts().sort_index())}")
        print(f"Distribuição teste: {dict(pd.Series(y_test).value_counts().sort_index())}")
        
        # Treinar modelos
        models_to_train = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        for name, model in models_to_train.items():
            print(f"\n  Treinando {name} Multiclasse...")
            
            # Treinar modelo
            model.fit(X_train, y_train)
            
            # Predições
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
            
            print(f"    {name} Multiclasse: Acc = {accuracy:.3f}, Macro F1 = {f1:.3f}")
            
            # Armazenar resultados
            self.multiclass_results[name] = {
                'model': model,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            }
            
            # Matriz de confusão
            cm = confusion_matrix(y_test, y_pred)
            self.multiclass_results[name]['confusion_matrix'] = cm
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            self.multiclass_results[name]['classification_report'] = report
        
        self.models = {name: result['model'] for name, result in self.multiclass_results.items()}
        print(f"\nModelos treinados: {len(self.models)}")
        
        return X_test, y_test
    
    def create_multiclass_dashboard(self):
        """Cria dashboard multiclasse sem MMSE"""
        print("\nGerando dashboard completo sem MMSE...")
        
        # Criar figura com múltiplos subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.4)
        
        # 1. Matriz de confusão multiclasse (Random Forest)
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_multiclass_confusion_matrix(ax1)
        
        # 2. Resumo das métricas multiclasse
        ax2 = fig.add_subplot(gs[0, 2])
        self.plot_multiclass_metrics_summary(ax2)
        
        # 3. Comparação de modelos multiclasse
        ax3 = fig.add_subplot(gs[1, :])
        self.plot_multiclass_model_comparison(ax3)
        
        # 4. Distribuição das classes CDR
        ax4 = fig.add_subplot(gs[2, :2])
        self.plot_cdr_distribution(ax4)
        
        # 5. Features mais importantes (Random Forest)
        ax5 = fig.add_subplot(gs[2, 2])
        self.plot_feature_importance(ax5)
        
        # Título principal
        fig.suptitle('DASHBOARD MULTICLASSE CDR - ANÁLISE SEM MMSE\n'
                    'Classificação de Alzheimer/MCI baseada em biomarcadores estruturais',
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig('alzheimer_multiclass_cdr_dashboard_sem_mmse.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print("Dashboard multiclasse salvo: alzheimer_multiclass_cdr_dashboard_sem_mmse.png")
    
    def plot_multiclass_confusion_matrix(self, ax):
        """Plota matriz de confusão multiclasse"""
        rf_results = self.multiclass_results.get('Random Forest')
        if not rf_results:
            return
        
        cm = rf_results['confusion_matrix']
        y_test = rf_results['y_test']
        y_pred = rf_results['y_pred']
        
        # Labels das classes CDR
        cdr_labels = ['CDR 0\n(Normal)', 'CDR 0.5\n(MCI)', 'CDR 1\n(Leve)', 'CDR 2\n(Moderado)']
        
        # Plotar matriz de confusão
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        
        # Adicionar valores nas células
        for i in range(len(cdr_labels)):
            for j in range(len(cdr_labels)):
                text = ax.text(j, i, str(cm[i, j]), ha="center", va="center", 
                             color="white" if cm[i, j] > cm.max() / 2 else "black",
                             fontweight='bold', fontsize=12)
        
        # Configurar eixos
        ax.set_xticks(range(len(cdr_labels)))
        ax.set_yticks(range(len(cdr_labels)))
        ax.set_xticklabels(cdr_labels, fontsize=11, fontweight='bold')
        ax.set_yticklabels(cdr_labels, fontsize=11, fontweight='bold')
        
        # Labels dos eixos
        ax.set_xlabel('Classe Predita (CDR)', fontsize=12)
        ax.set_ylabel('Classe Real (CDR)', fontsize=12)
        
        # Título
        accuracy = rf_results['accuracy']
        ax.set_title(f'Matriz de Confusão Multiclasse CDR\n'
                    f'Random Forest - Acurácia: {accuracy:.3f}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Adicionar colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Número de Amostras', rotation=270, labelpad=20)
        
        # Estatísticas por classe
        stats_text = []
        for i, label in enumerate(cdr_labels):
            tp = cm[i, i]
            total_real = cm[i, :].sum()
            total_pred = cm[:, i].sum()
            
            precision = tp / total_pred if total_pred > 0 else 0
            recall = tp / total_real if total_real > 0 else 0
            
            stats_text.append(f'{label.replace(chr(10), " ")}: P={precision:.2f}, R={recall:.2f}')
        
        # Adicionar estatísticas
        ax.text(0.5, -0.45, 'Estatísticas por Classe:', transform=ax.transAxes, 
               ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(0.5, -0.70, '\n'.join(stats_text), transform=ax.transAxes, 
               ha='center', va='center', fontsize=9)
    
    def plot_multiclass_metrics_summary(self, ax):
        """Plota resumo das métricas multiclasse"""
        rf_results = self.multiclass_results.get('Random Forest')
        if not rf_results:
            return
        
        report = rf_results['classification_report']
        
        # Preparar dados para visualização
        classes = ['CDR 0', 'CDR 0.5', 'CDR 1', 'CDR 2']
        metrics = ['Precisão', 'Recall', 'F1-Score']
        
        # Extrair valores
        data = []
        for cls in ['0.0', '0.5', '1.0', '2.0']:
            if cls in report:
                row = [report[cls]['precision'], report[cls]['recall'], report[cls]['f1-score']]
                data.append(row)
            else:
                data.append([0, 0, 0])
        
        data = np.array(data)
        
        # Plotar heatmap
        im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # Adicionar valores
        for i in range(len(classes)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center',
                             color='white' if data[i, j] < 0.5 else 'black',
                             fontweight='bold', fontsize=10)
        
        # Configurar eixos
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(metrics, fontsize=10, fontweight='bold', rotation=45, ha='right')
        ax.set_yticklabels(classes, fontsize=10, fontweight='bold')
        
        # Título
        ax.set_title('Métricas por Classe CDR\n(Random Forest)', 
                    fontsize=12, fontweight='bold', pad=15)
        
        # Adicionar colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Score', rotation=270, labelpad=15)
    
    def plot_multiclass_model_comparison(self, ax):
        """Plota comparação de modelos multiclasse"""
        models = list(self.multiclass_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Preparar dados
        data = []
        for model in models:
            row = [self.multiclass_results[model][metric] for metric in metrics]
            data.append(row)
        
        data = np.array(data)
        
        # Plotar heatmap
        im = ax.imshow(data, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        
        # Adicionar valores
        for i in range(len(models)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center',
                             color='white' if data[i, j] < 0.5 else 'black',
                             fontweight='bold', fontsize=11)
        
        # Configurar eixos
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(models)))
        ax.set_xticklabels(['Acurácia', 'Precisão', 'Recall', 'F1-Score'], 
                          fontsize=11, fontweight='bold')
        ax.set_yticklabels(models, fontsize=11, fontweight='bold')
        
        # Título
        ax.set_title('Comparação de Modelos Multiclasse CDR\n(Performance por Métrica)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Adicionar colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Score', rotation=270, labelpad=20)
    
    def plot_cdr_distribution(self, ax):
        """Plota distribuição das classes CDR"""
        cdr_counts = self.data['cdr'].value_counts().sort_index()
        cdr_labels = ['CDR 0\n(Normal)', 'CDR 0.5\n(MCI)', 'CDR 1\n(Leve)', 'CDR 2\n(Moderado)']
        
        # Cores para cada classe
        colors = ['#2E8B57', '#FFD700', '#FF8C00', '#DC143C']
        
        # Plotar barras
        bars = ax.bar(range(len(cdr_counts)), cdr_counts.values, color=colors, alpha=0.8)
        
        # Adicionar valores nas barras
        for bar, count in zip(bars, cdr_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                   str(count), ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Configurar eixos
        ax.set_xticks(range(len(cdr_counts)))
        ax.set_xticklabels(cdr_labels, fontsize=11, fontweight='bold')
        ax.set_ylabel('Número de Sujeitos', fontsize=12, fontweight='bold')
        
        # Título
        ax.set_title('Distribuição das Classes CDR\n(Dataset sem MMSE)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Adicionar estatísticas
        total = len(self.data)
        stats_text = f'Total: {total} sujeitos\n'
        stats_text += f'Classes: {len(cdr_counts)}\n'
        stats_text += f'Balanceamento: {"Sim" if len(cdr_counts.unique()) == 1 else "Não"}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
               fontsize=10)
    
    def plot_feature_importance(self, ax):
        """Plota importância das features (Random Forest)"""
        rf_results = self.multiclass_results.get('Random Forest')
        if not rf_results:
            return
        
        model = rf_results['model']
        
        # Obter importância das features
        if hasattr(model, 'feature_importances_'):
            # Random Forest ou Gradient Boosting
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # SVM ou MLP
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            return
        
        # Preparar dados
        exclude_cols = ['subject_id', 'diagnosis', 'gender']
        feature_cols = [col for col in self.data.columns 
                       if col not in exclude_cols and 
                       self.data[col].dtype in [np.float64, np.int64]]
        
        # Top 10 features
        top_indices = np.argsort(importances)[-10:]
        top_features = [feature_cols[i] for i in top_indices]
        top_importances = [importances[i] for i in top_indices]
        
        # Plotar barras horizontais
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_importances, color='skyblue', alpha=0.8)
        
        # Configurar eixos
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace('_', '\n') for f in top_features], fontsize=9)
        ax.set_xlabel('Importância', fontsize=11, fontweight='bold')
        
        # Título
        ax.set_title('Top 10 Features Mais\nImportantes (Random Forest)', 
                    fontsize=12, fontweight='bold', pad=15)
        
        # Adicionar valores nas barras
        for bar, importance in zip(bars, top_importances):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center', fontsize=9)
    
    def generate_summary_report(self):
        """Gera relatório resumo da análise sem MMSE"""
        print("\n" + "="*60)
        print("RELATÓRIO RESUMO - ANÁLISE SEM MMSE")
        print("="*60)
        
        # Melhor modelo
        best_model = max(self.multiclass_results.keys(), 
                        key=lambda x: self.multiclass_results[x]['accuracy'])
        best_results = self.multiclass_results[best_model]
        
        print(f"MODELO COM MELHOR PERFORMANCE: {best_model}")
        print(f"Acurácia: {best_results['accuracy']:.3f}")
        print(f"Precisão Macro: {best_results['precision']:.3f}")
        print(f"Recall Macro: {best_results['recall']:.3f}")
        print(f"F1-Score Macro: {best_results['f1']:.3f}")
        
        print(f"\nFEATURES UTILIZADAS: {len(self.data.columns) - 4}")  # -4 para excluir colunas não-feature
        print("Tipo: Biomarcadores estruturais + características demográficas")
        print("Excluído: MMSE (Mini-Mental State Examination)")
        
        print(f"\nDISTRIBUIÇÃO CDR:")
        cdr_dist = self.data['cdr'].value_counts().sort_index()
        for cdr, count in cdr_dist.items():
            percentage = (count / len(self.data)) * 100
            print(f"   CDR {cdr}: {count} sujeitos ({percentage:.1f}%)")
        
        print(f"\nVANTAGENS DA ABORDAGEM SEM MMSE:")
        print("   • Foco em marcadores objetivos (estruturais)")
        print("   • Evita viés de avaliação cognitiva subjetiva")
        print("   • Útil para casos onde MMSE não está disponível")
        print("   • Permite análise puramente baseada em neuroimagem")
        
        print(f"\nARQUIVOS GERADOS:")
        print("   • alzheimer_dataset_sem_mmse.csv")
        print("   • alzheimer_multiclass_cdr_dashboard_sem_mmse.png")
        
        print(f"\nANÁLISE CONCLUÍDA COM SUCESSO!")

def main():
    """Função principal"""
    print("GERADOR DE DASHBOARD ALTERNATIVO - ANÁLISE SEM MMSE")
    print("=" * 60)
    print("Este script NÃO utiliza MMSE como métrica de entrada")
    print("Foca em biomarcadores estruturais e características demográficas")
    print("=" * 60)
    
    # Criar gerador
    dashboard = AlzheimerDashboardGeneratorSemMMSE()
    
    # Carregar/criar dados
    dashboard.load_or_create_data()
    
    # Treinar modelos
    X_test, y_test = dashboard.train_multiclass_models()
    
    # Gerar dashboard
    dashboard.create_multiclass_dashboard()
    
    # Relatório resumo
    dashboard.generate_summary_report()
    
    print(f"\nDASHBOARD GERADO COM SUCESSO!")
    print(f"Arquivo: alzheimer_multiclass_cdr_dashboard_sem_mmse.png")
    
    print(f"\nO dashboard inclui:")
    print(f"   • Matriz de confusão multiclasse (4 classes CDR)")
    print(f"   • Resumo das métricas multiclasse por classe")
    print(f"   • Comparação de modelos multiclasse")
    print(f"   • Distribuição das classes CDR")
    print(f"   • Features mais importantes (sem MMSE)")
    
    print(f"\nMODELOS UTILIZADOS:")
    for name in dashboard.models.keys():
        print(f"   • {name} Multiclasse")
    
    print(f"\nABORDAGEM SEM MMSE:")
    print(f"   • Biomarcadores estruturais (volumes cerebrais)")
    print(f"   • Características demográficas (idade, gênero, educação)")
    print(f"   • Status socioeconômico (SES)")
    print(f"   • Features derivadas e ratios")
    print(f"   • Intensidades T1")

if __name__ == "__main__":
    main()

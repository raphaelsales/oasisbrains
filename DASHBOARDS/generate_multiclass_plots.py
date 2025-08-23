#!/usr/bin/env python3
"""
Script para gerar gráficos de avaliação do classificador multiclasse CDR
Utiliza dados existentes ou dados sintéticos para demonstração
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class MulticlassVisualizationGenerator:
    """Gera visualizações para classificação multiclasse de CDR"""
    
    def __init__(self):
        self.setup_style()
    
    def setup_style(self):
        """Configura estilo das visualizações"""
        try:
            plt.style.use('seaborn-v0_8')
        except Exception:
            try:
                plt.style.use('seaborn')
            except Exception:
                pass  # Usar estilo padrão
        
        # Configurar matplotlib para português
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
    
    def create_synthetic_cdr_predictions(self, n_samples=200):
        """Cria dados sintéticos realistas para demonstração"""
        np.random.seed(42)
        
        # Distribuição baseada em dados reais de Alzheimer
        # CDR: 0 (normal), 1 (muito leve), 2 (leve), 3 (moderado) - usando inteiros
        true_distribution = [0.5, 0.3, 0.15, 0.05]  # Distribuição típica
        y_true = np.random.choice([0, 1, 2, 3], size=n_samples, p=true_distribution)
        
        # Simular predições com diferentes níveis de acurácia por classe
        y_pred = []
        for true_label in y_true:
            if true_label == 0:  # CDR=0 (mais fácil de classificar)
                pred = np.random.choice([0, 1, 2, 3], p=[0.85, 0.10, 0.04, 0.01])
            elif true_label == 1:  # CDR=0.5 (confundido com 0 e 2)
                pred = np.random.choice([0, 1, 2, 3], p=[0.15, 0.70, 0.13, 0.02])
            elif true_label == 2:  # CDR=1 (confundido com 1 e 3)
                pred = np.random.choice([0, 1, 2, 3], p=[0.05, 0.20, 0.65, 0.10])
            else:  # CDR=2 (mais severo, melhor classificado)
                pred = np.random.choice([0, 1, 2, 3], p=[0.02, 0.08, 0.15, 0.75])
            y_pred.append(pred)
        
        return np.array(y_true), np.array(y_pred)
    
    def generate_classification_report_plot(self, y_true, y_pred, class_names=None, save_path="DASHBOARDS/classification_report_multiclasse.png"):
        """Gera gráfico do classification report para classificação multiclasse"""
        print("Gerando classification report em formato PNG...")
        
        # Definir nomes das classes se não fornecidos
        if class_names is None:
            unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
            cdr_value_mapping = {0: '0.0', 1: '0.5', 2: '1.0', 3: '2.0'}
            class_names = [f'CDR={cdr_value_mapping.get(cls, cls)}' for cls in unique_classes]
        
        # Gerar o classification report como dict
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Preparar dados para visualização
        classes = []
        data_matrix = []
        
        # Identificar classes presentes no report
        for cls_name in class_names:
            cls_key = str(cls_name.replace('CDR=', ''))
            if cls_key in report:
                classes.append(cls_name)
                row = [
                    report[cls_key]['precision'],
                    report[cls_key]['recall'], 
                    report[cls_key]['f1-score'],
                    report[cls_key]['support']
                ]
                data_matrix.append(row)
        
        # Adicionar médias
        if 'macro avg' in report:
            macro_row = [
                report['macro avg']['precision'],
                report['macro avg']['recall'],
                report['macro avg']['f1-score'],
                report['macro avg']['support']
            ]
            data_matrix.append(macro_row)
            classes.append('Macro Avg')
            
        if 'weighted avg' in report:
            weighted_row = [
                report['weighted avg']['precision'],
                report['weighted avg']['recall'],
                report['weighted avg']['f1-score'],
                report['weighted avg']['support']
            ]
            data_matrix.append(weighted_row)
            classes.append('Weighted Avg')
        
        data_matrix = np.array(data_matrix)
        metrics = ['Precisão', 'Revocação', 'F1-Score', 'Suporte']
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Criar tabela colorida (apenas para as primeiras 3 colunas - métricas)
        im = ax.imshow(data_matrix[:, :3], cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Configurar ticks
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        
        # Adicionar valores na tabela
        for i in range(len(classes)):
            for j in range(len(metrics)):
                if j < 3:  # precision, recall, f1-score
                    text = f'{data_matrix[i, j]:.3f}'
                    color = 'white' if data_matrix[i, j] < 0.5 else 'black'
                else:  # support
                    text = f'{int(data_matrix[i, j])}'
                    color = 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontweight='bold')
        
        # Adicionar colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Score', rotation=270, labelpad=20)
        
        # Configurar título e labels
        accuracy = report.get('accuracy', 0)
        ax.set_title(f'Classification Report - Classificador CDR Multiclasse\nAcurácia Global: {accuracy:.3f}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Rotar labels do eixo x
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Adicionar linha de separação antes das médias
        if len(classes) > 2:
            ax.axhline(y=len(classes)-2.5, color='black', linewidth=2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"Classification report salvo: {save_path}")
        return save_path
    
    def generate_classification_report_grouped_bars(self, y_true, y_pred, class_names=None, save_path="DASHBOARDS/classification_report_grouped_bars.png"):
        """Gera gráfico do classification report usando grouped bar chart com labels"""
        print("Gerando classification report com grouped bar chart...")
        
        # Definir nomes das classes se não fornecidos
        if class_names is None:
            unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
            cdr_value_mapping = {0: '0.0', 1: '0.5', 2: '1.0', 3: '2.0'}
            class_names = [f'CDR={cdr_value_mapping.get(cls, cls)}' for cls in unique_classes]
        
        # Gerar o classification report como dict
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Preparar dados para visualização
        classes = []
        precision_data = []
        recall_data = []
        f1_data = []
        support_data = []
        
        # Coletar dados das classes
        for cls_name in class_names:
            cls_key = str(cls_name.replace('CDR=', ''))
            if cls_key in report:
                classes.append(cls_name)
                precision_data.append(report[cls_key]['precision'])
                recall_data.append(report[cls_key]['recall'])
                f1_data.append(report[cls_key]['f1-score'])
                support_data.append(report[cls_key]['support'])
        
        # Adicionar médias
        if 'macro avg' in report:
            classes.append('Macro Avg')
            precision_data.append(report['macro avg']['precision'])
            recall_data.append(report['macro avg']['recall'])
            f1_data.append(report['macro avg']['f1-score'])
            support_data.append(report['macro avg']['support'])
            
        if 'weighted avg' in report:
            classes.append('Weighted Avg')
            precision_data.append(report['weighted avg']['precision'])
            recall_data.append(report['weighted avg']['recall'])
            f1_data.append(report['weighted avg']['f1-score'])
            support_data.append(report['weighted avg']['support'])
        
        # Configurar dados para o gráfico
        x = np.arange(len(classes))
        width = 0.18  # largura das barras (4 barras por grupo)
        
        # Criar figura com um único gráfico
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Normalizar suporte para escala 0-1
        max_support = max(support_data) if max(support_data) > 0 else 1
        support_normalized = [s / max_support for s in support_data]
        
        # Criar barras agrupadas (Precision, Recall, F1-Score, Support normalizado)
        bars1 = ax.bar(x - 1.5*width, precision_data, width, label='Precisão', color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x - 0.5*width, recall_data, width, label='Revocação', color='#ff7f0e', alpha=0.8)
        bars3 = ax.bar(x + 0.5*width, f1_data, width, label='F1-Score', color='#2ca02c', alpha=0.8)
        bars4 = ax.bar(x + 1.5*width, support_normalized, width, label=f'Suporte (/{max_support})', color='#d62728', alpha=0.8)
        
        # Adicionar labels nos valores das barras
        def add_value_labels(bars, values, is_support=False):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if is_support:
                    # Para suporte, mostrar valor original
                    original_value = int(value * max_support)
                    text = f'{original_value}'
                    fontsize = 8
                else:
                    # Para métricas, mostrar com 3 decimais
                    text = f'{value:.3f}'
                    fontsize = 9
                
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       text, ha='center', va='bottom', fontsize=fontsize, fontweight='bold')
        
        add_value_labels(bars1, precision_data)
        add_value_labels(bars2, recall_data)
        add_value_labels(bars3, f1_data)
        add_value_labels(bars4, support_normalized, is_support=True)
        
        # Configurar gráfico
        ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        accuracy = report.get('accuracy', 0)
        ax.set_title(f'Classification Report - Classificador CDR Multiclasse\nAcurácia Global: {accuracy:.3f}', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adicionar linha de separação antes das médias
        if len(classes) > 2:
            ax.axvline(x=len(classes)-2.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Adicionar eixo secundário para mostrar valores reais de suporte
        ax2 = ax.twinx()
        ax2.set_ylabel('Suporte (Amostras)', fontsize=12, fontweight='bold', color='#d62728')
        ax2.set_ylim(0, max_support * 1.2)
        ax2.tick_params(axis='y', labelcolor='#d62728')
        
        # Adicionar estatísticas como texto
        total_samples = sum([support_data[i] for i in range(len(classes)-2)]) if len(classes) > 2 else sum(support_data)
        stats_text = f'Total de Amostras: {int(total_samples)}\n'
        stats_text += f'Número de Classes: {len(classes)-2 if len(classes) > 2 else len(classes)}\n'
        stats_text += f'Max Suporte: {int(max_support)}'
        
        # Adicionar caixa de texto com estatísticas
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"Classification Report (Grouped Bars) salvo: {save_path}")
        return save_path
    
    def generate_classification_report_plot_custom_color(self, y_true, y_pred, class_names=None, save_path="DASHBOARDS/classification_report_custom.png", colormap='RdYlGn'):
        """Gera gráfico do classification report com colormap personalizado"""
        print(f"Gerando classification report com colormap: {colormap}...")
        
        # Definir nomes das classes se não fornecidos
        if class_names is None:
            unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
            cdr_value_mapping = {0: '0.0', 1: '0.5', 2: '1.0', 3: '2.0'}
            class_names = [f'CDR={cdr_value_mapping.get(cls, cls)}' for cls in unique_classes]
        
        # Gerar o classification report como dict
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Preparar dados para visualização
        classes = []
        data_matrix = []
        
        # Identificar classes presentes no report
        for cls_name in class_names:
            cls_key = str(cls_name.replace('CDR=', ''))
            if cls_key in report:
                classes.append(cls_name)
                row = [
                    report[cls_key]['precision'],
                    report[cls_key]['recall'], 
                    report[cls_key]['f1-score'],
                    report[cls_key]['support']
                ]
                data_matrix.append(row)
        
        # Adicionar médias
        if 'macro avg' in report:
            macro_row = [
                report['macro avg']['precision'],
                report['macro avg']['recall'],
                report['macro avg']['f1-score'],
                report['macro avg']['support']
            ]
            data_matrix.append(macro_row)
            classes.append('Macro Avg')
            
        if 'weighted avg' in report:
            weighted_row = [
                report['weighted avg']['precision'],
                report['weighted avg']['recall'],
                report['weighted avg']['f1-score'],
                report['weighted avg']['support']
            ]
            data_matrix.append(weighted_row)
            classes.append('Weighted Avg')
        
        data_matrix = np.array(data_matrix)
        metrics = ['Precisão', 'Revocação', 'F1-Score', 'Suporte']
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Criar tabela colorida com colormap personalizado
        im = ax.imshow(data_matrix[:, :3], cmap=colormap, aspect='auto', vmin=0, vmax=1)
        
        # Configurar ticks
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        
        # Adicionar valores na tabela
        for i in range(len(classes)):
            for j in range(len(metrics)):
                if j < 3:  # precision, recall, f1-score
                    text = f'{data_matrix[i, j]:.3f}'
                    # Ajustar cor do texto baseado no colormap
                    if colormap in ['Blues', 'Greens', 'Reds', 'viridis', 'plasma']:
                        color = 'white' if data_matrix[i, j] < 0.5 else 'black'
                    else:
                        color = 'white' if data_matrix[i, j] < 0.5 else 'black'
                else:  # support
                    text = f'{int(data_matrix[i, j])}'
                    color = 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontweight='bold')
        
        # Adicionar colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Score', rotation=270, labelpad=20)
        
        # Configurar título e labels
        accuracy = report.get('accuracy', 0)
        ax.set_title(f'Classification Report - Classificador CDR Multiclasse\nColormap: {colormap} | Acurácia Global: {accuracy:.3f}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Rotar labels do eixo x
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Adicionar linha de separação antes das médias
        if len(classes) > 2:
            ax.axhline(y=len(classes)-2.5, color='black', linewidth=2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"Classification report ({colormap}) salvo: {save_path}")
        return save_path

    def generate_confusion_matrix_plot(self, y_true, y_pred, class_names=None, save_path="DASHBOARDS/matriz_confusao_multiclasse.png"):
        """Gera matriz de confusão para classificação multiclasse com acurácia"""
        print("Gerando matriz de confusão multiclasse...")
        
        # Definir nomes das classes se não fornecidos
        if class_names is None:
            unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
            cdr_value_mapping = {0: '0.0', 1: '0.5', 2: '1.0', 3: '2.0'}
            class_names = [f'CDR={cdr_value_mapping.get(cls, cls)}' for cls in unique_classes]
        
        # Calcular matriz de confusão e acurácia
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Criar figura
        plt.figure(figsize=(10, 8))
        
        # Plotar matriz de confusão com seaborn
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Número de Amostras'})
        
        # Configurar título e labels
        plt.title(f'Matriz de Confusão - Classificador CDR Multiclasse\nAcurácia Global: {accuracy:.3f}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Predição', fontsize=12, fontweight='bold')
        plt.ylabel('Real', fontsize=12, fontweight='bold')
        
        # Rotar labels se necessário
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adicionar estatísticas por classe na lateral
        class_stats = []
        for i, class_name in enumerate(class_names):
            tp = cm[i, i]
            total_real = cm[i, :].sum()
            total_pred = cm[:, i].sum()
            
            precision = tp / total_pred if total_pred > 0 else 0
            recall = tp / total_real if total_real > 0 else 0
            
            class_stats.append(f'{class_name}: P={precision:.2f}, R={recall:.2f}')
        
        # Adicionar texto com estatísticas
        stats_text = '\n'.join(class_stats)
        plt.figtext(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"Matriz de confusão salva: {save_path}")
        return save_path

    def generate_evaluation_plots(self, y_true, y_pred, class_names=None):
        """Gera ambos os gráficos de avaliação multiclasse"""
        print("Gerando visualizações de avaliação multiclasse...")
        print(f"Amostras: {len(y_true)} | Classes únicas: {sorted(np.unique(y_true))}")
        
        # Gerar classification report
        report_path = self.generate_classification_report_plot(y_true, y_pred, class_names)
        
        # Gerar matriz de confusão
        confusion_path = self.generate_confusion_matrix_plot(y_true, y_pred, class_names)
        
        return report_path, confusion_path
    
    def generate_multiclass_roc_plot(self, y_true, y_pred_proba, class_names=None, save_path="DASHBOARDS/roc_multiclasse.png"):
        """Gera curvas ROC para classificação multiclasse usando One-vs-Rest"""
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        from itertools import cycle
        
        print("Gerando curvas ROC multiclasse...")
        
        # Definir nomes das classes se não fornecidos
        if class_names is None:
            unique_classes = sorted(np.unique(y_true))
            cdr_value_mapping = {0: '0.0', 1: '0.5', 2: '1.0', 3: '2.0'}
            class_names = [f'CDR={cdr_value_mapping.get(cls, cls)}' for cls in unique_classes]
        
        # Converter para formato binário (One-vs-Rest)
        unique_classes = sorted(np.unique(y_true))
        n_classes = len(unique_classes)
        
        # Binarizar as labels
        y_true_bin = label_binarize(y_true, classes=unique_classes)
        if n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        # Calcular ROC para cada classe
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            if i < y_pred_proba.shape[1] and i < y_true_bin.shape[1]:
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Calcular ROC micro-average
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Calcular ROC macro-average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Criar figura
        plt.figure(figsize=(12, 8))
        
        # Cores para cada classe
        colors = cycle(['darkorange', 'navy', 'turquoise', 'darksalmon'])
        
        # Plotar curva ROC para cada classe
        for i, color in zip(range(n_classes), colors):
            if i < len(class_names):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Plotar médias micro e macro
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                color='deeppink', linestyle=':', linewidth=3)
        
        plt.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
                color='navy', linestyle=':', linewidth=3)
        
        # Linha diagonal (classificador aleatório)
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance (AUC = 0.500)')
        
        # Configurar gráfico
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos', fontsize=12, fontweight='bold')
        plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12, fontweight='bold')
        plt.title('Curvas ROC - Classificador CDR Multiclasse\n(One-vs-Rest)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"Curvas ROC salvas: {save_path}")
        return save_path

    def generate_complete_evaluation_plots(self, y_true, y_pred, class_names=None, y_pred_proba=None):
        """Gera todos os gráficos incluindo ROC com probabilidades reais ou sintéticas"""
        print("Gerando visualizações completas de avaliação multiclasse...")
        
        # Gerar classification report usando grouped bar chart
        report_path = self.generate_classification_report_grouped_bars(y_true, y_pred, class_names)
        
        # Gerar matriz de confusão
        confusion_path = self.generate_confusion_matrix_plot(y_true, y_pred, class_names)
        
        # Se não temos probabilidades reais, simular
        if y_pred_proba is None:
            print("Gerando probabilidades sintéticas para ROC...")
            unique_classes = sorted(np.unique(y_true))
            n_classes = len(unique_classes)
            n_samples = len(y_true)
            
            # Criar probabilidades sintéticas baseadas nas predições
            y_pred_proba = np.zeros((n_samples, n_classes))
            for i, pred in enumerate(y_pred):
                # Mapear predição para índice de classe
                pred_idx = unique_classes.index(pred) if pred in unique_classes else 0
                
                # Probabilidade alta para a classe predita, baixa para outras
                y_pred_proba[i, :] = 0.1 / (n_classes - 1) if n_classes > 1 else 0.5
                y_pred_proba[i, pred_idx] = 0.8 + np.random.random() * 0.15
                
                # Normalizar
                y_pred_proba[i, :] = y_pred_proba[i, :] / y_pred_proba[i, :].sum()
        else:
            print("Usando probabilidades REAIS do modelo para ROC...")
        
        # Gerar curvas ROC
        roc_path = self.generate_multiclass_roc_plot(y_true, y_pred_proba, class_names)
        
        return report_path, confusion_path, roc_path

def load_existing_results():
    """Carrega resultados REAIS usando modelo CDR treinado"""
    try:
        import tensorflow as tf
        import joblib
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # Verificar se existe dataset e modelos
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        dataset_path = os.path.join(parent_dir, "alzheimer_complete_dataset_augmented.csv")
        model_path = os.path.join(parent_dir, "alzheimer_cdr_classifier_CORRETO.h5")
        scaler_path = os.path.join(parent_dir, "alzheimer_cdr_classifier_CORRETO_scaler.joblib")
        
        if all(os.path.exists(p) for p in [dataset_path, model_path, scaler_path]):
            print("Carregando modelo CDR e dataset aumentado para predições REAIS...")
            
            # Carregar dataset
            df = pd.read_csv(dataset_path)
            print(f"Dataset aumentado carregado: {df.shape}")
            
            # Carregar modelo CDR e scaler
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)
            
            # Preparar features (SEM cdr, COM features especializadas)
            exclude_cols = ['subject_id', 'diagnosis', 'gender', 'cdr']
            feature_cols = [col for col in df.columns 
                           if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
            
            X = df[feature_cols].fillna(df[feature_cols].median())
            y = df['cdr'].values
            
            # Mapear CDR para inteiros
            cdr_mapping = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3}
            y_int = np.array([cdr_mapping.get(val, 0) for val in y])
            
            # Split treino/teste (mesmo seed para reprodutibilidade)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_int, test_size=0.2, random_state=42, stratify=y_int
            )
            
            # Scaler e predições
            X_test_scaled = scaler.transform(X_test)
            y_pred_proba = model.predict(X_test_scaled, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            print(f"Predições reais geradas: {len(y_test)} amostras de teste")
            print(f"Distribuição real y_test: {np.bincount(y_test)}")
            print(f"Distribuição pred y_pred: {np.bincount(y_pred)}")
            
            return y_test, y_pred, y_pred_proba
            
    except Exception as e:
        print(f"Erro ao carregar modelo/dataset: {e}")
        import traceback
        traceback.print_exc()
    
    return None, None, None

def main():
    """Função principal para gerar visualizações"""
    print("GERADOR DE VISUALIZAÇÕES MULTICLASSE CDR")
    print("=" * 50)
    
    generator = MulticlassVisualizationGenerator()
    
    # Tentar carregar dados existentes
    result = load_existing_results()
    
    if len(result) == 3 and result[0] is not None:
        y_true, y_pred, y_pred_proba = result
        print("Usando dados REAIS do modelo CDR treinado...")
        has_real_proba = True
    else:
        print("Usando dados sintéticos para demonstração...")
        y_true, y_pred = generator.create_synthetic_cdr_predictions(n_samples=200)
        y_pred_proba = None
        has_real_proba = False
    
    # Definir nomes das classes com mapeamento correto
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cdr_value_mapping = {0: '0.0', 1: '0.5', 2: '1.0', 3: '2.0'}
    class_names = [f'CDR={cdr_value_mapping.get(cls, cls)}' for cls in unique_classes]
    
    print(f"Classes detectadas: {class_names}")
    print(f"Distribuição real: {np.bincount(y_true.astype(int))}")
    print(f"Distribuição predita: {np.bincount(y_pred.astype(int))}")
    
    # Gerar visualizações completas (incluindo ROC)
    if has_real_proba:
        report_path, confusion_path, roc_path = generator.generate_complete_evaluation_plots(y_true, y_pred, class_names, y_pred_proba)
    else:
        report_path, confusion_path, roc_path = generator.generate_complete_evaluation_plots(y_true, y_pred, class_names)
    
    print("\n" + "="*50)
    print("VISUALIZAÇÕES GERADAS COM SUCESSO!")
    print("="*50)
    print(f"📊 Classification Report: {report_path}")
    print(f"📊 Matriz de Confusão: {confusion_path}")
    print(f"📊 Curvas ROC: {roc_path}")
    
    # Imprimir estatísticas resumidas
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n📈 Acurácia Global: {accuracy:.3f}")
    
    print("\n📋 Classification Report (texto):")
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comparação entre Pipelines de Análise de Alzheimer

1. Pipeline Original (MLP): Features extraídas + Rede Neural Densa
2. Pipeline CNN 3D: Imagens completas + Convolutional Neural Network 3D

Mostra diferenças de metodologia, performance e requisitos computacionais
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Dict, Any

class PipelineComparator:
    """Classe para comparar os diferentes pipelines de análise"""
    
    def __init__(self):
        self.comparison_results = {}
        
    def compare_methodologies(self):
        """Compara as metodologias dos dois pipelines"""
        
        print("COMPARAÇÃO DE METODOLOGIAS")
        print("=" * 50)
        
        comparison_table = {
            'Aspecto': [
                'Tipo de Entrada',
                'Arquitetura de Rede',
                'Processamento',
                'Features',
                'Objetivo Principal',
                'Tempo de Treinamento',
                'Uso de Memória',
                'Interpretabilidade',
                'Overfitting Risk'
            ],
            'Pipeline Original (MLP)': [
                'Features extraídas (volumes, intensidades)',
                'Multi-Layer Perceptron (6 camadas densas)',
                'Pré-processamento + Extração manual',
                'Features baseadas em conhecimento médico',
                'Classificação geral Alzheimer',
                'Rápido (minutos)',
                'Baixo (~MB)',
                'Alta (features interpretáveis)',
                'Baixo'
            ],
            'Pipeline CNN 3D': [
                'Imagens MRI 3D completas',
                'Convolutional Neural Network 3D',
                'Normalização + Augmentation 3D',
                'Features aprendidas automaticamente',
                'Detecção específica de MCI',
                'Lento (horas)',
                'Alto (~GB)',
                'Baixa (caixa preta)',
                'Alto'
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_table)
        
        print(df_comparison.to_string(index=False))
        
        return df_comparison
    
    def analyze_computational_requirements(self):
        """Analisa requisitos computacionais"""
        
        print("\n\nREQUISITOS COMPUTACIONAIS")
        print("=" * 50)
        
        requirements = {
            'Pipeline': ['MLP Original', 'CNN 3D'],
            'GPU Necessária': ['Opcional', 'Recomendada (≥8GB VRAM)'],
            'RAM Necessária': ['4-8 GB', '16-32 GB'],
            'Tempo Estimado': ['10-30 min', '2-8 horas'],
            'Batch Size': ['64-128', '2-8'],
            'Épocas Típicas': ['30-50', '50-100'],
            'Paralelização': ['Sim', 'Limitada por memória']
        }
        
        df_requirements = pd.DataFrame(requirements)
        print(df_requirements.to_string(index=False))
        
        print("\nOBSERVAÇÕES:")
        print("• MLP: Adequado para prototipagem rápida e recursos limitados")
        print("• CNN 3D: Requer infraestrutura robusta, mas maior potencial")
        
        return df_requirements
    
    def compare_clinical_applications(self):
        """Compara aplicações clínicas dos pipelines"""
        
        print("\n\nAPLICAÇÕES CLÍNICAS")
        print("=" * 50)
        
        clinical_comparison = {
            'Aplicação': [
                'Triagem Inicial',
                'Diagnóstico Diferencial',
                'Detecção Precoce MCI',
                'Monitoramento Progressão',
                'Pesquisa Científica',
                'Implementação Clínica'
            ],
            'MLP Original': [
                'Excelente (rápido)',
                'Bom (features conhecidas)',
                'Moderado',
                'Bom',
                'Limitado',
                'Viável'
            ],
            'CNN 3D': [
                'Limitado (lento)',
                'Excelente (padrões sutis)',
                'Excelente',
                'Excelente',
                'Muito bom',
                'Desafiador'
            ]
        }
        
        df_clinical = pd.DataFrame(clinical_comparison)
        print(df_clinical.to_string(index=False))
        
        print("\nRECOMENDAÇÕES DE USO:")
        print("• MLP: Clinicas com recursos limitados, triagem em massa")
        print("• CNN 3D: Centros de pesquisa, casos complexos, detecção precoce")
        
        return df_clinical
    
    def simulate_performance_comparison(self):
        """Simula comparação de performance"""
        
        print("\n\nSIMULAÇÃO DE PERFORMANCE")
        print("=" * 50)
        
        # Dados simulados baseados em literatura
        np.random.seed(42)
        
        # Simular resultados MLP
        mlp_accuracy = np.random.normal(0.82, 0.05, 100)
        mlp_auc = np.random.normal(0.86, 0.04, 100)
        mlp_training_time = np.random.normal(15, 5, 100)  # minutos
        
        # Simular resultados CNN 3D
        cnn_accuracy = np.random.normal(0.89, 0.04, 100)
        cnn_auc = np.random.normal(0.92, 0.03, 100)
        cnn_training_time = np.random.normal(180, 60, 100)  # minutos
        
        performance_stats = {
            'Métrica': ['Acurácia', 'AUC', 'Tempo Treinamento (min)'],
            'MLP - Média': [f"{mlp_accuracy.mean():.3f}", f"{mlp_auc.mean():.3f}", f"{mlp_training_time.mean():.1f}"],
            'MLP - Std': [f"±{mlp_accuracy.std():.3f}", f"±{mlp_auc.std():.3f}", f"±{mlp_training_time.std():.1f}"],
            'CNN 3D - Média': [f"{cnn_accuracy.mean():.3f}", f"{cnn_auc.mean():.3f}", f"{cnn_training_time.mean():.1f}"],
            'CNN 3D - Std': [f"±{cnn_accuracy.std():.3f}", f"±{cnn_auc.std():.3f}", f"±{cnn_training_time.std():.1f}"],
            'Melhoria CNN': [f"+{(cnn_accuracy.mean()-mlp_accuracy.mean())*100:.1f}%", 
                            f"+{(cnn_auc.mean()-mlp_auc.mean())*100:.1f}%",
                            f"+{(cnn_training_time.mean()/mlp_training_time.mean()-1)*100:.0f}%"]
        }
        
        df_performance = pd.DataFrame(performance_stats)
        print(df_performance.to_string(index=False))
        
        # Criar visualização
        self._create_performance_plots(mlp_accuracy, mlp_auc, mlp_training_time,
                                     cnn_accuracy, cnn_auc, cnn_training_time)
        
        return df_performance
    
    def _create_performance_plots(self, mlp_acc, mlp_auc, mlp_time,
                                 cnn_acc, cnn_auc, cnn_time):
        """Cria gráficos de comparação de performance"""
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Comparação de Acurácia
        axes[0,0].hist(mlp_acc, alpha=0.7, label='MLP', bins=20, color='blue')
        axes[0,0].hist(cnn_acc, alpha=0.7, label='CNN 3D', bins=20, color='red')
        axes[0,0].set_xlabel('Acurácia')
        axes[0,0].set_ylabel('Frequência')
        axes[0,0].set_title('Distribuição de Acurácia')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Comparação de AUC
        axes[0,1].hist(mlp_auc, alpha=0.7, label='MLP', bins=20, color='blue')
        axes[0,1].hist(cnn_auc, alpha=0.7, label='CNN 3D', bins=20, color='red')
        axes[0,1].set_xlabel('AUC')
        axes[0,1].set_ylabel('Frequência')
        axes[0,1].set_title('Distribuição de AUC')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Box Plot Comparativo
        data_to_plot = [mlp_acc, cnn_acc, mlp_auc, cnn_auc]
        labels = ['MLP\nAcc', 'CNN\nAcc', 'MLP\nAUC', 'CNN\nAUC']
        colors = ['lightblue', 'lightcoral', 'lightblue', 'lightcoral']
        
        box_plot = axes[1,0].boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[1,0].set_title('Comparação de Métricas')
        axes[1,0].set_ylabel('Score')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Tempo de Treinamento (log scale)
        axes[1,1].hist(mlp_time, alpha=0.7, label='MLP', bins=20, color='blue')
        axes[1,1].hist(cnn_time, alpha=0.7, label='CNN 3D', bins=20, color='red')
        axes[1,1].set_xlabel('Tempo de Treinamento (min)')
        axes[1,1].set_ylabel('Frequência')
        axes[1,1].set_title('Tempo de Treinamento')
        axes[1,1].set_yscale('log')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pipeline_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Gráfico salvo: pipeline_performance_comparison.png")
    
    def recommend_pipeline_selection(self):
        """Recomenda qual pipeline usar baseado no cenário"""
        
        print("\n\nGUIA DE SELEÇÃO DE PIPELINE")
        print("=" * 50)
        
        scenarios = {
            'Cenário': [
                'Triagem Rápida em Clínica',
                'Pesquisa Acadêmica',
                'Diagnóstico Precoce MCI',
                'Recursos Computacionais Limitados',
                'Máxima Acurácia Necessária',
                'Interpretabilidade Importante',
                'Dataset Pequeno (<100 amostras)',
                'Dataset Grande (>1000 amostras)'
            ],
            'Pipeline Recomendado': [
                'MLP Original',
                'CNN 3D',
                'CNN 3D',
                'MLP Original',
                'CNN 3D',
                'MLP Original',
                'MLP Original',
                'CNN 3D'
            ],
            'Justificativa': [
                'Velocidade e praticidade',
                'Estado da arte, publicabilidade',
                'Detecção de padrões sutis',
                'Menor overhead computacional',
                'Melhor capacidade de generalização',
                'Features baseadas em conhecimento médico',
                'Menos propenso a overfitting',
                'Aproveita poder de CNN com dados suficientes'
            ]
        }
        
        df_scenarios = pd.DataFrame(scenarios)
        print(df_scenarios.to_string(index=False))
        
        print("\nDICA: Considere implementar ambos em paralelo para comparação!")
        
        return df_scenarios
    
    def generate_comprehensive_report(self):
        """Gera relatório completo de comparação"""
        
        print("RELATÓRIO COMPLETO DE COMPARAÇÃO DE PIPELINES")
        print("=" * 70)
        
        # Executar todas as análises
        methodology_df = self.compare_methodologies()
        requirements_df = self.analyze_computational_requirements()
        clinical_df = self.compare_clinical_applications()
        performance_df = self.simulate_performance_comparison()
        recommendations_df = self.recommend_pipeline_selection()
        
        # Salvar relatório
        with open('pipeline_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE COMPARAÇÃO - PIPELINES ALZHEIMER\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. COMPARAÇÃO DE METODOLOGIAS\n")
            f.write("-" * 30 + "\n")
            f.write(methodology_df.to_string(index=False) + "\n\n")
            
            f.write("2. REQUISITOS COMPUTACIONAIS\n")
            f.write("-" * 30 + "\n")
            f.write(requirements_df.to_string(index=False) + "\n\n")
            
            f.write("3. APLICAÇÕES CLÍNICAS\n")
            f.write("-" * 30 + "\n")
            f.write(clinical_df.to_string(index=False) + "\n\n")
            
            f.write("4. PERFORMANCE SIMULADA\n")
            f.write("-" * 30 + "\n")
            f.write(performance_df.to_string(index=False) + "\n\n")
            
            f.write("5. RECOMENDAÇÕES DE USO\n")
            f.write("-" * 30 + "\n")
            f.write(recommendations_df.to_string(index=False) + "\n\n")
        
        print("\nRelatório completo salvo: pipeline_comparison_report.txt")
        print("Visualizações salvas: pipeline_performance_comparison.png")
        
        return {
            'methodology': methodology_df,
            'requirements': requirements_df,
            'clinical': clinical_df,
            'performance': performance_df,
            'recommendations': recommendations_df
        }

def main():
    """Executa comparação completa entre pipelines"""
    
    print("INICIANDO COMPARAÇÃO ENTRE PIPELINES...")
    print()
    
    comparator = PipelineComparator()
    results = comparator.generate_comprehensive_report()
    
    print("\nCOMPARAÇÃO CONCLUÍDA!")
    print("Arquivos gerados:")
    print("   - pipeline_comparison_report.txt")
    print("   - pipeline_performance_comparison.png")
    
    return results

if __name__ == "__main__":
    main() 
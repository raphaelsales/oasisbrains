#!/usr/bin/env python3
"""
Script para gerar diagramas da arquitetura do pipeline de Alzheimer
Cria imagens PNG dos diagramas de arquitetura usando matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_model_architecture_diagram():
    """Cria diagrama da arquitetura dos modelos"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    
    # Modelo Binário (esquerda)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.set_title('Modelo Binário\n(Demente vs Normal)', fontsize=16, fontweight='bold', pad=20)
    
    # Camadas do modelo binário
    binary_layers = [
        ('Features Entrada\n(~50-80)', 11, '#e1f5fe'),
        ('Dense 256 + ReLU\nDropout 0.4 + BatchNorm', 9.5, '#e8f5e8'),
        ('Dense 128 + ReLU\nDropout 0.3 + BatchNorm', 8, '#e8f5e8'),
        ('Dense 64 + ReLU\nDropout 0.3 + BatchNorm', 6.5, '#e8f5e8'),
        ('Dense 32 + ReLU\nDropout 0.2', 5, '#e8f5e8'),
        ('Dense 16 + ReLU\nDropout 0.1', 3.5, '#e8f5e8'),
        ('Saída Binária\nDense 1 + Sigmoid', 2, '#fce4ec')
    ]
    
    for i, (text, y, color) in enumerate(binary_layers):
        rect = FancyBboxPatch((1, y-0.4), 8, 0.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color, 
                             edgecolor='black',
                             linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(5, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Setas conectoras
        if i < len(binary_layers) - 1:
            ax1.arrow(5, y-0.5, 0, -0.6, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Modelo Multiclasse CDR (direita)
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 12)
    ax2.set_title('Modelo Multiclasse CDR\n(Arquitetura Especializada)', fontsize=16, fontweight='bold', pad=20)
    
    # Entrada
    rect = FancyBboxPatch((4, 10.5), 4, 0.8, boxstyle="round,pad=0.1", 
                         facecolor='#e1f5fe', edgecolor='black', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(6, 10.9, 'Features Entrada\n(~80 após engineering)', ha='center', va='center', 
             fontsize=10, fontweight='bold')
    
    # Camada de atenção
    rect = FancyBboxPatch((4, 9), 4, 0.8, boxstyle="round,pad=0.1", 
                         facecolor='#fff3e0', edgecolor='black', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(6, 9.4, 'Dense 256 + ReLU\nDropout 0.4 + BatchNorm\n(Camada Atenção)', 
             ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Branches
    # Branch principal
    rect1 = FancyBboxPatch((1, 7), 3.5, 0.8, boxstyle="round,pad=0.1", 
                          facecolor='#e8f5e8', edgecolor='black', linewidth=1.5)
    ax2.add_patch(rect1)
    ax2.text(2.75, 7.4, 'Branch Principal\nDense 128 + ReLU\nDropout 0.3 + BatchNorm', 
             ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Branch intermediário
    rect2 = FancyBboxPatch((7.5, 7), 3.5, 0.8, boxstyle="round,pad=0.1", 
                          facecolor='#fff3e0', edgecolor='black', linewidth=1.5)
    ax2.add_patch(rect2)
    ax2.text(9.25, 7.4, 'Branch Intermediário\nDense 64 + ReLU\nDropout 0.3 + BatchNorm\n(CDR 0.5-1.0)', 
             ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Concatenação
    rect = FancyBboxPatch((4, 5.5), 4, 0.8, boxstyle="round,pad=0.1", 
                         facecolor='#f3e5f5', edgecolor='black', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(6, 5.9, 'Concatenate\n(128 + 64 = 192)', ha='center', va='center', 
             fontsize=10, fontweight='bold')
    
    # Camadas finais
    final_layers = [
        ('Dense 64 + ReLU\nDropout 0.3 + BatchNorm', 4, '#e8f5e8'),
        ('Dense 32 + ReLU\nDropout 0.2', 2.5, '#e8f5e8'),
        ('Dense 16 + ReLU\nDropout 0.1', 1, '#e8f5e8')
    ]
    
    for text, y, color in final_layers:
        rect = FancyBboxPatch((4, y-0.4), 4, 0.8, boxstyle="round,pad=0.1", 
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(6, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Saída final
    rect = FancyBboxPatch((3.5, -0.9), 5, 0.8, boxstyle="round,pad=0.1", 
                         facecolor='#fce4ec', edgecolor='black', linewidth=2)
    ax2.add_patch(rect)
    ax2.text(6, -0.5, 'Saída CDR\nDense 4 + Softmax\n(CDR: 0, 0.5, 1.0, 2.0)', 
             ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Setas
    # Entrada para atenção
    ax2.arrow(6, 10.4, 0, -0.6, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Atenção para branches
    ax2.arrow(5, 8.6, -1.8, -1, head_width=0.15, head_length=0.1, fc='black', ec='black')
    ax2.arrow(7, 8.6, 1.8, -1, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Branches para concatenação
    ax2.arrow(2.75, 6.6, 2.5, -0.7, head_width=0.15, head_length=0.1, fc='black', ec='black')
    ax2.arrow(9.25, 6.6, -2.5, -0.7, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Camadas finais
    for i in range(len(final_layers)):
        if i == 0:
            ax2.arrow(6, 5.1, 0, -0.7, head_width=0.15, head_length=0.1, fc='black', ec='black')
        else:
            y_start = final_layers[i-1][1] - 0.4
            y_end = final_layers[i][1] + 0.4
            ax2.arrow(6, y_start, 0, y_end - y_start - 0.2, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Final para saída
    ax2.arrow(6, 0.6, 0, -1.1, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('/app/alzheimer/arquitetura_modelos_alzheimer.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    print("Diagrama salvo: arquitetura_modelos_alzheimer.png")

def create_pipeline_flow_diagram():
    """Cria diagrama do fluxo completo do pipeline"""
    fig, ax = plt.subplots(figsize=(16, 20))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 22)
    ax.set_title('Pipeline Completo de IA para Análise de Alzheimer', 
                 fontsize=18, fontweight='bold', pad=30)
    
    # Definir estágios do pipeline
    stages = [
        # (texto, x, y, largura, altura, cor)
        ('Dados OASIS\nImagens RM + Metadados', 6, 20.5, 4, 1, '#e1f5fe'),
        ('Segmentação FreeSurfer\naparc+aseg.mgz', 2, 18.5, 3, 1, '#e8f5e8'),
        ('Imagem T1\nT1.mgz', 6, 18.5, 3, 1, '#e8f5e8'),
        ('Metadados Clínicos\nCDR, MMSE, Idade', 10, 18.5, 3, 1, '#e8f5e8'),
        
        ('Extração Features\nVolumétrica', 2, 16.5, 3, 1, '#fff3e0'),
        ('Extração Features\nIntensidade', 6, 16.5, 3, 1, '#fff3e0'),
        ('Features\nDemográficas', 10, 16.5, 3, 1, '#fff3e0'),
        
        ('8 Regiões Cerebrais:\n• Hipocampo L/R\n• Amígdala L/R\n• Entorrinal L/R\n• Temporal L/R', 
         2, 14.5, 3, 1.5, '#f3e5f5'),
        ('Estatísticas T1:\n• Intensidade média\n• Desvio padrão\n• Por região', 
         6, 14.5, 3, 1.5, '#f3e5f5'),
        ('Dados Clínicos:\n• Idade, Educação\n• MMSE, CDR\n• Status socioeconômico', 
         10, 14.5, 3, 1.5, '#f3e5f5'),
        
        ('Feature Engineering\nCombinação e Normalização', 6, 12, 4, 1, '#e8f5e8'),
        
        ('Features Especializadas CDR=1:\n• Razão hipocampo/amígdala\n• Assimetria temporal\n• Score cognitivo-anatômico\n• Índice deterioração\n• Score intensidade global', 
         6, 10, 6, 1.5, '#fce4ec'),
        
        ('Data Augmentation Direcionado', 6, 8, 4, 1, '#fff3e0'),
        
        ('Transformações Geométricas:\n• Rotação ±15°\n• Zoom 0.8-1.2x\n• Translação ±10%\n• Flip horizontal', 
         3, 6.5, 3.5, 1.5, '#f9fbe7'),
        ('Transformações Fotométricas:\n• Brilho ±20%\n• Contraste ±20%\n• Preservar realismo', 
         8.5, 6.5, 3.5, 1.5, '#f9fbe7'),
        
        ('Normalização StandardScaler\nDivisão Treino/Teste (80/20)', 6, 4.5, 5, 1, '#e8f5e8'),
        
        ('Treinamento Modelos\nGPU + Mixed Precision', 6, 3, 4, 1, '#fff3e0'),
        
        ('Modelo Binário\nDemente vs Normal', 3, 1.5, 3, 1, '#e1f5fe'),
        ('Modelo Multiclasse\nCDR 0/0.5/1.0/2.0', 9, 1.5, 3, 1, '#e1f5fe'),
        
        ('Avaliação e Métricas\nVisualizações', 6, 0, 4, 1, '#fce4ec')
    ]
    
    # Desenhar caixas
    for text, x, y, w, h, color in stages:
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color, 
                             edgecolor='black',
                             linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=9, fontweight='bold', wrap=True)
    
    # Adicionar setas conectoras principais
    connections = [
        (6, 20, 6, 19),  # Dados para segmentação central
        (6, 18, 2, 17),  # Para segmentação
        (6, 18, 6, 17),  # Para T1
        (6, 18, 10, 17), # Para metadados
        (2, 16, 2, 15.25),  # Extração para regiões
        (6, 16, 6, 15.25),  # Extração para stats
        (10, 16, 10, 15.25), # Extração para clínicos
        (2, 13.25, 6, 12.5),  # Regiões para engineering
        (6, 13.25, 6, 12.5),  # Stats para engineering
        (10, 13.25, 6, 12.5), # Clínicos para engineering
        (6, 11.5, 6, 10.75),  # Engineering para especializadas
        (6, 9.25, 6, 8.5),    # Especializadas para augmentation
        (6, 7.5, 3, 7.25),    # Augmentation para geométricas
        (6, 7.5, 8.5, 7.25),  # Augmentation para fotométricas
        (3, 5.75, 6, 5),      # Geométricas para normalização
        (8.5, 5.75, 6, 5),    # Fotométricas para normalização
        (6, 4, 6, 3.5),       # Normalização para treinamento
        (6, 2.5, 3, 2),       # Treinamento para binário
        (6, 2.5, 9, 2),       # Treinamento para multiclasse
        (3, 1, 6, 0.5),       # Binário para avaliação
        (9, 1, 6, 0.5)        # Multiclasse para avaliação
    ]
    
    for x1, y1, x2, y2 in connections:
        ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.1, head_length=0.1, 
                fc='black', ec='black', alpha=0.7)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/app/alzheimer/pipeline_completo_alzheimer.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("Diagrama salvo: pipeline_completo_alzheimer.png")

def create_feature_engineering_diagram():
    """Cria diagrama específico do feature engineering"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_title('Feature Engineering para Detecção de Alzheimer', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Features originais
    original_features = [
        ('Volumes Cerebrais\n• Hipocampo L/R\n• Amígdala L/R\n• Entorrinal L/R\n• Temporal L/R', 
         2, 8, 3, 1.5, '#e1f5fe'),
        ('Intensidades T1\n• Média por região\n• Desvio padrão\n• Contraste', 
         7, 8, 3, 1.5, '#e8f5e8'),
        ('Dados Clínicos\n• Idade, Educação\n• MMSE, CDR\n• Status socioecon.', 
         12, 8, 3, 1.5, '#fff3e0')
    ]
    
    # Features engenhadas
    engineered_features = [
        ('Razão Hipocampo/Amígdala\n(Indicador atrofia relativa)', 2, 5, 3.5, 1, '#fce4ec'),
        ('Assimetria Temporal\n(Diferenças L/R)', 6.5, 5, 3, 1, '#fce4ec'),
        ('Score Cognitivo-Anatômico\n(MMSE × volume)', 10.5, 5, 3.5, 1, '#fce4ec'),
        ('Índice Deterioração\n(Média regiões afetadas)', 2, 2.5, 3.5, 1, '#f3e5f5'),
        ('Score Intensidade Global\n(Padrão geral sinal)', 6.5, 2.5, 3, 1, '#f3e5f5'),
        ('Features Normalizadas\n(Volume/Cérebro total)', 10.5, 2.5, 3.5, 1, '#f3e5f5')
    ]
    
    # Desenhar features
    all_features = original_features + engineered_features
    for text, x, y, w, h, color in all_features:
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                             boxstyle="round,pad=0.1",
                             facecolor=color,
                             edgecolor='black',
                             linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=9, fontweight='bold')
    
    # Setas de derivação
    arrows = [
        (2, 7.25, 2, 5.5),     # Volumes para razão
        (2, 7.25, 6.5, 5.5),  # Volumes para assimetria
        (7, 7.25, 10.5, 5.5), # Intensidades para score cognitivo
        (2, 7.25, 2, 3),      # Volumes para deterioração
        (7, 7.25, 6.5, 3),    # Intensidades para score global
        (2, 7.25, 10.5, 3)    # Volumes para normalização
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.1, head_length=0.1,
                fc='blue', ec='blue', alpha=0.6)
    
    # Legenda
    ax.text(7, 0.5, 'Features Especializadas para Detecção Precoce de Alzheimer (CDR=1)',
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/app/alzheimer/feature_engineering_alzheimer.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("Diagrama salvo: feature_engineering_alzheimer.png")

if __name__ == "__main__":
    print("Gerando diagramas da arquitetura do pipeline de Alzheimer...")
    print("=" * 60)
    
    # Gerar todos os diagramas
    create_model_architecture_diagram()
    create_pipeline_flow_diagram()
    create_feature_engineering_diagram()
    
    print("\n" + "=" * 60)
    print("DIAGRAMAS GERADOS:")
    print("1. arquitetura_modelos_alzheimer.png - Arquitetura dos modelos neurais")
    print("2. pipeline_completo_alzheimer.png - Fluxo completo do pipeline")
    print("3. feature_engineering_alzheimer.png - Engenharia de características")
    print("\nTodos os arquivos foram salvos no diretório /app/alzheimer/")


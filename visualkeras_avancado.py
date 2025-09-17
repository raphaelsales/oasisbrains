#!/usr/bin/env python3
"""
Visualização Avançada com Visualkeras - Baseado nas melhores práticas
Inspirado no exemplo: https://www.kaggle.com/code/devsubhash/visualize-deep-learning-models-using-visualkeras
Criado para os modelos de Alzheimer
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import visualkeras
from visualkeras import SpacingDummyLayer
import warnings
warnings.filterwarnings('ignore')

# Configurar TensorFlow para CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_binary_model_detailed():
    """Cria modelo binário com nomes de camadas mais descritivos"""
    model = keras.Sequential([
        # Entrada e primeira camada densa
        layers.Dense(256, activation='relu', input_shape=(50,), name='Input_Dense_256'),
        layers.Dropout(0.4, name='Regularization_Dropout_40pct'),
        layers.BatchNormalization(name='Normalization_BatchNorm_1'),
        
        # Camadas intermediárias
        layers.Dense(128, activation='relu', name='Hidden_Dense_128'),
        layers.Dropout(0.3, name='Regularization_Dropout_30pct_A'),
        layers.BatchNormalization(name='Normalization_BatchNorm_2'),
        
        layers.Dense(64, activation='relu', name='Hidden_Dense_64'),
        layers.Dropout(0.3, name='Regularization_Dropout_30pct_B'),
        layers.BatchNormalization(name='Normalization_BatchNorm_3'),
        
        layers.Dense(32, activation='relu', name='Hidden_Dense_32'),
        layers.Dropout(0.2, name='Regularization_Dropout_20pct'),
        
        layers.Dense(16, activation='relu', name='Hidden_Dense_16'),
        layers.Dropout(0.1, name='Regularization_Dropout_10pct'),
        
        # Camada de saída
        layers.Dense(1, activation='sigmoid', name='Output_Binary_Classification')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_cdr_model_detailed():
    """Cria modelo CDR com arquitetura dual-branch detalhada"""
    # Entrada
    inputs = layers.Input(shape=(80,), name='Medical_Features_Input_80D')
    
    # Camada de atenção
    attention = layers.Dense(256, activation='relu', name='Attention_Layer_256')(inputs)
    attention = layers.Dropout(0.4, name='Attention_Dropout_40pct')(attention)
    attention = layers.BatchNormalization(name='Attention_BatchNorm')(attention)
    
    # Branch principal - características gerais
    main_branch = layers.Dense(128, activation='relu', name='Main_Branch_General_128')(attention)
    main_branch = layers.Dropout(0.3, name='Main_Branch_Dropout_30pct')(main_branch)
    main_branch = layers.BatchNormalization(name='Main_Branch_BatchNorm')(main_branch)
    
    # Branch especializado - CDR intermediário (0.5 e 1.0)
    specialized_branch = layers.Dense(64, activation='relu', name='Specialized_CDR_Branch_64')(attention)
    specialized_branch = layers.Dropout(0.3, name='Specialized_Dropout_30pct')(specialized_branch)
    specialized_branch = layers.BatchNormalization(name='Specialized_BatchNorm')(specialized_branch)
    
    # Concatenação das branches
    merged = layers.Concatenate(name='Merge_Dual_Branches')([main_branch, specialized_branch])
    
    # Camadas de fusão
    fusion = layers.Dense(64, activation='relu', name='Fusion_Layer_64')(merged)
    fusion = layers.Dropout(0.3, name='Fusion_Dropout_30pct')(fusion)
    fusion = layers.BatchNormalization(name='Fusion_BatchNorm')(fusion)
    
    # Camadas finais de classificação
    classifier = layers.Dense(32, activation='relu', name='Classifier_Dense_32')(fusion)
    classifier = layers.Dropout(0.2, name='Classifier_Dropout_20pct')(classifier)
    
    classifier = layers.Dense(16, activation='relu', name='Classifier_Dense_16')(classifier)
    classifier = layers.Dropout(0.1, name='Classifier_Dropout_10pct')(classifier)
    
    # Saída multiclasse
    outputs = layers.Dense(4, activation='softmax', name='CDR_4_Classes_Output')(classifier)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='Alzheimer_CDR_Classifier')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_color_schemes():
    """Define esquemas de cores personalizados para diferentes tipos de visualização"""
    
    # Esquema 1: Baseado em função das camadas
    functional_colors = {
        # Camadas de entrada
        'Input': '#E8F5E8',
        'Medical_Features_Input_80D': '#E8F5E8',
        
        # Camadas densas por tamanho
        'Input_Dense_256': '#4CAF50',
        'Attention_Layer_256': '#388E3C',
        'Hidden_Dense_128': '#66BB6A',
        'Main_Branch_General_128': '#66BB6A',
        'Hidden_Dense_64': '#81C784',
        'Specialized_CDR_Branch_64': '#81C784',
        'Fusion_Layer_64': '#81C784',
        'Hidden_Dense_32': '#A5D6A7',
        'Classifier_Dense_32': '#A5D6A7',
        'Hidden_Dense_16': '#C8E6C9',
        'Classifier_Dense_16': '#C8E6C9',
        
        # Camadas de regularização
        'Dropout': '#FF9800',
        'BatchNormalization': '#2196F3',
        
        # Camadas especiais
        'Concatenate': '#F44336',
        'Merge_Dual_Branches': '#F44336',
        
        # Saídas
        'Output_Binary_Classification': '#9C27B0',
        'CDR_4_Classes_Output': '#9C27B0'
    }
    
    # Esquema 2: Baseado na intensidade (gradiente)
    gradient_colors = {
        'Input_Dense_256': '#1A237E',
        'Hidden_Dense_128': '#303F9F',
        'Hidden_Dense_64': '#3F51B5',
        'Hidden_Dense_32': '#5C6BC0',
        'Hidden_Dense_16': '#7986CB',
        'Output_Binary_Classification': '#9FA8DA',
        'Dropout': '#FFC107',
        'BatchNormalization': '#FF5722',
        'Concatenate': '#E91E63'
    }
    
    # Esquema 3: Baseado no papel na arquitetura
    role_based_colors = {
        # Processamento inicial
        'Input': '#E1F5FE',
        'Attention': '#B3E5FC',
        
        # Branches
        'Main_Branch': '#4CAF50',
        'Specialized': '#FF9800',
        
        # Fusão
        'Merge': '#F44336',
        'Fusion': '#9C27B0',
        
        # Classificação final
        'Classifier': '#2196F3',
        'Output': '#607D8B',
        
        # Regularização
        'Dropout': '#FFEB3B',
        'BatchNormalization': '#00BCD4'
    }
    
    return functional_colors, gradient_colors, role_based_colors

def create_advanced_visualizations():
    """Cria visualizações avançadas usando diferentes estilos e configurações"""
    print("Criando visualizações avançadas com Visualkeras...")
    
    # Obter esquemas de cores
    functional_colors, gradient_colors, role_based_colors = create_color_schemes()
    
    # Criar modelos
    binary_model = create_binary_model_detailed()
    cdr_model = create_cdr_model_detailed()
    
    print(f"Modelo Binário: {binary_model.count_params():,} parâmetros")
    print(f"Modelo CDR: {cdr_model.count_params():,} parâmetros")
    
    # ==========================================
    # VISUALIZAÇÕES DO MODELO BINÁRIO
    # ==========================================
    
    print("\n1. Visualizações do Modelo Binário...")
    
    # 1.1 Visualização Layered Clássica
    binary_classic = visualkeras.layered_view(
        binary_model,
        legend=True,
        spacing=50,
        one_dim_orientation='y',
        draw_volume=True
    )
    binary_classic.save('/app/alzheimer/vk_binary_classic.png')
    
    # 1.2 Visualização com cores funcionais
    binary_functional = visualkeras.layered_view(
        binary_model,
        legend=True,
        color_map=functional_colors,
        spacing=60,
        one_dim_orientation='y',
        draw_volume=True
    )
    binary_functional.save('/app/alzheimer/vk_binary_functional.png')
    
    # 1.3 Visualização compacta horizontal
    binary_horizontal = visualkeras.layered_view(
        binary_model,
        legend=True,
        color_map=gradient_colors,
        spacing=30,
        one_dim_orientation='x',
        draw_volume=False
    )
    binary_horizontal.save('/app/alzheimer/vk_binary_horizontal.png')
    
    # ==========================================
    # VISUALIZAÇÕES DO MODELO CDR
    # ==========================================
    
    print("\n2. Visualizações do Modelo CDR...")
    
    # 2.1 Visualização Layered detalhada
    cdr_detailed = visualkeras.layered_view(
        cdr_model,
        legend=True,
        spacing=70,
        one_dim_orientation='y',
        draw_volume=True
    )
    cdr_detailed.save('/app/alzheimer/vk_cdr_detailed.png')
    
    # 2.2 Visualização com cores baseadas em função
    cdr_functional = visualkeras.layered_view(
        cdr_model,
        legend=True,
        color_map=functional_colors,
        spacing=80,
        one_dim_orientation='y',
        draw_volume=True
    )
    cdr_functional.save('/app/alzheimer/vk_cdr_functional.png')
    
    # 2.3 Visualização compacta
    cdr_compact = visualkeras.layered_view(
        cdr_model,
        legend=False,
        spacing=40,
        one_dim_orientation='y',
        draw_volume=False
    )
    cdr_compact.save('/app/alzheimer/vk_cdr_compact.png')
    
    # ==========================================
    # VISUALIZAÇÕES COMPARATIVAS
    # ==========================================
    
    print("\n3. Criando visualizações comparativas...")
    
    # Criar montagem comparativa
    create_comparative_montage(binary_model, cdr_model)
    
    return binary_model, cdr_model

def create_comparative_montage(binary_model, cdr_model):
    """Cria uma montagem comparativa dos dois modelos"""
    
    # Criar visualizações lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    
    # Informações dos modelos
    binary_info = f"""
Modelo Binário para Alzheimer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Arquitetura: Sequencial Linear
🔢 Parâmetros: {binary_model.count_params():,}
🎯 Classes: 2 (Demente vs Normal)
⚡ Complexidade: Baixa

📋 Estrutura:
• Entrada: 50 features
• Dense: 256 → 128 → 64 → 32 → 16
• Regularização: Dropout + BatchNorm
• Saída: 1 neurônio (Sigmoid)

🔬 Aplicação Clínica:
• Triagem populacional
• Screening inicial
• Detecção binária rápida
"""
    
    cdr_info = f"""
Modelo CDR Multiclasse para Alzheimer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Arquitetura: Dual-Branch Especializada
🔢 Parâmetros: {cdr_model.count_params():,}
🎯 Classes: 4 (CDR 0/0.5/1.0/2.0)
⚡ Complexidade: Média-Alta

📋 Estrutura:
• Entrada: 80 features (+ especializadas)
• Atenção: Dense 256
• Branch Principal: 128 neurônios
• Branch Especializado: 64 neurônios (CDR 0.5-1.0)
• Fusão: Concatenação + classificação
• Saída: 4 neurônios (Softmax)

🔬 Aplicação Clínica:
• Estadiamento detalhado
• Diagnóstico especializado
• Monitoramento progressão
"""
    
    # Configurar eixos
    ax1.text(0.05, 0.95, binary_info, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#E8F5E8', alpha=0.8))
    
    ax2.text(0.05, 0.95, cdr_info, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#FFF3E0', alpha=0.8))
    
    ax1.set_title('🧠 Modelo Binário - Classificação Simplificada', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_title('🧠 Modelo CDR - Classificação Especializada', 
                  fontsize=16, fontweight='bold', pad=20)
    
    ax1.axis('off')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('/app/alzheimer/vk_comparative_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

def create_architecture_infographic():
    """Cria infográfico detalhado das arquiteturas"""
    fig = plt.figure(figsize=(16, 20))
    
    # Título principal
    fig.suptitle('🧠 Pipeline de IA para Detecção de Alzheimer\nArquiteturas de Deep Learning Especializadas', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Layout em grid
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 2, 2, 1], hspace=0.3, wspace=0.2)
    
    # Seção 1: Visão Geral
    ax_overview = fig.add_subplot(gs[0, :])
    overview_text = """
🎯 OBJETIVO: Desenvolvimento de modelos de IA especializados para detecção e classificação de Alzheimer
📊 DADOS: Dataset OASIS com neuroimagens (RM) + metadados clínicos + features engineered
🔬 ABORDAGEM: Dual-model approach - Binário para triagem + Multiclasse para estadiamento
⚡ OTIMIZAÇÃO: GPU-accelerated training com Mixed Precision + Data Augmentation direcionado
"""
    ax_overview.text(0.5, 0.5, overview_text, ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='#E1F5FE', alpha=0.8))
    ax_overview.axis('off')
    
    # Seção 2: Modelo Binário
    ax_binary = fig.add_subplot(gs[1, 0])
    binary_details = """
🔹 MODELO BINÁRIO
━━━━━━━━━━━━━━━━━━━━

📐 Arquitetura: Feedforward Neural Network
🎯 Objetivo: Demente vs Normal
📊 Features: ~50 (volumétricas + intensidade)
🔢 Parâmetros: 58,625

🏗️ ESTRUTURA:
Input (50) → Dense(256) → Dropout(0.4) → BatchNorm
           → Dense(128) → Dropout(0.3) → BatchNorm  
           → Dense(64)  → Dropout(0.3) → BatchNorm
           → Dense(32)  → Dropout(0.2)
           → Dense(16)  → Dropout(0.1)
           → Dense(1)   → Sigmoid

⚙️ CONFIGURAÇÃO:
• Otimizador: Adam (LR=0.001)
• Loss: Binary Crossentropy
• Métricas: Accuracy, AUC-ROC
• Regularização: Dropout progressivo + BatchNorm

🎯 APLICAÇÃO CLÍNICA:
• Triagem populacional em larga escala
• Screening inicial em unidades básicas
• Identificação rápida de casos suspeitos
• Ferramenta de apoio para médicos generalistas
"""
    ax_binary.text(0.05, 0.95, binary_details, transform=ax_binary.transAxes, 
                  fontsize=9, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F5E8', alpha=0.9))
    ax_binary.axis('off')
    
    # Seção 3: Modelo CDR
    ax_cdr = fig.add_subplot(gs[1, 1])
    cdr_details = """
🔸 MODELO CDR MULTICLASSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━

📐 Arquitetura: Dual-Branch Neural Network
🎯 Objetivo: CDR 0/0.5/1.0/2.0 (estadiamento)
📊 Features: ~80 (incluindo 5 especializadas)
🔢 Parâmetros: 87,156

🏗️ ESTRUTURA DUAL-BRANCH:
Input (80) → Dense(256) → Attention Layer
           ↓
    ┌─ Main Branch ────┐    ┌─ Specialized Branch ─┐
    │ Dense(128)       │    │ Dense(64)            │
    │ Dropout(0.3)     │    │ Dropout(0.3)         │
    │ BatchNorm        │    │ BatchNorm             │
    │ (General)        │    │ (CDR 0.5-1.0 focus)  │
    └──────────────────┘    └───────────────────────┘
           │                         │
           └─── Concatenate(192) ────┘
                       │
              Dense(64) → Dense(32) → Dense(16)
                       │
                   Dense(4) → Softmax

⚙️ CONFIGURAÇÃO ESPECIALIZADA:
• Otimizador: Adam (LR=0.0005)
• Loss: Sparse Categorical Crossentropy
• Pesos de classe: CDR=1 com peso 1.5x
• Features especializadas para CDR=1

🎯 APLICAÇÃO CLÍNICA:
• Estadiamento preciso em centros especializados
• Monitoramento de progressão da doença
• Suporte para decisões terapêuticas
• Pesquisa clínica e ensaios farmacológicos
"""
    ax_cdr.text(0.05, 0.95, cdr_details, transform=ax_cdr.transAxes, 
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFF3E0', alpha=0.9))
    ax_cdr.axis('off')
    
    # Seção 4: Features Especializadas
    ax_features = fig.add_subplot(gs[2, :])
    features_text = """
🔬 FEATURES ESPECIALIZADAS PARA DETECÇÃO DE CDR=1 (Alzheimer Leve)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ RAZÃO HIPOCAMPO/AMÍGDALA
   • Indicador de atrofia relativa entre estruturas límbicas
   • Crucial para detecção de estágios intermediários
   • Hipocampo mais afetado que amígdala no Alzheimer

2️⃣ ASSIMETRIA TEMPORAL  
   • Diferenças volumétricas entre hemisférios cerebrais
   • Indicador precoce de disfunção neuronal
   • Padrão específico no córtex temporal

3️⃣ SCORE COGNITIVO-ANATÔMICO
   • Combinação: MMSE × Volume do Hipocampo
   • Correlação função cognitiva vs estrutura cerebral
   • Biomarcador híbrido funcional-anatômico

4️⃣ ÍNDICE DE DETERIORAÇÃO VOLUMÉTRICA
   • Média ponderada de regiões cerebrais afetadas
   • Padrão global de atrofia específico do Alzheimer
   • Considera hipocampo, entorrinal, temporal

5️⃣ SCORE DE INTENSIDADE GLOBAL
   • Padrão de sinal nas imagens de ressonância magnética
   • Detecta alterações microestruturais sutis
   • Complementa informações volumétricas
"""
    ax_features.text(0.05, 0.95, features_text, transform=ax_features.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='#F3E5F5', alpha=0.9))
    ax_features.axis('off')
    
    # Seção 5: Tecnologias e Performance
    ax_tech = fig.add_subplot(gs[3, :])
    tech_text = """
⚡ TECNOLOGIAS E OTIMIZAÇÕES IMPLEMENTADAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🖥️ GPU ACCELERATION: TensorFlow + CUDA + Mixed Precision (float16) | ⚡ Data Augmentation Direcionado: Transformações geométricas e fotométricas
🧠 ARQUITETURA: Dual-branch com atenção especializada para CDR=1 | 📊 Balanceamento: Pesos de classe + oversampling inteligente  
🔄 REGULARIZAÇÃO: Dropout progressivo + BatchNormalization | 📈 MONITORAMENTO: TensorBoard + Early Stopping + LR scheduling
"""
    ax_tech.text(0.5, 0.5, tech_text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", facecolor='#E0F2F1', alpha=0.9))
    ax_tech.axis('off')
    
    plt.savefig('/app/alzheimer/vk_architecture_infographic.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

def main():
    """Executa todas as visualizações avançadas"""
    print("🧠 VISUALIZAÇÃO AVANÇADA DE ARQUITETURAS - ALZHEIMER AI")
    print("=" * 70)
    print("Baseado nas melhores práticas do Kaggle Visualkeras")
    print("=" * 70)
    
    # 1. Criar visualizações avançadas
    print("\n📊 Fase 1: Criando visualizações Visualkeras avançadas...")
    binary_model, cdr_model = create_advanced_visualizations()
    
    # 2. Criar infográfico
    print("\n📋 Fase 2: Gerando infográfico detalhado...")
    create_architecture_infographic()
    
    # 3. Listar arquivos gerados
    print("\n📁 ARQUIVOS GERADOS:")
    print("-" * 50)
    
    visualizations = [
        "🔵 vk_binary_classic.png - Modelo binário clássico",
        "🔵 vk_binary_functional.png - Modelo binário com cores funcionais", 
        "🔵 vk_binary_horizontal.png - Modelo binário horizontal",
        "🔸 vk_cdr_detailed.png - Modelo CDR detalhado",
        "🔸 vk_cdr_functional.png - Modelo CDR funcional",
        "🔸 vk_cdr_compact.png - Modelo CDR compacto",
        "🎯 vk_comparative_analysis.png - Análise comparativa",
        "📊 vk_architecture_infographic.png - Infográfico completo"
    ]
    
    for viz in visualizations:
        print(f"   {viz}")
    
    print(f"\n📍 Localização: /app/alzheimer/")
    print(f"\n📈 ESTATÍSTICAS:")
    print(f"   🔢 Modelo Binário: {binary_model.count_params():,} parâmetros")
    print(f"   🔢 Modelo CDR: {cdr_model.count_params():,} parâmetros")
    print(f"   📊 Total de visualizações: {len(visualizations)}")
    
    print("\n✅ VISUALIZAÇÕES AVANÇADAS CONCLUÍDAS COM SUCESSO!")
    print("🎨 Use essas imagens para apresentações e documentação científica")

if __name__ == "__main__":
    main()


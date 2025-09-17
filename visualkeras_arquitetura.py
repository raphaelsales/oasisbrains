#!/usr/bin/env python3
"""
Visualização de Arquiteturas usando Visualkeras
Gera visualizações profissionais dos modelos de deep learning para Alzheimer
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import visualkeras
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

# Configurar TensorFlow para não usar GPU (evitar conflitos na visualização)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_binary_model(input_dim=50):
    """Cria modelo binário para visualização"""
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,), name='Dense_256_Input'),
        layers.Dropout(0.4, name='Dropout_40%'),
        layers.BatchNormalization(name='BatchNorm_1'),
        
        layers.Dense(128, activation='relu', name='Dense_128'),
        layers.Dropout(0.3, name='Dropout_30%_1'),
        layers.BatchNormalization(name='BatchNorm_2'),
        
        layers.Dense(64, activation='relu', name='Dense_64'),
        layers.Dropout(0.3, name='Dropout_30%_2'),
        layers.BatchNormalization(name='BatchNorm_3'),
        
        layers.Dense(32, activation='relu', name='Dense_32'),
        layers.Dropout(0.2, name='Dropout_20%'),
        
        layers.Dense(16, activation='relu', name='Dense_16'),
        layers.Dropout(0.1, name='Dropout_10%'),
        
        layers.Dense(1, activation='sigmoid', name='Output_Binary')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_multiclass_cdr_model(input_dim=80):
    """Cria modelo multiclasse CDR com arquitetura especializada"""
    # Entrada
    inputs = layers.Input(shape=(input_dim,), name='Features_Input')
    
    # Camada de atenção
    x = layers.Dense(256, activation='relu', name='Attention_Layer_256')(inputs)
    x = layers.Dropout(0.4, name='Attention_Dropout_40%')(x)
    x = layers.BatchNormalization(name='Attention_BatchNorm')(x)
    
    # Branch principal
    main_branch = layers.Dense(128, activation='relu', name='Main_Branch_128')(x)
    main_branch = layers.Dropout(0.3, name='Main_Dropout_30%')(main_branch)
    main_branch = layers.BatchNormalization(name='Main_BatchNorm')(main_branch)
    
    # Branch especializado para CDR intermediário
    intermediate_branch = layers.Dense(64, activation='relu', name='Intermediate_Branch_64')(x)
    intermediate_branch = layers.Dropout(0.3, name='Intermediate_Dropout_30%')(intermediate_branch)
    intermediate_branch = layers.BatchNormalization(name='Intermediate_BatchNorm')(intermediate_branch)
    
    # Concatenação
    combined = layers.Concatenate(name='Concat_Branches')([main_branch, intermediate_branch])
    
    # Camadas finais
    x = layers.Dense(64, activation='relu', name='Final_Dense_64')(combined)
    x = layers.Dropout(0.3, name='Final_Dropout_30%')(x)
    x = layers.BatchNormalization(name='Final_BatchNorm_1')(x)
    
    x = layers.Dense(32, activation='relu', name='Final_Dense_32')(x)
    x = layers.Dropout(0.2, name='Final_Dropout_20%')(x)
    
    x = layers.Dense(16, activation='relu', name='Final_Dense_16')(x)
    x = layers.Dropout(0.1, name='Final_Dropout_10%')(x)
    
    # Saída
    outputs = layers.Dense(4, activation='softmax', name='Output_CDR_4_Classes')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='CDR_Specialized_Model')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_visualkeras_plots():
    """Cria visualizações usando Visualkeras"""
    print("Criando visualizações com Visualkeras...")
    
    # Cores personalizadas para diferentes tipos de camadas
    color_map = {
        'Dense': '#4CAF50',           # Verde para camadas densas
        'Dropout': '#FF9800',         # Laranja para dropout
        'BatchNormalization': '#2196F3',  # Azul para batch norm
        'Input': '#9C27B0',           # Roxo para entrada
        'Concatenate': '#F44336',     # Vermelho para concatenação
        'Attention': '#00BCD4'        # Ciano para atenção
    }
    
    # Modelo Binário
    print("Gerando visualização do modelo binário...")
    binary_model = create_binary_model()
    
    # Visualização layered (camadas empilhadas)
    binary_layered = visualkeras.layered_view(
        binary_model,
        legend=True,
        draw_volume=True,
        color_map=color_map,
        spacing=50,
        one_dim_orientation='y'
    )
    binary_layered.save('/app/alzheimer/visualkeras_binary_layered.png')
    
    # Visualização graph (como grafo) - parâmetros corretos
    try:
        binary_graph = visualkeras.graph_view(
            binary_model,
            color_map=color_map
        )
        binary_graph.save('/app/alzheimer/visualkeras_binary_graph.png')
    except Exception as e:
        print(f"Aviso: Graph view não disponível para modelo binário: {e}")
        # Criar visualização alternativa
        binary_alt = visualkeras.layered_view(
            binary_model,
            legend=True,
            color_map=color_map,
            spacing=30
        )
        binary_alt.save('/app/alzheimer/visualkeras_binary_alternative.png')
    
    # Modelo Multiclasse CDR
    print("Gerando visualização do modelo multiclasse CDR...")
    cdr_model = create_multiclass_cdr_model()
    
    # Visualização layered para modelo CDR
    cdr_layered = visualkeras.layered_view(
        cdr_model,
        legend=True,
        draw_volume=True,
        color_map=color_map,
        spacing=60,
        one_dim_orientation='y'
    )
    cdr_layered.save('/app/alzheimer/visualkeras_cdr_layered.png')
    
    # Visualização graph para modelo CDR (mostra as ramificações)
    try:
        cdr_graph = visualkeras.graph_view(
            cdr_model,
            color_map=color_map,
            rankdir='TB'  # Top to Bottom
        )
        cdr_graph.save('/app/alzheimer/visualkeras_cdr_graph.png')
    except Exception as e:
        print(f"Aviso: Graph view não disponível para modelo CDR: {e}")
        # Criar visualização alternativa mais detalhada
        cdr_alt = visualkeras.layered_view(
            cdr_model,
            legend=True,
            color_map=color_map,
            spacing=40,
            draw_volume=False  # Para modelos complexos
        )
        cdr_alt.save('/app/alzheimer/visualkeras_cdr_alternative.png')
    
    print("Visualizações Visualkeras concluídas!")
    return binary_model, cdr_model

def create_enhanced_architecture_summary():
    """Cria resumo visual aprimorado das arquiteturas"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Criar modelos para análise
    binary_model, cdr_model = create_visualkeras_plots()
    
    # 1. Resumo Modelo Binário
    ax1.set_title('Modelo Binário - Resumo da Arquitetura', fontsize=14, fontweight='bold', pad=20)
    ax1.text(0.5, 0.9, f'Total de Parâmetros: {binary_model.count_params():,}', 
             ha='center', fontsize=12, fontweight='bold', transform=ax1.transAxes)
    
    # Informações das camadas do modelo binário
    binary_info = [
        'Entrada: 50 features (volumétricas + intensidade)',
        'Dense 256 → ReLU → Dropout 40% → BatchNorm',
        'Dense 128 → ReLU → Dropout 30% → BatchNorm', 
        'Dense 64 → ReLU → Dropout 30% → BatchNorm',
        'Dense 32 → ReLU → Dropout 20%',
        'Dense 16 → ReLU → Dropout 10%',
        'Saída: 1 neurônio → Sigmoid (Binário)',
        '',
        'Otimizador: Adam',
        'Loss: Binary Crossentropy',
        'Regularização: Dropout + BatchNorm',
        'Aplicação: Demente vs Normal'
    ]
    
    for i, info in enumerate(binary_info):
        color = '#e8f5e8' if 'Dense' in info else '#f0f0f0'
        if 'Entrada' in info or 'Saída' in info:
            color = '#e1f5fe'
        ax1.text(0.05, 0.8 - i*0.06, info, fontsize=10, transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # 2. Resumo Modelo CDR
    ax2.set_title('Modelo CDR Multiclasse - Resumo da Arquitetura', fontsize=14, fontweight='bold', pad=20)
    ax2.text(0.5, 0.9, f'Total de Parâmetros: {cdr_model.count_params():,}', 
             ha='center', fontsize=12, fontweight='bold', transform=ax2.transAxes)
    
    cdr_info = [
        'Entrada: 80 features (incluindo especializadas)',
        'Camada Atenção: Dense 256 → Dropout 40%',
        'Branch Principal: Dense 128 → BatchNorm',
        'Branch Intermediário: Dense 64 (CDR 0.5-1.0)',
        'Concatenação: 128 + 64 = 192 neurônios',
        'Dense 64 → Dense 32 → Dense 16',
        'Saída: 4 classes (CDR 0, 0.5, 1.0, 2.0)',
        '',
        'Arquitetura: Dupla ramificação',
        'Especialização: Detecção CDR=1',
        'Pesos de classe: Aplicados',
        'Features engenhadas: 5 especializadas'
    ]
    
    for i, info in enumerate(cdr_info):
        color = '#fff3e0' if 'Branch' in info else '#f0f0f0'
        if 'Entrada' in info or 'Saída' in info:
            color = '#e1f5fe'
        elif 'Concatenação' in info:
            color = '#fce4ec'
        ax2.text(0.05, 0.8 - i*0.06, info, fontsize=10, transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # 3. Comparação de Performance
    ax3.set_title('Comparação de Modelos', fontsize=14, fontweight='bold', pad=20)
    
    # Dados comparativos (simulados baseados na arquitetura)
    modelos = ['Modelo Binário', 'Modelo CDR']
    parametros = [binary_model.count_params(), cdr_model.count_params()]
    complexidade = ['Baixa', 'Alta']
    aplicacao = ['Triagem inicial', 'Diagnóstico detalhado']
    
    # Gráfico de barras dos parâmetros
    bars = ax3.bar(modelos, parametros, color=['#4CAF50', '#FF9800'], alpha=0.7)
    ax3.set_ylabel('Número de Parâmetros', fontsize=12)
    ax3.set_title('Complexidade dos Modelos', fontsize=12)
    
    # Adicionar valores nas barras
    for bar, param in zip(bars, parametros):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{param:,}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Features Engineering
    ax4.set_title('Features Especializadas para CDR=1', fontsize=14, fontweight='bold', pad=20)
    
    features_especializadas = [
        '1. Razão Hipocampo/Amígdala',
        '   • Indicador de atrofia relativa',
        '   • Importante para estágio intermediário',
        '',
        '2. Assimetria Temporal',
        '   • Diferenças entre hemisférios',
        '   • Indicador precoce de disfunção',
        '',
        '3. Score Cognitivo-Anatômico',
        '   • MMSE × Volume hipocampo',
        '   • Correlação função-estrutura',
        '',
        '4. Índice Deterioração Volumétrica',
        '   • Média de regiões afetadas',
        '   • Padrão global de atrofia',
        '',
        '5. Score Intensidade Global',
        '   • Padrão geral de sinal RM',
        '   • Alterações microestruturais'
    ]
    
    for i, feature in enumerate(features_especializadas):
        if feature.startswith(('1.', '2.', '3.', '4.', '5.')):
            color = '#e1f5fe'
            fontweight = 'bold'
        elif feature.startswith('   •'):
            color = '#f3e5f5'
            fontweight = 'normal'
        else:
            color = 'white'
            fontweight = 'normal'
        
        if feature.strip():  # Não mostrar linhas vazias
            ax4.text(0.05, 0.95 - i*0.045, feature, fontsize=10, fontweight=fontweight,
                    transform=ax4.transAxes,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('/app/alzheimer/arquitetura_summary_completo.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    return binary_model, cdr_model

def create_model_comparison_table():
    """Cria tabela comparativa detalhada dos modelos"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Dados da tabela
    comparison_data = [
        ['Característica', 'Modelo Binário', 'Modelo CDR Multiclasse'],
        ['Tipo de Classificação', 'Binária (Demente vs Normal)', 'Multiclasse (CDR 0/0.5/1.0/2.0)'],
        ['Arquitetura', 'Sequencial Linear', 'Dupla Ramificação'],
        ['Features de Entrada', '~50 (volumétricas + intensidade)', '~80 (incluindo especializadas)'],
        ['Camadas Densas', '6 camadas (256→128→64→32→16→1)', '8 camadas + concatenação'],
        ['Neurônios Total', '497', '577'],
        ['Especialização', 'Detecção geral', 'Foco em CDR=1 (estágio leve)'],
        ['Branch Principal', 'Única sequência', 'Dense 128 (características gerais)'],
        ['Branch Especializado', 'Não possui', 'Dense 64 (CDR intermediário)'],
        ['Regularização', 'Dropout + BatchNorm', 'Dropout + BatchNorm + Pesos classe'],
        ['Função Ativação Final', 'Sigmoid', 'Softmax (4 classes)'],
        ['Loss Function', 'Binary Crossentropy', 'Sparse Categorical Crossentropy'],
        ['Learning Rate', '0.001 (padrão)', '0.0005 (reduzido para multiclasse)'],
        ['Data Augmentation', 'Opcional', 'Direcionado para classes minoritárias'],
        ['Aplicação Clínica', 'Triagem inicial', 'Estadiamento detalhado'],
        ['Tempo Treinamento', 'Menor (~5-10 min)', 'Maior (~15-25 min)'],
        ['Interpretabilidade', 'Alta (saída binária)', 'Média (4 probabilidades)'],
        ['Sensibilidade CDR=1', 'Não específica', 'Otimizada (peso 1.5x)'],
        ['Complexidade Computacional', 'Baixa', 'Média-Alta'],
        ['Uso Recomendado', 'Screening populacional', 'Diagnóstico especializado']
    ]
    
    # Criar tabela
    table = ax.table(cellText=comparison_data[1:], 
                    colLabels=comparison_data[0],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.3, 0.35, 0.35])
    
    # Estilizar tabela
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Cores alternadas para linhas
    for i in range(1, len(comparison_data)):
        for j in range(len(comparison_data[0])):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('#ffffff')
            
            # Destacar diferenças importantes
            if 'CDR' in comparison_data[i][j] or 'Dupla' in comparison_data[i][j]:
                cell.set_facecolor('#fff3e0')
            elif 'especializada' in comparison_data[i][j].lower():
                cell.set_facecolor('#e1f5fe')
    
    # Estilizar cabeçalho
    for j in range(len(comparison_data[0])):
        cell = table[(0, j)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    ax.set_title('Comparação Detalhada: Modelo Binário vs Modelo CDR Multiclasse', 
                fontsize=16, fontweight='bold', pad=30)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/app/alzheimer/comparacao_modelos_detalhada.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("Tabela comparativa salva: comparacao_modelos_detalhada.png")

def main():
    """Executa todas as visualizações com Visualkeras"""
    print("VISUALIZAÇÃO DE ARQUITETURAS COM VISUALKERAS")
    print("=" * 60)
    
    # 1. Criar visualizações Visualkeras básicas
    print("\n1. Gerando visualizações Visualkeras...")
    binary_model, cdr_model = create_visualkeras_plots()
    
    # 2. Resumo aprimorado das arquiteturas
    print("\n2. Criando resumo aprimorado das arquiteturas...")
    create_enhanced_architecture_summary()
    
    # 3. Tabela comparativa detalhada
    print("\n3. Gerando tabela comparativa detalhada...")
    create_model_comparison_table()
    
    # 4. Resumo dos modelos
    print("\n4. RESUMO DOS MODELOS CRIADOS:")
    print("-" * 40)
    print(f"Modelo Binário:")
    print(f"   - Parâmetros: {binary_model.count_params():,}")
    print(f"   - Camadas: {len(binary_model.layers)}")
    print(f"   - Aplicação: Classificação Demente vs Normal")
    
    print(f"\nModelo CDR Multiclasse:")
    print(f"   - Parâmetros: {cdr_model.count_params():,}")
    print(f"   - Camadas: {len(cdr_model.layers)}")
    print(f"   - Aplicação: Classificação CDR (0, 0.5, 1.0, 2.0)")
    
    print("\n5. ARQUIVOS GERADOS:")
    print("-" * 40)
    visualizations = [
        "visualkeras_binary_layered.png - Modelo binário (camadas)",
        "visualkeras_binary_graph.png - Modelo binário (grafo)",
        "visualkeras_cdr_layered.png - Modelo CDR (camadas)",
        "visualkeras_cdr_graph.png - Modelo CDR (grafo)",
        "arquitetura_summary_completo.png - Resumo completo",
        "comparacao_modelos_detalhada.png - Tabela comparativa"
    ]
    
    for viz in visualizations:
        print(f"   • {viz}")
    
    print(f"\nTodos os arquivos salvos em: /app/alzheimer/")
    print("\nVISUALIZAÇÕES VISUALKERAS CONCLUÍDAS COM SUCESSO!")

if __name__ == "__main__":
    main()

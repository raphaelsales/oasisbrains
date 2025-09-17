#!/usr/bin/env python3
"""
Galeria de Visualizações - Compilação Final
Organiza todas as visualizações criadas em uma galeria navegável
"""

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def create_visualization_gallery():
    """Cria galeria organizada de todas as visualizações"""
    
    # Encontrar todas as visualizações
    viz_files = {
        'Visualkeras Clássico': [
            ('vk_binary_classic.png', 'Modelo Binário - Clássico'),
            ('vk_cdr_detailed.png', 'Modelo CDR - Detalhado')
        ],
        'Visualkeras Funcional': [
            ('vk_binary_functional.png', 'Modelo Binário - Cores Funcionais'),
            ('vk_cdr_functional.png', 'Modelo CDR - Cores Funcionais')
        ],
        'Visualkeras Compacto': [
            ('vk_binary_horizontal.png', 'Modelo Binário - Horizontal'),
            ('vk_cdr_compact.png', 'Modelo CDR - Compacto')
        ],
        'Análises Comparativas': [
            ('vk_comparative_analysis.png', 'Análise Comparativa Detalhada'),
            ('vk_architecture_infographic.png', 'Infográfico Completo')
        ],
        'Diagramas Personalizados': [
            ('arquitetura_modelos_alzheimer.png', 'Arquitetura Personalizada'),
            ('pipeline_completo_alzheimer.png', 'Pipeline Completo'),
            ('feature_engineering_alzheimer.png', 'Feature Engineering')
        ]
    }
    
    print("📸 CRIANDO GALERIA DE VISUALIZAÇÕES")
    print("=" * 50)
    
    for category, files in viz_files.items():
        print(f"\n📂 {category}:")
        for filename, description in files:
            if os.path.exists(filename):
                size = os.path.getsize(filename) / 1024  # KB
                print(f"   ✅ {description} ({size:.1f} KB)")
            else:
                print(f"   ❌ {description} (não encontrado)")
    
    # Criar overview das visualizações
    create_overview_grid()
    
    # Criar índice de navegação
    create_navigation_index()

def create_overview_grid():
    """Cria grid de overview de todas as visualizações principais"""
    
    # Arquivos principais para overview
    main_files = [
        ('vk_binary_classic.png', 'Modelo Binário\n(Visualkeras)'),
        ('vk_cdr_detailed.png', 'Modelo CDR\n(Visualkeras)'),
        ('arquitetura_modelos_alzheimer.png', 'Arquitetura\n(Personalizada)'),
        ('pipeline_completo_alzheimer.png', 'Pipeline\n(Completo)')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('🧠 Galeria de Visualizações - Pipeline de IA para Alzheimer', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    
    for i, (filename, title) in enumerate(main_files):
        ax = axes[i]
        
        if os.path.exists(filename):
            try:
                img = mpimg.imread(filename)
                ax.imshow(img)
                ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
                
                # Adicionar bordas coloridas
                colors = ['#4CAF50', '#FF9800', '#2196F3', '#9C27B0']
                rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                               linewidth=5, edgecolor=colors[i], 
                               facecolor='none', alpha=0.8)
                ax.add_patch(rect)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Erro ao carregar:\n{filename}\n{str(e)}', 
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
        else:
            ax.text(0.5, 0.5, f'Arquivo não encontrado:\n{filename}', 
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
        
        ax.axis('off')
    
    # Adicionar legenda
    legend_text = """
📊 TIPOS DE VISUALIZAÇÃO:
🔵 Visualkeras: Biblioteca especializada para arquiteturas neurais
🔸 Personalizada: Diagramas criados com matplotlib  
🎯 Pipeline: Fluxo completo do processamento
🧠 Comparativa: Análise lado-a-lado dos modelos
"""
    
    fig.text(0.02, 0.02, legend_text, fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#F0F8FF', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('/app/alzheimer/galeria_overview.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("✅ Overview da galeria criado: galeria_overview.png")

def create_navigation_index():
    """Cria índice navegável de todas as visualizações"""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    index_content = """
🧠 ÍNDICE COMPLETO DE VISUALIZAÇÕES - PIPELINE ALZHEIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📂 CATEGORIA 1: VISUALKERAS - BIBLIOTECAS ESPECIALIZADAS
   🔵 vk_binary_classic.png           → Modelo binário em estilo clássico
   🔵 vk_binary_functional.png        → Modelo binário com cores por função
   🔵 vk_binary_horizontal.png        → Modelo binário em layout horizontal
   🔸 vk_cdr_detailed.png             → Modelo CDR detalhado vertical
   🔸 vk_cdr_functional.png           → Modelo CDR com cores funcionais
   🔸 vk_cdr_compact.png              → Modelo CDR em formato compacto

📂 CATEGORIA 2: ANÁLISES COMPARATIVAS E INFOGRÁFICOS
   🎯 vk_comparative_analysis.png     → Comparação detalhada dos dois modelos
   📊 vk_architecture_infographic.png → Infográfico completo da arquitetura
   📋 comparacao_modelos_detalhada.png → Tabela comparativa detalhada
   📈 arquitetura_summary_completo.png → Resumo técnico das arquiteturas

📂 CATEGORIA 3: DIAGRAMAS DE PIPELINE E FLUXO
   🔄 pipeline_completo_alzheimer.png → Fluxo completo do pipeline (22 etapas)
   🏗️ arquitetura_modelos_alzheimer.png → Arquitetura detalhada dos modelos
   🔬 feature_engineering_alzheimer.png → Engenharia de características
   📊 galeria_overview.png            → Overview da galeria (este arquivo)

📂 CATEGORIA 4: VISUALIZAÇÕES LEGADAS (PRIMEIRAS VERSÕES)
   📋 visualkeras_binary_layered.png  → Primeira versão modelo binário
   📋 visualkeras_cdr_layered.png     → Primeira versão modelo CDR
   📋 visualkeras_binary_graph.png    → Tentativa de visualização em grafo
   📋 visualkeras_cdr_alternative.png → Versão alternativa modelo CDR

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 LOCALIZAÇÃO: /app/alzheimer/
📊 TOTAL DE VISUALIZAÇÕES: ~16 arquivos
🎨 FORMATOS: PNG (alta resolução, 300 DPI)
💾 TAMANHO TOTAL: ~8-10 MB

🎯 RECOMENDAÇÕES DE USO:
   📑 Apresentações: Use vk_comparative_analysis.png e vk_architecture_infographic.png
   📚 Documentação: Use pipeline_completo_alzheimer.png e feature_engineering_alzheimer.png  
   🔬 Papers científicos: Use vk_cdr_detailed.png e comparacao_modelos_detalhada.png
   🎓 Ensino: Use galeria_overview.png e arquitetura_modelos_alzheimer.png

🔧 FERRAMENTAS UTILIZADAS:
   • Visualkeras: Visualização especializada de redes neurais
   • Matplotlib: Diagramas personalizados e infográficos
   • Mermaid: Diagramas de fluxo (documentação)
   • TensorFlow/Keras: Criação dos modelos para visualização
   • PIL/Pillow: Processamento de imagens
"""
    
    ax.text(0.05, 0.95, index_content, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', alpha=0.95))
    
    ax.set_title('📚 Índice de Navegação - Visualizações Pipeline Alzheimer', 
                fontsize=16, fontweight='bold', pad=30)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/app/alzheimer/indice_navegacao.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("✅ Índice de navegação criado: indice_navegacao.png")

def generate_usage_guide():
    """Gera guia de uso das visualizações"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    guide_content = """
🎨 GUIA DE USO DAS VISUALIZAÇÕES - PIPELINE ALZHEIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 PARA APRESENTAÇÕES EXECUTIVAS (15-20 min):
   1️⃣ galeria_overview.png - Slide de abertura com visão geral
   2️⃣ vk_comparative_analysis.png - Comparação dos dois modelos
   3️⃣ vk_architecture_infographic.png - Infográfico técnico completo
   4️⃣ pipeline_completo_alzheimer.png - Fluxo do pipeline

🎓 PARA APRESENTAÇÕES ACADÊMICAS (30-45 min):
   1️⃣ pipeline_completo_alzheimer.png - Metodologia completa
   2️⃣ feature_engineering_alzheimer.png - Engenharia de características
   3️⃣ vk_cdr_detailed.png - Arquitetura especializada CDR
   4️⃣ comparacao_modelos_detalhada.png - Análise técnica detalhada
   5️⃣ vk_architecture_infographic.png - Resumo e conclusões

📚 PARA DOCUMENTAÇÃO TÉCNICA:
   • README.md: galeria_overview.png
   • Arquitetura: vk_cdr_detailed.png + vk_binary_classic.png  
   • Pipeline: pipeline_completo_alzheimer.png
   • Features: feature_engineering_alzheimer.png
   • Comparações: comparacao_modelos_detalhada.png

🔬 PARA PAPERS CIENTÍFICOS:
   • Figure 1: pipeline_completo_alzheimer.png (Methodology)
   • Figure 2: feature_engineering_alzheimer.png (Feature Engineering)  
   • Figure 3: vk_cdr_detailed.png (Network Architecture)
   • Figure 4: comparacao_modelos_detalhada.png (Model Comparison)
   • Supplementary: vk_architecture_infographic.png (Technical Details)

💡 DICAS DE FORMATAÇÃO:
   📐 Todas as imagens estão em 300 DPI (qualidade de impressão)
   🎨 Cores consistentes e paleta profissional
   📝 Texto legível em tamanhos reduzidos
   🔄 Layouts responsivos para diferentes mídias

⚙️ COMO BAIXAR E USAR:
   1. Navegue para /app/alzheimer/ no explorador do Cursor
   2. Clique com botão direito nos arquivos PNG
   3. Selecione "Download" ou "Salvar como..."
   4. Use em PowerPoint, LaTeX, Word, etc.

🎨 PERSONALIZAÇÃO:
   • Scripts Python disponíveis para modificações
   • Cores e layouts podem ser ajustados
   • Novos modelos podem ser adicionados facilmente
   • Visualkeras suporta diferentes estilos
"""
    
    ax.text(0.05, 0.95, guide_content, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F5E8', alpha=0.9))
    
    ax.set_title('📖 Guia de Uso das Visualizações', 
                fontsize=16, fontweight='bold', pad=30)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/app/alzheimer/guia_uso_visualizacoes.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("✅ Guia de uso criado: guia_uso_visualizacoes.png")

def main():
    """Executa criação da galeria completa"""
    print("🎨 COMPILANDO GALERIA FINAL DE VISUALIZAÇÕES")
    print("=" * 60)
    
    # Verificar arquivos existentes
    png_files = glob.glob("*.png")
    print(f"📁 Total de arquivos PNG encontrados: {len(png_files)}")
    
    # Criar galeria organizada
    create_visualization_gallery()
    
    # Gerar guia de uso
    generate_usage_guide()
    
    print("\n🎉 GALERIA COMPLETA FINALIZADA!")
    print("=" * 50)
    print("📂 Novos arquivos criados:")
    print("   • galeria_overview.png - Overview visual de 4 visualizações principais")
    print("   • indice_navegacao.png - Índice completo de todas as visualizações")
    print("   • guia_uso_visualizacoes.png - Guia de como usar cada visualização")
    
    print(f"\n📊 ESTATÍSTICAS FINAIS:")
    print(f"   🖼️ Visualizações Visualkeras: 6 arquivos")
    print(f"   📋 Diagramas personalizados: 4 arquivos")
    print(f"   📚 Documentação visual: 3 arquivos")
    print(f"   🎯 Análises comparativas: 3 arquivos")
    print(f"   📁 Total estimado: ~16 visualizações")
    
    print("\n✨ Todas as visualizações estão prontas para uso profissional!")

if __name__ == "__main__":
    main()


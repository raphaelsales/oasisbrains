#!/usr/bin/env python3
"""
Análise Estatística Completa dos Resultados do Hipocampo
Script para analisar os resultados gerados pelo analisar_hipocampo_direto.py

Funcionalidades:
1. Estatísticas descritivas detalhadas
2. Visualizações gráficas
3. Análise de correlações
4. Detecção de outliers
5. Classificação por grupos etários/volumes
6. Relatórios para publicação
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuração dos gráficos
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """Carrega e valida os dados"""
    print("📊 Carregando dados...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dados carregados: {len(df)} sujeitos")
        
        # Validações básicas
        print(f"📋 Colunas disponíveis: {len(df.columns)}")
        print(f"🔍 Valores ausentes: {df.isnull().sum().sum()}")
        
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return None

def generate_descriptive_statistics(df: pd.DataFrame):
    """Gera estatísticas descritivas completas"""
    print("\n📈 ESTATÍSTICAS DESCRITIVAS DETALHADAS")
    print("=" * 50)
    
    # Variáveis principais
    volume_vars = ['left_hippo_volume', 'right_hippo_volume', 'total_hippo_volume']
    intensity_vars = ['intensity_mean', 'intensity_median', 'intensity_std']
    other_vars = ['asymmetry_ratio', 'bayesian_prob_mean', 'voxel_volume']
    
    # Estatísticas por categoria
    categories = {
        'Volumes do Hipocampo (mm³)': volume_vars,
        'Intensidades': intensity_vars,
        'Outras Métricas': other_vars
    }
    
    for category_name, variables in categories.items():
        print(f"\n🧠 {category_name}:")
        print("-" * 30)
        
        for var in variables:
            if var in df.columns:
                data = df[var]
                print(f"  {var}:")
                print(f"    Média: {data.mean():.2f} ± {data.std():.2f}")
                print(f"    Mediana: {data.median():.2f}")
                print(f"    Min-Max: {data.min():.2f} - {data.max():.2f}")
                print(f"    Q1-Q3: {data.quantile(0.25):.2f} - {data.quantile(0.75):.2f}")
                print()

def detect_outliers(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Detecta outliers usando método IQR"""
    Q1 = df[variable].quantile(0.25)
    Q3 = df[variable].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[variable] < lower_bound) | (df[variable] > upper_bound)]
    return outliers

def create_visualizations(df: pd.DataFrame, save_plots: bool = True):
    """Cria visualizações completas"""
    print("\n📊 Gerando visualizações...")
    
    # 1. Distribuições dos volumes
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribuições dos Volumes do Hipocampo', fontsize=16, fontweight='bold')
    
    # Histograma volume total
    axes[0,0].hist(df['total_hippo_volume'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Volume Total do Hipocampo')
    axes[0,0].set_xlabel('Volume (mm³)')
    axes[0,0].set_ylabel('Frequência')
    axes[0,0].axvline(df['total_hippo_volume'].mean(), color='red', linestyle='--', label='Média')
    axes[0,0].legend()
    
    # Comparação esquerdo vs direito
    axes[0,1].scatter(df['left_hippo_volume'], df['right_hippo_volume'], alpha=0.6, color='green')
    axes[0,1].plot([df['left_hippo_volume'].min(), df['left_hippo_volume'].max()], 
                   [df['left_hippo_volume'].min(), df['left_hippo_volume'].max()], 
                   'r--', label='Linha de igualdade')
    axes[0,1].set_title('Volume Esquerdo vs Direito')
    axes[0,1].set_xlabel('Volume Hipocampo Esquerdo (mm³)')
    axes[0,1].set_ylabel('Volume Hipocampo Direito (mm³)')
    axes[0,1].legend()
    
    # Boxplot assimetria
    axes[1,0].boxplot(df['asymmetry_ratio'], vert=True)
    axes[1,0].set_title('Distribuição da Assimetria')
    axes[1,0].set_ylabel('Razão de Assimetria')
    
    # Intensidades vs volumes
    axes[1,1].scatter(df['total_hippo_volume'], df['intensity_mean'], alpha=0.6, color='purple')
    axes[1,1].set_title('Volume vs Intensidade Média')
    axes[1,1].set_xlabel('Volume Total (mm³)')
    axes[1,1].set_ylabel('Intensidade Média')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('distribuicoes_hipocampo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Análise de correlações
    plt.figure(figsize=(12, 10))
    correlation_vars = ['total_hippo_volume', 'left_hippo_volume', 'right_hippo_volume', 
                       'asymmetry_ratio', 'intensity_mean', 'intensity_std', 
                       'bayesian_prob_mean', 'bayesian_prob_std']
    
    corr_matrix = df[correlation_vars].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlação - Métricas do Hipocampo', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_plots:
        plt.savefig('correlacoes_hipocampo.png', dpi=300, bbox_inches='tight')
    plt.show()

def perform_clustering_analysis(df: pd.DataFrame):
    """Realiza análise de clustering dos sujeitos"""
    print("\n🎯 ANÁLISE DE CLUSTERING")
    print("=" * 30)
    
    # Preparar dados para clustering
    clustering_vars = ['total_hippo_volume', 'asymmetry_ratio', 'intensity_mean', 'bayesian_prob_mean']
    X = df[clustering_vars].copy()
    
    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df['cluster'] = clusters
    
    # Análise dos clusters
    print("📊 Características dos clusters:")
    cluster_summary = df.groupby('cluster')[clustering_vars].agg(['mean', 'std']).round(2)
    print(cluster_summary)
    
    # Contagem por cluster
    cluster_counts = df['cluster'].value_counts().sort_index()
    print(f"\n📈 Distribuição dos sujeitos:")
    for i, count in cluster_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   Cluster {i}: {count} sujeitos ({percentage:.1f}%)")
    
    # Visualização dos clusters
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA para visualização
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    axes[0].set_title('Clusters - Visualização PCA')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} da variância)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} da variância)')
    plt.colorbar(scatter, ax=axes[0])
    
    # Volume vs assimetria por cluster
    colors = ['red', 'blue', 'green']
    for i in range(3):
        cluster_data = df[df['cluster'] == i]
        axes[1].scatter(cluster_data['total_hippo_volume'], cluster_data['asymmetry_ratio'], 
                       c=colors[i], label=f'Cluster {i}', alpha=0.7)
    
    axes[1].set_title('Clusters - Volume vs Assimetria')
    axes[1].set_xlabel('Volume Total do Hipocampo (mm³)')
    axes[1].set_ylabel('Razão de Assimetria')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('clustering_analise.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def identify_potential_cases(df: pd.DataFrame):
    """Identifica casos potencialmente interessantes"""
    print("\n🔍 IDENTIFICAÇÃO DE CASOS ESPECIAIS")
    print("=" * 40)
    
    # Definir thresholds baseados na literatura
    volume_threshold_low = df['total_hippo_volume'].quantile(0.10)  # 10% menores volumes
    volume_threshold_high = df['total_hippo_volume'].quantile(0.90)  # 10% maiores volumes
    asymmetry_threshold = df['asymmetry_ratio'].quantile(0.90)  # 10% mais assimétricos
    
    print(f"📊 Thresholds definidos:")
    print(f"   Volume baixo: < {volume_threshold_low:.0f} mm³")
    print(f"   Volume alto: > {volume_threshold_high:.0f} mm³")
    print(f"   Alta assimetria: > {asymmetry_threshold:.3f}")
    
    # Casos de interesse
    low_volume_cases = df[df['total_hippo_volume'] < volume_threshold_low]
    high_asymmetry_cases = df[df['asymmetry_ratio'] > asymmetry_threshold]
    
    print(f"\n🔬 Casos identificados:")
    print(f"   📉 Volumes baixos (possível atrofia): {len(low_volume_cases)} casos")
    print(f"   ⚖️  Alta assimetria: {len(high_asymmetry_cases)} casos")
    
    # Casos extremos (combinação de fatores)
    extreme_cases = df[
        (df['total_hippo_volume'] < volume_threshold_low) & 
        (df['asymmetry_ratio'] > df['asymmetry_ratio'].median())
    ]
    
    print(f"   🚨 Casos extremos (volume baixo + assimetria): {len(extreme_cases)} casos")
    
    if len(extreme_cases) > 0:
        print(f"\n🎯 Sujeitos para investigação detalhada:")
        for _, case in extreme_cases.head(5).iterrows():
            print(f"   {case['subject_id']}: Volume={case['total_hippo_volume']:.0f}mm³, "
                  f"Assimetria={case['asymmetry_ratio']:.3f}")
    
    return {
        'low_volume': low_volume_cases,
        'high_asymmetry': high_asymmetry_cases,
        'extreme_cases': extreme_cases
    }

def generate_summary_report(df: pd.DataFrame, special_cases: dict):
    """Gera relatório final resumido"""
    print("\n📋 RELATÓRIO FINAL - ANÁLISE DO HIPOCAMPO")
    print("=" * 50)
    
    # Estatísticas gerais
    print(f"🧠 DATASET GERAL:")
    print(f"   Total de sujeitos analisados: {len(df)}")
    print(f"   Volume médio do hipocampo: {df['total_hippo_volume'].mean():.0f} ± {df['total_hippo_volume'].std():.0f} mm³")
    print(f"   Faixa de volumes: {df['total_hippo_volume'].min():.0f} - {df['total_hippo_volume'].max():.0f} mm³")
    print(f"   Assimetria média: {df['asymmetry_ratio'].mean():.3f} ± {df['asymmetry_ratio'].std():.3f}")
    
    # Casos especiais
    print(f"\n🔍 CASOS DE INTERESSE CLÍNICO:")
    print(f"   Volumes baixos (possível atrofia): {len(special_cases['low_volume'])} ({len(special_cases['low_volume'])/len(df)*100:.1f}%)")
    print(f"   Alta assimetria: {len(special_cases['high_asymmetry'])} ({len(special_cases['high_asymmetry'])/len(df)*100:.1f}%)")
    print(f"   Casos críticos: {len(special_cases['extreme_cases'])} ({len(special_cases['extreme_cases'])/len(df)*100:.1f}%)")
    
    # Recomendações
    print(f"\n💡 PRÓXIMOS PASSOS RECOMENDADOS:")
    print(f"   1. Análise longitudinal dos casos com volume baixo")
    print(f"   2. Correlação com dados clínicos (CDR, MMSE)")
    print(f"   3. Comparação com controles saudáveis")
    print(f"   4. Análise de outras regiões cerebrais")
    print(f"   5. Aplicação de machine learning para classificação")

def save_special_cases(df: pd.DataFrame, special_cases: dict):
    """Salva casos especiais em arquivos separados"""
    print("\n💾 Salvando casos especiais...")
    
    # Salvar casos de baixo volume
    special_cases['low_volume'].to_csv('casos_volume_baixo.csv', index=False)
    print(f"✅ Casos de volume baixo salvos: casos_volume_baixo.csv ({len(special_cases['low_volume'])} casos)")
    
    # Salvar casos de alta assimetria
    special_cases['high_asymmetry'].to_csv('casos_alta_assimetria.csv', index=False)
    print(f"✅ Casos de alta assimetria salvos: casos_alta_assimetria.csv ({len(special_cases['high_asymmetry'])} casos)")
    
    # Salvar casos extremos
    special_cases['extreme_cases'].to_csv('casos_extremos.csv', index=False)
    print(f"✅ Casos extremos salvos: casos_extremos.csv ({len(special_cases['extreme_cases'])} casos)")

def main():
    """Função principal"""
    print("🧠 ANÁLISE ESTATÍSTICA COMPLETA DO HIPOCAMPO")
    print("=" * 50)
    
    # Carregar dados
    df = load_and_validate_data('hippocampus_analysis_results.csv')
    if df is None:
        return
    
    # Executar análises
    print("\n🚀 Executando análises...")
    
    # 1. Estatísticas descritivas
    generate_descriptive_statistics(df)
    
    # 2. Visualizações
    create_visualizations(df)
    
    # 3. Clustering
    df = perform_clustering_analysis(df)
    
    # 4. Identificar casos especiais
    special_cases = identify_potential_cases(df)
    
    # 5. Relatório final
    generate_summary_report(df, special_cases)
    
    # 6. Salvar resultados
    save_special_cases(df, special_cases)
    
    # Salvar dataset com clusters
    df.to_csv('hippocampus_analysis_with_clusters.csv', index=False)
    print(f"\n✅ Dataset com clusters salvo: hippocampus_analysis_with_clusters.csv")
    
    print(f"\n🎉 Análise completa!")
    print(f"📁 Arquivos gerados:")
    print(f"   - distribuicoes_hipocampo.png")
    print(f"   - correlacoes_hipocampo.png") 
    print(f"   - clustering_analise.png")
    print(f"   - casos_volume_baixo.csv")
    print(f"   - casos_alta_assimetria.csv")
    print(f"   - casos_extremos.csv")
    print(f"   - hippocampus_analysis_with_clusters.csv")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Análise Estatística das Métricas FastSurfer para Detecção de MCI
Foca especificamente na identificação dos biomarcadores mais discriminativos
entre CDR=0 (Normal) e CDR=0.5 (MCI)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class FastSurferMCIAnalyzer:
    """
    Analisador especializado em métricas FastSurfer para MCI
    """
    
    def __init__(self, dataset_path: str = "alzheimer_complete_dataset.csv"):
        """Inicializa com dataset existente"""
        self.dataset_path = dataset_path
        self.df = None
        self.mci_df = None
        self.feature_importance = None
        
    def load_and_prepare_data(self):
        """Carrega e prepara dados para análise MCI"""
        print("📊 Carregando dataset para análise FastSurfer...")
        
        # Carregar dataset existente
        self.df = pd.read_csv(self.dataset_path)
        print(f"✓ Dataset carregado: {len(self.df)} sujeitos")
        
        # Filtrar apenas Normal (CDR=0) e MCI (CDR=0.5)
        self.mci_df = self.df[self.df['cdr'].isin([0.0, 0.5])].copy()
        print(f"✓ Filtrado para estudo MCI: {len(self.mci_df)} sujeitos")
        print(f"  - Normal (CDR=0): {len(self.mci_df[self.mci_df['cdr']==0])}")
        print(f"  - MCI (CDR=0.5): {len(self.mci_df[self.mci_df['cdr']==0.5])}")
        
        return self.mci_df
    
    def identify_fastsurfer_features(self):
        """Identifica todas as métricas FastSurfer disponíveis"""
        
        # Palavras-chave para métricas FastSurfer
        fastsurfer_keywords = [
            'hippocampus', 'amygdala', 'entorhinal', 'temporal',
            'volume', 'intensity', 'asymmetry', 'brain_ratio',
            'thickness', 'surface_area', 'curvature'
        ]
        
        # Identificar colunas FastSurfer
        fastsurfer_cols = []
        for col in self.mci_df.columns:
            if any(keyword in col.lower() for keyword in fastsurfer_keywords):
                if self.mci_df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    # Verificar se tem dados suficientes (>80% válidos)
                    if self.mci_df[col].notna().sum() / len(self.mci_df) > 0.8:
                        fastsurfer_cols.append(col)
        
        print(f"🧠 Métricas FastSurfer identificadas: {len(fastsurfer_cols)}")
        for i, col in enumerate(fastsurfer_cols[:10]):
            print(f"  {i+1:2d}. {col}")
        if len(fastsurfer_cols) > 10:
            print(f"  ... e mais {len(fastsurfer_cols)-10} métricas")
        
        return fastsurfer_cols
    
    def statistical_analysis(self, fastsurfer_cols: list):
        """Análise estatística completa das métricas FastSurfer"""
        
        print("\n📈 ANÁLISE ESTATÍSTICA DETALHADA")
        print("=" * 45)
        
        # Preparar dados
        normal_data = self.mci_df[self.mci_df['cdr'] == 0]
        mci_data = self.mci_df[self.mci_df['cdr'] == 0.5]
        
        results = []
        
        print(f"Comparando {len(normal_data)} normais vs {len(mci_data)} MCI...")
        
        for feature in fastsurfer_cols:
            # Dados válidos
            normal_vals = normal_data[feature].dropna()
            mci_vals = mci_data[feature].dropna()
            
            if len(normal_vals) < 5 or len(mci_vals) < 5:
                continue
            
            # Teste t de Student
            t_stat, p_value = stats.ttest_ind(normal_vals, mci_vals)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                ((len(normal_vals)-1)*normal_vals.var() + (len(mci_vals)-1)*mci_vals.var()) / 
                (len(normal_vals) + len(mci_vals) - 2)
            )
            
            if pooled_std > 0:
                cohens_d = abs(normal_vals.mean() - mci_vals.mean()) / pooled_std
            else:
                cohens_d = 0
            
            # Mann-Whitney U test (não paramétrico)
            u_stat, u_p_value = stats.mannwhitneyu(normal_vals, mci_vals, alternative='two-sided')
            
            # Análise de classificação
            direction = 'decreased' if mci_vals.mean() < normal_vals.mean() else 'increased'
            percent_change = ((mci_vals.mean() - normal_vals.mean()) / normal_vals.mean()) * 100
            
            # Significância clínica (effect size)
            if cohens_d >= 0.8:
                clinical_significance = 'Large'
            elif cohens_d >= 0.5:
                clinical_significance = 'Medium'
            elif cohens_d >= 0.2:
                clinical_significance = 'Small'
            else:
                clinical_significance = 'Negligible'
            
            # Interpretar região anatômica
            anatomical_region = self._identify_anatomical_region(feature)
            
            results.append({
                'feature': feature,
                'anatomical_region': anatomical_region,
                'normal_mean': normal_vals.mean(),
                'normal_std': normal_vals.std(),
                'mci_mean': mci_vals.mean(),
                'mci_std': mci_vals.std(),
                'percent_change': percent_change,
                'direction': direction,
                'p_value_ttest': p_value,
                'p_value_mannwhitney': u_p_value,
                'cohens_d': cohens_d,
                'clinical_significance': clinical_significance,
                'n_normal': len(normal_vals),
                'n_mci': len(mci_vals)
            })
        
        # Criar DataFrame dos resultados
        self.feature_importance = pd.DataFrame(results)
        
        # Ordenar por significância estatística e effect size
        self.feature_importance['combined_score'] = (
            (1 / (self.feature_importance['p_value_ttest'] + 1e-10)) * 
            self.feature_importance['cohens_d']
        )
        
        self.feature_importance = self.feature_importance.sort_values(
            'combined_score', ascending=False
        )
        
        return self.feature_importance
    
    def _identify_anatomical_region(self, feature_name: str) -> str:
        """Identifica a região anatômica baseada no nome da feature"""
        
        feature_lower = feature_name.lower()
        
        if 'hippocampus' in feature_lower:
            return 'Hipocampo'
        elif 'amygdala' in feature_lower:
            return 'Amígdala'
        elif 'entorhinal' in feature_lower:
            return 'Córtex Entorrinal'
        elif 'temporal' in feature_lower:
            return 'Lobo Temporal'
        elif 'brain_ratio' in feature_lower or 'brain' in feature_lower:
            return 'Cérebro Total'
        else:
            return 'Outra'
    
    def generate_detailed_report(self):
        """Gera relatório detalhado das descobertas"""
        
        print("\n🏆 TOP 15 BIOMARCADORES MAIS DISCRIMINATIVOS PARA MCI")
        print("=" * 65)
        
        top_features = self.feature_importance.head(15)
        
        for i, row in top_features.iterrows():
            significance_marker = "***" if row['p_value_ttest'] < 0.001 else "**" if row['p_value_ttest'] < 0.01 else "*" if row['p_value_ttest'] < 0.05 else ""
            
            print(f"\n{len(top_features) - list(top_features.index).index(i):2d}. {row['feature']} {significance_marker}")
            print(f"    Região: {row['anatomical_region']}")
            print(f"    Normal: {row['normal_mean']:.2f} ± {row['normal_std']:.2f}")
            print(f"    MCI:    {row['mci_mean']:.2f} ± {row['mci_std']:.2f}")
            print(f"    Mudança: {row['percent_change']:+.1f}% ({row['direction']})")
            print(f"    p-value: {row['p_value_ttest']:.4f}")
            print(f"    Effect size: {row['cohens_d']:.3f} ({row['clinical_significance']})")
        
        # Análise por região anatômica
        print(f"\n🧭 ANÁLISE POR REGIÃO ANATÔMICA")
        print("=" * 35)
        
        region_analysis = self.feature_importance.groupby('anatomical_region').agg({
            'cohens_d': ['mean', 'max', 'count'],
            'p_value_ttest': lambda x: (x < 0.05).sum()
        }).round(3)
        
        for region in region_analysis.index:
            n_features = region_analysis.loc[region, ('cohens_d', 'count')]
            n_significant = region_analysis.loc[region, ('p_value_ttest', '<lambda>')]
            avg_effect = region_analysis.loc[region, ('cohens_d', 'mean')]
            max_effect = region_analysis.loc[region, ('cohens_d', 'max')]
            
            print(f"{region}:")
            print(f"  Features analisadas: {n_features}")
            print(f"  Significativas (p<0.05): {n_significant}")
            print(f"  Effect size médio: {avg_effect:.3f}")
            print(f"  Effect size máximo: {max_effect:.3f}")
        
        # Recomendações clínicas
        print(f"\n💡 RECOMENDAÇÕES CLÍNICAS BASEADAS EM EVIDÊNCIAS")
        print("=" * 50)
        
        # Identificar features mais importantes
        critical_features = self.feature_importance[
            (self.feature_importance['p_value_ttest'] < 0.05) & 
            (self.feature_importance['cohens_d'] > 0.3)
        ]
        
        print(f"1. BIOMARCADORES CRÍTICOS PARA TRIAGEM:")
        for _, row in critical_features.head(5).iterrows():
            print(f"   • {row['anatomical_region']}: {row['percent_change']:+.1f}% em MCI")
        
        print(f"\n2. PROTOCOLO DE NEUROIMAGEM RECOMENDADO:")
        regions_affected = critical_features['anatomical_region'].unique()
        for region in regions_affected:
            print(f"   • {region}: Volumetria + análise de intensidade")
        
        print(f"\n3. CRITÉRIOS DE ALERTA PARA MCI:")
        severe_changes = critical_features[abs(critical_features['percent_change']) > 5]
        for _, row in severe_changes.head(3).iterrows():
            threshold_direction = "inferior" if row['direction'] == 'decreased' else "superior"
            change_magnitude = abs(row['percent_change'])
            print(f"   • {row['anatomical_region']}: Variação {threshold_direction} a {change_magnitude:.1f}%")
    
    def create_comprehensive_visualizations(self):
        """Cria visualizações abrangentes"""
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # 1. Top 10 features por effect size
        top_features = self.feature_importance.head(10)
        
        y_pos = np.arange(len(top_features))
        bars = axes[0, 0].barh(y_pos, top_features['cohens_d'], 
                               color=['red' if p < 0.05 else 'orange' for p in top_features['p_value_ttest']])
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels([feat.replace('_', ' ').title()[:20] for feat in top_features['feature']])
        axes[0, 0].set_xlabel("Cohen's d (Effect Size)")
        axes[0, 0].set_title("Top 10 Biomarcadores por Effect Size")
        axes[0, 0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium Effect')
        axes[0, 0].axvline(x=0.8, color='black', linestyle='--', alpha=0.5, label='Large Effect')
        axes[0, 0].legend()
        
        # 2. Distribuição de p-values
        axes[0, 1].hist(self.feature_importance['p_value_ttest'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 1].axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
        axes[0, 1].axvline(x=0.01, color='darkred', linestyle='--', label='p=0.01')
        axes[0, 1].set_xlabel('p-value')
        axes[0, 1].set_ylabel('Frequência')
        axes[0, 1].set_title('Distribuição de Significância Estatística')
        axes[0, 1].legend()
        
        # 3. Effect size por região anatômica
        region_effects = self.feature_importance.groupby('anatomical_region')['cohens_d'].mean().sort_values(ascending=False)
        region_effects.plot(kind='bar', ax=axes[0, 2], color='lightcoral')
        axes[0, 2].set_title('Effect Size Médio por Região Anatômica')
        axes[0, 2].set_ylabel("Cohen's d Médio")
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Volcano plot (Effect size vs p-value)
        x = -np.log10(self.feature_importance['p_value_ttest'] + 1e-10)
        y = self.feature_importance['cohens_d']
        
        colors = ['red' if (p < 0.05 and d > 0.3) else 'orange' if p < 0.05 else 'gray' 
                 for p, d in zip(self.feature_importance['p_value_ttest'], self.feature_importance['cohens_d'])]
        
        scatter = axes[1, 0].scatter(x, y, c=colors, alpha=0.6)
        axes[1, 0].axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(x=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('-log10(p-value)')
        axes[1, 0].set_ylabel("Cohen's d")
        axes[1, 0].set_title('Volcano Plot: Significância vs Effect Size')
        
        # Anotar top features
        for i, row in top_features.head(5).iterrows():
            x_pos = -np.log10(row['p_value_ttest'] + 1e-10)
            y_pos = row['cohens_d']
            axes[1, 0].annotate(row['feature'][:15], (x_pos, y_pos), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 5. Comparação Normal vs MCI para top feature
        top_feature = top_features.iloc[0]['feature']
        normal_vals = self.mci_df[self.mci_df['cdr'] == 0][top_feature].dropna()
        mci_vals = self.mci_df[self.mci_df['cdr'] == 0.5][top_feature].dropna()
        
        axes[1, 1].hist([normal_vals, mci_vals], bins=15, alpha=0.7, 
                       label=['Normal', 'MCI'], color=['blue', 'red'])
        axes[1, 1].set_xlabel(top_feature.replace('_', ' ').title())
        axes[1, 1].set_ylabel('Frequência')
        axes[1, 1].set_title(f'Distribuição: {top_feature}')
        axes[1, 1].legend()
        
        # 6. Heatmap de correlações entre top features
        top_feature_names = top_features.head(8)['feature'].tolist()
        corr_matrix = self.mci_df[top_feature_names].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   xticklabels=[f[:10] for f in top_feature_names],
                   yticklabels=[f[:10] for f in top_feature_names],
                   ax=axes[1, 2])
        axes[1, 2].set_title('Correlações entre Top Biomarcadores')
        
        # 7. Boxplot comparativo por região
        regions_to_plot = ['Hipocampo', 'Córtex Entorrinal', 'Amígdala']
        region_data = []
        region_labels = []
        
        for region in regions_to_plot:
            region_features = self.feature_importance[
                self.feature_importance['anatomical_region'] == region
            ]['feature'].tolist()
            
            if region_features:
                # Pegar primeira feature significativa da região
                feature = region_features[0]
                normal_vals = self.mci_df[self.mci_df['cdr'] == 0][feature].dropna()
                mci_vals = self.mci_df[self.mci_df['cdr'] == 0.5][feature].dropna()
                
                region_data.extend([normal_vals.tolist(), mci_vals.tolist()])
                region_labels.extend([f'{region}\nNormal', f'{region}\nMCI'])
        
        if region_data:
            bp = axes[2, 0].boxplot(region_data, labels=region_labels, patch_artist=True)
            colors = ['lightblue', 'lightcoral'] * (len(region_data)//2)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            axes[2, 0].set_title('Comparação por Região Anatômica')
            axes[2, 0].tick_params(axis='x', rotation=45)
        
        # 8. Mudança percentual por região
        region_changes = self.feature_importance.groupby('anatomical_region')['percent_change'].mean().sort_values()
        bars = axes[2, 1].bar(range(len(region_changes)), region_changes.values, 
                             color=['red' if x < 0 else 'green' for x in region_changes.values])
        axes[2, 1].set_xticks(range(len(region_changes)))
        axes[2, 1].set_xticklabels(region_changes.index, rotation=45)
        axes[2, 1].set_ylabel('Mudança Média (%)')
        axes[2, 1].set_title('Mudança Percentual por Região')
        axes[2, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 9. Resumo estatístico
        axes[2, 2].axis('off')
        
        n_significant = len(self.feature_importance[self.feature_importance['p_value_ttest'] < 0.05])
        n_large_effect = len(self.feature_importance[self.feature_importance['cohens_d'] > 0.8])
        n_medium_effect = len(self.feature_importance[self.feature_importance['cohens_d'] > 0.5])
        
        summary_text = f"""
RESUMO ESTATÍSTICO

Total de métricas analisadas: {len(self.feature_importance)}

Significância estatística:
• p < 0.05: {n_significant} métricas
• p < 0.01: {len(self.feature_importance[self.feature_importance['p_value_ttest'] < 0.01])} métricas

Effect size clínico:
• Large (>0.8): {n_large_effect} métricas
• Medium (>0.5): {n_medium_effect} métricas
• Small (>0.2): {len(self.feature_importance[self.feature_importance['cohens_d'] > 0.2])} métricas

Regiões mais afetadas:
{chr(10).join([f"• {region}: {count} métricas" for region, count in self.feature_importance['anatomical_region'].value_counts().head(3).items()])}

Biomarcador mais discriminativo:
• {top_features.iloc[0]['feature']}
• Effect size: {top_features.iloc[0]['cohens_d']:.3f}
• p-value: {top_features.iloc[0]['p_value_ttest']:.4f}
        """
        
        axes[2, 2].text(0.1, 0.9, summary_text, transform=axes[2, 2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle('Análise Abrangente de Biomarcadores FastSurfer para MCI', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('fastsurfer_mci_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualizações salvas: fastsurfer_mci_comprehensive_analysis.png")
    
    def save_results(self):
        """Salva resultados da análise"""
        
        # Salvar análise completa
        self.feature_importance.to_csv('fastsurfer_mci_statistical_analysis.csv', index=False)
        
        # Salvar apenas features significativas
        significant_features = self.feature_importance[self.feature_importance['p_value_ttest'] < 0.05]
        significant_features.to_csv('fastsurfer_mci_significant_features.csv', index=False)
        
        # Salvar top biomarcadores para uso clínico
        top_biomarkers = self.feature_importance.head(10)
        top_biomarkers[['feature', 'anatomical_region', 'percent_change', 'p_value_ttest', 'cohens_d']].to_csv(
            'fastsurfer_mci_top_biomarkers.csv', index=False
        )
        
        print("✓ Resultados salvos:")
        print("  - fastsurfer_mci_statistical_analysis.csv (análise completa)")
        print("  - fastsurfer_mci_significant_features.csv (features significativas)")
        print("  - fastsurfer_mci_top_biomarkers.csv (top biomarcadores)")

def main():
    """Função principal da análise FastSurfer para MCI"""
    
    print("🧠 ANÁLISE ESTATÍSTICA FASTSURFER PARA DETECÇÃO DE MCI")
    print("=" * 65)
    print("Identificação de biomarcadores discriminativos entre Normal e MCI")
    print("Baseado em métricas de volume, intensidade e assimetria cerebral")
    print("=" * 65)
    
    # Verificar se dataset existe
    import os
    if not os.path.exists("alzheimer_complete_dataset.csv"):
        print("❌ ERRO: Dataset não encontrado!")
        print("Execute primeiro: python3 alzheimer_cnn_pipeline_improved.py")
        return
    
    # Inicializar análise
    analyzer = FastSurferMCIAnalyzer()
    
    # Carregar dados
    mci_data = analyzer.load_and_prepare_data()
    
    if len(mci_data) < 10:
        print("❌ ERRO: Dados insuficientes para análise MCI!")
        return
    
    # Identificar métricas FastSurfer
    fastsurfer_features = analyzer.identify_fastsurfer_features()
    
    if len(fastsurfer_features) < 5:
        print("❌ ERRO: Métricas FastSurfer insuficientes!")
        return
    
    # Análise estatística
    feature_analysis = analyzer.statistical_analysis(fastsurfer_features)
    
    # Relatório detalhado
    analyzer.generate_detailed_report()
    
    # Visualizações
    analyzer.create_comprehensive_visualizations()
    
    # Salvar resultados
    analyzer.save_results()
    
    print(f"\n🎉 ANÁLISE CONCLUÍDA COM SUCESSO!")
    print(f"📊 {len(feature_analysis)} métricas FastSurfer analisadas")
    print(f"📈 {len(feature_analysis[feature_analysis['p_value_ttest'] < 0.05])} biomarcadores estatisticamente significativos")
    print(f"🏥 {len(feature_analysis[feature_analysis['cohens_d'] > 0.5])} com relevância clínica (effect size > 0.5)")
    
    # Próximos passos sugeridos
    print(f"\n💡 PRÓXIMOS PASSOS RECOMENDADOS:")
    print(f"1. Validar achados em coorte independente")
    print(f"2. Criar score composto com top biomarcadores")
    print(f"3. Integrar com dados longitudinais")
    print(f"4. Desenvolver protocolo de neuroimagem otimizado")

if __name__ == "__main__":
    main() 
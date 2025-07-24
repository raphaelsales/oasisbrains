#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISE APROFUNDADA PARA DIAGNÓSTICO PRECOCE DE ALZHEIMER
Foco: Comprometimento Cognitivo Leve (MCI) e Biomarcadores Neuroimagem

Autor: Sistema de IA - Pipeline Alzheimer
Data: 2025
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Carrega e prepara os dados para análise"""
    df = pd.read_csv('alzheimer_complete_dataset.csv')
    
    # Criar categorias de estágio
    df['stage'] = df['cdr'].map({
        0.0: 'Normal',
        0.5: 'MCI',           # Comprometimento Cognitivo Leve
        1.0: 'Mild_AD',       # Alzheimer Leve
        2.0: 'Moderate_AD'    # Alzheimer Moderado
    })
    
    # Criar grupos de risco
    df['risk_group'] = 'Unknown'
    df.loc[df['cdr'] == 0.0, 'risk_group'] = 'Low_Risk'
    df.loc[df['cdr'] == 0.5, 'risk_group'] = 'High_Risk_MCI'
    df.loc[df['cdr'] >= 1.0, 'risk_group'] = 'Established_AD'
    
    return df

def clinical_staging_analysis(df):
    """Análise detalhada dos estágios clínicos"""
    print("ANALISE DE ESTAGIOS CLINICOS")
    print("=" * 60)
    
    # Distribuição por estágio
    stage_counts = df['stage'].value_counts()
    total = len(df)
    
    print("DISTRIBUICAO POR ESTAGIO:")
    print("-" * 40)
    for stage, count in stage_counts.items():
        pct = (count/total)*100
        bar = "█" * int(pct/2)
        print(f"{stage:>12}: {bar:<25} {count:3d} ({pct:5.1f}%)")
    
    print()
    
    # Análise demográfica por estágio
    print("PERFIL DEMOGRAFICO POR ESTAGIO:")
    print("-" * 45)
    demo_analysis = df.groupby('stage').agg({
        'age': ['mean', 'std', 'min', 'max'],
        'mmse': ['mean', 'std'],
        'education': ['mean', 'std'],
        'gender': lambda x: (x == 'F').sum() / len(x) * 100  # % feminino
    }).round(2)
    
    for stage in stage_counts.index:
        stage_data = df[df['stage'] == stage]
        print(f"\n{stage.upper()}:")
        print(f"   Sujeitos: {len(stage_data)}")
        print(f"   Idade: {stage_data['age'].mean():.1f} +/- {stage_data['age'].std():.1f} anos")
        print(f"   MMSE: {stage_data['mmse'].mean():.1f} +/- {stage_data['mmse'].std():.1f}")
        print(f"   Educacao: {stage_data['education'].mean():.1f} +/- {stage_data['education'].std():.1f} anos")
        print(f"   Feminino: {(stage_data['gender'] == 'F').sum()}/{len(stage_data)} ({(stage_data['gender'] == 'F').mean()*100:.1f}%)")

def mci_deep_analysis(df):
    """Análise aprofundada do Comprometimento Cognitivo Leve"""
    print("\n\nANALISE APROFUNDADA - COMPROMETIMENTO COGNITIVO LEVE (MCI)")
    print("=" * 70)
    
    # Filtrar dados MCI
    mci_data = df[df['stage'] == 'MCI'].copy()
    normal_data = df[df['stage'] == 'Normal'].copy()
    ad_data = df[df['cdr'] >= 1.0].copy()
    
    print(f"POPULACAO MCI: {len(mci_data)} sujeitos (CDR = 0.5)")
    print(f"   Representa {len(mci_data)/len(df)*100:.1f}% da populacao total")
    print()
    
    # Características clínicas do MCI
    print("CARACTERISTICAS CLINICAS MCI:")
    print("-" * 40)
    print(f"   Idade media: {mci_data['age'].mean():.1f} +/- {mci_data['age'].std():.1f} anos")
    print(f"   MMSE medio: {mci_data['mmse'].mean():.1f} +/- {mci_data['mmse'].std():.1f}")
    print(f"      Range MMSE: {mci_data['mmse'].min():.1f} - {mci_data['mmse'].max():.1f}")
    print(f"   Educacao: {mci_data['education'].mean():.1f} +/- {mci_data['education'].std():.1f} anos")
    print(f"   Genero feminino: {(mci_data['gender'] == 'F').mean()*100:.1f}%")
    
    # Comparação MCI vs Normal vs AD
    print("\nCOMPARACAO ENTRE GRUPOS:")
    print("-" * 35)
    comparison_metrics = ['age', 'mmse', 'education']
    
    for metric in comparison_metrics:
        normal_mean = normal_data[metric].mean()
        mci_mean = mci_data[metric].mean()
        ad_mean = ad_data[metric].mean()
        
        print(f"\n{metric.upper()}:")
        print(f"   Normal:    {normal_mean:6.1f}")
        print(f"   MCI:       {mci_mean:6.1f} ({((mci_mean-normal_mean)/normal_mean)*100:+5.1f}%)")
        print(f"   AD:        {ad_mean:6.1f} ({((ad_mean-normal_mean)/normal_mean)*100:+5.1f}%)")

def neuroimaging_biomarkers_analysis(df):
    """Análise de biomarcadores de neuroimagem para diagnóstico precoce"""
    print("\n\nBIOMARCADORES NEUROIMAGEM - DIAGNOSTICO PRECOCE")
    print("=" * 65)
    
    # Identificar features de neuroimagem
    neuro_features = [col for col in df.columns if any(term in col for term in 
                     ['hippocampus', 'amygdala', 'entorhinal', 'temporal']) and 'volume' in col]
    
    # Remover features normalizadas (que são 0)
    neuro_features = [f for f in neuro_features if not f.endswith('_norm')]
    
    print(f"ANALISE DE {len(neuro_features)} BIOMARCADORES VOLUMETRICOS")
    print()
    
    # Calcular poder discriminativo para cada feature
    discriminative_power = []
    
    normal_data = df[df['stage'] == 'Normal']
    mci_data = df[df['stage'] == 'MCI']
    ad_data = df[df['cdr'] >= 1.0]
    
    print("PODER DISCRIMINATIVO NORMAL -> MCI -> AD:")
    print("-" * 55)
    print(f"{'Regiao':30s} {'Normal->MCI':>12s} {'MCI->AD':>10s} {'Total':>8s}")
    print("-" * 55)
    
    for feature in neuro_features:
        if feature in df.columns and df[feature].notna().sum() > 100:
            normal_mean = normal_data[feature].mean()
            mci_mean = mci_data[feature].mean()
            ad_mean = ad_data[feature].mean()
            
            # Calcular diferenças percentuais
            normal_to_mci = ((mci_mean - normal_mean) / normal_mean) * 100
            mci_to_ad = ((ad_mean - mci_mean) / mci_mean) * 100 if mci_mean != 0 else 0
            total_change = ((ad_mean - normal_mean) / normal_mean) * 100
            
            # Armazenar para ranking
            discriminative_power.append({
                'feature': feature,
                'normal_to_mci': abs(normal_to_mci),
                'mci_to_ad': abs(mci_to_ad),
                'total_change': abs(total_change),
                'normal_mean': normal_mean,
                'mci_mean': mci_mean,
                'ad_mean': ad_mean
            })
            
            # Formatar nome da região
            region_name = feature.replace('_volume', '').replace('_', ' ').title()
            print(f"{region_name:30s} {normal_to_mci:+8.1f}% {mci_to_ad:+8.1f}% {total_change:+6.1f}%")
    
    # Ranking dos melhores biomarcadores
    discriminative_power.sort(key=lambda x: x['normal_to_mci'], reverse=True)
    
    print(f"\nTOP 5 BIOMARCADORES PARA DETECCAO PRECOCE (Normal -> MCI):")
    print("-" * 60)
    for i, marker in enumerate(discriminative_power[:5], 1):
        region = marker['feature'].replace('_volume', '').replace('_', ' ').title()
        change = marker['normal_to_mci']
        print(f"{i}. {region:25s}: {change:5.1f}% alteracao")
    
    return discriminative_power

def cognitive_correlation_analysis(df):
    """Análise de correlações cognitivas"""
    print("\n\nANALISE DE CORRELACOES COGNITIVAS")
    print("=" * 50)
    
    # Correlações importantes
    cognitive_vars = ['mmse', 'cdr', 'age', 'education']
    correlations = df[cognitive_vars].corr()
    
    print("MATRIZ DE CORRELACOES:")
    print("-" * 35)
    print(f"{'':8s}", end="")
    for var in cognitive_vars:
        print(f"{var:>8s}", end="")
    print()
    print("-" * 41)
    
    for i, var1 in enumerate(cognitive_vars):
        print(f"{var1:8s}", end="")
        for var2 in cognitive_vars:
            corr = correlations.loc[var1, var2]
            print(f"{corr:8.3f}", end="")
        print()
    
    print("\nINSIGHTS CLINICOS:")
    print("-" * 25)
    mmse_cdr_corr = correlations.loc['mmse', 'cdr']
    age_mmse_corr = correlations.loc['age', 'mmse']
    education_mmse_corr = correlations.loc['education', 'mmse']
    
    print(f"- MMSE <-> CDR: {mmse_cdr_corr:.3f} (correlacao {'forte' if abs(mmse_cdr_corr) > 0.7 else 'moderada' if abs(mmse_cdr_corr) > 0.3 else 'fraca'})")
    print(f"- Idade <-> MMSE: {age_mmse_corr:.3f} (impacto {'significativo' if abs(age_mmse_corr) > 0.3 else 'moderado' if abs(age_mmse_corr) > 0.1 else 'minimo'})")
    print(f"- Educacao <-> MMSE: {education_mmse_corr:.3f} (efeito protetor {'forte' if education_mmse_corr > 0.3 else 'moderado' if education_mmse_corr > 0.1 else 'fraco'})")

def early_detection_risk_model(df):
    """Modelo de risco para detecção precoce"""
    print("\n\nMODELO DE RISCO - DETECCAO PRECOCE")
    print("=" * 50)
    
    # Definir limiares baseados nos dados
    mci_data = df[df['stage'] == 'MCI']
    normal_data = df[df['stage'] == 'Normal']
    
    # Calcular limiares
    mmse_threshold = normal_data['mmse'].quantile(0.1)  # 10% mais baixos
    age_threshold = normal_data['age'].quantile(0.9)     # 10% mais velhos
    
    print("FATORES DE RISCO IDENTIFICADOS:")
    print("-" * 40)
    
    # Análise de fatores de risco
    risk_factors = {
        'Idade > 80 anos': (df['age'] > 80).sum(),
        f'MMSE < {mmse_threshold:.1f}': (df['mmse'] < mmse_threshold).sum(),
        'Genero feminino': (df['gender'] == 'F').sum(),
        'Baixa escolaridade (< 12 anos)': (df['education'] < 12).sum(),
    }
    
    for factor, count in risk_factors.items():
        prevalence = count / len(df) * 100
        print(f"- {factor:25s}: {count:3d} casos ({prevalence:4.1f}%)")
    
    # Calcular risco composto
    print(f"\nANALISE DE RISCO COMPOSTO:")
    print("-" * 35)
    
    # Criar score de risco simples
    df['risk_score'] = 0
    df.loc[df['age'] > 75, 'risk_score'] += 1
    df.loc[df['mmse'] < 28, 'risk_score'] += 1
    df.loc[df['gender'] == 'F', 'risk_score'] += 1
    df.loc[df['education'] < 12, 'risk_score'] += 1
    
    # Analisar por score de risco
    for score in range(5):
        score_data = df[df['risk_score'] == score]
        if len(score_data) > 0:
            mci_rate = (score_data['stage'] == 'MCI').mean() * 100
            ad_rate = (score_data['cdr'] >= 1.0).mean() * 100
            print(f"Score {score}: {len(score_data):3d} sujeitos | MCI: {mci_rate:4.1f}% | AD: {ad_rate:4.1f}%")

def clinical_recommendations(df, discriminative_power):
    """Gerar recomendações clínicas"""
    print("\n\nRECOMENDACOES CLINICAS - DIAGNOSTICO PRECOCE")
    print("=" * 60)
    
    mci_data = df[df['stage'] == 'MCI']
    
    print("PROTOCOLO SUGERIDO PARA TRIAGEM:")
    print("-" * 45)
    print("1. AVALIACAO COGNITIVA:")
    print(f"   - MMSE < {mci_data['mmse'].quantile(0.75):.1f}: Investigacao adicional")
    print(f"   - CDR = 0.5: Alto risco para progressao")
    print()
    
    print("2. BIOMARCADORES NEUROIMAGEM PRIORITARIOS:")
    top_markers = discriminative_power[:3]
    for i, marker in enumerate(top_markers, 1):
        region = marker['feature'].replace('_volume', '').replace('_', ' ').title()
        print(f"   {i}. {region}: {marker['normal_to_mci']:.1f}% reducao no MCI")
    print()
    
    print("3. PERFIL DE RISCO ALTO:")
    high_risk_age = df[df['stage'] == 'MCI']['age'].mean()
    print(f"   - Idade > {high_risk_age:.0f} anos")
    print(f"   - MMSE entre {mci_data['mmse'].min():.0f}-{mci_data['mmse'].max():.0f}")
    print(f"   - Genero feminino (maior prevalencia)")
    print()
    
    print("4. MONITORAMENTO LONGITUDINAL:")
    print("   - Avaliacao semestral para CDR 0.5")
    print("   - Neuroimagem anual")
    print("   - Biomarcadores LCR/sangue (se disponivel)")
    print()
    
    print("5. SINAIS DE ALERTA PARA PROGRESSAO:")
    print(f"   - Declinio MMSE > 2 pontos/ano")
    print(f"   - Reducao hipocampo > 2%/ano")
    print(f"   - Surgimento de CDR >= 1.0")

def generate_summary_report(df, discriminative_power):
    """Gerar relatório executivo"""
    print("\n\nRELATORIO EXECUTIVO - DIAGNOSTICO PRECOCE ALZHEIMER")
    print("=" * 65)
    
    total_subjects = len(df)
    mci_subjects = len(df[df['stage'] == 'MCI'])
    ad_subjects = len(df[df['cdr'] >= 1.0])
    
    print(f"POPULACAO ESTUDADA: {total_subjects} sujeitos")
    print(f"   MCI (CDR 0.5): {mci_subjects} ({mci_subjects/total_subjects*100:.1f}%)")
    print(f"   Alzheimer: {ad_subjects} ({ad_subjects/total_subjects*100:.1f}%)")
    print()
    
    print("PRINCIPAIS DESCOBERTAS:")
    print("-" * 30)
    print("1. Biomarcadores mais sensiveis para MCI:")
    for i, marker in enumerate(discriminative_power[:3], 1):
        region = marker['feature'].replace('_volume', '').replace('_', ' ').title()
        print(f"   {i}. {region}: {marker['normal_to_mci']:.1f}% alteracao")
    
    print()
    mmse_mci = df[df['stage'] == 'MCI']['mmse'].mean()
    print(f"2. MMSE no MCI: {mmse_mci:.1f} +/- {df[df['stage'] == 'MCI']['mmse'].std():.1f}")
    print(f"   Limiar sugerido para triagem: < {mmse_mci + df[df['stage'] == 'MCI']['mmse'].std():.1f}")
    
    print()
    age_mci = df[df['stage'] == 'MCI']['age'].mean()
    print(f"3. Idade de risco: > {age_mci:.0f} anos")
    
    print()
    female_mci_rate = (df[df['gender'] == 'F']['stage'] == 'MCI').mean() * 100
    male_mci_rate = (df[df['gender'] == 'M']['stage'] == 'MCI').mean() * 100
    print(f"4. Prevalencia MCI: Feminino {female_mci_rate:.1f}% vs Masculino {male_mci_rate:.1f}%")
    
    print("\nIMPACTO CLINICO:")
    print("-" * 20)
    print("- Deteccao precoce permite intervencao terapeutica")
    print("- Biomarcadores identificam 'janela de oportunidade'")
    print("- Protocolo pode reduzir progressao MCI -> AD")
    print("- Otimizacao de recursos diagnosticos")

def main():
    """Função principal - Análise completa"""
    print("SISTEMA DE ANALISE PARA DIAGNOSTICO PRECOCE DE ALZHEIMER")
    print("=" * 70)
    print("Foco: Comprometimento Cognitivo Leve (MCI) e Biomarcadores")
    print("Data:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    # Carregar dados
    print("Carregando dados...")
    df = load_data()
    print(f"{len(df)} sujeitos carregados com {df.shape[1]} features")
    
    # Executar análises
    clinical_staging_analysis(df)
    mci_deep_analysis(df)
    discriminative_power = neuroimaging_biomarkers_analysis(df)
    cognitive_correlation_analysis(df)
    early_detection_risk_model(df)
    clinical_recommendations(df, discriminative_power)
    generate_summary_report(df, discriminative_power)
    
    print(f"\nANALISE CONCLUIDA!")
    print("=" * 50)
    print("Use este relatorio para:")
    print("   - Protocolo de triagem clinica")
    print("   - Selecao de biomarcadores")
    print("   - Estratificacao de risco")
    print("   - Monitoramento longitudinal")

if __name__ == "__main__":
    main() 
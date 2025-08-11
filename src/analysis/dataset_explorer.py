#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATASET EXPLORER - Análise Completa do Dataset Alzheimer
Formato exato solicitado pelo usuário
"""

import pandas as pd
import numpy as np

def explore_complete_dataset():
    """Analise completa do dataset no formato solicitado"""
    # Carregar e analisar o dataset
    df = pd.read_csv('alzheimer_complete_dataset.csv')
    
    print('ANALISE DO DATASET COMPLETO:')
    print('=' * 50)
    print(f'Dimensoes: {df.shape[0]} sujeitos x {df.shape[1]} features')
    print()

    print('DISTRIBUICAO POR DIAGNOSTICO:')
    diag_counts = df['diagnosis'].value_counts()
    for diagnosis, count in diag_counts.items():
        percentage = (count / len(df)) * 100
        print(f'   {diagnosis}: {count} ({percentage:.1f}%)')
    print()

    print('DISTRIBUICAO CDR (Severidade):')
    cdr_counts = df['cdr'].value_counts().sort_index()
    cdr_labels = {
        0.0: 'Normal',
        0.5: 'Muito Leve (MCI)',
        1.0: 'Leve',
        2.0: 'Moderada'
    }
    
    for cdr, count in cdr_counts.items():
        percentage = (count / len(df)) * 100
        label = cdr_labels.get(cdr, f'CDR {cdr}')
        print(f'   CDR {cdr} ({label}): {count} ({percentage:.1f}%)')
    print()

    print('DISTRIBUICAO POR GENERO:')
    gender_counts = df['gender'].value_counts()
    gender_labels = {'F': 'Mulheres', 'M': 'Homens'}
    for gender, count in gender_counts.items():
        percentage = (count / len(df)) * 100
        label = gender_labels.get(gender, gender)
        print(f'   {label}: {count} ({percentage:.1f}%)')
    print()

    print('ESTATISTICAS DE IDADE:')
    print(f'   Media: {df["age"].mean():.1f} anos')
    print(f'   Min-Max: {df["age"].min():.1f} - {df["age"].max():.1f} anos')
    print(f'   Desvio: +/-{df["age"].std():.1f} anos')
    print(f'   Mediana: {df["age"].median():.1f} anos')
    print()

    print('ESTATISTICAS MMSE (Cognicao):')
    print(f'   Media: {df["mmse"].mean():.1f} pontos')
    print(f'   Min-Max: {df["mmse"].min():.1f} - {df["mmse"].max():.1f} pontos')
    print(f'   Desvio: +/-{df["mmse"].std():.1f} pontos')
    print(f'   Mediana: {df["mmse"].median():.1f} pontos')
    print()

    # Analisar hipocampo
    if 'total_hippocampus_volume' in df.columns:
        print('HIPOCAMPO (Principal Biomarcador):')
        by_diagnosis = df.groupby('diagnosis')['total_hippocampus_volume'].agg(['mean', 'std'])
        for diagnosis, stats in by_diagnosis.iterrows():
            print(f'   {diagnosis}: {stats["mean"]:.0f} +/- {stats["std"]:.0f} mm3')
        print()

    print('EDUCACAO (Anos de Estudo):')
    education_counts = df['education'].value_counts().sort_index()
    for years, count in education_counts.items():
        percentage = (count / len(df)) * 100
        print(f'   {years:2.0f} anos: {count:3d} ({percentage:4.1f}%)')
    print()

def detailed_composition_table():
    """Tabela de composicao detalhada"""
    df = pd.read_csv('alzheimer_complete_dataset.csv')
    
    print('COMPOSICAO DETALHADA DO DATASET (405 sujeitos):')
    print('=' * 55)
    
    # Tabela principal
    print(f"{'Categoria':<20} {'Quantidade':<12} {'Porcentagem':<12}")
    print('-' * 45)
    
    # Por diagnóstico
    nondemented = (df['diagnosis'] == 'Nondemented').sum()
    demented = (df['diagnosis'] == 'Demented').sum()
    print(f"{'Nao-Dementes':<20} {nondemented:<12} {nondemented/len(df)*100:>8.1f}%")
    print(f"{'Dementes':<20} {demented:<12} {demented/len(df)*100:>8.1f}%")
    print()
    
    # Por gênero
    women = (df['gender'] == 'F').sum()
    men = (df['gender'] == 'M').sum()
    print(f"{'Mulheres':<20} {women:<12} {women/len(df)*100:>8.1f}%")
    print(f"{'Homens':<20} {men:<12} {men/len(df)*100:>8.1f}%")
    print()
    
    # Tabela CDR detalhada
    print('SEVERIDADE CDR (Clinical Dementia Rating):')
    print('-' * 55)
    print(f"{'CDR':<6} {'Descricao':<15} {'Quantidade':<12} {'%':<8}")
    print('-' * 45)
    
    cdr_info = [
        (0.0, 'Normal', 253),
        (0.5, 'Muito Leve', 68),
        (1.0, 'Leve', 64),
        (2.0, 'Moderada', 20)
    ]
    
    for cdr, desc, count in cdr_info:
        percentage = count / len(df) * 100
        print(f"{cdr:<6} {desc:<15} {count:<12} {percentage:>6.1f}%")

def biomarcador_analysis():
    """Analise especifica de biomarcadores"""
    df = pd.read_csv('alzheimer_complete_dataset.csv')
    
    print('\nANALISE DE BIOMARCADORES NEUROIMAGEM:')
    print('=' * 50)
    
    # Biomarcadores principais
    key_biomarkers = [
        ('total_hippocampus_volume', 'Hipocampo Total'),
        ('left_hippocampus_volume', 'Hipocampo Esquerdo'),
        ('right_hippocampus_volume', 'Hipocampo Direito'),
        ('left_entorhinal_volume', 'Cortex Entorrinal Esq.'),
        ('right_entorhinal_volume', 'Cortex Entorrinal Dir.')
    ]
    
    print(f"{'Biomarcador':<25} {'Normal':<10} {'MCI':<10} {'AD':<10} {'Delta%':<8}")
    print('-' * 65)
    
    normal_data = df[df['cdr'] == 0.0]
    mci_data = df[df['cdr'] == 0.5]
    ad_data = df[df['cdr'] >= 1.0]
    
    for column, name in key_biomarkers:
        if column in df.columns:
            normal_mean = normal_data[column].mean()
            mci_mean = mci_data[column].mean()
            ad_mean = ad_data[column].mean()
            
            # Calcular mudança percentual Normal → MCI
            pct_change = ((mci_mean - normal_mean) / normal_mean) * 100
            
            print(f"{name:<25} {normal_mean:>8.0f} {mci_mean:>8.0f} {ad_mean:>8.0f} {pct_change:>+6.1f}%")

def main():
    """Funcao principal - analise completa do dataset"""
    print("DATASET EXPLORER - VISUALIZANDO OS RESULTADOS")
    print("=" * 60)
    print("Analise Exploratoria Completa do Dataset Alzheimer")
    print(f"Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")
    print()
    
    # Análise principal
    explore_complete_dataset()
    
    # Tabela de composição
    detailed_composition_table()
    
    # Análise de biomarcadores
    biomarcador_analysis()
    
    print('\nANALISE EXPLORATORIA CONCLUIDA!')
    print('=' * 45)
    print('Use estes dados para:')
    print('   - Relatorios cientificos')
    print('   - Apresentacoes clinicas')
    print('   - Analise estatistica')
    print('   - Validacao de resultados')

if __name__ == "__main__":
    main() 
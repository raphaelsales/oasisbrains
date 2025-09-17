#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para abrir e processar dados do DATASUS usando PySUS
Arquivo: DNTO2023.dbc (Declarações de Nascidos Vivos de 2023)
"""

import pandas as pd
import pyreaddbc
from dbfread import DBF
import os
import tempfile

def abrir_dados_datasus(caminho_arquivo):
    """
    Abre e processa arquivos .dbc do DATASUS
    
    Args:
        caminho_arquivo (str): Caminho para o arquivo .dbc
        
    Returns:
        pandas.DataFrame: DataFrame com os dados processados
    """
    try:
        # Verificar se o arquivo existe
        if not os.path.exists(caminho_arquivo):
            print(f"Erro: Arquivo não encontrado - {caminho_arquivo}")
            return None
            
        print(f"Abrindo arquivo: {caminho_arquivo}")
        
        # Converter .dbc para .dbf temporariamente
        nome_base = os.path.splitext(os.path.basename(caminho_arquivo))[0]
        temp_dbf = f"temp_{nome_base}.dbf"
        
        print("Convertendo .dbc para .dbf...")
        pyreaddbc.dbc2dbf(caminho_arquivo, temp_dbf)
        
        # Ler o arquivo .dbf convertido
        print("Lendo dados convertidos...")
        dbf = DBF(temp_dbf, encoding='iso-8859-1')
        df = pd.DataFrame(iter(dbf))
        
        # Remover arquivo temporário
        os.remove(temp_dbf)
        
        print(f"Dados carregados com sucesso!")
        print(f"Número de registros: {len(df)}")
        print(f"Número de colunas: {len(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"Erro ao abrir o arquivo: {str(e)}")
        return None

def explorar_dados(df):
    """
    Explora e exibe informações básicas sobre os dados
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados
    """
    if df is None:
        return
        
    print("\n" + "="*50)
    print("EXPLORAÇÃO DOS DADOS")
    print("="*50)
    
    # Informações gerais
    print(f"\nShape dos dados: {df.shape}")
    
    # Primeiras linhas
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    
    # Informações sobre as colunas
    print("\nInformações sobre as colunas:")
    print(df.info())
    
    # Colunas disponíveis
    print(f"\nColunas disponíveis ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    # Estatísticas descritivas para colunas numéricas
    print("\nEstatísticas descritivas:")
    print(df.describe())

def main():
    """Função principal"""
    # Caminho para o arquivo (mesmo diretório do script)
    caminho_arquivo = "DNTO2023.dbc"
    
    print("Iniciando processamento dos dados DATASUS...")
    print("Arquivo: DNTO2023.dbc (Declarações de Nascidos Vivos 2023)")
    
    # Abrir os dados
    df = abrir_dados_datasus(caminho_arquivo)
    
    if df is not None:
        # Explorar os dados
        explorar_dados(df)
        
        # Salvar em formato CSV para facilitar análises futuras
        arquivo_csv = "DNTO2023.csv"
        print(f"\nSalvando dados em formato CSV: {arquivo_csv}")
        df.to_csv(arquivo_csv, index=False, encoding='utf-8')
        print("Arquivo CSV salvo com sucesso!")
        
        # Exemplo de análise básica
        print("\n" + "="*50)
        print("ANÁLISE BÁSICA")
        print("="*50)
        
        # Verificar colunas comuns do SINASC
        colunas_analise = {
            'SEXO': 'Distribuição por sexo',
            'PESO': 'Estatísticas de peso',
            'CODMUNNASC': 'Municípios de nascimento',
            'DTNASC': 'Datas de nascimento',
            'APGAR1': 'APGAR 1º minuto',
            'APGAR5': 'APGAR 5º minuto'
        }
        
        for coluna, descricao in colunas_analise.items():
            if coluna in df.columns:
                print(f"\n{descricao} ({coluna}):")
                if coluna == 'PESO':
                    # Tentar converter para numérico, ignorando erros
                    try:
                        peso_numerico = pd.to_numeric(df[coluna], errors='coerce')
                        peso_valido = peso_numerico.dropna()
                        if len(peso_valido) > 0:
                            print(f"  Peso médio: {peso_valido.mean():.2f}g")
                            print(f"  Peso mínimo: {peso_valido.min()}g")
                            print(f"  Peso máximo: {peso_valido.max()}g")
                            print(f"  Registros com peso válido: {len(peso_valido)}/{len(df)}")
                        else:
                            print(f"  Nenhum valor de peso válido encontrado")
                    except:
                        print(f"  Erro ao processar pesos - valores não numéricos")
                elif df[coluna].nunique() < 20:
                    print(df[coluna].value_counts().head(10))
                else:
                    print(f"  Total de valores únicos: {df[coluna].nunique()}")
                    print(f"  Primeiros valores: {df[coluna].unique()[:5]}")
        
        return df
    else:
        print("Falha ao carregar os dados.")
        return None

if __name__ == "__main__":
    # Executar função principal
    dados = main()
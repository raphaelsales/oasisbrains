#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemplo simples para abrir dados DATASUS com PySUS
"""

# Primeiro, instale as dependências necessárias:
# pip install pysus pandas

from pysus.utilities.readdbc import read_dbc
import pandas as pd

# Caminho para o arquivo
arquivo_dbc = "/app/alzheimer/Outros/Datasus/DNTO2023.dbc"

# Abrir o arquivo .dbc
print("Carregando dados do DATASUS...")
df = read_dbc(arquivo_dbc, encoding='iso-8859-1')

# Exibir informações básicas
print(f"Dados carregados! Shape: {df.shape}")
print("\nPrimeiras 5 linhas:")
print(df.head())

print("\nColunas disponíveis:")
print(df.columns.tolist())

# Salvar como CSV (opcional)
df.to_csv("/app/alzheimer/Outros/Datasus/dados_nascimentos_2023.csv", 
          index=False, encoding='utf-8')
print("\nDados salvos como CSV!")

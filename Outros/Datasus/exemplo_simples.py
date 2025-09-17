#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemplo SIMPLES para abrir dados DATASUS com PySUS
"""

# Instale primeiro: pip install pysus pandas

import pyreaddbc
import pandas as pd
from dbfread import DBF
import os
import tempfile

# Abrir o arquivo .dbc
print("Carregando dados do DATASUS...")

# Converter .dbc para .dbf temporariamente
temp_dbf = "temp_DNTO2023.dbf"
pyreaddbc.dbc2dbf("DNTO2023.dbc", temp_dbf)

# Ler o arquivo .dbf convertido
dbf = DBF(temp_dbf, encoding='iso-8859-1')
df = pd.DataFrame(iter(dbf))

# Remover arquivo temporário
os.remove(temp_dbf)

# Exibir informações básicas
print(f"Dados carregados! Shape: {df.shape}")
print(f"Colunas: {len(df.columns)}")

print("\nPrimeiras 3 linhas:")
print(df.head(3))

print("\nColunas disponíveis:")
print(df.columns.tolist())

# Salvar como CSV
df.to_csv("dados_nascimentos_2023.csv", index=False, encoding='utf-8')
print("\nDados salvos como CSV!")

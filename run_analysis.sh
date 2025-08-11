#!/bin/bash

# SCRIPT PRINCIPAL DE EXECUÇÃO - PROJETO ALZHEIMER
# Executa o sistema de análise principal

echo "PROJETO ALZHEIMER - SISTEMA DE ANÁLISE"
echo "======================================"
echo

# Verificar se estamos no diretório correto
if [ ! -f "alzheimer_analysis_suite.sh" ]; then
    echo "ERRO: Execute este script do diretório raiz do projeto"
    exit 1
fi

# Executar o sistema principal
./alzheimer_analysis_suite.sh

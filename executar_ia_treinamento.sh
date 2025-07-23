#!/bin/bash

echo "🧠 SISTEMA DE TREINAMENTO DE IA PARA ALZHEIMER"
echo "=============================================="
echo ""

# Verificar se os dados processados existem
DATA_DIR="/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Diretório de dados não encontrado: $DATA_DIR"
    echo "   Execute primeiro o processamento FastSurfer"
    exit 1
fi

# Contar sujeitos processados
PROCESSED_COUNT=$(find "$DATA_DIR" -name "aparc+aseg.mgz" | wc -l)
echo "📊 Sujeitos processados encontrados: $PROCESSED_COUNT"

if [ $PROCESSED_COUNT -lt 50 ]; then
    echo "⚠️  Poucos sujeitos processados ($PROCESSED_COUNT)"
    echo "   Recomendado: pelo menos 50 sujeitos para treinamento"
    echo "   Deseja continuar mesmo assim? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Cancelado pelo usuário"
        exit 1
    fi
fi

echo ""
echo "🔧 Verificando dependências Python..."

# Verificar e instalar dependências
REQUIRED_PACKAGES=(
    "pandas"
    "numpy" 
    "nibabel"
    "scikit-learn"
    "matplotlib"
    "seaborn"
    "tensorflow"
    "joblib"
)

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import ${package}" 2>/dev/null; then
        echo "✅ $package: OK"
    else
        echo "❌ $package: FALTANDO"
        echo "   Instalando $package..."
        pip install "$package" || {
            echo "❌ Falha ao instalar $package"
            echo "   Tente: pip install $package"
            exit 1
        }
    fi
done

echo ""
echo "🚀 ESCOLHA O TIPO DE TREINAMENTO:"
echo "================================="
echo "1. 🔬 Extração básica de features (Rápido)"
echo "2. 🧠 Pipeline completo de Alzheimer (Completo)"
echo "3. 📊 Apenas análise exploratória"
echo ""
echo -n "Digite sua escolha (1-3): "
read -r choice

case $choice in
    1)
        echo ""
        echo "🔬 EXECUTANDO EXTRAÇÃO BÁSICA DE FEATURES..."
        python3 extrair_features_ia.py
        ;;
    2)
        echo ""
        echo "🧠 EXECUTANDO PIPELINE COMPLETO DE ALZHEIMER..."
        python3 alzheimer_ai_pipeline.py
        ;;
    3)
        echo ""
        echo "📊 EXECUTANDO APENAS ANÁLISE EXPLORATÓRIA..."
        python3 -c "
from alzheimer_ai_pipeline import AlzheimerBrainAnalyzer, AlzheimerAnalysisReport
import pandas as pd

# Carregar dados se já existir o CSV
try:
    df = pd.read_csv('alzheimer_complete_dataset.csv')
    print('📁 Carregando dataset existente...')
except:
    print('📊 Criando novo dataset...')
    analyzer = AlzheimerBrainAnalyzer('$DATA_DIR')
    df = analyzer.create_comprehensive_dataset()
    df.to_csv('alzheimer_complete_dataset.csv', index=False)

# Gerar relatório
report = AlzheimerAnalysisReport(df)
report.generate_exploratory_analysis()
print('✅ Análise exploratória concluída!')
"
        ;;
    *)
        echo "❌ Opção inválida. Escolha 1, 2 ou 3."
        exit 1
        ;;
esac

echo ""
echo "✅ TREINAMENTO CONCLUÍDO!"
echo ""
echo "📁 ARQUIVOS GERADOS:"
ls -la *.csv *.png *.h5 *.joblib 2>/dev/null || echo "   Nenhum arquivo gerado"

echo ""
echo "💡 PRÓXIMOS PASSOS:"
echo "=================="
echo "1. 📊 Analise os datasets CSV gerados"
echo "2. 🔍 Examine as visualizações PNG"
echo "3. 🤖 Use os modelos .h5 para predições"
echo "4. 📋 Leia os relatórios de performance"
echo ""
echo "🚀 Para usar os modelos treinados:"
echo "   python3 -c \"import joblib; import tensorflow as tf; model = tf.keras.models.load_model('modelo.h5')\"" 
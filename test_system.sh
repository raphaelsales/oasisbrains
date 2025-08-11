#!/bin/bash

echo "TESTE DO SISTEMA APÓS CORREÇÃO"
echo "=============================="
echo

# Testar se os arquivos Python funcionam
echo "Testando scripts Python..."

# Teste 1: Dataset Explorer
echo "1. Testando dataset_explorer.py..."
if python3 -c "import sys; sys.path.append('.'); from src.analysis.dataset_explorer import *; print('✅ dataset_explorer.py importado com sucesso')" 2>/dev/null; then
    echo "   ✅ dataset_explorer.py OK"
else
    echo "   ❌ dataset_explorer.py com problemas"
fi

# Teste 2: MCI Clinical Insights
echo "2. Testando mci_clinical_insights.py..."
if python3 -c "import sys; sys.path.append('.'); from src.analysis.mci_clinical_insights import *; print('✅ mci_clinical_insights.py importado com sucesso')" 2>/dev/null; then
    echo "   ✅ mci_clinical_insights.py OK"
else
    echo "   ❌ mci_clinical_insights.py com problemas"
fi

# Teste 3: Alzheimer Early Diagnosis
echo "3. Testando alzheimer_early_diagnosis_analysis.py..."
if python3 -c "import sys; sys.path.append('.'); from src.analysis.alzheimer_early_diagnosis_analysis import *; print('✅ alzheimer_early_diagnosis_analysis.py importado com sucesso')" 2>/dev/null; then
    echo "   ✅ alzheimer_early_diagnosis_analysis.py OK"
else
    echo "   ❌ alzheimer_early_diagnosis_analysis.py com problemas"
fi

# Teste 4: Verificar arquivos CSV
echo "4. Verificando arquivos CSV..."
csv_count=0
for csv_file in *.csv; do
    if [ -f "$csv_file" ]; then
        echo "   ✅ $csv_file encontrado"
        ((csv_count++))
    fi
done
echo "   Total de arquivos CSV: $csv_count"

# Teste 5: Verificar modelos
echo "5. Verificando modelos..."
model_count=0
for model_file in *.h5 *.joblib; do
    if [ -f "$model_file" ]; then
        echo "   ✅ $model_file encontrado"
        ((model_count++))
    fi
done
echo "   Total de modelos: $model_count"

echo
echo "TESTE CONCLUÍDO!"
echo "Para testar o sistema completo:"
echo "  ./alzheimer_analysis_suite.sh"

#!/bin/bash

# SCRIPT PARA CORRIGIR CAMINHOS NOS ARQUIVOS PYTHON APÓS REORGANIZAÇÃO
# Atualiza os caminhos dos arquivos CSV e outros recursos

echo "CORRIGINDO CAMINHOS NOS ARQUIVOS PYTHON"
echo "======================================"
echo

# Verificar se estamos no diretório correto
if [ ! -d "src" ]; then
    echo "❌ ERRO: Execute este script do diretório raiz do projeto"
    exit 1
fi

echo "✅ Diretório correto detectado"

# Lista de arquivos CSV que podem estar sendo referenciados
csv_files=(
    "alzheimer_complete_dataset.csv"
    "mci_subjects_metadata.csv"
    "mci_subjects_metadata_improved.csv"
    "hybrid_mci_subjects_metadata.csv"
    "fastsurfer_mci_significant_features.csv"
    "fastsurfer_mci_statistical_analysis.csv"
    "fastsurfer_mci_top_biomarkers.csv"
)

# Verificar se os arquivos CSV existem em data/processed/
echo "Verificando arquivos CSV em data/processed/..."
for csv_file in "${csv_files[@]}"; do
    if [ -f "data/processed/$csv_file" ]; then
        echo "  ✅ $csv_file encontrado em data/processed/"
    else
        echo "  ❌ $csv_file não encontrado em data/processed/"
    fi
done

# Verificar se os arquivos CSV existem no diretório raiz (para compatibilidade)
echo
echo "Verificando arquivos CSV no diretório raiz..."
for csv_file in "${csv_files[@]}"; do
    if [ -f "$csv_file" ]; then
        echo "  ✅ $csv_file encontrado no diretório raiz"
    else
        echo "  ❌ $csv_file não encontrado no diretório raiz"
    fi
done

# Criar links simbólicos para compatibilidade
echo
echo "Criando links simbólicos para compatibilidade..."
for csv_file in "${csv_files[@]}"; do
    if [ -f "data/processed/$csv_file" ] && [ ! -f "$csv_file" ]; then
        echo "  Criando link simbólico: $csv_file -> data/processed/$csv_file"
        ln -sf "data/processed/$csv_file" "$csv_file"
    fi
done

# Verificar arquivos de modelo
echo
echo "Verificando arquivos de modelo..."
model_files=(
    "*.h5"
)

for pattern in "${model_files[@]}"; do
    if ls models/checkpoints/$pattern 2>/dev/null | grep -q .; then
        echo "  ✅ Modelos encontrados em models/checkpoints/"
        # Criar links simbólicos para compatibilidade
        for model_file in models/checkpoints/$pattern; do
            if [ -f "$model_file" ]; then
                filename=$(basename "$model_file")
                if [ ! -f "$filename" ]; then
                    echo "    Criando link: $filename -> $model_file"
                    ln -sf "$model_file" "$filename"
                fi
            fi
        done
    fi
done

# Verificar arquivos scaler
echo
echo "Verificando arquivos scaler..."
scaler_files=(
    "*.joblib"
)

for pattern in "${scaler_files[@]}"; do
    if ls models/scalers/$pattern 2>/dev/null | grep -q .; then
        echo "  ✅ Scalers encontrados em models/scalers/"
        # Criar links simbólicos para compatibilidade
        for scaler_file in models/scalers/$pattern; do
            if [ -f "$scaler_file" ]; then
                filename=$(basename "$scaler_file")
                if [ ! -f "$filename" ]; then
                    echo "    Criando link: $filename -> $scaler_file"
                    ln -sf "$scaler_file" "$filename"
                fi
            fi
        done
    fi
done

# Verificar arquivos de relatório
echo
echo "Verificando arquivos de relatório..."
report_files=(
    "*_report_*.txt"
    "*_report_*.png"
    "*_analysis.png"
    "*_performance_report.png"
)

for pattern in "${report_files[@]}"; do
    if ls data/results/$pattern 2>/dev/null | grep -q .; then
        echo "  ✅ Relatórios encontrados em data/results/"
        # Criar links simbólicos para compatibilidade
        for report_file in data/results/$pattern; do
            if [ -f "$report_file" ]; then
                filename=$(basename "$report_file")
                if [ ! -f "$filename" ]; then
                    echo "    Criando link: $filename -> $report_file"
                    ln -sf "$report_file" "$filename"
                fi
            fi
        done
    fi
done

# Verificar diretório oasis_data
echo
echo "Verificando diretório oasis_data..."
if [ -d "data/raw/oasis_data" ] && [ ! -d "oasis_data" ]; then
    echo "  Criando link simbólico: oasis_data -> data/raw/oasis_data"
    ln -sf "data/raw/oasis_data" "oasis_data"
fi

# Verificar logs
echo
echo "Verificando logs..."
if [ -d "logs/tensorboard" ] && [ ! -d "logs" ]; then
    echo "  Criando link simbólico: logs -> logs/tensorboard"
    ln -sf "logs/tensorboard" "logs"
fi

# Atualizar alzheimer_analysis_suite.sh para usar caminhos corretos
echo
echo "Atualizando alzheimer_analysis_suite.sh..."

# Fazer backup
cp alzheimer_analysis_suite.sh alzheimer_analysis_suite.sh.before_fix

# Atualizar caminhos para modelos
sed -i 's|ls -lh \*\.h5|ls -lh models/checkpoints/*.h5 2>/dev/null \|\| ls -lh *.h5|g' alzheimer_analysis_suite.sh
sed -i 's|ls -lh \*\.joblib|ls -lh models/scalers/*.joblib 2>/dev/null \|\| ls -lh *.joblib|g' alzheimer_analysis_suite.sh
sed -i 's|ls -lh \*\.csv \*\.png|ls -lh data/results/*.csv data/results/*.png 2>/dev/null \|\| ls -lh *.csv *.png|g' alzheimer_analysis_suite.sh

echo "✅ alzheimer_analysis_suite.sh atualizado"

# Criar script de teste
echo
echo "Criando script de teste..."
cat > test_system.sh << 'EOF'
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
EOF

chmod +x test_system.sh

echo "✅ Script de teste criado: test_system.sh"

# Resumo final
echo
echo "CORREÇÃO CONCLUÍDA!"
echo "=================="
echo
echo "Ações realizadas:"
echo "  📁 Criados links simbólicos para compatibilidade"
echo "  🔧 Atualizado alzheimer_analysis_suite.sh"
echo "  🧪 Criado script de teste"
echo
echo "Para testar o sistema:"
echo "  ./test_system.sh"
echo "  ./alzheimer_analysis_suite.sh"
echo
echo "Arquivos de backup:"
echo "  - alzheimer_analysis_suite.sh.before_fix"

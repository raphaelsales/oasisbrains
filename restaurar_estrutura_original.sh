#!/bin/bash

# SCRIPT DE RESTAURAÇÃO DA ESTRUTURA ORIGINAL
# Restaura a estrutura original caso a reorganização cause problemas

echo "RESTAURAÇÃO DA ESTRUTURA ORIGINAL"
echo "================================="
echo

# Verificar se existe backup
if [ ! -f "alzheimer_analysis_suite.sh.backup" ]; then
    echo "❌ ERRO: Backup do alzheimer_analysis_suite.sh não encontrado"
    echo "   Isso indica que a reorganização não foi feita ou o backup foi removido"
    exit 1
fi

echo "✅ Backup encontrado"

# Verificar se existe estrutura reorganizada
if [ ! -d "src" ] && [ ! -d "scripts" ]; then
    echo "❌ ERRO: Estrutura reorganizada não encontrada"
    echo "   Não há nada para restaurar"
    exit 1
fi

echo "✅ Estrutura reorganizada detectada"

# Confirmar restauração
echo
echo "⚠️  ATENÇÃO: Esta operação irá:"
echo "   - Restaurar alzheimer_analysis_suite.sh original"
echo "   - Mover todos os arquivos de volta para o diretório raiz"
echo "   - Remover a estrutura organizada"
echo "   - Manter backup do alzheimer_analysis_suite.sh atual"
echo
read -p "Deseja continuar com a restauração? (s/n): " -r
if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    echo "Restauração cancelada."
    exit 1
fi

echo
echo "Iniciando restauração..."

# Fazer backup do alzheimer_analysis_suite.sh atual
echo "Fazendo backup do arquivo atual..."
cp alzheimer_analysis_suite.sh alzheimer_analysis_suite.sh.reorganizado

# Restaurar alzheimer_analysis_suite.sh original
echo "Restaurando alzheimer_analysis_suite.sh original..."
cp alzheimer_analysis_suite.sh.backup alzheimer_analysis_suite.sh

# Mover arquivos Python de volta
echo "Movendo arquivos Python de volta..."

# CNN Models
if [ -d "src/cnn" ]; then
    echo "  - CNN Models..."
    mv src/cnn/*.py . 2>/dev/null || true
fi

# OpenAI Integration
if [ -d "src/openai" ]; then
    echo "  - OpenAI Integration..."
    mv src/openai/*.py . 2>/dev/null || true
fi

# Analysis Scripts
if [ -d "src/analysis" ]; then
    echo "  - Analysis Scripts..."
    mv src/analysis/*.py . 2>/dev/null || true
fi

# Utils
if [ -d "src/utils" ]; then
    echo "  - Utils..."
    mv src/utils/*.py . 2>/dev/null || true
fi

# Scripts de Setup
if [ -d "scripts/setup" ]; then
    echo "  - Setup Scripts..."
    mv scripts/setup/*.sh . 2>/dev/null || true
fi

# Scripts de Processamento
if [ -d "scripts/processing" ]; then
    echo "  - Processing Scripts..."
    mv scripts/processing/*.sh . 2>/dev/null || true
fi

# Scripts de Monitoramento
if [ -d "scripts/monitoring" ]; then
    echo "  - Monitoring Scripts..."
    mv scripts/monitoring/*.sh . 2>/dev/null || true
fi

# Scripts de Teste
if [ -d "scripts/testing" ]; then
    echo "  - Testing Scripts..."
    mv scripts/testing/*.sh . 2>/dev/null || true
fi

# Modelos e Checkpoints
if [ -d "models/checkpoints" ]; then
    echo "  - Models and Checkpoints..."
    mv models/checkpoints/*.h5 . 2>/dev/null || true
fi

if [ -d "models/scalers" ]; then
    echo "  - Scalers..."
    mv models/scalers/*.joblib . 2>/dev/null || true
fi

# Dados
if [ -d "data/processed" ]; then
    echo "  - Data Files..."
    mv data/processed/*.csv . 2>/dev/null || true
fi

if [ -d "data/raw" ]; then
    echo "  - Raw Data..."
    mv data/raw/oasis_data . 2>/dev/null || true
fi

# Resultados
if [ -d "data/results" ]; then
    echo "  - Results..."
    mv data/results/* . 2>/dev/null || true
fi

# Logs
if [ -d "logs/tensorboard" ]; then
    echo "  - Logs..."
    mv logs/tensorboard/* logs/ 2>/dev/null || true
fi

# Documentação
if [ -d "docs/manuais" ]; then
    echo "  - Documentation..."
    mv docs/manuais/*.md . 2>/dev/null || true
fi

if [ -d "docs/relatorios" ]; then
    echo "  - Reports..."
    mv docs/relatorios/* . 2>/dev/null || true
fi

if [ -d "docs/configuracoes" ]; then
    echo "  - Configurations..."
    mv docs/configuracoes/* . 2>/dev/null || true
fi

# Configurações
if [ -d "config" ]; then
    echo "  - Configuration..."
    mv config/* . 2>/dev/null || true
fi

# Remover diretórios vazios
echo "Removendo diretórios vazios..."
find . -type d -empty -delete 2>/dev/null || true

# Remover diretórios da estrutura organizada
echo "Removendo estrutura organizada..."
rm -rf src scripts models data docs logs config tests 2>/dev/null || true

# Remover arquivos de reorganização
echo "Removendo arquivos de reorganização..."
rm -f run_analysis.sh ESTRUTURA_REORGANIZADA.md 2>/dev/null || true

# Verificar se a restauração foi bem-sucedida
echo
echo "Verificando restauração..."

critical_files=(
    "alzheimer_analysis_suite.sh"
    "openai_fastsurfer_analyzer.py"
    "config_openai.py"
    "alzheimer_cnn_pipeline.py"
    "mci_detection_cnn_optimized.py"
    "alzheimer_early_diagnosis_analysis.py"
    "mci_clinical_insights.py"
    "dataset_explorer.py"
)

restored_files=0
for file in "${critical_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
        ((restored_files++))
    else
        echo "  ❌ $file (NÃO RESTAURADO)"
    fi
done

echo
echo "RESULTADO DA RESTAURAÇÃO"
echo "======================="
echo "Arquivos críticos restaurados: $restored_files/${#critical_files[@]}"

if [ $restored_files -eq ${#critical_files[@]} ]; then
    echo "✅ RESTAURAÇÃO COMPLETA!"
    echo
    echo "A estrutura original foi restaurada com sucesso."
    echo "Arquivos de backup mantidos:"
    echo "  - alzheimer_analysis_suite.sh.backup (original)"
    echo "  - alzheimer_analysis_suite.sh.reorganizado (versão reorganizada)"
    echo
    echo "Para usar o sistema:"
    echo "  ./alzheimer_analysis_suite.sh"
else
    echo "⚠️  RESTAURAÇÃO PARCIAL!"
    echo
    echo "Alguns arquivos não foram restaurados."
    echo "Verifique se há problemas ou se os arquivos foram movidos para outros locais."
fi

echo
echo "Restauração concluída!"

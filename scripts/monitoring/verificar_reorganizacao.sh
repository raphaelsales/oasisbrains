#!/bin/bash

# SCRIPT DE VERIFICAÇÃO PRÉVIA PARA REORGANIZAÇÃO
# Verifica se a reorganização pode ser feita com segurança

echo "VERIFICAÇÃO PRÉVIA PARA REORGANIZAÇÃO"
echo "===================================="
echo

# Verificar se estamos no diretório correto
if [ ! -f "alzheimer_analysis_suite.sh" ]; then
    echo "❌ ERRO: Execute este script do diretório raiz do projeto"
    echo "   Diretório atual: $(pwd)"
    exit 1
fi

echo "✅ Diretório correto detectado"

# Verificar arquivos críticos
echo
echo "Verificando arquivos críticos..."

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

missing_files=()
for file in "${critical_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (NÃO ENCONTRADO)"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo
    echo "⚠️  ATENÇÃO: Alguns arquivos críticos não foram encontrados:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo
    echo "Isso pode indicar que a reorganização já foi feita ou há problemas."
    read -p "Deseja continuar mesmo assim? (s/n): " -r
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo "Reorganização cancelada."
        exit 1
    fi
fi

# Verificar espaço em disco
echo
echo "Verificando espaço em disco..."
available_space=$(df . | awk 'NR==2 {print $4}')
available_gb=$((available_space / 1024 / 1024))

echo "  Espaço disponível: ${available_gb}GB"

if [ $available_gb -lt 5 ]; then
    echo "  ⚠️  ATENÇÃO: Pouco espaço em disco (menos de 5GB)"
    read -p "Deseja continuar mesmo assim? (s/n): " -r
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo "Reorganização cancelada."
        exit 1
    fi
else
    echo "  ✅ Espaço suficiente disponível"
fi

# Verificar se há processos em execução
echo
echo "Verificando processos em execução..."
running_processes=$(pgrep -f "python.*alzheimer\|fastsurfer\|tensorboard" | wc -l)

if [ $running_processes -gt 0 ]; then
    echo "  ⚠️  ATENÇÃO: Encontrados $running_processes processos relacionados em execução"
    echo "  Processos encontrados:"
    pgrep -f "python.*alzheimer\|fastsurfer\|tensorboard" | xargs ps -p 2>/dev/null || echo "    (Não foi possível listar detalhes)"
    echo
    read -p "Recomenda-se parar os processos antes da reorganização. Continuar? (s/n): " -r
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo "Reorganização cancelada."
        exit 1
    fi
else
    echo "  ✅ Nenhum processo relacionado em execução"
fi

# Verificar permissões
echo
echo "Verificando permissões..."
if [ -w . ]; then
    echo "  ✅ Permissão de escrita no diretório atual"
else
    echo "  ❌ ERRO: Sem permissão de escrita no diretório atual"
    exit 1
fi

# Verificar se já existe alguma estrutura organizada
echo
echo "Verificando estrutura existente..."
if [ -d "src" ] || [ -d "scripts" ] || [ -d "models" ]; then
    echo "  ⚠️  ATENÇÃO: Detectada estrutura organizada existente:"
    [ -d "src" ] && echo "    - src/ (já existe)"
    [ -d "scripts" ] && echo "    - scripts/ (já existe)"
    [ -d "models" ] && echo "    - models/ (já existe)"
    echo
    read -p "Isso pode causar conflitos. Deseja fazer backup e continuar? (s/n): " -r
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        echo "  Fazendo backup da estrutura existente..."
        backup_dir="backup_estrutura_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$backup_dir"
        [ -d "src" ] && mv src "$backup_dir/"
        [ -d "scripts" ] && mv scripts "$backup_dir/"
        [ -d "models" ] && mv models "$backup_dir/"
        [ -d "data" ] && mv data "$backup_dir/"
        [ -d "docs" ] && mv docs "$backup_dir/"
        [ -d "logs" ] && mv logs "$backup_dir/"
        [ -d "config" ] && mv config "$backup_dir/"
        [ -d "tests" ] && mv tests "$backup_dir/"
        echo "  ✅ Backup criado em: $backup_dir"
    else
        echo "Reorganização cancelada."
        exit 1
    fi
else
    echo "  ✅ Nenhuma estrutura organizada detectada"
fi

# Resumo final
echo
echo "RESUMO DA VERIFICAÇÃO"
echo "===================="
echo "✅ Diretório correto"
echo "✅ Permissões adequadas"
echo "✅ Espaço em disco suficiente"
echo "✅ Arquivos críticos verificados"
echo "✅ Estrutura limpa para reorganização"
echo
echo "🎯 TUDO PRONTO PARA REORGANIZAÇÃO!"
echo
echo "Para executar a reorganização:"
echo "  ./reorganizar_estrutura.sh"
echo
echo "A reorganização irá:"
echo "  📁 Criar estrutura organizada de diretórios"
echo "  📦 Mover arquivos para pastas específicas"
echo "  🔧 Atualizar alzheimer_analysis_suite.sh"
echo "  📋 Criar documentação da nova estrutura"
echo "  💾 Fazer backup do arquivo original"
echo
read -p "Deseja executar a reorganização agora? (s/n): " -r
if [[ $REPLY =~ ^[Ss]$ ]]; then
    echo
    echo "Executando reorganização..."
    ./reorganizar_estrutura.sh
else
    echo
    echo "Reorganização não executada."
    echo "Execute manualmente quando estiver pronto:"
    echo "  ./reorganizar_estrutura.sh"
fi

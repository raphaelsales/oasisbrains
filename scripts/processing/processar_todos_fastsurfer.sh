#!/bin/bash

echo "🚀 FASTSURFER - PROCESSAMENTO DEFINITIVO TODOS OS SUJEITOS"
echo "=========================================================="
echo ""

# Configurações
DATA_BASE="/app/alzheimer/oasis_data"
OUTPUT_DIR="${DATA_BASE}/outputs_fastsurfer_definitivo_todos"
LOG_DIR="${DATA_BASE}/logs_fastsurfer_definitivo_todos"
LICENSE_FILE="/app/alzheimer/freesurfer_license_oficial.txt"

# Criar diretórios
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Contadores
TOTAL_SUBJECTS=0
PROCESSED=0
FAILED=0
SKIPPED=0

# Função para processar um sujeito
process_subject() {
    local subject_dir=$1
    local subject=$(basename "$subject_dir")
    local disc=$(basename "$(dirname "$subject_dir")")
    
    echo "🔄 Processando: $subject ($disc)"
    
    # Verificar se já foi processado
    if [ -d "$OUTPUT_DIR/$subject" ]; then
        echo "  ⏭️  Já processado, pulando..."
        ((SKIPPED++))
        return 0
    fi
    
    # Verificar se arquivo T1 existe
    local t1_file="${subject_dir}/mri/T1.mgz"
    if [ ! -f "$t1_file" ]; then
        echo "  ❌ Arquivo T1.mgz não encontrado"
        ((FAILED++))
        return 1
    fi
    
    # Criar diretório temporário para o arquivo T1
    local temp_dir="/tmp/fastsurfer_${subject}_$$"
    mkdir -p "$temp_dir"
    
    # Copiar arquivo T1 (resolve problema Git Annex)
    cp "$t1_file" "$temp_dir/T1.mgz"
    
    local start_time=$(date +%s)
    
    # Executar FastSurfer
    docker run --rm \
        --user $(id -u):$(id -g) \
        -v "$temp_dir:/input" \
        -v "$OUTPUT_DIR:/output" \
        -v "$LICENSE_FILE:/license.txt" \
        deepmi/fastsurfer:latest \
        --fs_license /license.txt \
        --t1 /input/T1.mgz \
        --sid "$subject" \
        --sd /output \
        --device cpu \
        --py python3 \
        --threads 6 \
        2>&1 | tee "${LOG_DIR}/${subject}.log"
    
    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Limpar diretório temporário
    rm -rf "$temp_dir"
    
    if [ $exit_code -eq 0 ]; then
        echo "  ✅ Sucesso em ${duration}s"
        ((PROCESSED++))
        
        # Verificar se arquivos foram gerados
        local result_files=$(find "$OUTPUT_DIR/$subject" -type f 2>/dev/null | wc -l)
        echo "  📊 Arquivos gerados: $result_files"
        
    else
        echo "  ❌ Falhou (exit code: $exit_code) em ${duration}s"
        ((FAILED++))
    fi
    
    # Mostrar progresso
    show_progress
}

# Função para mostrar progresso
show_progress() {
    local remaining=$((TOTAL_SUBJECTS - PROCESSED - FAILED - SKIPPED))
    local elapsed=$(($(date +%s) - START_TIME))
    
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "📊 PROGRESSO:"
    echo "   ✅ Processados: $PROCESSED"
    echo "   ❌ Falharam: $FAILED" 
    echo "   ⏭️  Pulados: $SKIPPED"
    echo "   ⏳ Restantes: $remaining"
    echo "   📈 Total: $TOTAL_SUBJECTS"
    echo ""
    
    if [ $PROCESSED -gt 0 ]; then
        local avg_time=$((elapsed / (PROCESSED + FAILED)))
        local eta=$((remaining * avg_time))
        echo "⏱️  TEMPO:"
        echo "   🕐 Decorrido: $(date -ud @$elapsed '+%Hh %Mm %Ss')"
        echo "   ⚡ Médio por sujeito: $(date -ud @$avg_time '+%Mm %Ss')"
        echo "   🔮 Estimativa restante: $(date -ud @$eta '+%Hh %Mm')"
    fi
    
    echo "═══════════════════════════════════════════════════════"
    echo ""
}

# Inicialização
START_TIME=$(date +%s)

echo "🔍 Coletando sujeitos..."

# Coletar todos os sujeitos
declare -a SUBJECTS
for disc in {1..11}; do
    DISK_DIR="${DATA_BASE}/disc${disc}"
    if [ -d "$DISK_DIR" ]; then
        for subject_dir in "${DISK_DIR}"/OAS1_*_MR1; do
            if [ -d "$subject_dir" ] && [ -f "${subject_dir}/mri/T1.mgz" ]; then
                SUBJECTS+=("$subject_dir")
                ((TOTAL_SUBJECTS++))
            fi
        done
    fi
done

echo "📋 Total de sujeitos encontrados: $TOTAL_SUBJECTS"
echo "📁 Resultados serão salvos em: $OUTPUT_DIR"
echo "📝 Logs serão salvos em: $LOG_DIR"
echo ""

if [ $TOTAL_SUBJECTS -eq 0 ]; then
    echo "❌ Nenhum sujeito encontrado!"
    exit 1
fi

echo "🚀 INICIANDO PROCESSAMENTO..."
echo "⏱️  Início: $(date)"
echo ""

# Processar cada sujeito sequencialmente
for subject_dir in "${SUBJECTS[@]}"; do
    process_subject "$subject_dir"
done

# Resumo final
TOTAL_TIME=$(($(date +%s) - START_TIME))

echo ""
echo "🎉 PROCESSAMENTO CONCLUÍDO!"
echo "=========================="
echo "⏱️  Tempo total: $(date -ud @$TOTAL_TIME '+%Hh %Mm %Ss')"
echo "✅ Processados com sucesso: $PROCESSED"
echo "❌ Falharam: $FAILED"
echo "⏭️  Pulados (já processados): $SKIPPED"
echo "📈 Total: $TOTAL_SUBJECTS"

if [ $PROCESSED -gt 0 ]; then
    local avg_time=$((TOTAL_TIME / PROCESSED))
    echo "⚡ Tempo médio por sujeito: $(date -ud @$avg_time '+%Mm %Ss')"
fi

echo ""
echo "📁 Resultados disponíveis em: $OUTPUT_DIR"
echo "📝 Logs disponíveis em: $LOG_DIR" 
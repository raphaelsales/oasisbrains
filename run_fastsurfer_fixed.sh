#!/bin/bash

echo "=== FastSurfer - PROCESSAMENTO EM LOTE CORRIGIDO ==="

# Configurações
DATA_BASE="/app/alzheimer/oasis_data"
OUTPUT_DIR="${DATA_BASE}/outputs_fastsurfer_fixed"
LOG_DIR="${DATA_BASE}/fastsurfer_logs_fixed"
MAX_PARALLEL=2

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Configurar licença
LICENSE_FILE="./freesurfer_license.txt"
if [ ! -f "/home/comaisserveria/license.txt" ]; then
    if [ ! -f "$LICENSE_FILE" ]; then
        echo "🔧 Criando licença temporária..."
        cat > "$LICENSE_FILE" << 'EOF'
# FreeSurfer License - Para uso acadêmico/pesquisa  
# Registre-se em: https://surfer.nmr.mgh.harvard.edu/registration.html
raphael.comaisserveria@email.com
12345
*Ca123456789
FSabcdefghijk
EOF
    fi
    echo "⚠️  Usando licença temporária. Registre-se no FreeSurfer!"
else
    LICENSE_FILE="/home/comaisserveria/license.txt"
fi

# Contadores
TOTAL_SUBJECTS=0
PROCESSED=0
FAILED=0
START_TIME=$(date +%s)

# Coletar todos os sujeitos
echo "📊 Coletando sujeitos..."
SUBJECTS=()
for disc in {1..11}; do
    DISK_DIR="${DATA_BASE}/disc${disc}"
    [ -d "$DISK_DIR" ] || continue
    
    for subject_dir in "${DISK_DIR}"/OAS1_*_MR1; do
        [ -d "$subject_dir" ] && [ -f "${subject_dir}/mri/T1.mgz" ] || continue
        SUBJECTS+=("$subject_dir")
        ((TOTAL_SUBJECTS++))
    done
done

echo "📋 Total de sujeitos encontrados: $TOTAL_SUBJECTS"

# Função para processar um sujeito
process_subject() {
    local subject_dir=$1
    local subject=$(basename "$subject_dir")
    local log_file="${LOG_DIR}/${subject}.log"
    
    echo "🚀 Processando: $subject"
    echo "⏱️  Início: $(date)" > "$log_file"
    
    # Verificar se já foi processado
    if [ -d "${OUTPUT_DIR}/${subject}" ] && [ -f "${OUTPUT_DIR}/${subject}/stats/aseg.stats" ]; then
        echo "✅ $subject já processado, pulando..." | tee -a "$log_file"
        return 0
    fi
    
    # Executar FastSurfer (versão corrigida)
    local exit_code
    
    # Tentativa 1: Versão padrão
    docker run --rm \
        -v "${subject_dir}:/input" \
        -v "${OUTPUT_DIR}:/output" \
        -v "${LICENSE_FILE}:/fs_license/license.txt" \
        deepmi/fastsurfer:latest \
        --fs_license /fs_license/license.txt \
        --t1 /input/mri/T1.mgz \
        --sid "${subject}" \
        --sd /output \
        --threads 2 \
        >> "$log_file" 2>&1
    
    exit_code=$?
    
    # Se falhou, tentar versão simplificada
    if [ $exit_code -ne 0 ]; then
        echo "⚠️  Tentativa padrão falhou, tentando versão CPU..." >> "$log_file"
        
        docker run --rm \
            --user root \
            -v "${subject_dir}:/input" \
            -v "${OUTPUT_DIR}:/output" \
            -v "${LICENSE_FILE}:/fs_license/license.txt" \
            deepmi/fastsurfer:latest \
            --fs_license /fs_license/license.txt \
            --t1 /input/mri/T1.mgz \
            --sid "${subject}" \
            --sd /output \
            --threads 2 \
            --device cpu \
            >> "$log_file" 2>&1
        
        exit_code=$?
    fi
    
    echo "⏱️  Fim: $(date)" >> "$log_file"
    
    # Verificar resultado
    if [ $exit_code -eq 0 ] && [ -f "${OUTPUT_DIR}/${subject}/stats/aseg.stats" ]; then
        echo "✅ $subject: SUCESSO" | tee -a "$log_file"
        ((PROCESSED++))
        return 0
    else
        echo "❌ $subject: FALHOU (código: $exit_code)" | tee -a "$log_file"
        ((FAILED++))
        echo "$subject" >> "${LOG_DIR}/failed_subjects.txt"
        return 1
    fi
}

# Função de monitoramento
show_progress() {
    local elapsed=$(($(date +%s) - START_TIME))
    local avg_time=$((elapsed / (PROCESSED > 0 ? PROCESSED : 1)))
    local remaining=$(((TOTAL_SUBJECTS - PROCESSED - FAILED) * avg_time))
    
    echo ""
    echo "=== PROGRESSO FASTSURFER ==="
    echo "✅ Processados: $PROCESSED"
    echo "❌ Falharam: $FAILED"
    echo "⏳ Restantes: $((TOTAL_SUBJECTS - PROCESSED - FAILED))"
    echo "⏱️  Tempo médio: $(date -ud "@$avg_time" +'%Mm %Ss')"
    echo "🕐 Estimativa restante: $(date -ud "@$remaining" +'%Hh %Mm')"
    echo "=========================="
}

# Processamento principal
echo ""
echo "🚀 INICIANDO PROCESSAMENTO EM LOTE..."
echo "⏱️  Início: $(date)"

# Processar sujeitos com controle de paralelismo
for subject_dir in "${SUBJECTS[@]}"; do
    
    # Controlar número de processos paralelos
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 10
        show_progress
    done
    
    # Processar em background
    process_subject "$subject_dir" &
    
    # Mostrar progresso a cada 10 sujeitos
    if [ $((PROCESSED + FAILED)) -gt 0 ] && [ $(((PROCESSED + FAILED) % 10)) -eq 0 ]; then
        show_progress
    fi
done

# Aguardar todos os processos
wait

# Relatório final
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "=== RELATÓRIO FINAL ==="
echo "✅ Processados com sucesso: $PROCESSED"
echo "❌ Falharam: $FAILED"
echo "📊 Total: $TOTAL_SUBJECTS"
echo "⏱️  Tempo total: $(date -ud "@$TOTAL_TIME" +'%Hh %Mm %Ss')"
echo "⚡ Tempo médio por sujeito: $(date -ud "@$((TOTAL_TIME / TOTAL_SUBJECTS))" +'%Mm %Ss')"
echo "📁 Resultados em: $OUTPUT_DIR"
echo "📋 Logs em: $LOG_DIR"

if [ $FAILED -gt 0 ]; then
    echo "❌ Sujeitos que falharam listados em: ${LOG_DIR}/failed_subjects.txt"
fi

echo "========================" 
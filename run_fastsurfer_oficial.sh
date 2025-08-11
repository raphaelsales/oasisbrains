#!/bin/bash

<<<<<<< HEAD
# PROCESSAMENTO OFICIAL FASTSURFER - VERSÃO DEFINITIVA
# Processa todos os sujeitos do OASIS com configurações otimizadas

set -e

# Configurações principais
OASIS_DIR="/app/alzheimer/oasis_data"
OUTPUT_DIR="/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos"
STATE_FILE="/app/alzheimer/processing_state.txt"
LICENSE_FILE="/app/alzheimer/freesurfer_license.txt"

# Verificar licença
if [ ! -f "$LICENSE_FILE" ]; then
    echo "Licença oficial não encontrada: $LICENSE_FILE"
    echo "Certifique-se de que o arquivo existe com a licença do FreeSurfer"
    exit 1
fi

echo "Licença oficial configurada: $LICENSE_FILE"

# Criar diretórios
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$STATE_FILE")"

# Inicializar ou carregar estado
PROCESSED=0
if [ -f "$STATE_FILE" ]; then
    PROCESSED=$(cat "$STATE_FILE")
fi

if [ $PROCESSED -gt 0 ]; then
    echo "Carregando estado anterior..."
    echo "$PROCESSED sujeitos já processados anteriormente"
fi

# Coletar todos os sujeitos disponíveis
echo "Coletando sujeitos disponíveis..."

cd "$OASIS_DIR"
ALL_SUBJECTS=()

# Procurar em todas as estruturas possíveis
for pattern in "OAS1_*_MR1" "disc*/OAS1_*_MR1" "*/OAS1_*_MR1"; do
    while IFS= read -r -d '' subject_dir; do
        if [ -f "$subject_dir/mri/T1.mgz" ]; then
            subject_id=$(basename "$subject_dir")
            ALL_SUBJECTS+=("$subject_id:$subject_dir")
        fi
    done < <(find . -path "./$pattern" -type d -print0 2>/dev/null)
done

TOTAL_SUBJECTS=${#ALL_SUBJECTS[@]}
echo "Total de sujeitos encontrados: $TOTAL_SUBJECTS"
echo "Resultados serão salvos em: $OUTPUT_DIR"

# Configurações Docker otimizadas
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Processar cada sujeito
for subject_entry in "${ALL_SUBJECTS[@]}"; do
    IFS=':' read -r subject subject_path <<< "$subject_entry"
    
    RESULT_DIR="$OUTPUT_DIR/$subject"
    
    # Verificar se já foi processado
    if [ -d "$RESULT_DIR" ] && [ -f "$RESULT_DIR/stats/aseg.stats" ]; then
        echo "$subject: JÁ PROCESSADO (pulando)"
        continue
    fi
    
    # Verificar se resultado já existe (mesmo sem stats completos)
    if [ -d "$RESULT_DIR" ] && [ "$(find "$RESULT_DIR" -name "*.mgz" | wc -l)" -gt 5 ]; then
        echo "$subject: RESULTADO EXISTE (pulando)"
        continue
    fi
    
    PROCESSED=$((PROCESSED + 1))
    echo "$PROCESSED sujeitos já processados anteriormente"
    
    echo "Processando: $subject"
    echo "  Entrada: $subject_path/mri/T1.mgz"
    echo "  Saída: $RESULT_DIR"
    echo "  Progresso: $PROCESSED/$TOTAL_SUBJECTS"
    
    # Comando FastSurfer otimizado
    if timeout 7200 docker run --rm \
        --gpus all \
        --user "$USER_ID:$GROUP_ID" \
        -v "$OASIS_DIR:/data" \
        -v "$OUTPUT_DIR:/output" \
        -v "$LICENSE_FILE:/fs_license/license.txt:ro" \
        deepmi/fastsurfer:latest \
        --fs_license /fs_license/license.txt \
        --t1 "/data/${subject_path#./}/mri/T1.mgz" \
        --sid "$subject" \
        --sd /output \
        --parallel \
        --3T; then
        
        echo "  SUCESSO: $subject processado"
        
        # Verificar arquivos críticos
        if [ -f "$RESULT_DIR/stats/aseg.stats" ] && [ -f "$RESULT_DIR/mri/aparc+aseg.mgz" ]; then
            echo "  VALIDADO: Arquivos principais confirmados"
        else
            echo "  AVISO: Processamento pode estar incompleto"
        fi
        
    else
        echo "  ERRO: Falha no processamento de $subject"
        
        # Limpeza em caso de erro
        if [ -d "$RESULT_DIR" ]; then
            rm -rf "$RESULT_DIR"
            echo "  LIMPEZA: Diretório removido após erro"
        fi
        
        continue
    fi
    
    # Salvar progresso
    echo "$PROCESSED" > "$STATE_FILE"
    
    # Status a cada 10 sujeitos
    if [ $((PROCESSED % 10)) -eq 0 ]; then
        echo
        echo "CHECKPOINT: $PROCESSED/$TOTAL_SUBJECTS sujeitos processados"
        echo "Progresso: $(echo "scale=1; $PROCESSED * 100 / $TOTAL_SUBJECTS" | bc -l)%"
        echo
    fi
    
    # Pausa entre processamentos
    sleep 2
    
done

echo
echo "PROCESSAMENTO CONCLUÍDO!"
echo "========================"
echo "Total processado: $PROCESSED sujeitos"
echo "Diretório de saída: $OUTPUT_DIR"

# Estatísticas finais
SUCCESSFUL=$(find "$OUTPUT_DIR" -name "aseg.stats" | wc -l)
echo "Sucessos confirmados: $SUCCESSFUL"

if [ $SUCCESSFUL -gt 0 ]; then
    echo
    echo "PRÓXIMOS PASSOS:"
    echo "1. Verificar resultados: ls -la $OUTPUT_DIR"
    echo "2. Executar análise IA: python3 alzheimer_ai_pipeline.py"
    echo "3. Gerar relatórios: python3 dataset_explorer.py"
    echo
    echo "PROCESSAMENTO OFICIAL CONCLUÍDO COM SUCESSO!"
else
    echo
    echo "NENHUM SUJEITO FOI PROCESSADO COM SUCESSO"
    echo "Verifique:"
    echo "1. Licença FreeSurfer"
    echo "2. Espaço em disco"
    echo "3. Configuração Docker"
    echo "4. Logs de erro"
fi 
=======
echo "=== FastSurfer - PROCESSAMENTO EM LOTE OFICIAL ==="

# Configurações
DATA_BASE="/app/alzheimer/oasis_data"
OUTPUT_DIR="${DATA_BASE}/outputs_fastsurfer_oficial"
LOG_DIR="${DATA_BASE}/fastsurfer_logs_oficial"
MAX_PARALLEL=2

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Usar licença oficial
LICENSE_FILE="./freesurfer_license_oficial.txt"
if [ ! -f "$LICENSE_FILE" ]; then
    echo "❌ Licença oficial não encontrada: $LICENSE_FILE"
    echo "💡 Certifique-se de que o arquivo existe com a licença do FreeSurfer"
    exit 1
fi

echo "✅ Licença oficial configurada: $LICENSE_FILE"

# Contadores e estatísticas
TOTAL_SUBJECTS=0
PROCESSED=0
FAILED=0
SKIPPED=0
START_TIME=$(date +%s)

# Arquivo de controle
STATE_FILE="${LOG_DIR}/processamento_estado.txt"
FAILED_FILE="${LOG_DIR}/failed_subjects.txt"

# Carregar estado anterior se existir
if [ -f "$STATE_FILE" ]; then
    echo "📋 Carregando estado anterior..."
    PROCESSED=$(wc -l < "$STATE_FILE" 2>/dev/null || echo 0)
    echo "✅ $PROCESSED sujeitos já processados anteriormente"
fi

# Coletar todos os sujeitos
echo "📊 Coletando sujeitos disponíveis..."
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
echo "📁 Resultados serão salvos em: $OUTPUT_DIR"
echo "📝 Logs serão salvos em: $LOG_DIR"

# Função para processar um sujeito
process_subject() {
    local subject_dir=$1
    local subject=$(basename "$subject_dir")
    local log_file="${LOG_DIR}/${subject}.log"
    
    # Verificar se já foi processado
    if [ -f "$STATE_FILE" ] && grep -q "^$subject$" "$STATE_FILE"; then
        echo "✅ $subject: JÁ PROCESSADO (pulando)"
        ((SKIPPED++))
        return 0
    fi
    
    # Verificar se resultado já existe e está completo
    if [ -d "${OUTPUT_DIR}/${subject}" ] && [ -f "${OUTPUT_DIR}/${subject}/stats/aseg.stats" ]; then
        echo "✅ $subject: RESULTADO EXISTE (pulando)"
        echo "$subject" >> "$STATE_FILE"
        ((SKIPPED++))
        return 0
    fi
    
    echo "🚀 Processando: $subject"
    echo "⏱️  Início: $(date)" > "$log_file"
    echo "📁 Input: $subject_dir" >> "$log_file"
    echo "📁 Output: ${OUTPUT_DIR}/${subject}" >> "$log_file"
    echo "======================================" >> "$log_file"
    
    # Executar FastSurfer
    docker run --rm \
        -v "${subject_dir}:/input" \
        -v "${OUTPUT_DIR}:/output" \
        -v "${LICENSE_FILE}:/fs_license/license.txt" \
        deepmi/fastsurfer:latest \
        --fs_license /fs_license/license.txt \
        --t1 /input/mri/T1.mgz \
        --sid "${subject}" \
        --sd /output \
        --threads 4 \
        --parallel \
        >> "$log_file" 2>&1
    
    local exit_code=$?
    echo "⏱️  Fim: $(date)" >> "$log_file"
    echo "🔍 Exit code: $exit_code" >> "$log_file"
    
    # Verificar resultado
    if [ $exit_code -eq 0 ] && [ -f "${OUTPUT_DIR}/${subject}/stats/aseg.stats" ]; then
        echo "✅ $subject: SUCESSO" | tee -a "$log_file"
        echo "$subject" >> "$STATE_FILE"
        ((PROCESSED++))
        
        # Validação adicional dos arquivos importantes
        local validation_ok=true
        if [ ! -f "${OUTPUT_DIR}/${subject}/mri/aparc+aseg.mgz" ]; then
            echo "⚠️  $subject: aparc+aseg.mgz não encontrado" | tee -a "$log_file"
            validation_ok=false
        fi
        
        if [ ! -d "${OUTPUT_DIR}/${subject}/surf" ] || [ $(ls "${OUTPUT_DIR}/${subject}/surf" | wc -l) -lt 8 ]; then
            echo "⚠️  $subject: Superfícies incompletas" | tee -a "$log_file"
            validation_ok=false
        fi
        
        if [ "$validation_ok" = true ]; then
            echo "🎉 $subject: VALIDAÇÃO COMPLETA" | tee -a "$log_file"
        fi
        
        return 0
    else
        echo "❌ $subject: FALHOU (código: $exit_code)" | tee -a "$log_file"
        echo "$subject" >> "$FAILED_FILE"
        ((FAILED++))
        return 1
    fi
}

# Função de monitoramento
show_progress() {
    local current_total=$((PROCESSED + FAILED + SKIPPED))
    local elapsed=$(($(date +%s) - START_TIME))
    local avg_time=$((elapsed / (current_total > 0 ? current_total : 1)))
    local remaining=$(((TOTAL_SUBJECTS - current_total) * avg_time))
    
    clear
    echo "════════════════════════════════════════════════════════════"
    echo "🚀 FASTSURFER - PROCESSAMENTO EM LOTE OFICIAL"
    echo "════════════════════════════════════════════════════════════"
    echo "📊 PROGRESSO:"
    echo "   ✅ Processados com sucesso: $PROCESSED"
    echo "   ❌ Falharam: $FAILED"
    echo "   ⏭️  Pulados (já processados): $SKIPPED"
    echo "   ⏳ Restantes: $((TOTAL_SUBJECTS - PROCESSED - FAILED - SKIPPED))"
    echo "   📈 Total: $TOTAL_SUBJECTS"
    echo ""
    echo "⏱️  TEMPO:"
    echo "   🕐 Decorrido: $(date -ud "@$elapsed" +'%Hh %Mm %Ss')"
    echo "   ⚡ Médio por sujeito: $(date -ud "@$avg_time" +'%Mm %Ss')"
    echo "   🔮 Estimativa restante: $(date -ud "@$remaining" +'%Hh %Mm')"
    echo ""
    echo "📁 ARQUIVOS:"
    echo "   📂 Resultados: $OUTPUT_DIR"
    echo "   📝 Logs: $LOG_DIR"
    echo "   📋 Estado: $STATE_FILE"
    echo ""
    echo "🔄 Processos ativos: $(jobs -r | wc -l)/$MAX_PARALLEL"
    echo "════════════════════════════════════════════════════════════"
}

# Processamento principal
echo ""
echo "🚀 INICIANDO PROCESSAMENTO EM LOTE COM LICENÇA OFICIAL..."
echo "⏱️  Início: $(date)"
echo ""

# Loop de monitoramento em background
(
    while true; do
        show_progress
        sleep 30
    done
) &
MONITOR_PID=$!

# Processar sujeitos com controle de paralelismo
for subject_dir in "${SUBJECTS[@]}"; do
    
    # Controlar número de processos paralelos
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 5
    done
    
    # Processar em background
    process_subject "$subject_dir" &
    
    # Pequena pausa para evitar sobrecarga
    sleep 2
done

# Aguardar todos os processos
wait

# Parar monitoramento
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

# Relatório final
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

clear
echo "════════════════════════════════════════════════════════════"
echo "🎉 PROCESSAMENTO FASTSURFER CONCLUÍDO!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📊 ESTATÍSTICAS FINAIS:"
echo "   ✅ Processados com sucesso: $PROCESSED"
echo "   ❌ Falharam: $FAILED"
echo "   ⏭️  Pulados (já existiam): $SKIPPED"
echo "   📈 Total de sujeitos: $TOTAL_SUBJECTS"
echo ""
echo "⏱️  TEMPO TOTAL: $(date -ud "@$TOTAL_TIME" +'%Hh %Mm %Ss')"
if [ $PROCESSED -gt 0 ]; then
    echo "   ⚡ Tempo médio por sujeito: $(date -ud "@$((TOTAL_TIME / PROCESSED))" +'%Mm %Ss')"
fi
echo ""
echo "📁 RESULTADOS:"
echo "   📂 Dados processados: $OUTPUT_DIR"
echo "   📝 Logs detalhados: $LOG_DIR"
echo "   📋 Estado salvo em: $STATE_FILE"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "❌ SUJEITOS QUE FALHARAM:"
    echo "   📄 Lista completa: $FAILED_FILE"
    echo "   🔧 Recomendação: Verificar logs individuais para diagnóstico"
fi

if [ $PROCESSED -gt 0 ]; then
    echo ""
    echo "🎉 SUCESSO! Você agora tem $PROCESSED sujeitos processados com FastSurfer!"
    echo "💡 PRÓXIMOS PASSOS:"
    echo "   1. Use os resultados em: $OUTPUT_DIR"
    echo "   2. Atualize seu processar_T1_discos.py para usar estes dados"
    echo "   3. Execute suas análises de hipocampo"
fi

echo ""
echo "════════════════════════════════════════════════════════════" 
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)

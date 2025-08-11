#!/bin/bash

# ---------------------------------------
# CONFIGURAÇÕES GLOBAIS
# ---------------------------------------
DATA_BASE="/app/alzheimer/oasis_data"
MAX_PARALLEL=2
LOG_DIR="${DATA_BASE}/processing_logs"
STATE_FILE="${LOG_DIR}/state.txt"
FAILED_FILE="${LOG_DIR}/failed_subjects.txt"
LOCK_FILE="${LOG_DIR}/.lock"

# Criação de diretórios e arquivos necessários
mkdir -p "${LOG_DIR}"
touch "${STATE_FILE}" "${FAILED_FILE}"
exec 200>"${LOCK_FILE}"

# Carregando o estado anterior
declare -A PROCESSED
while IFS= read -r line; do
    PROCESSED["$line"]=1
done < "${STATE_FILE}"

TOTAL_SUBJECTS=0
COMPLETED=${#PROCESSED[@]}
START_TIME=$(date +%s)

# ---------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------

# Atualiza o state.txt e o log de progresso
update_state() {
    local subject=$1
    flock -x 200
    echo "$subject" >> "${STATE_FILE}"
    echo "$(date) - [STATUS] Progresso: $COMPLETED/$TOTAL_SUBJECTS" >> "${LOG_DIR}/progresso.log"
}

# Função principal de processamento para cada sujeito
process_subject() {
    local DISK_DIR=$1
    local subject=$2
    local subject_dir=$3
    local log_file="${LOG_DIR}/${subject}.log"

    export DISK_DIR subject subject_dir log_file

    (
        echo "===== INÍCIO: $(date) ====="
        echo "Subject: ${subject}"
        echo "Disco: ${DISK_DIR}"
        echo "Diretório: ${subject_dir}"

        # Verificar arquivo de entrada
        if [[ ! -f "${subject_dir}/mri/T1.mgz" ]]; then
            echo "[ERRO] Arquivo T1.mgz não encontrado!"
            return 1
        fi

        # Se houver diretório de saída anterior, removê-lo
        rm -rf "${DISK_DIR}/outputs/${subject}"

        # Executa o container Docker do FastSurfer com o flag --allow_root para forçar execução como root
        if ! docker run --gpus all --rm \
            -u $(id -u):$(id -g) \
            -v "${DISK_DIR}:/data" \
            -v "$HOME/license.txt:/license.txt" \
            deepmi/fastsurfer:latest \
            --allow_root \
            --fs_license /license.txt \
            --t1 "/data/${subject}/mri/T1.mgz" \
            --sid "${subject}" \
            --sd "/data/outputs" \
            --threads 4; then

            echo "[ERRO] Falha no container Docker"
            return 2
        fi

        # Valida se o resultado foi gerado corretamente
        if [[ ! -f "${DISK_DIR}/outputs/${subject}/stats/aseg.stats" ]]; then
            echo "[ERRO] Arquivo de saída não gerado"
            return 3
        fi

        echo "===== SUCESSO: $(date) ====="
        return 0
    ) 2>&1 | tee -a "$log_file"
}

# Monitoramento em tempo real de jobs ativos
monitor_jobs() {
    while true; do
        clear
        echo "===== STATUS PROCESSAMENTO ====="
        echo "Subjects processados: $COMPLETED/$TOTAL_SUBJECTS"
        echo "Tempo decorrido: $(date -ud @$(( $(date +%s) - START_TIME )) +'%Hh %Mm %Ss')"
        echo ""
        echo "=== Processos em execução ==="
        jobs -r -l | while read -r job_line; do
            echo "$job_line"
        done
        sleep 5
    done
}

# ---------------------------------------
# COLETA E FILTRAGEM DE SUJEITOS
# ---------------------------------------
declare -a SUBJECT_QUEUE

for disc in {1..11}; do
    DISK_DIR="${DATA_BASE}/disc${disc}"
    [[ -d "${DISK_DIR}" ]] || continue

    while IFS= read -r -d $'\0' subject_dir; do
        subject=$(basename "${subject_dir}")
        # Se ainda não foi processado, adiciona na fila
        if [[ -z "${PROCESSED[$subject]}" ]]; then
            SUBJECT_QUEUE+=("${DISK_DIR} ${subject} ${subject_dir}")
            ((TOTAL_SUBJECTS++))
        fi
    done < <(find "${DISK_DIR}" -maxdepth 1 -type d -name "OAS1_*_MR1" -print0)
done

# ---------------------------------------
# LOG INICIAL
# ---------------------------------------
echo "===== INÍCIO DO PROCESSAMENTO =====" | tee -a "${LOG_DIR}/progresso.log"
echo "Subjects a processar: ${TOTAL_SUBJECTS}" | tee -a "${LOG_DIR}/progresso.log"
echo "Já processados: ${COMPLETED}" | tee -a "${LOG_DIR}/progresso.log"

# Trata interrupções (Ctrl+C) para parar e salvar estado
trap 'echo "Interrupção recebida. Encerrando..."; kill $(jobs -p); exit 2' SIGINT SIGTERM

# Inicia thread de monitoramento em segundo plano
monitor_jobs &
MONITOR_PID=$!

# ---------------------------------------
# PROCESSAMENTO PRINCIPAL
# ---------------------------------------
for subject_info in "${SUBJECT_QUEUE[@]}"; do

    # Limita a quantidade de processos simultâneos
    while [[ $(jobs -r | wc -l) -ge MAX_PARALLEL ]]; do
        sleep 1
    done

    read -r DISK_DIR subject subject_dir <<< "${subject_info}"

    # Dispara o processamento em segundo plano
    (
        echo "Iniciando processamento: Disco=${DISK_DIR}, Subject=${subject}"
        if process_subject "${DISK_DIR}" "${subject}" "${subject_dir}"; then
            flock -x 200
            ((COMPLETED++))
            update_state "$subject"
        else
            flock -x 200
            echo "$subject" >> "${FAILED_FILE}"
        fi
    ) &
done

# Espera todos os processos terminarem
wait

# Finaliza o monitor de jobs
kill "$MONITOR_PID" 2>/dev/null
wait "$MONITOR_PID" 2>/dev/null

# ---------------------------------------
# RELATÓRIO FINAL
# ---------------------------------------
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

{
    echo "===== PROCESSAMENTO CONCLUÍDO ====="
    echo "Tempo Total: $(date -ud "@$TOTAL_DURATION" +'%Hh %Mm %Ss')"
    echo "Subjects com erro: $(wc -l < "${FAILED_FILE}")"
    echo "Último Subject Processado: $(tail -n 1 "${STATE_FILE}")"
} | tee -a "${LOG_DIR}/progresso.log"

exit 0
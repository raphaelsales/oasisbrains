#!/bin/bash

# Configurações globais
DATA_BASE="/app/alzheimer/oasis_data"
MAX_PARALLEL=2
LOG_DIR="${DATA_BASE}/processing_logs"
STATE_FILE="${LOG_DIR}/state.txt"
FAILED_FILE="${LOG_DIR}/failed_subjects.txt"
LOCK_FILE="${LOG_DIR}/.lock"

# Inicialização segura
mkdir -p "${LOG_DIR}"
touch "${STATE_FILE}" "${FAILED_FILE}"
exec 200>"${LOCK_FILE}"

# Carregar estado anterior
declare -A PROCESSED
while IFS= read -r line; do
    PROCESSED["$line"]=1
done < "${STATE_FILE}"

TOTAL_SUBJECTS=0
COMPLETED=${#PROCESSED[@]}
START_TIME=$(date +%s)

# Função thread-safe para atualização de estado
update_state() {
    local subject=$1
    (
        flock -x 200
        echo "$subject" >> "${STATE_FILE}"
        echo "$(date) - [STATUS] Progresso: $COMPLETED/$TOTAL_SUBJECTS" >> "${LOG_DIR}/progresso.log"
    ) 200>"${LOCK_FILE}"
}

# Função de progresso com cálculo preciso
update_progress() {
    (
        flock -x 200
        local elapsed=$(( $(date +%s) - START_TIME ))
        local avg_time=$(( elapsed / (COMPLETED > 0 ? COMPLETED : 1) ))
        local remaining=$(( (TOTAL_SUBJECTS - COMPLETED) * avg_time ))

        printf "\n=== Processamento Alzheimer ===\n"
        printf "Progresso: %d/%d\n" "$COMPLETED" "$TOTAL_SUBJECTS"
        printf "Tempo Médio: %s\n" "$(date -ud "@$avg_time" +'%Hh %Mm %Ss')"
        printf "Estimado Restante: %s\n" "$(date -ud "@$remaining" +'%Hh %Mm %Ss')"
        printf "Última Atualização: %s\n" "$(date +'%d/%m/%Y %H:%M:%S')"
    ) 200>"${LOCK_FILE}" | tee -a "${LOG_DIR}/progresso.log"
}

# Loop contínuo de progresso
progress_loop() {
    while true; do
        update_progress
        sleep 60
    done
}

# Função principal de processamento
process_subject() {
    local DISK_DIR=$1
    local subject=$2
    local subject_dir=$3
    local log_file="${LOG_DIR}/${subject}.log"

    # Configuração do ambiente
    export DISK_DIR subject subject_dir log_file

    (
        echo "===== INÍCIO: $(date) ====="
        echo "Subject: ${subject}"
        echo "Disco: ${DISK_DIR}"
        echo "Diretório: ${subject_dir}"

        # Verificar arquivo de entrada
        if [[ ! -f "${subject_dir}/mri/T1.mgz" ]]; then
            echo "[ERRO] Arquivo T1.mgz não encontrado!" | tee -a "$log_file"
            return 1
        fi

        # Remover saída anterior se existir
        rm -rf "${DISK_DIR}/outputs/${subject}"

        # Processamento com Docker
        if ! docker run --gpus all --rm \
            -u $(id -u):$(id -g) \
            -v "${DISK_DIR}:/data" \
            -v "$HOME/license.txt:/license.txt" \
            deepmi/fastsurfer:latest \
            --fs_license /license.txt \
            --t1 "/data/${subject}/mri/T1.mgz" \
            --sid "${subject}" \
            --sd "/data/outputs" \
            --threads 4 >> "$log_file" 2>&1; then

            echo "[ERRO] Falha no container Docker" | tee -a "$log_file"
            return 2
        fi

        # Validação da saída
        if [[ ! -f "${DISK_DIR}/outputs/${subject}/stats/aseg.stats" ]]; then
            echo "[ERRO] Arquivo de saída não gerado" | tee -a "$log_file"
            return 3
        fi

        echo "===== SUCESSO: $(date) ====="
        return 0
    ) >> "$log_file" 2>&1
}

# Spinner visual
spinner() {
    local pid=$1
    local delay=0.2
    local spinstr='|/-\'
    tput civis
    while kill -0 "$pid" 2>/dev/null; do
        local temp=${spinstr#?}
        printf " [%c] Processando..." "$spinstr"
        spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\r"
    done
    tput cnorm
    echo ""
}

# Coletar e filtrar subjects
declare -a SUBJECT_QUEUE
for disc in {1..11}; do
    DISK_DIR="${DATA_BASE}/disc${disc}"
    [[ -d "${DISK_DIR}" ]] || continue

    while IFS= read -r -d $'\0' subject_dir; do
        subject=$(basename "${subject_dir}")
        if [[ -z "${PROCESSED[$subject]}" ]]; then
            SUBJECT_QUEUE+=("${DISK_DIR} ${subject} ${subject_dir}")
            ((TOTAL_SUBJECTS++))
        fi
    done < <(find "${DISK_DIR}" -maxdepth 1 -type d -name "OAS1_*_MR1" -print0)
done

# Registro inicial
echo "===== INÍCIO DO PROCESSAMENTO =====" | tee -a "${LOG_DIR}/progresso.log"
echo "Subjects a processar: ${TOTAL_SUBJECTS}" | tee -a "${LOG_DIR}/progresso.log"
echo "Já processados: ${COMPLETED}" | tee -a "${LOG_DIR}/progresso.log"

# Armadilha para interrupções
trap 'echo "INTERRUPÇÃO RECEBIDA! Salvando estado..."; exit 2' SIGINT SIGTERM

# Inicia progresso contínuo
progress_loop &
PROGRESS_PID=$!

# Processamento principal em background
(
    for subject_info in "${SUBJECT_QUEUE[@]}"; do
        while [[ $(jobs -r | wc -l) -ge MAX_PARALLEL ]]; do
            sleep $(( RANDOM % 5 + 1 ))
        done

        read -r DISK_DIR subject subject_dir <<< "${subject_info}"

        (
            if process_subject "$DISK_DIR" "$subject" "$subject_dir"; then
                (
                    flock -x 200
                    ((COMPLETED++))
                    update_state "$subject"
                ) 200>"${LOCK_FILE}"
            else
                (
                    flock -x 200
                    echo "$subject" >> "${FAILED_FILE}"
                ) 200>"${LOCK_FILE}"
            fi
        ) &
    done
    wait
) &
MAIN_PID=$!

# Mostrar spinner
spinner "$MAIN_PID"

# Finaliza progresso
kill "$PROGRESS_PID" 2>/dev/null
wait "$PROGRESS_PID" 2>/dev/null

# Relatório final
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

(
    echo "===== PROCESSAMENTO CONCLUÍDO ====="
    echo "Tempo Total: $(date -ud "@$TOTAL_DURATION" +'%Hh %Mm %Ss')"
    echo "Subjects com erro: $(wc -l < "${FAILED_FILE}")"
    echo "Último Subject Processado: $(tail -n 1 "${STATE_FILE}")"
) | tee -a "${LOG_DIR}/progresso.log"

exit 0
#!/bin/bash

# Script otimizado usando FastSurfer
DATA_BASE="/app/alzheimer/oasis_data"
OUTPUT_DIR="${DATA_BASE}/outputs_fastsurfer"
LOG_DIR="${DATA_BASE}/fastsurfer_logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Função para processar um sujeito com FastSurfer
process_with_fastsurfer() {
    local subject_dir=$1
    local subject=$(basename "$subject_dir")
    local t1_file="${subject_dir}/mri/T1.mgz"
    
    echo "Processando $subject com FastSurfer..."
    
    if [ ! -f "$t1_file" ]; then
        echo "❌ Arquivo T1.mgz não encontrado para $subject"
        return 1
    fi
    
    # Executar FastSurfer
    docker run --gpus all --rm \
        -v "${subject_dir}:/input" \
        -v "${OUTPUT_DIR}:/output" \
        -v "$HOME/license.txt:/license.txt" \
        deepmi/fastsurfer:latest \
        --fs_license /license.txt \
        --t1 /input/mri/T1.mgz \
        --sid "${subject}" \
        --sd /output \
        --parallel \
        --threads 4 \
        2>&1 | tee "${LOG_DIR}/${subject}_fastsurfer.log"
    
    if [ $? -eq 0 ]; then
        echo "✅ $subject processado com sucesso"
        return 0
    else
        echo "❌ Erro ao processar $subject"
        return 1
    fi
}

# Processar todos os sujeitos pendentes
echo "Iniciando processamento com FastSurfer..."
for disc in {1..11}; do
    DISK_DIR="${DATA_BASE}/disc${disc}"
    [ -d "$DISK_DIR" ] || continue
    
    for subject_dir in "${DISK_DIR}"/OAS1_*_MR1; do
        [ -d "$subject_dir" ] || continue
        process_with_fastsurfer "$subject_dir"
    done
done

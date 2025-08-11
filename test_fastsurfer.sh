#!/bin/bash

echo "=== Teste FastSurfer - Um Sujeito ==="

# Configurações
DATA_BASE="/app/alzheimer/oasis_data"
OUTPUT_DIR="${DATA_BASE}/test_fastsurfer_output"
LOG_DIR="${DATA_BASE}/test_logs"

# Criar diretórios
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Selecionar primeiro sujeito disponível
TEST_SUBJECT=""
for disc in {1..11}; do
    DISK_DIR="${DATA_BASE}/disc${disc}"
    if [ -d "$DISK_DIR" ]; then
        for subject_dir in "${DISK_DIR}"/OAS1_*_MR1; do
            if [ -d "$subject_dir" ] && [ -f "${subject_dir}/mri/T1.mgz" ]; then
                TEST_SUBJECT="$subject_dir"
                break 2
            fi
        done
    fi
done

if [ -z "$TEST_SUBJECT" ]; then
    echo "❌ Nenhum sujeito com T1.mgz encontrado!"
    exit 1
fi

SUBJECT_ID=$(basename "$TEST_SUBJECT")
echo "📋 Testando com sujeito: $SUBJECT_ID"
echo "📁 Diretório: $TEST_SUBJECT"
echo "📁 Saída: $OUTPUT_DIR"

# Verificar arquivo T1
T1_FILE="${TEST_SUBJECT}/mri/T1.mgz"
if [ ! -f "$T1_FILE" ]; then
    echo "❌ Arquivo T1.mgz não encontrado: $T1_FILE"
    exit 1
fi

echo "✅ Arquivo T1.mgz encontrado: $T1_FILE"
echo "📊 Tamanho: $(du -h "$T1_FILE" | cut -f1)"

# Verificar licença
LICENSE_FILE="$HOME/license.txt"
if [ ! -f "$LICENSE_FILE" ]; then
    echo "⚠️  Licença não encontrada em: $LICENSE_FILE"
    echo "Tentando localizar licença..."
    find /home -name "license.txt" 2>/dev/null | head -3
    LICENSE_FILE="./license.txt"
fi

# Verificar Docker
echo "🐳 Verificando Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker não encontrado!"
    exit 1
fi

if ! docker images | grep -q fastsurfer; then
    echo "📥 Baixando imagem FastSurfer..."
    docker pull deepmi/fastsurfer:latest
fi

echo "✅ Docker configurado"

# EXECUTAR FASTSURFER - TESTE
echo ""
echo "🚀 INICIANDO TESTE FASTSURFER..."
echo "⏱️  Início: $(date)"
echo ""

# Comando FastSurfer
docker run --gpus all --rm \
    -v "${TEST_SUBJECT}:/input" \
    -v "${OUTPUT_DIR}:/output" \
    -v "${LICENSE_FILE}:/license.txt" \
    deepmi/fastsurfer:latest \
    --fs_license /license.txt \
    --t1 /input/mri/T1.mgz \
    --sid "${SUBJECT_ID}_test" \
    --sd /output \
    --parallel \
    --threads 4 \
    2>&1 | tee "${LOG_DIR}/${SUBJECT_ID}_test.log"

EXIT_CODE=$?

echo ""
echo "⏱️  Fim: $(date)"

# Verificar resultado
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TESTE SUCESSO!"
    echo "📁 Resultados em: $OUTPUT_DIR/${SUBJECT_ID}_test"
    
    # Verificar arquivos gerados
    if [ -d "$OUTPUT_DIR/${SUBJECT_ID}_test" ]; then
        echo "📊 Arquivos gerados:"
        ls -la "$OUTPUT_DIR/${SUBJECT_ID}_test/"
        
        # Verificar arquivos importantes
        if [ -f "$OUTPUT_DIR/${SUBJECT_ID}_test/stats/aseg.stats" ]; then
            echo "✅ Arquivo aseg.stats gerado com sucesso"
        else
            echo "⚠️  Arquivo aseg.stats não encontrado"
        fi
    fi
    
    echo ""
    echo "🎉 TESTE CONCLUÍDO COM SUCESSO!"
    echo "💡 Agora você pode executar para todos os sujeitos:"
    echo "   ./run_fastsurfer_optimized.sh"
    
else
    echo "❌ TESTE FALHOU (código: $EXIT_CODE)"
    echo "📋 Verificar log: ${LOG_DIR}/${SUBJECT_ID}_test.log"
    echo ""
    echo "🔧 Possíveis soluções:"
    echo "1. Verificar licença do FreeSurfer"
    echo "2. Verificar se GPU está disponível (nvidia-smi)"
    echo "3. Tentar sem GPU: remover --gpus all"
fi

echo ""
echo "=== FIM DO TESTE ===" 
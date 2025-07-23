#!/bin/bash

echo "=== Teste FastSurfer - VERSÃO CORRIGIDA ==="

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
echo "✅ Arquivo T1.mgz encontrado: $T1_FILE"
echo "📊 Tamanho: $(du -h "$T1_FILE" | cut -f1)"

# CORREÇÃO 1: Criar licença temporária se não existir
LICENSE_FILE="./freesurfer_license.txt"
if [ ! -f "/home/comaisserveria/license.txt" ]; then
    echo "🔧 Criando licença temporária..."
    cat > "$LICENSE_FILE" << 'EOF'
# FreeSurfer License - Para uso acadêmico/pesquisa
# Registre-se em: https://surfer.nmr.mgh.harvard.edu/registration.html
# Este é um placeholder - substitua pela sua licença real
raphael.comaisserveria@email.com
12345
*Ca123456789
FSabcdefghijk
EOF
    echo "⚠️  ATENÇÃO: Licença temporária criada. Registre-se no FreeSurfer para licença oficial!"
else
    LICENSE_FILE="/home/comaisserveria/license.txt"
fi

echo "✅ Licença configurada: $LICENSE_FILE"

# Verificar Docker
echo "🐳 Verificando Docker..."
if ! docker images | grep -q fastsurfer; then
    echo "📥 Baixando imagem FastSurfer..."
    docker pull deepmi/fastsurfer:latest
fi

# CORREÇÃO 2: Comando Docker SEM problemas de usuário
echo ""
echo "🚀 INICIANDO TESTE FASTSURFER (VERSÃO CORRIGIDA)..."
echo "⏱️  Início: $(date)"
echo ""

# VERSÃO 1: Tentar com usuário atual
echo "🔧 Tentativa 1: Docker com usuário atual"
docker run --rm \
    -v "${TEST_SUBJECT}:/input" \
    -v "${OUTPUT_DIR}:/output" \
    -v "${LICENSE_FILE}:/fs_license/license.txt" \
    deepmi/fastsurfer:latest \
    --fs_license /fs_license/license.txt \
    --t1 /input/mri/T1.mgz \
    --sid "${SUBJECT_ID}_test" \
    --sd /output \
    --threads 2 \
    2>&1 | tee "${LOG_DIR}/${SUBJECT_ID}_test_v1.log"

EXIT_CODE_V1=$?

if [ $EXIT_CODE_V1 -ne 0 ]; then
    echo "⚠️  Tentativa 1 falhou. Tentando versão simplificada..."
    
    # VERSÃO 2: Sem GPU, usuário root
    echo "🔧 Tentativa 2: Docker como root, sem GPU"
    docker run --rm \
        --user root \
        -v "${TEST_SUBJECT}:/input" \
        -v "${OUTPUT_DIR}:/output" \
        -v "${LICENSE_FILE}:/fs_license/license.txt" \
        deepmi/fastsurfer:latest \
        --fs_license /fs_license/license.txt \
        --t1 /input/mri/T1.mgz \
        --sid "${SUBJECT_ID}_test_v2" \
        --sd /output \
        --threads 2 \
        --device cpu \
        2>&1 | tee "${LOG_DIR}/${SUBJECT_ID}_test_v2.log"
    
    EXIT_CODE=$?
    SUBJECT_TEST="${SUBJECT_ID}_test_v2"
else
    EXIT_CODE=$EXIT_CODE_V1
    SUBJECT_TEST="${SUBJECT_ID}_test"
fi

echo ""
echo "⏱️  Fim: $(date)"

# Verificar resultado
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TESTE SUCESSO!"
    echo "📁 Resultados em: $OUTPUT_DIR/$SUBJECT_TEST"
    
    # Verificar arquivos gerados
    if [ -d "$OUTPUT_DIR/$SUBJECT_TEST" ]; then
        echo "📊 Estrutura gerada:"
        find "$OUTPUT_DIR/$SUBJECT_TEST" -type f | head -10
        
        # Verificar arquivos importantes
        if [ -f "$OUTPUT_DIR/$SUBJECT_TEST/stats/aseg.stats" ]; then
            echo "✅ Arquivo aseg.stats gerado com sucesso"
            echo "📊 Tamanho: $(du -h "$OUTPUT_DIR/$SUBJECT_TEST/stats/aseg.stats" | cut -f1)"
        fi
        
        if [ -f "$OUTPUT_DIR/$SUBJECT_TEST/mri/aparc+aseg.mgz" ]; then
            echo "✅ Segmentação aparc+aseg.mgz gerada"
        fi
    fi
    
    echo ""
    echo "🎉 FASTSURFER FUNCIONANDO!"
    echo "⚡ Tempo estimado por sujeito: 30-60 minutos"
    echo "💡 Agora você pode processar todos os sujeitos"
    
else
    echo "❌ TESTE FALHOU (código: $EXIT_CODE)"
    echo "📋 Logs disponíveis:"
    ls -la "${LOG_DIR}"/*.log 2>/dev/null || echo "Nenhum log encontrado"
    
    echo ""
    echo "🔧 Diagnóstico adicional:"
    echo "1. Verificar GPU: nvidia-smi"
    echo "2. Verificar espaço em disco: df -h"
    echo "3. Verificar memória: free -h"
fi

echo ""
echo "=== FIM DO TESTE CORRIGIDO ===" 
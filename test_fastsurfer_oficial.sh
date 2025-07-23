#!/bin/bash

echo "=== Teste FastSurfer - VERSÃO OFICIAL ==="

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

# Usar licença oficial
LICENSE_FILE="./freesurfer_license_oficial.txt"
if [ ! -f "$LICENSE_FILE" ]; then
    echo "❌ Arquivo de licença oficial não encontrado: $LICENSE_FILE"
    exit 1
fi

echo "✅ Licença oficial configurada: $LICENSE_FILE"

# Verificar Docker
echo "🐳 Verificando Docker..."
if ! docker images | grep -q fastsurfer; then
    echo "📥 Baixando imagem FastSurfer..."
    docker pull deepmi/fastsurfer:latest
fi

echo "✅ Docker configurado"

# Limpar teste anterior se existir
rm -rf "${OUTPUT_DIR}/${SUBJECT_ID}_test_oficial"

echo ""
echo "🚀 INICIANDO TESTE FASTSURFER COM LICENÇA OFICIAL..."
echo "⏱️  Início: $(date)"
echo ""

# COMANDO FASTSURFER - VERSÃO OFICIAL
docker run --rm \
    -v "${TEST_SUBJECT}:/input" \
    -v "${OUTPUT_DIR}:/output" \
    -v "${LICENSE_FILE}:/fs_license/license.txt" \
    deepmi/fastsurfer:latest \
    --fs_license /fs_license/license.txt \
    --t1 /input/mri/T1.mgz \
    --sid "${SUBJECT_ID}_test_oficial" \
    --sd /output \
    --threads 4 \
    --parallel \
    2>&1 | tee "${LOG_DIR}/${SUBJECT_ID}_test_oficial.log"

EXIT_CODE=$?

echo ""
echo "⏱️  Fim: $(date)"

# Verificar resultado
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TESTE COM LICENÇA OFICIAL: SUCESSO!"
    echo "📁 Resultados em: $OUTPUT_DIR/${SUBJECT_ID}_test_oficial"
    
    # Verificar estrutura gerada
    if [ -d "$OUTPUT_DIR/${SUBJECT_ID}_test_oficial" ]; then
        echo ""
        echo "📊 ESTRUTURA GERADA:"
        echo "├── Diretórios:"
        find "$OUTPUT_DIR/${SUBJECT_ID}_test_oficial" -type d | head -10
        echo ""
        echo "├── Arquivos importantes:"
        
        # Verificar arquivos cruciais
        RESULT_DIR="$OUTPUT_DIR/${SUBJECT_ID}_test_oficial"
        
        if [ -f "$RESULT_DIR/stats/aseg.stats" ]; then
            echo "  ✅ aseg.stats ($(du -h "$RESULT_DIR/stats/aseg.stats" | cut -f1))"
        else
            echo "  ❌ aseg.stats AUSENTE"
        fi
        
        if [ -f "$RESULT_DIR/mri/aparc+aseg.mgz" ]; then
            echo "  ✅ aparc+aseg.mgz ($(du -h "$RESULT_DIR/mri/aparc+aseg.mgz" | cut -f1))"
        else
            echo "  ❌ aparc+aseg.mgz AUSENTE"
        fi
        
        if [ -f "$RESULT_DIR/mri/T1.mgz" ]; then
            echo "  ✅ T1.mgz processado ($(du -h "$RESULT_DIR/mri/T1.mgz" | cut -f1))"
        fi
        
        if [ -d "$RESULT_DIR/surf" ]; then
            echo "  ✅ Superfícies geradas ($(ls "$RESULT_DIR/surf" | wc -l) arquivos)"
        fi
        
        echo ""
        echo "📈 ESTATÍSTICAS DO PROCESSAMENTO:"
        
        # Mostrar algumas estatísticas se disponíveis
        if [ -f "$RESULT_DIR/stats/aseg.stats" ]; then
            echo "  📊 Volume total do cérebro:"
            grep "BrainVol" "$RESULT_DIR/stats/aseg.stats" | head -3
        fi
        
        echo ""
        echo "🎉 FASTSURFER OFICIAL FUNCIONANDO PERFEITAMENTE!"
        echo "⚡ Tempo estimado por sujeito: 30-60 minutos"
        echo "🚀 Speedup vs FreeSurfer tradicional: ~10x"
        echo ""
        echo "💡 PRÓXIMO PASSO: Execute o processamento em lote:"
        echo "   chmod +x run_fastsurfer_oficial.sh"
        echo "   ./run_fastsurfer_oficial.sh"
        
    else
        echo "⚠️  Diretório de resultados não encontrado"
    fi
    
else
    echo "❌ TESTE FALHOU (código: $EXIT_CODE)"
    echo "📋 Log detalhado: ${LOG_DIR}/${SUBJECT_ID}_test_oficial.log"
    echo ""
    echo "🔧 Diagnóstico:"
    
    # Mostrar últimas linhas do log
    if [ -f "${LOG_DIR}/${SUBJECT_ID}_test_oficial.log" ]; then
        echo "📄 Últimas 10 linhas do log:"
        tail -10 "${LOG_DIR}/${SUBJECT_ID}_test_oficial.log"
    fi
    
    echo ""
    echo "💡 Possíveis soluções:"
    echo "1. Verificar se a licença está correta"
    echo "2. Verificar espaço em disco: df -h"
    echo "3. Verificar memória: free -h"
    echo "4. Tentar sem paralelismo: remover --parallel"
fi

echo ""
echo "=== FIM DO TESTE OFICIAL ===" 
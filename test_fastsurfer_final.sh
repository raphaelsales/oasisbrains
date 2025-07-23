#!/bin/bash

echo "=== Teste FastSurfer - VERSÃO FINAL (SEM ERRO DOCKER) ==="

# Configurações
DATA_BASE="/app/alzheimer/oasis_data"
OUTPUT_DIR="${DATA_BASE}/test_fastsurfer_final"
LOG_DIR="${DATA_BASE}/test_logs_final"

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
echo "✅ Licença oficial: $LICENSE_FILE"

# Limpar teste anterior
rm -rf "${OUTPUT_DIR}/${SUBJECT_ID}_final"

echo ""
echo "🚀 TESTANDO FASTSURFER (VERSÃO SEM ERRO DOCKER)..."
echo "⏱️  Início: $(date)"

# SOLUÇÃO DEFINITIVA: Usar --user com IDs numéricos específicos
USER_ID=$(id -u)
GROUP_ID=$(id -g)

echo "🔧 Usando User ID: $USER_ID, Group ID: $GROUP_ID"
echo ""

# COMANDO FASTSURFER CORRIGIDO
docker run --rm \
    --user "$USER_ID:$GROUP_ID" \
    -v "${TEST_SUBJECT}:/input" \
    -v "${OUTPUT_DIR}:/output" \
    -v "${LICENSE_FILE}:/license.txt" \
    deepmi/fastsurfer:latest \
    --fs_license /license.txt \
    --t1 /input/mri/T1.mgz \
    --sid "${SUBJECT_ID}_final" \
    --sd /output \
    --threads 4 \
    --device cpu \
    2>&1 | tee "${LOG_DIR}/${SUBJECT_ID}_final.log"

EXIT_CODE=$?

echo ""
echo "⏱️  Fim: $(date)"
echo "🔍 Exit code: $EXIT_CODE"

# Verificar resultado
RESULT_DIR="${OUTPUT_DIR}/${SUBJECT_ID}_final"

if [ $EXIT_CODE -eq 0 ] && [ -d "$RESULT_DIR" ]; then
    echo ""
    echo "✅ TESTE FINAL: SUCESSO!"
    echo "📁 Resultados em: $RESULT_DIR"
    
    # Verificar estrutura
    echo ""
    echo "📊 ARQUIVOS GERADOS:"
    
    # Contar arquivos por tipo
    local mri_count=$(find "$RESULT_DIR" -name "*.mgz" 2>/dev/null | wc -l)
    local stats_count=$(find "$RESULT_DIR" -name "*.stats" 2>/dev/null | wc -l)
    local surf_count=$(find "$RESULT_DIR" -name "*.surf" -o -name "lh.*" -o -name "rh.*" 2>/dev/null | wc -l)
    
    echo "  📈 Arquivos MRI (.mgz): $mri_count"
    echo "  📊 Arquivos Stats: $stats_count"
    echo "  🧠 Arquivos Surface: $surf_count"
    
    # Verificar arquivos chave
    echo ""
    echo "🔍 VALIDAÇÃO DOS ARQUIVOS PRINCIPAIS:"
    
    if [ -f "$RESULT_DIR/stats/aseg.stats" ]; then
        echo "  ✅ aseg.stats: $(du -h "$RESULT_DIR/stats/aseg.stats" | cut -f1)"
    else
        echo "  ❌ aseg.stats: AUSENTE"
    fi
    
    if [ -f "$RESULT_DIR/mri/aparc+aseg.mgz" ]; then
        echo "  ✅ aparc+aseg.mgz: $(du -h "$RESULT_DIR/mri/aparc+aseg.mgz" | cut -f1)"
    else
        echo "  ⚠️  aparc+aseg.mgz: AUSENTE (pode estar em processamento)"
    fi
    
    if [ -f "$RESULT_DIR/mri/T1.mgz" ]; then
        echo "  ✅ T1.mgz: $(du -h "$RESULT_DIR/mri/T1.mgz" | cut -f1)"
    fi
    
    # Mostrar estrutura de diretórios
    echo ""
    echo "📂 ESTRUTURA DE DIRETÓRIOS:"
    find "$RESULT_DIR" -type d | head -10 | sed 's|^|  |'
    
    echo ""
    echo "🎉 FASTSURFER FUNCIONANDO!"
    echo "⚡ O problema de usuário Docker foi resolvido!"
    echo "💡 Pronto para processar todos os sujeitos!"
    
elif [ $EXIT_CODE -eq 0 ]; then
    echo "⚠️  PROCESSAMENTO CONCLUÍDO mas diretório não encontrado"
    echo "🔍 Verificando se foi criado em outro local..."
    find "$OUTPUT_DIR" -name "*$SUBJECT_ID*" -type d 2>/dev/null | head -5
    
else
    echo "❌ TESTE FALHOU (código: $EXIT_CODE)"
    echo ""
    echo "📋 DIAGNÓSTICO:"
    
    # Mostrar últimas linhas do log
    if [ -f "${LOG_DIR}/${SUBJECT_ID}_final.log" ]; then
        echo "📄 Últimas 20 linhas do log:"
        echo "----------------------------------------"
        tail -20 "${LOG_DIR}/${SUBJECT_ID}_final.log"
        echo "----------------------------------------"
    fi
    
    echo ""
    echo "🔧 TENTATIVAS ALTERNATIVAS:"
    echo "1. Verificar GPU: nvidia-smi"
    echo "2. Tentar sem threads: remover --threads 4"  
    echo "3. Verificar espaço: df -h"
    echo "4. Tentar como root: --user root"
fi

echo ""
echo "📝 Log completo salvo em: ${LOG_DIR}/${SUBJECT_ID}_final.log"
echo ""
echo "=== FIM DO TESTE FINAL ===" 
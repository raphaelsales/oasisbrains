#!/bin/bash

echo "=== Teste FastSurfer - VERSÃO CORRIGIDA (VOLUMES) ==="

# Configurações
DATA_BASE="/app/alzheimer/oasis_data"
OUTPUT_DIR="${DATA_BASE}/test_fastsurfer_corrigido"
LOG_DIR="${DATA_BASE}/test_logs_corrigido"

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

# Verificar arquivo T1 - DIAGNÓSTICO DETALHADO
T1_FILE="${TEST_SUBJECT}/mri/T1.mgz"
echo ""
echo "🔍 DIAGNÓSTICO DO ARQUIVO T1:"
echo "📁 Caminho: $T1_FILE"

if [ -L "$T1_FILE" ]; then
    echo "🔗 TIPO: Link simbólico"
    echo "🎯 Destino: $(readlink -f "$T1_FILE")"
    REAL_T1_FILE=$(readlink -f "$T1_FILE")
else
    echo "📄 TIPO: Arquivo regular"
    REAL_T1_FILE="$T1_FILE"
fi

echo "📊 Tamanho: $(du -h "$REAL_T1_FILE" | cut -f1)"
echo "✅ Arquivo T1 verificado"

# Usar licença oficial
LICENSE_FILE="./freesurfer_license_oficial.txt"
echo "✅ Licença oficial: $LICENSE_FILE"

# Limpar teste anterior
rm -rf "${OUTPUT_DIR}/${SUBJECT_ID}_corrigido"

echo ""
echo "🚀 TESTANDO FASTSURFER (MAPEAMENTO CORRIGIDO)..."
echo "⏱️  Início: $(date)"

# SOLUÇÃO: Mapear diretório pai inteiro para evitar problemas de links simbólicos
USER_ID=$(id -u)
GROUP_ID=$(id -g)

echo "🔧 User ID: $USER_ID, Group ID: $GROUP_ID"
echo "🗂️  Mapeando diretório base: $DATA_BASE"

# ESTRATÉGIA 1: Mapear todo o diretório base
echo ""
echo "📂 ESTRATÉGIA 1: Mapeamento completo do diretório base"

docker run --rm \
    --user "$USER_ID:$GROUP_ID" \
    -v "${DATA_BASE}:/data" \
    -v "${OUTPUT_DIR}:/output" \
    -v "${LICENSE_FILE}:/license.txt" \
    deepmi/fastsurfer:latest \
    --fs_license /license.txt \
    --t1 "/data/disc1/${SUBJECT_ID}/mri/T1.mgz" \
    --sid "${SUBJECT_ID}_corrigido" \
    --sd /output \
    --threads 4 \
    --device cpu \
    2>&1 | tee "${LOG_DIR}/${SUBJECT_ID}_corrigido.log"

EXIT_CODE_1=$?

echo ""
echo "🔍 Resultado Estratégia 1: $EXIT_CODE_1"

# Se falhar, tentar ESTRATÉGIA 2: Copiar arquivo localmente
if [ $EXIT_CODE_1 -ne 0 ]; then
    echo ""
    echo "📂 ESTRATÉGIA 2: Cópia local do arquivo T1"
    
    # Criar diretório temporário
    TEMP_DIR="/tmp/fastsurfer_test_$$"
    mkdir -p "$TEMP_DIR"
    
    # Copiar arquivo T1 para local temporário
    cp "$REAL_T1_FILE" "$TEMP_DIR/T1.mgz"
    
    echo "📋 Arquivo copiado para: $TEMP_DIR/T1.mgz"
    echo "📊 Tamanho: $(du -h "$TEMP_DIR/T1.mgz" | cut -f1)"
    
    docker run --rm \
        --user "$USER_ID:$GROUP_ID" \
        -v "${TEMP_DIR}:/input" \
        -v "${OUTPUT_DIR}:/output" \
        -v "${LICENSE_FILE}:/license.txt" \
        deepmi/fastsurfer:latest \
        --fs_license /license.txt \
        --t1 /input/T1.mgz \
        --sid "${SUBJECT_ID}_corrigido2" \
        --sd /output \
        --threads 4 \
        --device cpu \
        2>&1 | tee "${LOG_DIR}/${SUBJECT_ID}_corrigido2.log"
    
    EXIT_CODE_2=$?
    
    # Limpar arquivo temporário
    rm -rf "$TEMP_DIR"
    
    echo ""
    echo "🔍 Resultado Estratégia 2: $EXIT_CODE_2"
    
    FINAL_EXIT_CODE=$EXIT_CODE_2
    FINAL_RESULT_DIR="${OUTPUT_DIR}/${SUBJECT_ID}_corrigido2"
else
    FINAL_EXIT_CODE=$EXIT_CODE_1
    FINAL_RESULT_DIR="${OUTPUT_DIR}/${SUBJECT_ID}_corrigido"
fi

echo ""
echo "⏱️  Fim: $(date)"
echo "🔍 Exit code final: $FINAL_EXIT_CODE"

# Verificar resultado
if [ $FINAL_EXIT_CODE -eq 0 ] && [ -d "$FINAL_RESULT_DIR" ]; then
    echo ""
    echo "✅ TESTE CORRIGIDO: SUCESSO!"
    echo "📁 Resultados em: $FINAL_RESULT_DIR"
    
    # Verificar estrutura
    echo ""
    echo "📊 ARQUIVOS GERADOS:"
    
    # Contar arquivos por tipo
    mri_count=$(find "$FINAL_RESULT_DIR" -name "*.mgz" 2>/dev/null | wc -l)
    stats_count=$(find "$FINAL_RESULT_DIR" -name "*.stats" 2>/dev/null | wc -l)
    surf_count=$(find "$FINAL_RESULT_DIR" -name "*.surf" -o -name "lh.*" -o -name "rh.*" 2>/dev/null | wc -l)
    
    echo "  📈 Arquivos MRI (.mgz): $mri_count"
    echo "  📊 Arquivos Stats: $stats_count"
    echo "  🧠 Arquivos Surface: $surf_count"
    
    # Verificar arquivos chave
    echo ""
    echo "🔍 VALIDAÇÃO DOS ARQUIVOS PRINCIPAIS:"
    
    if [ -f "$FINAL_RESULT_DIR/stats/aseg.stats" ]; then
        echo "  ✅ aseg.stats: $(du -h "$FINAL_RESULT_DIR/stats/aseg.stats" | cut -f1)"
    else
        echo "  ❌ aseg.stats: AUSENTE"
    fi
    
    if [ -f "$FINAL_RESULT_DIR/mri/aparc+aseg.mgz" ]; then
        echo "  ✅ aparc+aseg.mgz: $(du -h "$FINAL_RESULT_DIR/mri/aparc+aseg.mgz" | cut -f1)"
    else
        echo "  ⚠️  aparc+aseg.mgz: AUSENTE (pode estar em processamento)"
    fi
    
    if [ -f "$FINAL_RESULT_DIR/mri/T1.mgz" ]; then
        echo "  ✅ T1.mgz: $(du -h "$FINAL_RESULT_DIR/mri/T1.mgz" | cut -f1)"
    fi
    
    # Mostrar estrutura de diretórios
    echo ""
    echo "📂 ESTRUTURA DE DIRETÓRIOS:"
    find "$FINAL_RESULT_DIR" -type d | head -10 | sed 's|^|  |'
    
    echo ""
    echo "🎉 FASTSURFER FUNCIONANDO PERFEITAMENTE!"
    echo "⚡ Problema de mapeamento de volumes resolvido!"
    echo "💡 Pronto para processar todos os sujeitos!"
    
elif [ $FINAL_EXIT_CODE -eq 0 ]; then
    echo "⚠️  PROCESSAMENTO CONCLUÍDO mas diretório não encontrado"
    echo "🔍 Verificando todos os locais..."
    find "$OUTPUT_DIR" -name "*$SUBJECT_ID*" -type d 2>/dev/null | head -10
    
else
    echo "❌ TESTE FALHOU (código: $FINAL_EXIT_CODE)"
    echo ""
    echo "📋 DIAGNÓSTICO COMPLETO:"
    
    # Mostrar logs de ambas as estratégias
    echo ""
    echo "📄 LOG ESTRATÉGIA 1 (últimas 10 linhas):"
    if [ -f "${LOG_DIR}/${SUBJECT_ID}_corrigido.log" ]; then
        tail -10 "${LOG_DIR}/${SUBJECT_ID}_corrigido.log"
    fi
    
    if [ $EXIT_CODE_1 -ne 0 ] && [ -f "${LOG_DIR}/${SUBJECT_ID}_corrigido2.log" ]; then
        echo ""
        echo "📄 LOG ESTRATÉGIA 2 (últimas 10 linhas):"
        tail -10 "${LOG_DIR}/${SUBJECT_ID}_corrigido2.log"
    fi
    
    echo ""
    echo "🔧 POSSÍVEIS SOLUÇÕES:"
    echo "1. Verificar se Docker tem acesso ao diretório"
    echo "2. Tentar como root: --user root"
    echo "3. Verificar espaço em disco: df -h"
    echo "4. Testar FreeSurfer nativo: ./test_freesurfer_nativo.sh"
fi

echo ""
echo "📝 Logs salvos em: ${LOG_DIR}/"
echo ""
echo "=== FIM DO TESTE CORRIGIDO ===" 
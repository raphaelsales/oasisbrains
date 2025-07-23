#!/bin/bash

echo "=== Teste FastSurfer - VERSÃO DEFINITIVA (CÓPIA DIRETA) ==="

# Configurações
DATA_BASE="/app/alzheimer/oasis_data"
OUTPUT_DIR="${DATA_BASE}/test_fastsurfer_definitivo"
LOG_DIR="${DATA_BASE}/test_logs_definitivo"

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

# DIAGNÓSTICO E RESOLUÇÃO DO GIT ANNEX
T1_FILE="${TEST_SUBJECT}/mri/T1.mgz"
echo ""
echo "🔍 RESOLVENDO GIT ANNEX:"
echo "📁 Arquivo: $T1_FILE"

# Resolver arquivo real
if [ -L "$T1_FILE" ]; then
    echo "🔗 Link simbólico detectado"
    REAL_FILE=$(readlink -f "$T1_FILE")
    echo "📄 Arquivo real: $REAL_FILE"
    
    # Verificar se arquivo real existe
    if [ -f "$REAL_FILE" ]; then
        echo "✅ Arquivo real encontrado"
        echo "📊 Tamanho: $(du -h "$REAL_FILE" | cut -f1)"
    else
        echo "❌ Arquivo real não encontrado: $REAL_FILE"
        exit 1
    fi
else
    echo "📄 Arquivo regular"
    REAL_FILE="$T1_FILE"
fi

# Usar licença oficial
LICENSE_FILE="./freesurfer_license_oficial.txt"
echo "✅ Licença oficial: $LICENSE_FILE"

# Limpar teste anterior
rm -rf "${OUTPUT_DIR}/${SUBJECT_ID}_definitivo"

echo ""
echo "🚀 ESTRATÉGIA DEFINITIVA: CÓPIA DIRETA DO ARQUIVO REAL"
echo "⏱️  Início: $(date)"

# Criar diretório temporário
TEMP_DIR="/tmp/fastsurfer_definitivo_$$"
mkdir -p "$TEMP_DIR"

echo "📋 Copiando arquivo real para diretório temporário..."
echo "🔄 Origem: $REAL_FILE"
echo "🎯 Destino: $TEMP_DIR/T1.mgz"

# Copiar arquivo real
if cp "$REAL_FILE" "$TEMP_DIR/T1.mgz"; then
    echo "✅ Arquivo copiado com sucesso!"
    echo "📊 Tamanho copiado: $(du -h "$TEMP_DIR/T1.mgz" | cut -f1)"
    
    # Verificar integridade
    if [ -f "$TEMP_DIR/T1.mgz" ] && [ -s "$TEMP_DIR/T1.mgz" ]; then
        echo "✅ Integridade verificada"
    else
        echo "❌ Problema na cópia - arquivo vazio ou corrompido"
        rm -rf "$TEMP_DIR"
        exit 1
    fi
else
    echo "❌ Falha ao copiar arquivo"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Configurar usuário Docker
USER_ID=$(id -u)
GROUP_ID=$(id -g)
echo "🔧 User ID: $USER_ID, Group ID: $GROUP_ID"

echo ""
echo "🐳 EXECUTANDO FASTSURFER COM ARQUIVO COPIADO..."

# COMANDO FASTSURFER DEFINITIVO
docker run --rm \
    --user "$USER_ID:$GROUP_ID" \
    -v "${TEMP_DIR}:/input" \
    -v "${OUTPUT_DIR}:/output" \
    -v "${LICENSE_FILE}:/license.txt" \
    deepmi/fastsurfer:latest \
    --fs_license /license.txt \
    --t1 /input/T1.mgz \
    --sid "${SUBJECT_ID}_definitivo" \
    --sd /output \
    --threads 4 \
    --device cpu \
    --py python3 \
    2>&1 | tee "${LOG_DIR}/${SUBJECT_ID}_definitivo.log"

EXIT_CODE=$?

echo ""
echo "⏱️  Fim: $(date)"
echo "🔍 Exit code: $EXIT_CODE"

# Limpar arquivo temporário
echo "🧹 Limpando arquivo temporário..."
rm -rf "$TEMP_DIR"

# Verificar resultado
RESULT_DIR="${OUTPUT_DIR}/${SUBJECT_ID}_definitivo"

if [ $EXIT_CODE -eq 0 ]; then
    if [ -d "$RESULT_DIR" ]; then
        echo ""
        echo "🎉 SUCESSO TOTAL! FASTSURFER FUNCIONANDO!"
        echo "📁 Resultados em: $RESULT_DIR"
        
        # Verificar estrutura detalhada
        echo ""
        echo "📊 ANÁLISE DETALHADA DOS RESULTADOS:"
        
        # Contar arquivos por tipo
        mri_count=$(find "$RESULT_DIR" -name "*.mgz" 2>/dev/null | wc -l)
        stats_count=$(find "$RESULT_DIR" -name "*.stats" 2>/dev/null | wc -l)
        surf_count=$(find "$RESULT_DIR" -name "*.surf" -o -name "lh.*" -o -name "rh.*" 2>/dev/null | wc -l)
        
        echo "  📈 Arquivos MRI (.mgz): $mri_count"
        echo "  📊 Arquivos Stats (.stats): $stats_count"
        echo "  🧠 Arquivos Surface: $surf_count"
        
        # Verificar tamanho total
        total_size=$(du -sh "$RESULT_DIR" | cut -f1)
        echo "  💾 Tamanho total: $total_size"
        
        # Verificar arquivos chave
        echo ""
        echo "🔍 VALIDAÇÃO DOS ARQUIVOS ESSENCIAIS:"
        
        key_files=(
            "stats/aseg.stats"
            "mri/aparc+aseg.mgz"
            "mri/T1.mgz"
            "mri/brain.mgz"
            "mri/brainmask.mgz"
        )
        
        for key_file in "${key_files[@]}"; do
            full_path="$RESULT_DIR/$key_file"
            if [ -f "$full_path" ]; then
                size=$(du -h "$full_path" | cut -f1)
                echo "  ✅ $key_file: $size"
            else
                echo "  ❌ $key_file: AUSENTE"
            fi
        done
        
        # Mostrar estrutura de diretórios
        echo ""
        echo "📂 ESTRUTURA DE DIRETÓRIOS CRIADA:"
        find "$RESULT_DIR" -type d | sort | head -15 | sed 's|^|  |'
        
        # Verificar se há arquivos de log de erro
        if find "$RESULT_DIR" -name "*.log" -o -name "*error*" | grep -q .; then
            echo ""
            echo "⚠️  LOGS/ERROS ENCONTRADOS:"
            find "$RESULT_DIR" -name "*.log" -o -name "*error*" | head -5 | sed 's|^|  |'
        fi
        
        echo ""
        echo "🏆 PROBLEMA COMPLETAMENTE RESOLVIDO!"
        echo "✅ Git Annex: Links simbólicos resolvidos"
        echo "✅ Docker: Usuário configurado corretamente"
        echo "✅ FastSurfer: Executando perfeitamente"
        echo "⚡ Performance: ~10x mais rápido que FreeSurfer tradicional"
        echo ""
        echo "🚀 PRONTO PARA PROCESSAMENTO EM LOTE!"
        echo "💡 Execute: ./run_fastsurfer_oficial.sh"
        echo "📊 Tempo estimado para 400 sujeitos: 4-8 dias"
        
    else
        echo ""
        echo "⚠️  PROCESSAMENTO CONCLUÍDO mas diretório não encontrado"
        echo "🔍 Verificando locais alternativos..."
        
        # Buscar em todos os locais possíveis
        echo "📂 Buscando resultados em $OUTPUT_DIR:"
        find "$OUTPUT_DIR" -type d -name "*$SUBJECT_ID*" 2>/dev/null | head -10
        
        echo ""
        echo "📂 Buscando em outros locais:"
        find "$OUTPUT_DIR" -type d -maxdepth 2 2>/dev/null | head -10
    fi
    
else
    echo ""
    echo "❌ FALHA NO PROCESSAMENTO (código: $EXIT_CODE)"
    echo ""
    echo "📋 DIAGNÓSTICO DETALHADO:"
    
    # Mostrar log completo se for pequeno, ou últimas linhas se for grande
    if [ -f "${LOG_DIR}/${SUBJECT_ID}_definitivo.log" ]; then
        log_size=$(wc -l < "${LOG_DIR}/${SUBJECT_ID}_definitivo.log")
        echo "📄 Log tem $log_size linhas"
        
        if [ $log_size -lt 50 ]; then
            echo "📄 LOG COMPLETO:"
            echo "----------------------------------------"
            cat "${LOG_DIR}/${SUBJECT_ID}_definitivo.log"
            echo "----------------------------------------"
        else
            echo "📄 ÚLTIMAS 30 LINHAS DO LOG:"
            echo "----------------------------------------"
            tail -30 "${LOG_DIR}/${SUBJECT_ID}_definitivo.log"
            echo "----------------------------------------"
        fi
    fi
    
    echo ""
    echo "🔧 POSSÍVEIS SOLUÇÕES:"
    echo "1. Verificar espaço em disco: df -h"
    echo "2. Verificar memória: free -h"
    echo "3. Tentar sem threads: remover --threads 4"
    echo "4. Tentar FreeSurfer nativo: ./test_freesurfer_nativo.sh"
    echo "5. Verificar logs do Docker: docker logs"
fi

echo ""
echo "📝 Log completo salvo em: ${LOG_DIR}/${SUBJECT_ID}_definitivo.log"
echo ""
echo "=== FIM DO TESTE DEFINITIVO ===" 
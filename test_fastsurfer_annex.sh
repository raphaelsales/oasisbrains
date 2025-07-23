#!/bin/bash

echo "=== Teste FastSurfer - SOLUÇÃO GIT ANNEX ==="

# Configurações
DATA_BASE="/app/alzheimer/oasis_data"
OUTPUT_DIR="${DATA_BASE}/test_fastsurfer_annex"
LOG_DIR="${DATA_BASE}/test_logs_annex"
ANNEX_DIR="/app/alzheimer/.git/annex/objects"

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

# DIAGNÓSTICO COMPLETO DO GIT ANNEX
T1_FILE="${TEST_SUBJECT}/mri/T1.mgz"
echo ""
echo "🔍 DIAGNÓSTICO GIT ANNEX:"
echo "📁 Arquivo: $T1_FILE"

# Verificar se é link simbólico
if [ -L "$T1_FILE" ]; then
    echo "🔗 CONFIRMADO: É um link simbólico"
    LINK_TARGET=$(readlink "$T1_FILE")
    echo "🎯 Link aponta para: $LINK_TARGET"
    
    # Resolver caminho absoluto
    REAL_FILE=$(readlink -f "$T1_FILE")
    echo "📄 Arquivo real: $REAL_FILE"
    
    # Verificar se está no Git Annex
    if [[ "$REAL_FILE" == *".git/annex/objects"* ]]; then
        echo "✅ CONFIRMADO: Arquivo está no Git Annex"
        echo "📂 Diretório Annex: $ANNEX_DIR"
        
        # Verificar se diretório annex existe
        if [ -d "$ANNEX_DIR" ]; then
            echo "✅ Diretório Git Annex encontrado"
            echo "📊 Tamanho do arquivo real: $(du -h "$REAL_FILE" | cut -f1)"
        else
            echo "❌ Diretório Git Annex não encontrado: $ANNEX_DIR"
        fi
    else
        echo "⚠️  Arquivo não está no Git Annex"
    fi
else
    echo "📄 Arquivo regular (não é link simbólico)"
    REAL_FILE="$T1_FILE"
fi

# Usar licença oficial
LICENSE_FILE="./freesurfer_license_oficial.txt"
echo "✅ Licença oficial: $LICENSE_FILE"

# Limpar teste anterior
rm -rf "${OUTPUT_DIR}/${SUBJECT_ID}_annex"

echo ""
echo "🚀 TESTANDO FASTSURFER (SOLUÇÃO GIT ANNEX)..."
echo "⏱️  Início: $(date)"

USER_ID=$(id -u)
GROUP_ID=$(id -g)

echo "🔧 User ID: $USER_ID, Group ID: $GROUP_ID"

# ESTRATÉGIA DEFINITIVA: Mapear todos os diretórios necessários
echo ""
echo "📂 ESTRATÉGIA: Mapeamento completo incluindo Git Annex"

# Verificar se precisa mapear Git Annex
if [ -d "$ANNEX_DIR" ]; then
    echo "🗂️  Mapeando Git Annex: $ANNEX_DIR"
    
    docker run --rm \
        --user "$USER_ID:$GROUP_ID" \
        -v "${DATA_BASE}:/data" \
        -v "${ANNEX_DIR}:/annex" \
        -v "${OUTPUT_DIR}:/output" \
        -v "${LICENSE_FILE}:/license.txt" \
        deepmi/fastsurfer:latest \
        --fs_license /license.txt \
        --t1 "/data/disc1/${SUBJECT_ID}/mri/T1.mgz" \
        --sid "${SUBJECT_ID}_annex" \
        --sd /output \
        --threads 4 \
        --device cpu \
        2>&1 | tee "${LOG_DIR}/${SUBJECT_ID}_annex.log"
    
    EXIT_CODE=$?
    
else
    echo "⚠️  Git Annex não encontrado, usando estratégia de cópia"
    
    # FALLBACK: Copiar arquivo real para local temporário
    TEMP_DIR="/tmp/fastsurfer_annex_$$"
    mkdir -p "$TEMP_DIR"
    
    echo "📋 Copiando arquivo real para: $TEMP_DIR/T1.mgz"
    cp "$REAL_FILE" "$TEMP_DIR/T1.mgz"
    
    if [ $? -eq 0 ]; then
        echo "✅ Arquivo copiado com sucesso"
        echo "📊 Tamanho: $(du -h "$TEMP_DIR/T1.mgz" | cut -f1)"
        
        docker run --rm \
            --user "$USER_ID:$GROUP_ID" \
            -v "${TEMP_DIR}:/input" \
            -v "${OUTPUT_DIR}:/output" \
            -v "${LICENSE_FILE}:/license.txt" \
            deepmi/fastsurfer:latest \
            --fs_license /license.txt \
            --t1 /input/T1.mgz \
            --sid "${SUBJECT_ID}_annex" \
            --sd /output \
            --threads 4 \
            --device cpu \
            2>&1 | tee "${LOG_DIR}/${SUBJECT_ID}_annex.log"
        
        EXIT_CODE=$?
        
        # Limpar arquivo temporário
        rm -rf "$TEMP_DIR"
    else
        echo "❌ Falha ao copiar arquivo"
        EXIT_CODE=1
    fi
fi

echo ""
echo "⏱️  Fim: $(date)"
echo "🔍 Exit code: $EXIT_CODE"

# Verificar resultado
RESULT_DIR="${OUTPUT_DIR}/${SUBJECT_ID}_annex"

if [ $EXIT_CODE -eq 0 ] && [ -d "$RESULT_DIR" ]; then
    echo ""
    echo "✅ TESTE GIT ANNEX: SUCESSO!"
    echo "📁 Resultados em: $RESULT_DIR"
    
    # Verificar estrutura
    echo ""
    echo "📊 ARQUIVOS GERADOS:"
    
    # Contar arquivos por tipo
    mri_count=$(find "$RESULT_DIR" -name "*.mgz" 2>/dev/null | wc -l)
    stats_count=$(find "$RESULT_DIR" -name "*.stats" 2>/dev/null | wc -l)
    surf_count=$(find "$RESULT_DIR" -name "*.surf" -o -name "lh.*" -o -name "rh.*" 2>/dev/null | wc -l)
    
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
    echo "🎉 PROBLEMA DO GIT ANNEX RESOLVIDO!"
    echo "✅ FastSurfer funcionando com links simbólicos!"
    echo "💡 Pronto para processar todos os sujeitos!"
    
elif [ $EXIT_CODE -eq 0 ]; then
    echo "⚠️  PROCESSAMENTO CONCLUÍDO mas diretório não encontrado"
    echo "🔍 Verificando todos os locais..."
    find "$OUTPUT_DIR" -name "*$SUBJECT_ID*" -type d 2>/dev/null | head -10
    
else
    echo "❌ TESTE FALHOU (código: $EXIT_CODE)"
    echo ""
    echo "📋 DIAGNÓSTICO:"
    
    # Mostrar últimas linhas do log
    if [ -f "${LOG_DIR}/${SUBJECT_ID}_annex.log" ]; then
        echo "📄 Últimas 15 linhas do log:"
        echo "----------------------------------------"
        tail -15 "${LOG_DIR}/${SUBJECT_ID}_annex.log"
        echo "----------------------------------------"
    fi
    
    echo ""
    echo "🔧 SOLUÇÕES ALTERNATIVAS:"
    echo "1. Usar 'git annex unlock' para converter links em arquivos"
    echo "2. Usar 'git annex get' para baixar arquivos"
    echo "3. Testar FreeSurfer nativo: ./test_freesurfer_nativo.sh"
    echo "4. Verificar permissões do Git Annex"
fi

echo ""
echo "📝 Log completo salvo em: ${LOG_DIR}/${SUBJECT_ID}_annex.log"
echo ""
echo "=== FIM DO TESTE GIT ANNEX ==="

# BONUS: Mostrar comandos úteis do Git Annex
echo ""
echo "🔧 COMANDOS ÚTEIS DO GIT ANNEX:"
echo "   git annex whereis T1.mgz    # Ver onde está o arquivo"
echo "   git annex unlock T1.mgz     # Converter link em arquivo"
echo "   git annex get T1.mgz        # Baixar arquivo"
echo "   git annex status            # Ver status do repositório" 
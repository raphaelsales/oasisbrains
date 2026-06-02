#!/bin/bash

echo "=== Teste FreeSurfer NATIVO (SEM DOCKER) ==="

# Configurações
DATA_BASE="/app/alzheimer/oasis_data"
OUTPUT_DIR="${DATA_BASE}/test_freesurfer_nativo"
LOG_DIR="${DATA_BASE}/test_logs_nativo"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Configurar ambiente FreeSurfer
echo "🔧 Configurando ambiente FreeSurfer..."

# Localizar FreeSurfer instalado
FREESURFER_PATHS=(
    "/usr/local/freesurfer"
    "$HOME/freesurfer/freesurfer"
    "/opt/freesurfer"
)

FREESURFER_HOME=""
for path in "${FREESURFER_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/bin/recon-all" ]; then
        FREESURFER_HOME="$path"
        echo "✅ FreeSurfer encontrado em: $FREESURFER_HOME"
        break
    fi
done

if [ -z "$FREESURFER_HOME" ]; then
    echo "❌ FreeSurfer nativo não encontrado!"
    echo "🐳 Recomendação: Use a versão Docker"
    exit 1
fi

# Configurar variáveis de ambiente
export FREESURFER_HOME
export PATH="$FREESURFER_HOME/bin:$PATH"
export SUBJECTS_DIR="$OUTPUT_DIR"

# Configurar licença
if [ -f "./freesurfer_license_oficial.txt" ]; then
    cp "./freesurfer_license_oficial.txt" "$FREESURFER_HOME/license.txt"
    echo "✅ Licença oficial configurada"
else
    echo "⚠️  Licença oficial não encontrada"
fi

# Verificar instalação
echo "🔍 Verificando instalação FreeSurfer..."
echo "FREESURFER_HOME: $FREESURFER_HOME"
echo "SUBJECTS_DIR: $SUBJECTS_DIR"

# Testar comando recon-all
if ! command -v recon-all &> /dev/null; then
    echo "❌ Comando recon-all não encontrado no PATH"
    exit 1
fi

echo "✅ recon-all disponível: $(which recon-all)"

# Selecionar sujeito para teste
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
    echo "❌ Nenhum sujeito encontrado!"
    exit 1
fi

SUBJECT_ID=$(basename "$TEST_SUBJECT")_nativo
T1_FILE="${TEST_SUBJECT}/mri/T1.mgz"

echo "📋 Testando com sujeito: $SUBJECT_ID"
echo "📁 Input T1: $T1_FILE"
echo "📁 Output: $OUTPUT_DIR/$SUBJECT_ID"

# Limpar teste anterior
rm -rf "$OUTPUT_DIR/$SUBJECT_ID"

echo ""
echo "🚀 INICIANDO TESTE FREESURFER NATIVO..."
echo "⏱️  Início: $(date)"

# COMANDO FREESURFER NATIVO (TESTE RÁPIDO)
# Usar apenas as etapas essenciais para validar funcionamento
recon-all \
    -i "$T1_FILE" \
    -s "$SUBJECT_ID" \
    -sd "$OUTPUT_DIR" \
    -autorecon1 \
    -noskullstrip \
    -no-isrunning \
    2>&1 | tee "${LOG_DIR}/${SUBJECT_ID}.log"

EXIT_CODE=$?

echo ""
echo "⏱️  Fim: $(date)"
echo "🔍 Exit code: $EXIT_CODE"

# Verificar resultado
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TESTE FREESURFER NATIVO: SUCESSO!"
    
    RESULT_DIR="$OUTPUT_DIR/$SUBJECT_ID"
    if [ -d "$RESULT_DIR" ]; then
        echo "📁 Resultados em: $RESULT_DIR"
        
        echo ""
        echo "📊 ESTRUTURA CRIADA:"
        find "$RESULT_DIR" -type d | head -10
        
        echo ""
        echo "📄 ARQUIVOS GERADOS:"
        find "$RESULT_DIR" -type f | head -15
        
        # Verificar se T1 foi processado
        if [ -f "$RESULT_DIR/mri/orig.mgz" ]; then
            echo ""
            echo "✅ Arquivo orig.mgz criado com sucesso"
            echo "📊 Tamanho: $(du -h "$RESULT_DIR/mri/orig.mgz" | cut -f1)"
        fi
        
        echo ""
        echo "🎉 FREESURFER NATIVO FUNCIONANDO!"
        echo "💡 Para processamento completo, use:"
        echo "   recon-all -s $SUBJECT_ID -all"
        
    else
        echo "⚠️  Diretório de resultado não encontrado"
    fi
    
else
    echo "❌ TESTE FALHOU (código: $EXIT_CODE)"
    
    if [ -f "${LOG_DIR}/${SUBJECT_ID}.log" ]; then
        echo ""
        echo "📄 Últimas linhas do log:"
        tail -15 "${LOG_DIR}/${SUBJECT_ID}.log"
    fi
fi

echo ""
echo "📝 Log completo: ${LOG_DIR}/${SUBJECT_ID}.log"
echo ""
echo "=== FIM DO TESTE NATIVO ===" 
#!/bin/bash

echo "=== TESTE RÁPIDO DO FASTSURFER ==="
echo "⏱️  $(date)"

# Configurações
SUBJECT_DIR="/app/alzheimer/oasis_data/disc1/OAS1_0001_MR1"
OUTPUT_DIR="/app/alzheimer/oasis_data/test_rapido_fastsurfer"
LOG_DIR="/app/alzheimer/oasis_data/test_logs_rapido"
LICENSE_FILE="./freesurfer_license_oficial.txt"

# Criar diretórios
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Verificar se o sujeito existe
if [ ! -d "$SUBJECT_DIR" ]; then
    echo "❌ Sujeito não encontrado: $SUBJECT_DIR"
    exit 1
fi

# Verificar se o arquivo T1.mgz existe
if [ ! -f "$SUBJECT_DIR/mri/T1.mgz" ]; then
    echo "❌ Arquivo T1.mgz não encontrado: $SUBJECT_DIR/mri/T1.mgz"
    exit 1
fi

# Verificar licença
if [ ! -f "$LICENSE_FILE" ]; then
    echo "❌ Licença não encontrada: $LICENSE_FILE"
    exit 1
fi

echo "✅ Configuração verificada:"
echo "   📁 Sujeito: $SUBJECT_DIR"
echo "   📄 T1.mgz: $(ls -lh "$SUBJECT_DIR/mri/T1.mgz" | awk '{print $5}')"
echo "   📂 Output: $OUTPUT_DIR"
echo "   📝 Logs: $LOG_DIR"
echo "   🔑 Licença: $LICENSE_FILE"

# Verificar se Docker está funcionando
echo ""
echo "🐳 Verificando Docker..."
if ! docker --version >/dev/null 2>&1; then
    echo "❌ Docker não está funcionando"
    exit 1
fi

echo "✅ Docker está funcionando"

# Verificar se a imagem FastSurfer existe
echo ""
echo "🔍 Verificando imagem FastSurfer..."
if ! docker images | grep -q "deepmi/fastsurfer"; then
    echo "⚠️  Imagem FastSurfer não encontrada, baixando..."
    docker pull deepmi/fastsurfer:latest
fi

echo "✅ Imagem FastSurfer disponível"

# Executar teste
echo ""
echo "🚀 INICIANDO TESTE DO FASTSURFER..."
echo "⏱️  Início: $(date)"

SUBJECT_ID="OAS1_0001_MR1"
LOG_FILE="$LOG_DIR/${SUBJECT_ID}_teste_rapido.log"

echo "📋 Executando comando:"
echo "docker run --rm \\"
echo "  -v \"$SUBJECT_DIR:/input\" \\"
echo "  -v \"$OUTPUT_DIR:/output\" \\"
echo "  -v \"$LICENSE_FILE:/fs_license/license.txt\" \\"
echo "  deepmi/fastsurfer:latest \\"
echo "  --fs_license /fs_license/license.txt \\"
echo "  --t1 /input/mri/T1.mgz \\"
echo "  --sid \"$SUBJECT_ID\" \\"
echo "  --sd /output \\"
echo "  --threads 2 \\"
echo "  --parallel"

echo ""
echo "🔄 Processando (isso pode demorar 20-30 minutos)..."

# Configurar usuário Docker
USER_ID=$(id -u)
GROUP_ID=$(id -g)
echo "🔧 Configuração Docker: User ID: $USER_ID, Group ID: $GROUP_ID"

# Criar diretório temporário e copiar arquivo
TEMP_DIR="/tmp/fastsurfer_test_$$"
mkdir -p "$TEMP_DIR"

echo "📋 Copiando arquivo T1.mgz para diretório temporário..."
if ! cp "$SUBJECT_DIR/mri/T1.mgz" "$TEMP_DIR/T1.mgz"; then
    echo "❌ Falha ao copiar arquivo T1.mgz"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Executar FastSurfer (CONFIGURAÇÃO CORRIGIDA)
docker run --rm \
    --user "$USER_ID:$GROUP_ID" \
    -v "$TEMP_DIR:/input" \
    -v "$OUTPUT_DIR:/output" \
    -v "$LICENSE_FILE:/license.txt" \
    deepmi/fastsurfer:latest \
    --fs_license /license.txt \
    --t1 /input/T1.mgz \
    --sid "$SUBJECT_ID" \
    --sd /output \
    --threads 2 \
    --device cpu \
    --py python3 \
    > "$LOG_FILE" 2>&1

EXIT_CODE=$?

# Limpar arquivo temporário
rm -rf "$TEMP_DIR"

echo ""
echo "⏱️  Fim: $(date)"
echo "🔍 Exit code: $EXIT_CODE"

# Verificar resultado
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ TESTE CONCLUÍDO COM SUCESSO!"
    
    # Verificar se os arquivos foram criados
    RESULT_DIR="$OUTPUT_DIR/$SUBJECT_ID"
    if [ -d "$RESULT_DIR" ]; then
        echo ""
        echo "📂 RESULTADOS GERADOS:"
        echo "   📁 Diretório: $RESULT_DIR"
        echo "   📊 Arquivos totais: $(find "$RESULT_DIR" -type f | wc -l)"
        echo "   💾 Tamanho total: $(du -sh "$RESULT_DIR" | cut -f1)"
        
        # Verificar arquivos importantes
        echo ""
        echo "🔍 ARQUIVOS IMPORTANTES:"
        if [ -f "$RESULT_DIR/stats/aseg.stats" ]; then
            echo "   ✅ aseg.stats: $(ls -lh "$RESULT_DIR/stats/aseg.stats" | awk '{print $5}')"
        else
            echo "   ❌ aseg.stats: NÃO ENCONTRADO"
        fi
        
        if [ -f "$RESULT_DIR/mri/aparc+aseg.mgz" ]; then
            echo "   ✅ aparc+aseg.mgz: $(ls -lh "$RESULT_DIR/mri/aparc+aseg.mgz" | awk '{print $5}')"
        else
            echo "   ❌ aparc+aseg.mgz: NÃO ENCONTRADO"
        fi
        
        if [ -d "$RESULT_DIR/surf" ]; then
            surf_count=$(ls "$RESULT_DIR/surf" | wc -l)
            echo "   ✅ Superfícies: $surf_count arquivos"
        else
            echo "   ❌ Superfícies: DIRETÓRIO NÃO ENCONTRADO"
        fi
        
        echo ""
        echo "🎉 TESTE COMPLETAMENTE FUNCIONAL!"
        echo "💡 O FastSurfer está funcionando corretamente"
        echo "⚡ Agora você pode usar o processamento em lote"
        
    else
        echo ""
        echo "⚠️  PROCESSAMENTO CONCLUÍDO mas diretório não encontrado"
        echo "🔍 Verificando outros locais..."
        find "$OUTPUT_DIR" -name "*$SUBJECT_ID*" -type d 2>/dev/null || echo "   ❌ Nenhum resultado encontrado"
    fi
else
    echo ""
    echo "❌ TESTE FALHOU (código: $EXIT_CODE)"
    echo ""
    echo "📋 ÚLTIMAS LINHAS DO LOG:"
    echo "----------------------------------------"
    tail -20 "$LOG_FILE"
    echo "----------------------------------------"
fi

echo ""
echo "📝 Log completo salvo em: $LOG_FILE"
echo ""
echo "=== FIM DO TESTE RÁPIDO ===" 
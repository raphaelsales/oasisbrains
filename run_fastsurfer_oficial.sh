#!/bin/bash

# PROCESSAMENTO OFICIAL FASTSURFER - VERSÃO DEFINITIVA
# Processa todos os sujeitos do OASIS com configurações otimizadas

set -e

# Configurações principais
OASIS_DIR="/app/alzheimer/oasis_data"
OUTPUT_DIR="/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos"
STATE_FILE="/app/alzheimer/processing_state.txt"
LICENSE_FILE="/app/alzheimer/freesurfer_license.txt"

# Verificar licença
if [ ! -f "$LICENSE_FILE" ]; then
    echo "Licença oficial não encontrada: $LICENSE_FILE"
    echo "Certifique-se de que o arquivo existe com a licença do FreeSurfer"
    exit 1
fi

echo "Licença oficial configurada: $LICENSE_FILE"

# Criar diretórios
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$STATE_FILE")"

# Inicializar ou carregar estado
PROCESSED=0
if [ -f "$STATE_FILE" ]; then
    PROCESSED=$(cat "$STATE_FILE")
fi

if [ $PROCESSED -gt 0 ]; then
    echo "Carregando estado anterior..."
    echo "$PROCESSED sujeitos já processados anteriormente"
fi

# Coletar todos os sujeitos disponíveis
echo "Coletando sujeitos disponíveis..."

cd "$OASIS_DIR"
ALL_SUBJECTS=()

# Procurar em todas as estruturas possíveis
for pattern in "OAS1_*_MR1" "disc*/OAS1_*_MR1" "*/OAS1_*_MR1"; do
    while IFS= read -r -d '' subject_dir; do
        if [ -f "$subject_dir/mri/T1.mgz" ]; then
            subject_id=$(basename "$subject_dir")
            ALL_SUBJECTS+=("$subject_id:$subject_dir")
        fi
    done < <(find . -path "./$pattern" -type d -print0 2>/dev/null)
done

TOTAL_SUBJECTS=${#ALL_SUBJECTS[@]}
echo "Total de sujeitos encontrados: $TOTAL_SUBJECTS"
echo "Resultados serão salvos em: $OUTPUT_DIR"

# Configurações Docker otimizadas
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Processar cada sujeito
for subject_entry in "${ALL_SUBJECTS[@]}"; do
    IFS=':' read -r subject subject_path <<< "$subject_entry"
    
    RESULT_DIR="$OUTPUT_DIR/$subject"
    
    # Verificar se já foi processado
    if [ -d "$RESULT_DIR" ] && [ -f "$RESULT_DIR/stats/aseg.stats" ]; then
        echo "$subject: JÁ PROCESSADO (pulando)"
        continue
    fi
    
    # Verificar se resultado já existe (mesmo sem stats completos)
    if [ -d "$RESULT_DIR" ] && [ "$(find "$RESULT_DIR" -name "*.mgz" | wc -l)" -gt 5 ]; then
        echo "$subject: RESULTADO EXISTE (pulando)"
        continue
    fi
    
    PROCESSED=$((PROCESSED + 1))
    echo "$PROCESSED sujeitos já processados anteriormente"
    
    echo "Processando: $subject"
    echo "  Entrada: $subject_path/mri/T1.mgz"
    echo "  Saída: $RESULT_DIR"
    echo "  Progresso: $PROCESSED/$TOTAL_SUBJECTS"
    
    # Comando FastSurfer otimizado
    if timeout 7200 docker run --rm \
        --gpus all \
        --user "$USER_ID:$GROUP_ID" \
        -v "$OASIS_DIR:/data" \
        -v "$OUTPUT_DIR:/output" \
        -v "$LICENSE_FILE:/fs_license/license.txt:ro" \
        deepmi/fastsurfer:latest \
        --fs_license /fs_license/license.txt \
        --t1 "/data/${subject_path#./}/mri/T1.mgz" \
        --sid "$subject" \
        --sd /output \
        --parallel \
        --3T; then
        
        echo "  SUCESSO: $subject processado"
        
        # Verificar arquivos críticos
        if [ -f "$RESULT_DIR/stats/aseg.stats" ] && [ -f "$RESULT_DIR/mri/aparc+aseg.mgz" ]; then
            echo "  VALIDADO: Arquivos principais confirmados"
        else
            echo "  AVISO: Processamento pode estar incompleto"
        fi
        
    else
        echo "  ERRO: Falha no processamento de $subject"
        
        # Limpeza em caso de erro
        if [ -d "$RESULT_DIR" ]; then
            rm -rf "$RESULT_DIR"
            echo "  LIMPEZA: Diretório removido após erro"
        fi
        
        continue
    fi
    
    # Salvar progresso
    echo "$PROCESSED" > "$STATE_FILE"
    
    # Status a cada 10 sujeitos
    if [ $((PROCESSED % 10)) -eq 0 ]; then
        echo
        echo "CHECKPOINT: $PROCESSED/$TOTAL_SUBJECTS sujeitos processados"
        echo "Progresso: $(echo "scale=1; $PROCESSED * 100 / $TOTAL_SUBJECTS" | bc -l)%"
        echo
    fi
    
    # Pausa entre processamentos
    sleep 2
    
done

echo
echo "PROCESSAMENTO CONCLUÍDO!"
echo "========================"
echo "Total processado: $PROCESSED sujeitos"
echo "Diretório de saída: $OUTPUT_DIR"

# Estatísticas finais
SUCCESSFUL=$(find "$OUTPUT_DIR" -name "aseg.stats" | wc -l)
echo "Sucessos confirmados: $SUCCESSFUL"

if [ $SUCCESSFUL -gt 0 ]; then
    echo
    echo "PRÓXIMOS PASSOS:"
    echo "1. Verificar resultados: ls -la $OUTPUT_DIR"
    echo "2. Executar análise IA: python3 alzheimer_ai_pipeline.py"
    echo "3. Gerar relatórios: python3 dataset_explorer.py"
    echo
    echo "PROCESSAMENTO OFICIAL CONCLUÍDO COM SUCESSO!"
else
    echo
    echo "NENHUM SUJEITO FOI PROCESSADO COM SUCESSO"
    echo "Verifique:"
    echo "1. Licença FreeSurfer"
    echo "2. Espaço em disco"
    echo "3. Configuração Docker"
    echo "4. Logs de erro"
fi 
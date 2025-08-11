#!/bin/bash

echo "=== MONITOR EM TEMPO REAL FASTSURFER ==="
echo "Pressione Ctrl+C para sair"
echo ""

# Função para limpar a tela e mostrar status
show_status() {
    clear
    echo "🔄 MONITOR TEMPO REAL - $(date)"
    echo "========================================"
    
    # Processos Docker
    DOCKER_COUNT=$(ps aux | grep -E "(docker.*fastsurfer|deepmi/fastsurfer)" | grep -v grep | wc -l)
    echo "🐳 Processos FastSurfer ativos: $DOCKER_COUNT"
    
    # Containers ativos
    CONTAINER_COUNT=$(docker ps --filter ancestor=deepmi/fastsurfer:latest --quiet | wc -l)
    echo "📦 Containers Docker ativos: $CONTAINER_COUNT"
    
    # Progresso
    OUTPUT_DIR="/app/alzheimer/oasis_data/outputs_fastsurfer"
    if [ -d "$OUTPUT_DIR" ]; then
        PROCESSED=$(find "$OUTPUT_DIR" -name "aparc+aseg.mgz" | wc -l)
        TOTAL=405
        PERCENT=$((PROCESSED * 100 / TOTAL))
        echo "📊 Progresso: $PROCESSED/$TOTAL ($PERCENT%)"
        
        # Barra de progresso visual
        FILLED=$((PERCENT / 2))  # Escala para 50 caracteres
        printf "["
        for i in $(seq 1 50); do
            if [ $i -le $FILLED ]; then
                printf "="
            else
                printf " "
            fi
        done
        printf "] $PERCENT%%\n"
    fi
    
    # Log mais recente
    LOG_DIR="/app/alzheimer/oasis_data/processing_logs"
    if [ -d "$LOG_DIR" ]; then
        LATEST_LOG=$(find "$LOG_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -n "$LATEST_LOG" ]; then
            SUBJECT=$(basename "$LATEST_LOG" .log)
            LAST_MOD=$(stat -c %Y "$LATEST_LOG" 2>/dev/null)
            CURRENT_TIME=$(date +%s)
            TIME_DIFF=$((CURRENT_TIME - LAST_MOD))
            
            echo ""
            echo "📄 Processando: $SUBJECT"
            echo "⏰ Última atividade: há $(($TIME_DIFF/60)) min"
            
            # Status baseado na atividade
            if [ $TIME_DIFF -lt 300 ]; then  # 5 minutos
                echo "🟢 Status: ATIVO"
            elif [ $TIME_DIFF -lt 1800 ]; then  # 30 minutos
                echo "🟡 Status: LENTO"
            else
                echo "🔴 Status: POSSÍVEL PROBLEMA"
            fi
        fi
    fi
    
    # Recursos do sistema
    echo ""
    echo "💻 RECURSOS:"
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    RAM_USAGE=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
    echo "   CPU: ${CPU_USAGE}% | RAM: ${RAM_USAGE}%"
    
    # Últimas linhas do log atual
    if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
        echo ""
        echo "📝 ÚLTIMAS LINHAS DO LOG:"
        echo "----------------------------------------"
        tail -3 "$LATEST_LOG" 2>/dev/null | sed 's/^/   /'
    fi
    
    echo ""
    echo "🔄 Atualizando em 30 segundos... (Ctrl+C para sair)"
}

# Loop principal
while true; do
    show_status
    sleep 30
done 
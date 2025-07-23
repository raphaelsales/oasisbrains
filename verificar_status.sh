#!/bin/bash

echo "=== STATUS DOS PROCESSOS FASTSURFER ==="
echo "Data/Hora: $(date)"
echo ""

# 1. PROCESSOS EM EXECUÇÃO
echo "🔍 1. PROCESSOS EM EXECUÇÃO:"
echo "----------------------------------------"

# Verificar processos Docker relacionados ao FastSurfer
DOCKER_PROCS=$(ps aux | grep -E "(docker.*fastsurfer|deepmi/fastsurfer)" | grep -v grep)
if [ -n "$DOCKER_PROCS" ]; then
    echo "✅ Processos Docker FastSurfer encontrados:"
    echo "$DOCKER_PROCS"
    echo ""
    
    # Contar quantos processos
    NUM_PROCS=$(echo "$DOCKER_PROCS" | wc -l)
    echo "📊 Total de processos FastSurfer: $NUM_PROCS"
else
    echo "❌ Nenhum processo Docker FastSurfer em execução"
fi

echo ""

# 2. CONTAINERS DOCKER ATIVOS
echo "🐳 2. CONTAINERS DOCKER ATIVOS:"
echo "----------------------------------------"
CONTAINERS=$(docker ps --filter ancestor=deepmi/fastsurfer:latest --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}")
if [ $(echo "$CONTAINERS" | wc -l) -gt 1 ]; then
    echo "$CONTAINERS"
else
    echo "❌ Nenhum container FastSurfer ativo"
fi

echo ""

# 3. SCRIPTS DE PROCESSAMENTO EM EXECUÇÃO
echo "📜 3. SCRIPTS DE PROCESSAMENTO:"
echo "----------------------------------------"
SCRIPT_PROCS=$(ps aux | grep -E "(processar_todos_fastsurfer|iniciar_processamento|executar_processamento)" | grep -v grep)
if [ -n "$SCRIPT_PROCS" ]; then
    echo "✅ Scripts em execução:"
    echo "$SCRIPT_PROCS"
else
    echo "❌ Nenhum script de processamento em execução"
fi

echo ""

# 4. ÚLTIMOS LOGS DE ATIVIDADE
echo "📋 4. ÚLTIMAS ATIVIDADES (logs):"
echo "----------------------------------------"
LOG_DIR="/app/alzheimer/oasis_data/processing_logs"
if [ -d "$LOG_DIR" ]; then
    # Encontrar o log mais recente
    LATEST_LOG=$(find "$LOG_DIR" -name "*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$LATEST_LOG" ]; then
        SUBJECT=$(basename "$LATEST_LOG" .log)
        LAST_MOD=$(stat -c %Y "$LATEST_LOG")
        CURRENT_TIME=$(date +%s)
        TIME_DIFF=$((CURRENT_TIME - LAST_MOD))
        
        echo "📄 Log mais recente: $SUBJECT"
        echo "⏰ Última atividade: há $(($TIME_DIFF/60)) minutos"
        
        # Mostrar últimas linhas do log
        echo ""
        echo "📝 Últimas 5 linhas do log:"
        tail -5 "$LATEST_LOG" 2>/dev/null || echo "Erro ao ler log"
    else
        echo "❌ Nenhum log encontrado"
    fi
else
    echo "❌ Diretório de logs não encontrado: $LOG_DIR"
fi

echo ""

# 5. PROGRESSO GERAL
echo "📈 5. PROGRESSO GERAL:"
echo "----------------------------------------"
OUTPUT_DIR="/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos"
if [ -d "$OUTPUT_DIR" ]; then
    # Contar sujeitos processados
    PROCESSED=$(find "$OUTPUT_DIR" -name "aparc+aseg.mgz" | wc -l)
    
    # Contar total de sujeitos
    TOTAL=405  # Sabemos que são 405 sujeitos no dataset OASIS
    
    # Calcular porcentagem
    PERCENT=$((PROCESSED * 100 / TOTAL))
    
    echo "✅ Sujeitos processados: $PROCESSED / $TOTAL ($PERCENT%)"
    
    # Estimar tempo restante se houver progresso
    if [ $PROCESSED -gt 0 ]; then
        REMAINING=$((TOTAL - PROCESSED))
        ESTIMATED_HOURS=$((REMAINING * 20 / 60))  # 20 min por sujeito
        echo "⏱️  Estimativa de tempo restante: ~$ESTIMATED_HOURS horas"
    fi
else
    echo "❌ Diretório de saída não encontrado: $OUTPUT_DIR"
fi

echo ""

# 6. USO DE RECURSOS
echo "💻 6. USO DE RECURSOS:"
echo "----------------------------------------"
echo "🖥️  CPU: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"
echo "🧠 RAM: $(free -h | awk 'NR==2{printf "%.1f%% (%s/%s)\n", $3*100/$2, $3, $2}')"
echo "💾 Disco (/app): $(df -h /app | awk 'NR==2{print $3"/"$2" ("$5")"}')"

echo ""

# 7. COMANDO PARA PARAR TUDO
echo "🛑 7. CONTROLES RÁPIDOS:"
echo "----------------------------------------"
echo "Para parar todos os processos: ./parar_tudo.sh"
echo "Para ver este status novamente: ./verificar_status.sh"
echo "Para ver logs em tempo real: tail -f $LOG_DIR/\$(ls -t $LOG_DIR | head -1)"

echo ""
echo "=== FIM DO STATUS ===" 
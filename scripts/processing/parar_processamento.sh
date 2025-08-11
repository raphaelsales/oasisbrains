#!/bin/bash

echo "=== PARAR PROCESSAMENTO FASTSURFER ==="
echo "⏱️  $(date)"
echo ""

echo "🔍 VERIFICANDO PROCESSOS EM EXECUÇÃO..."
echo "----------------------------------------"

# Verificar processos relacionados ao FastSurfer
FASTSURFER_PIDS=$(ps aux | grep -E "(run_fastsurfer_oficial|fastsurfer)" | grep -v grep | awk '{print $2}')
DOCKER_CONTAINERS=$(docker ps -q --filter "ancestor=deepmi/fastsurfer" 2>/dev/null)

if [ -n "$FASTSURFER_PIDS" ] || [ -n "$DOCKER_CONTAINERS" ]; then
    echo "📋 Processos encontrados:"
    
    if [ -n "$FASTSURFER_PIDS" ]; then
        echo ""
        echo "🔄 Processos bash do FastSurfer:"
        ps aux | grep -E "(run_fastsurfer_oficial|fastsurfer)" | grep -v grep | head -10
    fi
    
    if [ -n "$DOCKER_CONTAINERS" ]; then
        echo ""
        echo "🐳 Containers Docker do FastSurfer:"
        docker ps --filter "ancestor=deepmi/fastsurfer"
    fi
    
    echo ""
    echo "⚠️  ATENÇÃO: Isso irá interromper o processamento em andamento!"
    echo "💾 Os dados já processados serão preservados"
    echo ""
    echo "Deseja continuar? (s/N)"
    read -r response
    
    if [[ "$response" =~ ^[Ss]$ ]]; then
        echo ""
        echo "🛑 INTERROMPENDO PROCESSAMENTO..."
        
        # Parar containers Docker primeiro
        if [ -n "$DOCKER_CONTAINERS" ]; then
            echo "🐳 Parando containers Docker..."
            for container in $DOCKER_CONTAINERS; do
                echo "   Parando container: $container"
                docker stop "$container" 2>/dev/null
                docker rm "$container" 2>/dev/null
            done
        fi
        
        # Parar processos bash
        if [ -n "$FASTSURFER_PIDS" ]; then
            echo "🔄 Parando processos bash..."
            for pid in $FASTSURFER_PIDS; do
                echo "   Parando processo: $pid"
                kill -TERM "$pid" 2>/dev/null
                sleep 2
                # Se não parou, forçar
                if kill -0 "$pid" 2>/dev/null; then
                    echo "   Forçando parada: $pid"
                    kill -KILL "$pid" 2>/dev/null
                fi
            done
        fi
        
        echo ""
        echo "✅ PROCESSAMENTO INTERROMPIDO"
        echo ""
        echo "📊 VERIFICANDO ESTADO FINAL..."
        
        # Verificar se ainda há processos
        remaining_pids=$(ps aux | grep -E "(run_fastsurfer_oficial|fastsurfer)" | grep -v grep | awk '{print $2}')
        remaining_containers=$(docker ps -q --filter "ancestor=deepmi/fastsurfer" 2>/dev/null)
        
        if [ -n "$remaining_pids" ] || [ -n "$remaining_containers" ]; then
            echo "⚠️  Ainda há processos em execução:"
            [ -n "$remaining_pids" ] && echo "   Processos: $remaining_pids"
            [ -n "$remaining_containers" ] && echo "   Containers: $remaining_containers"
            echo "💡 Pode ser necessário reiniciar o sistema"
        else
            echo "✅ Todos os processos foram interrompidos com sucesso"
        fi
        
        # Mostrar estado dos dados
        echo ""
        echo "📁 ESTADO DOS DADOS:"
        echo "----------------------------------------"
        
        STATE_FILE="/app/alzheimer/oasis_data/fastsurfer_logs_oficial/processamento_estado.txt"
        OUTPUT_DIR="/app/alzheimer/oasis_data/outputs_fastsurfer_oficial"
        
        if [ -f "$STATE_FILE" ]; then
            processed_count=$(wc -l < "$STATE_FILE" 2>/dev/null || echo "0")
            echo "✅ Sujeitos processados: $processed_count"
            echo "📋 Arquivo de estado preservado: $STATE_FILE"
        else
            echo "❌ Nenhum arquivo de estado encontrado"
        fi
        
        if [ -d "$OUTPUT_DIR" ]; then
            result_count=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "OAS1_*" 2>/dev/null | wc -l)
            echo "✅ Resultados preservados: $result_count sujeitos"
            echo "📂 Diretório de resultados: $OUTPUT_DIR"
        else
            echo "❌ Nenhum resultado encontrado"
        fi
        
        echo ""
        echo "🚀 PRÓXIMOS PASSOS:"
        echo "----------------------------------------"
        echo "1. 🔍 Verificar status: ./status_processamento.sh"
        echo "2. 🔄 Testar funcionamento: ./teste_rapido_fastsurfer.sh"
        echo "3. 🚀 Reiniciar processamento: ./run_fastsurfer_oficial.sh"
        echo "4. 💡 Os dados já processados serão pulados automaticamente"
        
    else
        echo ""
        echo "❌ OPERAÇÃO CANCELADA"
        echo "💡 Processamento continua em execução"
    fi
else
    echo "✅ NENHUM PROCESSO FASTSURFER EM EXECUÇÃO"
    echo ""
    echo "🔍 VERIFICAÇÃO ADICIONAL:"
    echo "----------------------------------------"
    
    # Verificar se há algum processo relacionado
    echo "📋 Processos bash gerais:"
    ps aux | grep bash | grep -v grep | head -5 || echo "   ❌ Nenhum processo bash encontrado"
    
    echo ""
    echo "📋 Containers Docker gerais:"
    docker ps | head -5 || echo "   ❌ Nenhum container Docker em execução"
    
    echo ""
    echo "💡 Se você esperava que houvesse processamento em andamento,"
    echo "   pode ser que tenha terminado ou travado."
    echo ""
    echo "🔍 Verificar status: ./status_processamento.sh"
fi

echo ""
echo "=== FIM DA VERIFICAÇÃO ===" 
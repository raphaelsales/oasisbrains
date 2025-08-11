#!/bin/bash

echo "=== STATUS DO PROCESSAMENTO FASTSURFER ==="
echo "⏱️  $(date)"
echo ""

# Verificar processos relacionados ao FastSurfer
echo "🔍 PROCESSOS EM EXECUÇÃO:"
echo "----------------------------------------"

# Verificar processos bash
echo "📋 Processos bash relacionados:"
ps aux | grep -E "(fastsurfer|run_fastsurfer)" | grep -v grep | head -10 || echo "   ❌ Nenhum processo encontrado"

echo ""
echo "📋 Processos Docker:"
docker ps | head -10 || echo "   ❌ Nenhum container Docker em execução"

echo ""
echo "📋 Containers Docker (todos):"
docker ps -a | head -10 || echo "   ❌ Nenhum container Docker encontrado"

echo ""
echo "🔍 ARQUIVOS DE ESTADO:"
echo "----------------------------------------"

STATE_FILE="/app/alzheimer/oasis_data/fastsurfer_logs_oficial/processamento_estado.txt"
LOG_DIR="/app/alzheimer/oasis_data/fastsurfer_logs_oficial"

if [ -f "$STATE_FILE" ]; then
    echo "✅ Arquivo de estado encontrado"
    echo "   📊 Sujeitos processados: $(wc -l < "$STATE_FILE")"
    echo "   🕐 Última modificação: $(stat -c %y "$STATE_FILE" 2>/dev/null || echo "N/A")"
    echo ""
    echo "📄 Últimos 5 sujeitos processados:"
    tail -5 "$STATE_FILE" | sed 's/^/   /' || echo "   ❌ Erro ao ler arquivo"
else
    echo "❌ Arquivo de estado não encontrado: $STATE_FILE"
fi

echo ""
echo "📁 Diretório de logs:"
if [ -d "$LOG_DIR" ]; then
    echo "✅ Diretório existe: $LOG_DIR"
    echo "   📊 Arquivos de log: $(ls "$LOG_DIR"/*.log 2>/dev/null | wc -l)"
    echo "   🕐 Última modificação: $(stat -c %y "$LOG_DIR" 2>/dev/null || echo "N/A")"
else
    echo "❌ Diretório de logs não encontrado: $LOG_DIR"
fi

echo ""
echo "🔍 RESULTADOS:"
echo "----------------------------------------"

OUTPUT_DIR="/app/alzheimer/oasis_data/outputs_fastsurfer_oficial"
if [ -d "$OUTPUT_DIR" ]; then
    echo "✅ Diretório de resultados existe: $OUTPUT_DIR"
    result_count=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "OAS1_*" 2>/dev/null | wc -l)
    echo "   📊 Sujeitos processados: $result_count"
    echo "   🕐 Última modificação: $(stat -c %y "$OUTPUT_DIR" 2>/dev/null || echo "N/A")"
    
    if [ $result_count -gt 0 ]; then
        echo ""
        echo "📄 Últimos 5 sujeitos processados:"
        find "$OUTPUT_DIR" -maxdepth 1 -type d -name "OAS1_*" 2>/dev/null | sort | tail -5 | sed 's/^/   /' || echo "   ❌ Erro ao listar resultados"
    fi
else
    echo "❌ Diretório de resultados não encontrado: $OUTPUT_DIR"
fi

echo ""
echo "🔍 RECURSOS DO SISTEMA:"
echo "----------------------------------------"

echo "💾 Uso de disco:"
df -h /app/alzheimer 2>/dev/null | head -2 || echo "   ❌ Erro ao verificar disco"

echo ""
echo "🧠 Uso de memória:"
free -h 2>/dev/null || echo "   ❌ Erro ao verificar memória"

echo ""
echo "🐳 Docker:"
docker --version 2>/dev/null || echo "   ❌ Docker não disponível"

echo ""
echo "🎯 DIAGNÓSTICO:"
echo "----------------------------------------"

# Verificar se há processos em execução
if ps aux | grep -E "(run_fastsurfer_oficial)" | grep -v grep >/dev/null 2>&1; then
    echo "✅ Script run_fastsurfer_oficial.sh está em execução"
    
    if [ -f "$STATE_FILE" ] && [ -s "$STATE_FILE" ]; then
        processed_count=$(wc -l < "$STATE_FILE" 2>/dev/null || echo "0")
        echo "✅ Processamento ativo ($processed_count sujeitos processados)"
    else
        echo "⚠️  Script rodando mas nenhum sujeito processado ainda"
        echo "💡 Pode estar na fase de inicialização"
    fi
else
    echo "❌ Script run_fastsurfer_oficial.sh NÃO está em execução"
fi

# Verificar se há containers Docker
if docker ps | grep -q "fastsurfer" 2>/dev/null; then
    echo "✅ Container FastSurfer está em execução"
else
    echo "⚠️  Nenhum container FastSurfer em execução"
fi

echo ""
echo "🚀 RECOMENDAÇÕES:"
echo "----------------------------------------"

if [ ! -f "$STATE_FILE" ] || [ ! -s "$STATE_FILE" ]; then
    echo "1. ⏰ Aguardar mais alguns minutos (pode estar inicializando)"
    echo "2. 🔄 Executar teste rápido: ./teste_rapido_fastsurfer.sh"
    echo "3. 🛑 Se não houver progresso, considere reiniciar o processamento"
else
    processed_count=$(wc -l < "$STATE_FILE" 2>/dev/null || echo "0")
    echo "1. ✅ Processamento está funcionando ($processed_count sujeitos processados)"
    echo "2. 📊 Verificar progresso periodicamente"
    echo "3. ⏰ Aguardar conclusão (pode demorar vários dias)"
fi

echo ""
echo "=== FIM DO STATUS ===" 
#!/bin/bash

echo "=== COMANDOS RÁPIDOS PARA MONITORAR FASTSURFER ==="
echo ""

echo "📋 COMANDOS ÚTEIS (copie e cole):"
echo "================================================"
echo ""

echo "🔍 1. Ver processos FastSurfer:"
echo "   ps aux | grep fastsurfer | grep -v grep"
echo ""

echo "🐳 2. Ver containers Docker ativos:"
echo "   docker ps"
echo ""

echo "📊 3. Contar sujeitos processados:"
echo "   find /app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos -name 'aparc+aseg.mgz' | wc -l"
echo ""

echo "📋 4. Ver último log:"
echo "   ls -t /app/alzheimer/oasis_data/processing_logs/*.log | head -1 | xargs tail -10"
echo ""

echo "💻 5. Ver uso de CPU:"
echo "   top -bn1 | head -5"
echo ""

echo "🧠 6. Ver uso de memória:"
echo "   free -h"
echo ""

echo "📄 7. Ver log em tempo real (do último sujeito):"
echo "   tail -f \$(ls -t /app/alzheimer/oasis_data/processing_logs/*.log | head -1)"
echo ""

echo "🛑 8. Parar todos os processos FastSurfer:"
echo "   docker stop \$(docker ps -q --filter ancestor=deepmi/fastsurfer:latest)"
echo ""

echo "🔄 9. Ver quantos scripts estão rodando:"
echo "   ps aux | grep -E '(processar_todos|iniciar_processamento)' | grep -v grep"
echo ""

echo "📈 10. Progresso detalhado:"
echo "    ./verificar_status.sh"
echo ""

echo "==============================================="
echo "💡 DICA: Use esses comandos diretamente no terminal"
echo "    sem precisar executar scripts grandes!"
echo ""

# Executar verificação básica automaticamente
echo "🚀 EXECUÇÃO AUTOMÁTICA - STATUS ATUAL:"
echo "==============================================="

echo ""
echo "Processos FastSurfer ativos:"
FASTSURFER_PROCS=$(ps aux | grep fastsurfer | grep -v grep | wc -l)
echo "   $FASTSURFER_PROCS processo(s) encontrado(s)"

echo ""
echo "Containers Docker:"
CONTAINERS=$(docker ps --filter ancestor=deepmi/fastsurfer:latest --quiet | wc -l)
echo "   $CONTAINERS container(s) ativo(s)"

echo ""
echo "Sujeitos processados:"
if [ -d "/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos" ]; then
    PROCESSED=$(find /app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos -name "aparc+aseg.mgz" 2>/dev/null | wc -l)
    echo "   $PROCESSED sujeitos completados"
else
    echo "   Diretório de saída não encontrado"
fi

echo ""
echo "===============================================" 
#!/bin/bash

echo "🚀 EXECUTANDO PROCESSAMENTO FASTSURFER OTIMIZADO"
echo "================================================"
echo ""

# Executar o processamento otimizado
echo "Iniciando processamento com nohup..."
nohup bash run_fastsurfer_otimizado.sh > "fastsurfer_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID=$!

# Salvar PID
echo $PID > processamento_atual.pid

echo "✅ Processamento iniciado!"
echo "   PID: $PID"
echo "   Log: fastsurfer_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "📋 Para acompanhar:"
echo "   tail -f fastsurfer_*.log"
echo ""
echo "🛑 Para parar:"
echo "   kill $PID"
echo "" 
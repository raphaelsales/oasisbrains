#!/bin/bash

echo "🚀 INICIANDO PROCESSAMENTO FASTSURFER DEFINITIVO"
echo "=============================================="
echo ""

# 1. Limpar processos anteriores
echo "🛑 Parando processos anteriores..."
bash parar_tudo.sh

echo ""
echo "⏳ Aguardando 5 segundos..."
sleep 5

# 2. Verificar pré-requisitos
echo "🔍 Verificando pré-requisitos..."

# Verificar Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker não está funcionando"
    exit 1
fi
echo "✅ Docker funcionando"

# Verificar licença
if [ ! -f "freesurfer_license_oficial.txt" ]; then
    echo "❌ Licença oficial não encontrada"
    exit 1
fi
echo "✅ Licença encontrada"

# Verificar espaço em disco
DISK_USAGE=$(df /app/alzheimer | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 90 ]; then
    echo "⚠️  Aviso: Uso de disco em ${DISK_USAGE}%"
    echo "   Recomenda-se ter pelo menos 10% livre"
fi
echo "✅ Espaço em disco: ${DISK_USAGE}% usado"

echo ""
echo "🎯 CONFIGURAÇÃO DO PROCESSAMENTO:"
echo "   • Método: Sequencial (1 por vez)"
echo "   • Threads por sujeito: 6"
echo "   • Tempo estimado por sujeito: ~20 min"
echo "   • Total estimado: ~135 horas (5-6 dias)"
echo "   • Vantagem: 100% confiável, baseado no teste que funcionou"
echo ""

# 3. Executar processamento
LOG_NAME="fastsurfer_definitivo_$(date +%Y%m%d_%H%M%S).log"

echo "🚀 Iniciando processamento definitivo..."
echo "📝 Log: $LOG_NAME"
echo ""

# Executar em background
nohup bash processar_todos_fastsurfer.sh > "$LOG_NAME" 2>&1 &
PID=$!

# Salvar PID
echo $PID > processamento_definitivo.pid

echo "✅ PROCESSAMENTO INICIADO!"
echo "   PID: $PID"
echo "   Log: $LOG_NAME"
echo ""
echo "📋 COMANDOS ÚTEIS:"
echo "   📊 Acompanhar progresso:"
echo "     tail -f $LOG_NAME"
echo ""
echo "   🔍 Verificar processo:"
echo "     ps -p $PID"
echo ""
echo "   🛑 Parar processamento:"
echo "     kill $PID"
echo ""
echo "   📈 Verificar resultados:"
echo "     ls -la oasis_data/outputs_fastsurfer_definitivo_todos/"
echo ""
echo "⚡ O processamento está rodando em background!"
echo "   Use os comandos acima para monitorar o progresso." 
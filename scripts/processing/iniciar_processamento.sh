#!/bin/bash

echo "=== INICIANDO PROCESSAMENTO FASTSURFER OTIMIZADO ==="
echo ""

# Verificar se há processos em execução
echo "🔍 Verificando processos existentes..."
EXISTING_PIDS=$(ps aux | grep -i fastsurfer | grep -v grep | wc -l)
if [ $EXISTING_PIDS -gt 0 ]; then
    echo "⚠️  Encontrados $EXISTING_PIDS processos FastSurfer em execução:"
    ps aux | grep -i fastsurfer | grep -v grep
    echo ""
    echo "❓ Deseja finalizar esses processos? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🛑 Finalizando processos..."
        pkill -f fastsurfer
        sleep 3
    else
        echo "❌ Cancelando - processos existentes ainda em execução"
        exit 1
    fi
fi

# Limpar PIDs antigos
echo "🧹 Limpando PIDs antigos..."
rm -f *.pid

# Verificar espaço em disco
echo "💾 Verificando espaço em disco..."
DISK_USAGE=$(df /app/alzheimer | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 85 ]; then
    echo "⚠️  Aviso: Uso de disco em ${DISK_USAGE}%"
    echo "   Considere liberar espaço antes de continuar"
fi

# Verificar Docker
echo "🐳 Verificando Docker..."
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker não está funcionando"
    exit 1
fi

# Verificar licença
echo "📄 Verificando licença..."
if [ ! -f "freesurfer_license_oficial.txt" ]; then
    echo "❌ Licença oficial não encontrada"
    exit 1
fi

# Verificar script otimizado
echo "📜 Verificando script otimizado..."
if [ ! -f "run_fastsurfer_otimizado.sh" ]; then
    echo "❌ Script otimizado não encontrado"
    exit 1
fi

# Criar diretórios necessários
echo "📁 Criando diretórios..."
mkdir -p oasis_data/outputs_fastsurfer_otimizado
mkdir -p oasis_data/logs_fastsurfer_otimizado

# Gerar nome do log
LOG_NAME="fastsurfer_otimizado_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ Tudo pronto para iniciar o processamento!"
echo ""
echo "📊 Configuração:"
echo "   • Script: run_fastsurfer_otimizado.sh"
echo "   • Processos paralelos: 4"
echo "   • Threads por processo: 6"
echo "   • Total de threads: 24"
echo "   • Sujeitos: ~405"
echo "   • Tempo estimado: 2-3 dias"
echo "   • Log: $LOG_NAME"
echo ""
echo "🚀 Iniciando processamento..."

# Executar o script otimizado
nohup bash run_fastsurfer_otimizado.sh > "$LOG_NAME" 2>&1 &
PID=$!

# Salvar PID
echo $PID > fastsurfer_otimizado_atual.pid

echo "✅ Processamento iniciado!"
echo "   PID: $PID"
echo "   Log: $LOG_NAME"
echo ""
echo "📋 Para acompanhar o progresso:"
echo "   tail -f $LOG_NAME"
echo ""
echo "🔍 Para verificar status:"
echo "   bash status_atual.sh"
echo ""
echo "🛑 Para parar o processamento:"
echo "   kill $PID"
echo "" 
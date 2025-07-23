#!/bin/bash

echo "🛑 PARANDO TODOS OS PROCESSOS FASTSURFER"
echo "======================================="

# Finalizar todos os processos relacionados
echo "Finalizando processos FastSurfer..."
pkill -f "run_fastsurfer"
pkill -f "fastsurfer"
pkill -9 -f "docker.*fastsurfer"

sleep 3

# Verificar se ainda há processos
echo "Verificando processos restantes..."
ps aux | grep -E "(fastsurfer|docker.*deepmi)" | grep -v grep || echo "✅ Nenhum processo encontrado"

# Limpar containers Docker órfãos
echo "Limpando containers Docker..."
docker container prune -f 2>/dev/null || true

# Remover PIDs antigos
echo "Removendo arquivos PID..."
rm -f *.pid

echo ""
echo "✅ LIMPEZA COMPLETA!"
echo ""
echo "🚀 Para iniciar processamento limpo, execute:"
echo "   bash test_fastsurfer_definitivo.sh"
echo ""
echo "📊 Para verificar status:"
echo "   htop ou ps aux | grep fastsurfer" 
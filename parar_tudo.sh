#!/bin/bash

<<<<<<< HEAD
echo "PARANDO TODOS OS PROCESSOS FASTSURFER"
echo "====================================="

echo "Parando containers Docker..."
docker stop $(docker ps -q --filter ancestor=deepmi/fastsurfer:latest) 2>/dev/null || true

echo "Parando processos relacionados..."
pkill -f fastsurfer 2>/dev/null || true
pkill -f "run_fastsurfer" 2>/dev/null || true
pkill -f "processar_todos" 2>/dev/null || true

echo "Verificando processos restantes..."
ps aux | grep -E "(fastsurfer|docker.*deepmi)" | grep -v grep || echo "Nenhum processo encontrado"

echo
echo "Limpando containers parados..."
docker container prune -f 2>/dev/null || true

echo
echo "Verificando espaço em disco..."
df -h | grep -E "(/$|/app|/tmp)"

echo
echo "LIMPEZA COMPLETA!"
echo
echo "Para iniciar processamento limpo, execute:"
echo "./run_fastsurfer_oficial.sh"
echo
echo "Para verificar status:"
echo "./status_processamento.sh" 
=======
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
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)

#!/bin/bash

echo "PARANDO TODOS OS PROCESSOS FASTSURFER"
echo "======================================="

echo "Finalizando processos FastSurfer..."
pkill -f "run_fastsurfer" 2>/dev/null || true
pkill -f "fastsurfer" 2>/dev/null || true
pkill -9 -f "docker.*fastsurfer" 2>/dev/null || true

sleep 3

echo "Verificando processos restantes..."
ps aux | grep -E "(fastsurfer|docker.*deepmi)" | grep -v grep || echo "Nenhum processo encontrado"

echo "Limpando containers Docker..."
docker container prune -f 2>/dev/null || true

echo "Removendo arquivos PID..."
rm -f *.pid

echo ""
echo "LIMPEZA COMPLETA!"
echo ""
echo "Para iniciar processamento limpo, execute:"
echo "   bash test_fastsurfer_definitivo.sh"
echo ""
echo "Para verificar status:"
echo "   htop ou ps aux | grep fastsurfer"

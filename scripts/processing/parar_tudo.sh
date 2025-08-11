#!/bin/bash

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
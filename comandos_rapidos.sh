#!/bin/bash

# COMANDOS ÚTEIS - CÓPIA E COLA DIRETO
echo "COMANDOS ÚTEIS (copie e cole):"
echo "================================"

echo
echo "1. Ver processos FastSurfer:"
echo "ps aux | grep fastsurfer"
echo "ps aux | grep docker"
echo "docker ps"

echo
echo "2. Ver logs em tempo real:"
echo "tail -f fastsurfer_processamento.log"

echo
echo "3. Contar sujeitos processados:"
echo "find outputs_fastsurfer_definitivo_todos -name 'aparc+aseg.mgz' | wc -l"

echo
echo "4. Ver último log:"
echo "ls -lt *.log | head -1"

echo
echo "5. Ver uso de CPU:"
echo "htop"
echo "top"

echo
echo "6. Ver uso de memória:"
echo "free -h"
echo "df -h"

echo
echo "7. Parar todos os processos:"
echo "./parar_tudo.sh"

echo
echo "8. Status atual:"
echo "./status_processamento.sh"

echo
echo "9. Ver quantos scripts estão rodando:"
echo "pgrep -f 'run_fastsurfer' | wc -l"

echo
echo "10. Verificar GPU:"
echo "nvidia-smi"

echo
echo "DICA: Use esses comandos diretamente no terminal"
echo

# AUTO-EXECUÇÃO DE ALGUNS COMANDOS ÚTEIS
echo
echo "EXECUÇÃO AUTOMÁTICA - STATUS ATUAL:"
echo "===================================="

echo
echo "Processos FastSurfer ativos:"
ps aux | grep -E "(fastsurfer|docker.*deepmi)" | grep -v grep || echo "Nenhum processo ativo"

echo
echo "Espaço em disco:"
df -h | grep -E "(/$|/app)"

echo
echo "Uso de memória:"
free -h

echo
echo "Sujeitos já processados:"
if [ -d "outputs_fastsurfer_definitivo_todos" ]; then
    find outputs_fastsurfer_definitivo_todos -name 'aparc+aseg.mgz' | wc -l
else
    echo "0 (diretório não encontrado)"
fi 
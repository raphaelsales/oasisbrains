#!/bin/bash

echo "Verificando processos do FreeSurfer..."

# Listar processos relacionados ao FreeSurfer
FREESURFER_PIDS=$(ps aux | grep -E "(recon-all|mri_|fs_)" | grep -v grep | awk '{print $2}')

if [ -n "$FREESURFER_PIDS" ]; then
    echo "Processos encontrados:"
    ps aux | grep -E "(recon-all|mri_|fs_)" | grep -v grep
    echo ""
    echo "Para finalizar todos os processos, execute:"
    echo "kill -9 $FREESURFER_PIDS"
    echo ""
    echo "⚠️  CUIDADO: Isso irá interromper todos os processamentos em andamento!"
else
    echo "✅ Nenhum processo do FreeSurfer em execução"
fi

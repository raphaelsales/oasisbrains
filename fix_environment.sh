#!/bin/bash

# Script para configurar ambiente FreeSurfer/FastSurfer
# Detecta automaticamente a instalação

echo "Configurando ambiente FreeSurfer..."

# Detectar FreeSurfer
if [ -d "/usr/local/freesurfer" ]; then
    export FREESURFER_HOME="/usr/local/freesurfer"
elif [ -d "/opt/freesurfer" ]; then
    export FREESURFER_HOME="/opt/freesurfer"
else
    echo "FreeSurfer não encontrado!"
    echo "Instale o FreeSurfer ou use apenas FastSurfer"
    exit 1
fi

# Configurar variáveis
export SUBJECTS_DIR="/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos"
export FUNCTIONALS_DIR="$FREESURFER_HOME/sessions"

# Configurar licença
export FS_LICENSE="/app/alzheimer/freesurfer_license.txt"

# Carregar ambiente FreeSurfer
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"

echo "Ambiente configurado:"
echo "  FREESURFER_HOME: $FREESURFER_HOME"
echo "  SUBJECTS_DIR: $SUBJECTS_DIR"
echo "  FS_LICENSE: $FS_LICENSE"

# Salvar configuração
echo "# Configuração FreeSurfer" > freesurfer_env.sh
echo "export FREESURFER_HOME=\"$FREESURFER_HOME\"" >> freesurfer_env.sh
echo "export SUBJECTS_DIR=\"$SUBJECTS_DIR\"" >> freesurfer_env.sh
echo "export FS_LICENSE=\"$FS_LICENSE\"" >> freesurfer_env.sh
echo "source \"\$FREESURFER_HOME/SetUpFreeSurfer.sh\"" >> freesurfer_env.sh

echo "Configuração salva em: freesurfer_env.sh"
echo "Para usar em outros scripts: source freesurfer_env.sh"

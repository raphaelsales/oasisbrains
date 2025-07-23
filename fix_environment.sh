#!/bin/bash

echo "Corrigindo configuração do ambiente..."

# Definir FreeSurfer Home baseado na instalação atual
if [ -d "/home/comaisserveria/freesurfer/freesurfer" ]; then
    export FREESURFER_HOME="/home/comaisserveria/freesurfer/freesurfer"
elif [ -d "/usr/local/freesurfer" ]; then
    export FREESURFER_HOME="/usr/local/freesurfer"
else
    echo "❌ FreeSurfer não encontrado!"
    exit 1
fi

export PATH="$FREESURFER_HOME/bin:$PATH"
export SUBJECTS_DIR="/app/alzheimer/oasis_data/subjects"

# Criar diretório de sujeitos se não existir
mkdir -p "$SUBJECTS_DIR"

echo "✅ Ambiente configurado:"
echo "   FREESURFER_HOME: $FREESURFER_HOME"
echo "   SUBJECTS_DIR: $SUBJECTS_DIR"
echo "   PATH: Atualizado"

# Salvar configuração
echo "export FREESURFER_HOME=\"$FREESURFER_HOME\"" > freesurfer_env.sh
echo "export PATH=\"\$FREESURFER_HOME/bin:\$PATH\"" >> freesurfer_env.sh
echo "export SUBJECTS_DIR=\"/app/alzheimer/oasis_data/subjects\"" >> freesurfer_env.sh

echo "✅ Configuração salva em: freesurfer_env.sh"
echo "   Para usar: source freesurfer_env.sh"

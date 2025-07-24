#!/bin/bash
# TEMPLATE DE CONFIGURAÇÃO DE AMBIENTE
# 
# INSTRUÇÕES:
# 1. Copie este arquivo para "env_config.sh" (gitignored)
# 2. Substitua os valores pelos seus dados reais
# 3. Execute: source env_config.sh
# 4. NUNCA commite env_config.sh no Git!

# =================
# CONFIGURAÇÕES DE USUÁRIO
# =================

# Diretório base do usuário (automático)
export USER_HOME="$HOME"

# FreeSurfer Configuration
export FREESURFER_HOME="$HOME/freesurfer/freesurfer"
export FREESURFER_LICENSE="$HOME/license.txt"

# Dados do projeto
export PROJECT_ROOT="/app/alzheimer"
export OASIS_DATA="$PROJECT_ROOT/oasis_data"

# Configurações de processamento
export THREADS_COUNT=4
export MEMORY_LIMIT="8G"

# =================
# SEGURANÇA - NUNCA EXPONHA ESTES DADOS
# =================

# Email institucional (para registros de licença)
export USER_EMAIL="SEU_EMAIL@INSTITUICAO.EDU"

# Configurações de Docker
export DOCKER_USER_ID=$(id -u)
export DOCKER_GROUP_ID=$(id -g)

# =================
# VALIDAÇÃO AUTOMÁTICA
# =================

echo "🔧 Configuração de ambiente carregada:"
echo "   FREESURFER_HOME: $FREESURFER_HOME"
echo "   FREESURFER_LICENSE: $FREESURFER_LICENSE"
echo "   PROJECT_ROOT: $PROJECT_ROOT"
echo "   USER_EMAIL: $USER_EMAIL"

# Verificar se arquivos críticos existem
if [ ! -f "$FREESURFER_LICENSE" ]; then
    echo "⚠️  ATENÇÃO: Licença FreeSurfer não encontrada em $FREESURFER_LICENSE"
    echo "   Registre-se em: https://surfer.nmr.mgh.harvard.edu/registration.html"
fi

if [ ! -d "$FREESURFER_HOME" ]; then
    echo "⚠️  ATENÇÃO: FreeSurfer não encontrado em $FREESURFER_HOME"
fi

echo "✅ Configuração concluída!" 
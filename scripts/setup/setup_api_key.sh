#!/bin/bash

echo "🔑 CONFIGURAÇÃO DA API KEY OPENAI"
echo "=================================="
echo

# Verificar se já existe arquivo .env
if [ -f ".env" ]; then
    echo "⚠️  Arquivo .env já existe!"
    echo "Conteúdo atual:"
    cat .env | grep -v "OPENAI_API_KEY" || echo "   (sem API key configurada)"
    echo
    
    read -p "Deseja sobrescrever? (s/N): " overwrite
    if [[ ! $overwrite =~ ^[Ss]$ ]]; then
        echo "❌ Configuração cancelada"
        exit 0
    fi
fi

echo "📋 INSTRUÇÕES:"
echo "1. Acesse: https://platform.openai.com/api-keys"
echo "2. Crie uma nova chave de API"
echo "3. Copie a chave (começa com 'sk-')"
echo

# Solicitar API key
read -s -p "Cole sua API key da OpenAI: " api_key
echo

# Validar formato da chave
if [[ ! $api_key =~ ^sk-[a-zA-Z0-9]{32,}$ ]]; then
    echo "❌ ERRO: Formato de API key inválido!"
    echo "A chave deve começar com 'sk-' e ter pelo menos 32 caracteres"
    exit 1
fi

# Criar arquivo .env
cat > .env << EOF
# CONFIGURAÇÃO OPENAI API
# Criado em: $(date)
# IMPORTANTE: NUNCA commite este arquivo no Git!

OPENAI_API_KEY=$api_key
GPT_MODEL=gpt-4
MAX_TOKENS=2000
TEMPERATURE=0.3
EOF

echo "✅ API key configurada com sucesso!"
echo "📁 Arquivo .env criado"
echo

# Testar configuração
echo "🧪 TESTANDO CONFIGURAÇÃO..."
python3 -c "
import os
from pathlib import Path

# Carregar .env
env_file = Path('.env')
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            if line.startswith('OPENAI_API_KEY='):
                api_key = line.split('=', 1)[1].strip().strip('\"\\'')
                break
else:
    api_key = os.getenv('OPENAI_API_KEY')

if api_key and api_key.startswith('sk-'):
    print('✅ API key válida detectada!')
    print(f'   Formato: {api_key[:10]}...{api_key[-4:]}')
else:
    print('❌ API key não encontrada ou inválida!')
"

echo
echo "🎯 PRÓXIMOS PASSOS:"
echo "1. Execute: python3 config_openai.py"
echo "2. Execute: python3 openai_fastsurfer_analyzer.py"
echo
echo "💡 DICAS:"
echo "• Use GPT-3.5-turbo para testes (mais barato)"
echo "• Use GPT-4 para análises finais (mais preciso)"
echo "• Monitore o uso em: https://platform.openai.com/usage"
echo
echo "🔒 SEGURANÇA:"
echo "• O arquivo .env está protegido pelo .gitignore"
echo "• Nunca compartilhe sua API key"
echo "• Configure limites de uso na plataforma OpenAI"

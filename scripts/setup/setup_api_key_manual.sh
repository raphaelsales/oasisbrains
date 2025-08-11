#!/bin/bash

echo "🔑 CONFIGURAÇÃO MANUAL DA API KEY OPENAI"
echo "========================================"
echo

echo "📋 MÉTODO MANUAL SIMPLES:"
echo "1. Vou abrir o editor nano"
echo "2. Cole sua API key no arquivo"
echo "3. Salve e saia (Ctrl+X, Y, Enter)"
echo

# Criar arquivo .env básico
cat > .env << 'EOF'
# CONFIGURAÇÃO OPENAI API
# IMPORTANTE: NUNCA commite este arquivo no Git!

OPENAI_API_KEY=sk-sua-chave-aqui
GPT_MODEL=gpt-4
MAX_TOKENS=2000
TEMPERATURE=0.3
EOF

echo "📝 Abrindo editor para você colar sua API key..."
echo "Substitua 'sk-sua-chave-aqui' pela sua chave real"
echo

# Abrir editor
if command -v nano >/dev/null 2>&1; then
    nano .env
elif command -v vim >/dev/null 2>&1; then
    vim .env
else
    echo "❌ Nenhum editor encontrado!"
    echo "Edite manualmente o arquivo .env"
    exit 1
fi

echo
echo "🧪 TESTANDO CONFIGURAÇÃO..."

# Testar se a API key foi configurada
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

if api_key and api_key.startswith('sk-') and len(api_key) > 32:
    print('✅ API key válida detectada!')
    print(f'   Formato: {api_key[:10]}...{api_key[-4:]}')
    print(f'   Comprimento: {len(api_key)} caracteres')
else:
    print('❌ API key não encontrada ou inválida!')
    print('Verifique se você substituiu "sk-sua-chave-aqui" pela sua chave real')
"

echo
echo "🎯 PRÓXIMOS PASSOS:"
echo "1. Execute: python3 config_openai.py"
echo "2. Execute: python3 openai_fastsurfer_analyzer.py"

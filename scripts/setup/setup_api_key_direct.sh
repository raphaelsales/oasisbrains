#!/bin/bash

echo "🔑 CONFIGURAÇÃO DIRETA DA API KEY OPENAI"
echo "========================================"
echo

echo "📋 MÉTODO DIRETO:"
echo "1. Cole sua API key no comando abaixo"
echo "2. Pressione Enter"
echo "3. Pronto!"
echo

echo "Exemplo:"
echo "echo 'OPENAI_API_KEY=sk-abc123...' > .env"
echo

echo "Cole o comando completo aqui (incluindo 'echo' e '> .env'):"
read -r command

# Executar o comando
eval "$command"

echo
echo "🧪 VERIFICANDO ARQUIVO CRIADO..."

if [ -f ".env" ]; then
    echo "✅ Arquivo .env criado!"
    echo "📄 Conteúdo:"
    cat .env
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

if api_key and api_key.startswith('sk-') and len(api_key) > 32:
    print('✅ API key válida detectada!')
    print(f'   Formato: {api_key[:10]}...{api_key[-4:]}')
    print(f'   Comprimento: {len(api_key)} caracteres')
else:
    print('❌ API key não encontrada ou inválida!')
    print('Verifique o formato da chave')
"
else
    echo "❌ Erro ao criar arquivo .env"
    echo "Tente novamente ou use o método manual"
fi

echo
echo "🎯 PRÓXIMOS PASSOS:"
echo "1. Execute: python3 config_openai.py"
echo "2. Execute: python3 openai_fastsurfer_analyzer.py"

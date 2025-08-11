#!/bin/bash

echo "🔧 INSTALAÇÃO DE DEPENDÊNCIAS OPENAI + FASTSURFER"
echo "================================================="
echo

echo "📦 Instalando bibliotecas Python..."
pip install openai pandas numpy matplotlib seaborn

echo
echo "✅ Verificando instalação..."
python3 -c "
try:
    import openai
    print('✅ OpenAI: OK')
except ImportError:
    print('❌ OpenAI: FALHOU')

try:
    import pandas
    print('✅ Pandas: OK')
except ImportError:
    print('❌ Pandas: FALHOU')

try:
    import numpy
    print('✅ NumPy: OK')
except ImportError:
    print('❌ NumPy: FALHOU')

try:
    import matplotlib
    print('✅ Matplotlib: OK')
except ImportError:
    print('❌ Matplotlib: FALHOU')

try:
    import seaborn
    print('✅ Seaborn: OK')
except ImportError:
    print('❌ Seaborn: FALHOU')
"

echo
echo "🎯 PRÓXIMOS PASSOS:"
echo "1. Configure sua API key: ./setup_api_key_manual.sh"
echo "2. Teste a configuração: python3 config_openai.py"
echo "3. Execute a análise: python3 openai_fastsurfer_analyzer.py"
echo
echo "💡 DICAS:"
echo "• Use GPT-3.5-turbo para testes (mais barato)"
echo "• Use GPT-4 para análises finais (mais preciso)"
echo "• Monitore custos em: https://platform.openai.com/usage"

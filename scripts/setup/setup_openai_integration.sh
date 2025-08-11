#!/bin/bash

echo "🔧 CONFIGURAÇÃO DA INTEGRAÇÃO OPENAI + FASTSURFER"
echo "================================================="
echo "Script para configurar análise interpretativa com ChatGPT"
echo

# Verificar se o Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 não encontrado!"
    echo "Instale o Python3 antes de continuar"
    exit 1
fi

echo "✅ Python3 encontrado: $(python3 --version)"

# Verificar se o pip está instalado
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 não encontrado!"
    echo "Instale o pip3 antes de continuar"
    exit 1
fi

echo "✅ pip3 encontrado"

# Instalar dependências
echo ""
echo "📦 INSTALANDO DEPENDÊNCIAS..."
echo "============================"

# Lista de dependências
dependencies=(
    "openai"
    "pandas"
    "numpy"
    "matplotlib"
    "seaborn"
)

for dep in "${dependencies[@]}"; do
    echo "Instalando $dep..."
    pip3 install "$dep" --quiet
    if [ $? -eq 0 ]; then
        echo "  ✅ $dep instalado"
    else
        echo "  ❌ Erro ao instalar $dep"
    fi
done

echo ""
echo "🔑 CONFIGURAÇÃO DA API KEY OPENAI"
echo "================================="

# Verificar se a API key já está configurada
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ OPENAI_API_KEY já configurada"
    echo "   Chave: ${OPENAI_API_KEY:0:10}..."
else
    echo "❌ OPENAI_API_KEY não configurada"
    echo ""
    echo "Para configurar sua API key:"
    echo "1. Acesse: https://platform.openai.com/api-keys"
    echo "2. Crie uma nova API key"
    echo "3. Execute um dos comandos abaixo:"
    echo ""
    echo "   Opção 1 - Variável de ambiente temporária:"
    echo "   export OPENAI_API_KEY='sua-chave-aqui'"
    echo ""
    echo "   Opção 2 - Adicionar ao .bashrc (permanente):"
    echo "   echo 'export OPENAI_API_KEY=\"sua-chave-aqui\"' >> ~/.bashrc"
    echo "   source ~/.bashrc"
    echo ""
    echo "   Opção 3 - Criar arquivo .env:"
    echo "   echo 'OPENAI_API_KEY=sua-chave-aqui' > .env"
    echo ""
fi

echo ""
echo "📁 VERIFICANDO ESTRUTURA DE DADOS"
echo "================================="

# Verificar se os dados FastSurfer existem
fastsurfer_dir="/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos"

if [ -d "$fastsurfer_dir" ]; then
    subject_count=$(find "$fastsurfer_dir" -maxdepth 1 -name "OAS1_*" -type d | wc -l)
    echo "✅ Diretório FastSurfer encontrado: $fastsurfer_dir"
    echo "   Sujeitos disponíveis: $subject_count"
else
    echo "❌ Diretório FastSurfer não encontrado: $fastsurfer_dir"
    echo "   Execute primeiro o processamento FastSurfer"
fi

echo ""
echo "🧪 TESTE DE CONECTIVIDADE OPENAI"
echo "================================"

# Criar script de teste
cat > test_openai_connection.py << 'EOF'
#!/usr/bin/env python3
import os
import openai

# Verificar API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("❌ OPENAI_API_KEY não configurada!")
    print("Configure a variável de ambiente OPENAI_API_KEY")
    exit(1)

print(f"✅ API Key configurada: {api_key[:10]}...")

try:
    # Testar conexão
    client = openai.OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Responda apenas 'OK' se esta mensagem foi recebida."}
        ],
        max_tokens=10
    )
    
    if response.choices[0].message.content.strip() == "OK":
        print("✅ Conexão com OpenAI estabelecida com sucesso!")
        print(f"   Modelo testado: gpt-3.5-turbo")
        print(f"   Tokens utilizados: {response.usage.total_tokens}")
    else:
        print("⚠️  Conexão estabelecida, mas resposta inesperada")
        
except Exception as e:
    print(f"❌ Erro na conexão: {e}")
    print("   Verifique sua API key e conexão com a internet")
EOF

# Executar teste
python3 test_openai_connection.py

echo ""
echo "📋 CRIANDO SCRIPTS DE EXECUÇÃO"
echo "=============================="

# Script para análise individual
cat > analyze_single_subject.sh << 'EOF'
#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Uso: $0 <subject_id>"
    echo "Exemplo: $0 OAS1_0001_MR1"
    exit 1
fi

SUBJECT_ID=$1

echo "🧠 ANÁLISE INDIVIDUAL COM OPENAI"
echo "Sujeito: $SUBJECT_ID"
echo "================================"

python3 -c "
import os
import sys
sys.path.append('.')
from openai_fastsurfer_analyzer import FastSurferDataExtractor, OpenAIFastSurferAnalyzer

# Configurações
fastsurfer_dir = '/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos'
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    print('❌ OPENAI_API_KEY não configurada!')
    exit(1)

# Extrair dados do sujeito
extractor = FastSurferDataExtractor(fastsurfer_dir)
metrics = extractor.extract_subject_metrics('$SUBJECT_ID')

if not metrics:
    print('❌ Sujeito não encontrado ou dados incompletos')
    exit(1)

# Analisar com OpenAI
analyzer = OpenAIFastSurferAnalyzer(api_key)
analysis = analyzer.analyze_subject(metrics)

print('\\n📊 ANÁLISE COMPLETA:')
print('=' * 50)
print(analysis['gpt_analysis'])
print('\\n' + '=' * 50)
print(f'Modelo: {analysis[\"model_used\"]}')
print(f'Tokens: {analysis[\"tokens_used\"]}')
"
EOF

chmod +x analyze_single_subject.sh

# Script para análise de coorte
cat > analyze_cohort.sh << 'EOF'
#!/bin/bash

MAX_SUBJECTS=${1:-10}
MAX_ANALYSES=${2:-5}

echo "🧠 ANÁLISE DE COORTE COM OPENAI"
echo "Sujeitos máximos: $MAX_SUBJECTS"
echo "Análises máximas: $MAX_ANALYSES"
echo "================================"

python3 openai_fastsurfer_analyzer.py
EOF

chmod +x analyze_cohort.sh

echo "✅ Scripts criados:"
echo "   • analyze_single_subject.sh - Análise individual"
echo "   • analyze_cohort.sh - Análise de coorte"

echo ""
echo "📖 MANUAL DE USO"
echo "================"

cat > MANUAL_OPENAI_FASTSURFER.md << 'EOF'
# MANUAL: ANÁLISE FASTSURFER COM OPENAI GPT

## Configuração Inicial

1. **Configure sua API Key:**
   ```bash
   export OPENAI_API_KEY="sua-chave-aqui"
   ```

2. **Execute o setup:**
   ```bash
   ./setup_openai_integration.sh
   ```

## Uso

### Análise Individual
```bash
./analyze_single_subject.sh OAS1_0001_MR1
```

### Análise de Coorte
```bash
./analyze_cohort.sh 20 5  # 20 sujeitos, 5 análises
```

### Execução Direta
```bash
python3 openai_fastsurfer_analyzer.py
```

## Funcionalidades

### 1. Extração de Dados FastSurfer
- Volumes subcorticais (hipocampo, amígdala, etc.)
- Métricas corticais (espessura, área, volume)
- Qualidade do processamento

### 2. Análise com GPT-4
- Interpretação clínica automatizada
- Detecção de padrões anômalos
- Recomendações clínicas
- Comparação com valores normativos

### 3. Relatórios
- Análises individuais detalhadas
- Relatórios de coorte agregados
- Dashboards visuais

## Vantagens vs CNN

### OpenAI GPT:
- ✅ Interpretação linguística natural
- ✅ Análise contextual avançada
- ✅ Recomendações personalizadas
- ✅ Linguagem médica apropriada
- ✅ Detecção de padrões sutis
- ❌ Custo por análise
- ❌ Dependência de internet

### CNN:
- ✅ Análise em tempo real
- ✅ Sem custo por predição
- ✅ Funciona offline
- ❌ Interpretabilidade limitada
- ❌ Requer treinamento extensivo

## Custos Estimados

- **GPT-4:** ~$0.03 por análise
- **GPT-3.5-turbo:** ~$0.002 por análise
- **100 análises:** $3.00 (GPT-4) ou $0.20 (GPT-3.5)

## Arquivos Gerados

- `openai_fastsurfer_analyses_YYYYMMDD_HHMM.csv`
- `openai_fastsurfer_cohort_report_YYYYMMDD_HHMM.txt`
- `openai_fastsurfer_dashboard_YYYYMMDD_HHMM.png`
EOF

echo "✅ Manual criado: MANUAL_OPENAI_FASTSURFER.md"

echo ""
echo "🎉 CONFIGURAÇÃO CONCLUÍDA!"
echo "=========================="
echo ""
echo "📋 PRÓXIMOS PASSOS:"
echo "1. Configure sua OPENAI_API_KEY"
echo "2. Execute: ./analyze_single_subject.sh OAS1_0001_MR1"
echo "3. Ou execute: ./analyze_cohort.sh"
echo ""
echo "📖 Consulte: MANUAL_OPENAI_FASTSURFER.md"
echo ""
echo "💡 DICA: Use GPT-3.5-turbo para testes (mais barato)"
echo "   Use GPT-4 para análises finais (mais preciso)"

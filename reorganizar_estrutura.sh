#!/bin/bash

# SCRIPT DE REORGANIZAÇÃO DA ESTRUTURA DO PROJETO ALZHEIMER
# Mantém a funcionalidade do alzheimer_analysis_suite.sh

echo "REORGANIZANDO ESTRUTURA DO PROJETO ALZHEIMER"
echo "============================================="
echo

# Criar estrutura de diretórios
echo "Criando estrutura de diretórios..."

# Diretórios principais
mkdir -p src/{cnn,openai,analysis,utils,visualization}
mkdir -p scripts/{setup,processing,monitoring,testing}
mkdir -p models/{checkpoints,scalers,reports}
mkdir -p data/{raw,processed,results}
mkdir -p docs/{manuais,relatorios,configuracoes}
mkdir -p logs/{tensorboard,processing,analysis}
mkdir -p config
mkdir -p tests

echo "✓ Estrutura de diretórios criada"

# Mover arquivos Python por categoria
echo
echo "Movendo arquivos Python..."

# CNN Models
echo "  - CNN Models..."
mv alzheimer_cnn_pipeline.py src/cnn/
mv alzheimer_cnn_pipeline_improved.py src/cnn/
mv alzheimer_cnn_improved_final.py src/cnn/
mv mci_detection_cnn_optimized.py src/cnn/
mv alzheimer_ai_pipeline.py src/cnn/

# OpenAI Integration
echo "  - OpenAI Integration..."
mv openai_fastsurfer_analyzer.py src/openai/
mv config_openai.py src/openai/
mv cnn_vs_openai_comparison.py src/openai/
mv test_openai_models.py src/openai/

# Analysis Scripts
echo "  - Analysis Scripts..."
mv alzheimer_early_diagnosis_analysis.py src/analysis/
mv mci_clinical_insights.py src/analysis/
mv analise_estatistica_hipocampo.py src/analysis/
mv machine_learning_hipocampo.py src/analysis/
mv fastsurfer_mci_analysis.py src/analysis/
mv dataset_explorer.py src/analysis/
mv pipeline_comparison.py src/analysis/

# Utils
echo "  - Utils..."
mv processar_T1_discos.py src/utils/
mv monitor_pipeline.py src/utils/
mv run_alzheimer.py src/utils/
mv run_pipeline_config.py src/utils/

# Visualization
echo "  - Visualization..."
# (arquivos de visualização já são gerados automaticamente)

# Scripts de Setup
echo "  - Setup Scripts..."
mv setup_*.sh scripts/setup/
mv install_openai_dependencies.sh scripts/setup/
mv setup_freesurfer.sh scripts/setup/

# Scripts de Processamento
echo "  - Processing Scripts..."
mv run_fastsurfer_*.sh scripts/processing/
mv processar_*.sh scripts/processing/
mv iniciar_processamento*.sh scripts/processing/
mv parar_processamento.sh scripts/processing/
mv parar_tudo.sh scripts/processing/

# Scripts de Monitoramento
echo "  - Monitoring Scripts..."
mv monitor_tempo_real.sh scripts/monitoring/
mv status_*.sh scripts/monitoring/
mv verificar_*.sh scripts/monitoring/

# Scripts de Teste
echo "  - Testing Scripts..."
mv test_*.sh scripts/testing/
mv teste_*.sh scripts/testing/

# Modelos e Checkpoints
echo "  - Models and Checkpoints..."
mv *.h5 models/checkpoints/ 2>/dev/null || true
mv *.joblib models/scalers/ 2>/dev/null || true
mv checkpoints/* models/checkpoints/ 2>/dev/null || true
mv checkpoints_*/* models/checkpoints/ 2>/dev/null || true

# Dados
echo "  - Data Files..."
mv *.csv data/processed/ 2>/dev/null || true
mv oasis_data data/raw/ 2>/dev/null || true

# Logs
echo "  - Logs..."
mv logs/* logs/tensorboard/ 2>/dev/null || true
mv logs_*/* logs/tensorboard/ 2>/dev/null || true

# Documentação
echo "  - Documentation..."
mv *.md docs/manuais/ 2>/dev/null || true
mv *_report_*.txt docs/relatorios/ 2>/dev/null || true
mv *_report_*.png docs/relatorios/ 2>/dev/null || true
mv env_example.txt docs/configuracoes/

# Configurações
echo "  - Configuration..."
mv .env* config/ 2>/dev/null || true
mv freesurfer_license*.txt config/ 2>/dev/null || true

# Resultados OpenAI
echo "  - OpenAI Results..."
mv openai_fastsurfer_*.csv data/results/ 2>/dev/null || true
mv openai_fastsurfer_*.txt data/results/ 2>/dev/null || true
mv openai_fastsurfer_*.png data/results/ 2>/dev/null || true

# Resultados de Análise
echo "  - Analysis Results..."
mv *_analysis.png data/results/ 2>/dev/null || true
mv *_performance_report.png data/results/ 2>/dev/null || true
mv *_report.png data/results/ 2>/dev/null || true

echo "✓ Arquivos movidos com sucesso"

# Criar __init__.py para tornar os diretórios Python packages
echo
echo "Criando __init__.py files..."
touch src/__init__.py
touch src/cnn/__init__.py
touch src/openai/__init__.py
touch src/analysis/__init__.py
touch src/utils/__init__.py
touch src/visualization/__init__.py
touch tests/__init__.py

echo "✓ __init__.py files criados"

# Atualizar alzheimer_analysis_suite.sh para usar os novos caminhos
echo
echo "Atualizando alzheimer_analysis_suite.sh..."

# Fazer backup do arquivo original
cp alzheimer_analysis_suite.sh alzheimer_analysis_suite.sh.backup

# Atualizar caminhos no script
sed -i 's|python3 dataset_explorer.py|python3 src/analysis/dataset_explorer.py|g' alzheimer_analysis_suite.sh
sed -i 's|python3 alzheimer_early_diagnosis_analysis.py|python3 src/analysis/alzheimer_early_diagnosis_analysis.py|g' alzheimer_analysis_suite.sh
sed -i 's|python3 mci_clinical_insights.py|python3 src/analysis/mci_clinical_insights.py|g' alzheimer_analysis_suite.sh

echo "✓ alzheimer_analysis_suite.sh atualizado"

# Criar script de execução principal
echo
echo "Criando script de execução principal..."
cat > run_analysis.sh << 'EOF'
#!/bin/bash

# SCRIPT PRINCIPAL DE EXECUÇÃO - PROJETO ALZHEIMER
# Executa o sistema de análise principal

echo "PROJETO ALZHEIMER - SISTEMA DE ANÁLISE"
echo "======================================"
echo

# Verificar se estamos no diretório correto
if [ ! -f "alzheimer_analysis_suite.sh" ]; then
    echo "ERRO: Execute este script do diretório raiz do projeto"
    exit 1
fi

# Executar o sistema principal
./alzheimer_analysis_suite.sh
EOF

chmod +x run_analysis.sh

echo "✓ Script principal criado: run_analysis.sh"

# Criar README da nova estrutura
echo
echo "Criando README da nova estrutura..."
cat > ESTRUTURA_REORGANIZADA.md << 'EOF'
# ESTRUTURA REORGANIZADA - PROJETO ALZHEIMER

## 📁 Estrutura de Diretórios

```
alzheimer/
├── src/                          # Código fonte Python
│   ├── cnn/                     # Modelos CNN
│   ├── openai/                  # Integração OpenAI
│   ├── analysis/                # Scripts de análise
│   ├── utils/                   # Utilitários
│   └── visualization/           # Visualizações
├── scripts/                     # Scripts Shell
│   ├── setup/                   # Configuração
│   ├── processing/              # Processamento
│   ├── monitoring/              # Monitoramento
│   └── testing/                 # Testes
├── models/                      # Modelos treinados
│   ├── checkpoints/             # Checkpoints dos modelos
│   ├── scalers/                 # Scalers salvos
│   └── reports/                 # Relatórios técnicos
├── data/                        # Dados
│   ├── raw/                     # Dados brutos
│   ├── processed/               # Dados processados
│   └── results/                 # Resultados
├── docs/                        # Documentação
│   ├── manuais/                 # Manuais
│   ├── relatorios/              # Relatórios
│   └── configuracoes/           # Configurações
├── logs/                        # Logs
│   ├── tensorboard/             # Logs TensorBoard
│   ├── processing/              # Logs de processamento
│   └── analysis/                # Logs de análise
├── config/                      # Configurações
├── tests/                       # Testes
├── alzheimer_analysis_suite.sh  # Sistema principal
└── run_analysis.sh              # Script de execução
```

## 🚀 Como Usar

### Execução Principal
```bash
./run_analysis.sh
```

### Execução Direta
```bash
./alzheimer_analysis_suite.sh
```

### Executar Scripts Específicos

#### CNN Models
```bash
python3 src/cnn/alzheimer_cnn_pipeline.py
python3 src/cnn/mci_detection_cnn_optimized.py
```

#### OpenAI Integration
```bash
python3 src/openai/openai_fastsurfer_analyzer.py
python3 src/openai/test_openai_models.py
```

#### Analysis
```bash
python3 src/analysis/alzheimer_early_diagnosis_analysis.py
python3 src/analysis/mci_clinical_insights.py
```

#### Processing Scripts
```bash
./scripts/processing/run_fastsurfer_oficial.sh
./scripts/processing/processar_todos_fastsurfer.sh
```

#### Setup Scripts
```bash
./scripts/setup/setup_openai_integration.sh
./scripts/setup/install_openai_dependencies.sh
```

## 📋 Principais Arquivos

- `alzheimer_analysis_suite.sh`: Sistema principal de análise
- `run_analysis.sh`: Script de execução principal
- `src/cnn/`: Modelos de deep learning
- `src/openai/`: Integração com OpenAI GPT
- `src/analysis/`: Análises clínicas e estatísticas
- `models/checkpoints/`: Modelos treinados
- `data/results/`: Resultados das análises
- `docs/relatorios/`: Relatórios gerados

## 🔧 Configuração

### OpenAI API
```bash
./scripts/setup/setup_api_key_manual.sh
```

### Dependências
```bash
./scripts/setup/install_openai_dependencies.sh
```

## 📊 Resultados

Os resultados são salvos em:
- `data/results/`: Arquivos CSV e PNG
- `docs/relatorios/`: Relatórios em texto
- `models/checkpoints/`: Modelos treinados

## 🔍 Monitoramento

```bash
./scripts/monitoring/monitor_tempo_real.sh
./scripts/monitoring/status_atual.sh
```
EOF

echo "✓ README da estrutura criado: ESTRUTURA_REORGANIZADA.md"

# Criar script de limpeza de diretórios vazios
echo
echo "Limpando diretórios vazios..."
find . -type d -empty -delete 2>/dev/null || true

echo "✓ Diretórios vazios removidos"

# Verificar se tudo está funcionando
echo
echo "Verificando integridade..."
if [ -f "alzheimer_analysis_suite.sh" ]; then
    echo "✓ alzheimer_analysis_suite.sh mantido"
else
    echo "✗ ERRO: alzheimer_analysis_suite.sh não encontrado"
fi

if [ -d "src" ]; then
    echo "✓ Estrutura src/ criada"
else
    echo "✗ ERRO: Estrutura src/ não criada"
fi

if [ -d "scripts" ]; then
    echo "✓ Estrutura scripts/ criada"
else
    echo "✗ ERRO: Estrutura scripts/ não criada"
fi

echo
echo "REORGANIZAÇÃO CONCLUÍDA!"
echo "======================="
echo
echo "Nova estrutura criada com sucesso!"
echo "Para usar o sistema:"
echo "  ./run_analysis.sh"
echo
echo "Arquivos importantes:"
echo "  - alzheimer_analysis_suite.sh (sistema principal)"
echo "  - run_analysis.sh (script de execução)"
echo "  - ESTRUTURA_REORGANIZADA.md (documentação)"
echo
echo "Backup do alzheimer_analysis_suite.sh original: alzheimer_analysis_suite.sh.backup"

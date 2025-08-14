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

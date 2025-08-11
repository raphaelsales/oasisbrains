# GUIA DE REORGANIZAÇÃO - PROJETO ALZHEIMER

## 📋 Visão Geral

Este guia explica como reorganizar a estrutura do seu projeto Alzheimer de forma segura, mantendo a funcionalidade do `alzheimer_analysis_suite.sh`.

## 🎯 Objetivos da Reorganização

- **Organizar** arquivos por categoria e função
- **Manter** funcionalidade do sistema principal
- **Facilitar** manutenção e desenvolvimento
- **Melhorar** navegação e compreensão do código
- **Preservar** todos os dados e resultados

## 📁 Nova Estrutura Proposta

```
alzheimer/
├── src/                          # Código fonte Python
│   ├── cnn/                     # Modelos CNN
│   │   ├── alzheimer_cnn_pipeline.py
│   │   ├── alzheimer_cnn_pipeline_improved.py
│   │   ├── alzheimer_cnn_improved_final.py
│   │   ├── mci_detection_cnn_optimized.py
│   │   └── alzheimer_ai_pipeline.py
│   ├── openai/                  # Integração OpenAI
│   │   ├── openai_fastsurfer_analyzer.py
│   │   ├── config_openai.py
│   │   ├── cnn_vs_openai_comparison.py
│   │   └── test_openai_models.py
│   ├── analysis/                # Scripts de análise
│   │   ├── alzheimer_early_diagnosis_analysis.py
│   │   ├── mci_clinical_insights.py
│   │   ├── analise_estatistica_hipocampo.py
│   │   ├── machine_learning_hipocampo.py
│   │   ├── fastsurfer_mci_analysis.py
│   │   ├── dataset_explorer.py
│   │   └── pipeline_comparison.py
│   ├── utils/                   # Utilitários
│   │   ├── processar_T1_discos.py
│   │   ├── monitor_pipeline.py
│   │   ├── run_alzheimer.py
│   │   └── run_pipeline_config.py
│   └── visualization/           # Visualizações
├── scripts/                     # Scripts Shell
│   ├── setup/                   # Configuração
│   │   ├── setup_*.sh
│   │   ├── install_openai_dependencies.sh
│   │   └── setup_freesurfer.sh
│   ├── processing/              # Processamento
│   │   ├── run_fastsurfer_*.sh
│   │   ├── processar_*.sh
│   │   ├── iniciar_processamento*.sh
│   │   ├── parar_processamento.sh
│   │   └── parar_tudo.sh
│   ├── monitoring/              # Monitoramento
│   │   ├── monitor_tempo_real.sh
│   │   ├── status_*.sh
│   │   └── verificar_*.sh
│   └── testing/                 # Testes
│       ├── test_*.sh
│       └── teste_*.sh
├── models/                      # Modelos treinados
│   ├── checkpoints/             # Checkpoints dos modelos
│   │   ├── *.h5
│   │   └── checkpoints_*/
│   ├── scalers/                 # Scalers salvos
│   │   └── *.joblib
│   └── reports/                 # Relatórios técnicos
│       └── *_technical_report.txt
├── data/                        # Dados
│   ├── raw/                     # Dados brutos
│   │   └── oasis_data/
│   ├── processed/               # Dados processados
│   │   └── *.csv
│   └── results/                 # Resultados
│       ├── openai_fastsurfer_*.csv
│       ├── openai_fastsurfer_*.txt
│       ├── openai_fastsurfer_*.png
│       ├── *_analysis.png
│       ├── *_performance_report.png
│       └── *_report.png
├── docs/                        # Documentação
│   ├── manuais/                 # Manuais
│   │   ├── *.md
│   │   └── *_guide.md
│   ├── relatorios/              # Relatórios
│   │   ├── *_report_*.txt
│   │   └── *_report_*.png
│   └── configuracoes/           # Configurações
│       └── env_example.txt
├── logs/                        # Logs
│   ├── tensorboard/             # Logs TensorBoard
│   │   ├── logs/
│   │   └── logs_*/
│   ├── processing/              # Logs de processamento
│   └── analysis/                # Logs de análise
├── config/                      # Configurações
│   ├── .env*
│   └── freesurfer_license*.txt
├── tests/                       # Testes
├── alzheimer_analysis_suite.sh  # Sistema principal
├── run_analysis.sh              # Script de execução
└── ESTRUTURA_REORGANIZADA.md    # Documentação da estrutura
```

## 🚀 Como Executar a Reorganização

### Passo 1: Verificação Prévia
```bash
./verificar_reorganizacao.sh
```

Este script irá:
- ✅ Verificar se está no diretório correto
- ✅ Verificar arquivos críticos
- ✅ Verificar espaço em disco
- ✅ Verificar processos em execução
- ✅ Verificar permissões
- ✅ Verificar estrutura existente

### Passo 2: Executar Reorganização
```bash
./reorganizar_estrutura.sh
```

Este script irá:
- 📁 Criar estrutura de diretórios
- 📦 Mover arquivos para pastas específicas
- 🔧 Atualizar `alzheimer_analysis_suite.sh`
- 📋 Criar documentação
- 💾 Fazer backup do arquivo original

### Passo 3: Verificar Funcionamento
```bash
./run_analysis.sh
```

## 🔧 Como Usar Após a Reorganização

### Execução Principal
```bash
./run_analysis.sh
# ou
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

## 🔄 Como Restaurar (Se Necessário)

Se algo der errado, você pode restaurar a estrutura original:

```bash
./restaurar_estrutura_original.sh
```

Este script irá:
- 🔙 Restaurar `alzheimer_analysis_suite.sh` original
- 📦 Mover todos os arquivos de volta
- 🗑️ Remover estrutura organizada
- 💾 Manter backups

## 📋 Arquivos de Backup

Após a reorganização, você terá:

- `alzheimer_analysis_suite.sh.backup` - Versão original
- `alzheimer_analysis_suite.sh.reorganizado` - Versão reorganizada (se restaurar)

## ⚠️ Considerações Importantes

### Antes da Reorganização
1. **Faça backup** do projeto completo
2. **Pare processos** em execução
3. **Verifique espaço** em disco
4. **Teste** o sistema atual

### Durante a Reorganização
1. **Não interrompa** o processo
2. **Aguarde** a conclusão
3. **Verifique** se não há erros

### Após a Reorganização
1. **Teste** o sistema principal
2. **Verifique** se todos os arquivos estão no lugar
3. **Teste** scripts específicos
4. **Documente** qualquer problema

## 🔍 Solução de Problemas

### Problema: "Arquivo não encontrado"
**Solução**: Verifique se o arquivo foi movido corretamente e se o caminho está atualizado.

### Problema: "Permissão negada"
**Solução**: Verifique permissões de escrita no diretório.

### Problema: "Estrutura já existe"
**Solução**: O script fará backup automático da estrutura existente.

### Problema: "Pouco espaço em disco"
**Solução**: Libere espaço ou use um diretório com mais espaço.

## 📞 Suporte

Se encontrar problemas:

1. **Verifique** os logs de erro
2. **Consulte** a documentação
3. **Restaure** a estrutura original se necessário
4. **Reporte** problemas específicos

## 🎉 Benefícios da Nova Estrutura

- **Organização**: Arquivos organizados por função
- **Manutenibilidade**: Fácil localização de arquivos
- **Escalabilidade**: Estrutura preparada para crescimento
- **Colaboração**: Mais fácil para outros desenvolvedores
- **Documentação**: Estrutura auto-documentada
- **Backup**: Sistema de backup e restauração

---

**Nota**: Esta reorganização mantém 100% da funcionalidade do `alzheimer_analysis_suite.sh` e adiciona organização sem quebrar o sistema existente.

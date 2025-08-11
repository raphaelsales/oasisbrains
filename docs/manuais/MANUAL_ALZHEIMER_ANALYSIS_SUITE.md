# MANUAL DO ALZHEIMER ANALYSIS SUITE

## Visão Geral

O Alzheimer Analysis Suite é um sistema integrado de análise para detecção precoce da Doença de Alzheimer, com foco específico no Comprometimento Cognitivo Leve (MCI) e biomarcadores neuroimagem. O sistema processa um dataset de 405 sujeitos utilizando inteligência artificial com GPU para classificação de alta precisão.

### Especificações Técnicas

- **Dataset**: 405 sujeitos com 42 features neuroimagem e clínicas
- **População MCI**: 68 sujeitos (16.8% do total)
- **Casos de Alzheimer**: 84 sujeitos (20.7% do total)
- **Classificador Binário**: 95.1% de acurácia (AUC: 0.992)
- **Classificador CDR**: 98.8% de acurácia
- **Processamento**: GPU NVIDIA RTX A4000 com Mixed Precision

## Iniciando o Sistema

Para iniciar o Alzheimer Analysis Suite, execute o comando:

```bash
./alzheimer_analysis_suite.sh
```

O sistema apresentará o menu principal com 9 opções organizadas em três categorias:

## Menu Principal - Opções Disponíveis

### ANÁLISES RÁPIDAS

#### Opção 1: Estatísticas Básicas

**Comando**: Digite `1` no menu principal

**Descrição**: Apresenta um resumo estatístico fundamental do dataset.

**Conteúdo Exibido**:
- Total de sujeitos no dataset (405)
- Número total de features (42)
- Distribuição por diagnóstico (Não-dementes vs Dementes)
- Idade média da população
- Pontuação MMSE média
- Tempo de execução: aproximadamente 5 segundos

**Indicado para**: Visão geral rápida do dataset, apresentações executivas, verificação inicial dos dados.

#### Opção 2: Análise Abrangente

**Comando**: Digite `2` no menu principal

**Descrição**: Análise estatística completa incluindo correlações, modelos e features.

**Conteúdo Exibido**:
- Estatísticas básicas completas
- Matriz de correlações entre variáveis clínicas (idade, MMSE, CDR)
- Resumo dos modelos treinados com tamanhos de arquivo
- Performance dos classificadores (95.1% binário, 98.8% CDR)
- Análise detalhada de features por categoria:
  - Hipocampo: 8 features
  - Amígdala: 8 features
  - Entorrinal: 8 features
  - Temporal: 8 features
  - Clínicas: 5 features
- Tempo de execução: aproximadamente 15 segundos

**Indicado para**: Análise científica, relatórios técnicos, avaliação completa do pipeline.

#### Opção 3: Explorador do Dataset (Completo)

**Comando**: Digite `3` no menu principal

**Descrição**: Exploração detalhada e visualização completa dos dados no formato de relatório científico.

**Conteúdo Exibido**:

1. **Dimensões do Dataset**:
   - 405 sujeitos × 42 features

2. **Distribuição por Diagnóstico**:
   - Não-dementes: 253 (62.5%)
   - Dementes: 152 (37.5%)

3. **Severidade CDR (Clinical Dementia Rating)**:
   - CDR 0.0 (Normal): 253 (62.5%)
   - CDR 0.5 (Muito Leve - MCI): 68 (16.8%)
   - CDR 1.0 (Leve): 64 (15.8%)
   - CDR 2.0 (Moderada): 20 (4.9%)

4. **Distribuição por Gênero**:
   - Mulheres: 244 (60.2%)
   - Homens: 161 (39.8%)

5. **Estatísticas de Idade**:
   - Média, mínimo, máximo, desvio padrão e mediana

6. **Estatísticas MMSE (Cognição)**:
   - Distribuição completa da pontuação cognitiva

7. **Análise do Hipocampo** (Principal Biomarcador):
   - Volumes médios por grupo diagnóstico

8. **Educação (Anos de Estudo)**:
   - Distribuição educacional da população

9. **Tabela de Composição Detalhada**:
   - Breakdown completo por categoria e porcentagem

10. **Análise de Biomarcadores Neuroimagem**:
    - Volumes regionais para regiões críticas
    - Comparação entre grupos (Normal, MCI, AD)

**Tempo de execução**: aproximadamente 20 segundos

**Indicado para**: Relatórios científicos, apresentações clínicas, análise estatística, validação de resultados.

### ANÁLISES ESPECIALIZADAS

#### Opção 4: Diagnóstico Precoce (Completo)

**Comando**: Digite `4` no menu principal

**Descrição**: Sistema especializado para diagnóstico precoce com foco em MCI e biomarcadores.

**Conteúdo Exibido**:

1. **Análise de Estágios Clínicos**:
   - Distribuição visual por estágio (Normal, MCI, Mild_AD, Moderate_AD)
   - Perfil demográfico detalhado por estágio

2. **Análise Aprofundada do MCI**:
   - População MCI: 68 sujeitos (16.8%)
   - Características clínicas específicas
   - Comparação entre grupos (Normal vs MCI vs AD)

3. **Biomarcadores Neuroimagem**:
   - Análise de 9 biomarcadores volumétricos
   - Poder discriminativo Normal → MCI → AD
   - Ranking dos 5 melhores biomarcadores para detecção precoce

4. **Análise de Correlações Cognitivas**:
   - Matriz de correlações entre MMSE, CDR, idade e educação
   - Insights clínicos sobre as relações

5. **Modelo de Risco para Detecção Precoce**:
   - Fatores de risco identificados
   - Análise de risco composto por score

6. **Recomendações Clínicas**:
   - Protocolo sugerido para triagem
   - Biomarcadores prioritários
   - Perfil de risco alto
   - Monitoramento longitudinal
   - Sinais de alerta para progressão

7. **Relatório Executivo**:
   - Principais descobertas
   - Impacto clínico esperado

**Tempo de execução**: aproximadamente 45 segundos

**Indicado para**: Protocolo de triagem clínica, seleção de biomarcadores, estratificação de risco, monitoramento longitudinal.

#### Opção 5: Análise Clínica MCI

**Comando**: Digite `5` no menu principal

**Descrição**: Sistema especializado em Comprometimento Cognitivo Leve com aplicação clínica direta.

**Conteúdo Exibido**:

1. **Perfil Clínico Detalhado - MCI**:
   - Amostra: 68 pacientes
   - Distribuição MMSE em quartis
   - Categorização de severidade MCI (Mild, Moderate, Severe)

2. **Achados Neuroanatômicos**:
   - Volumes regionais críticos (mm³)
   - Comparação Normal vs MCI com significância estatística
   - Identificação de regiões mais afetadas

3. **Diretrizes de Avaliação Cognitiva**:
   - Pontos de corte MMSE sugeridos
   - Sinais de alerta para progressão

4. **Estratificação de Risco**:
   - Fatores de risco para progressão MCI → AD
   - Classificação de risco (Baixo, Moderado, Alto)
   - Distribuição de pacientes por score de risco

5. **Protocolo Clínico de Manejo**:
   - Fluxo de atendimento sugerido
   - Triagem inicial
   - Investigação complementar
   - Monitoramento
   - Intervenções

6. **Resumo Executivo**:
   - Principais achados clínicos
   - Biomarcadores prioritários
   - Protocolo diagnóstico
   - Impacto clínico esperado

**Tempo de execução**: aproximadamente 35 segundos

**Indicado para**: Triagem em consulta, protocolos diagnósticos, monitoramento de pacientes, decisões terapêuticas.

### SISTEMA

#### Opção 6: Performance dos Modelos

**Comando**: Digite `6` no menu principal

**Descrição**: Informações técnicas sobre os modelos de IA treinados e performance do sistema.

**Conteúdo Exibido**:

1. **Modelos Treinados**:
   - alzheimer_binary_classifier.h5 (Classificador Binário: 95.1% accuracy)
   - alzheimer_cdr_classifier.h5 (Classificador CDR: 98.8% accuracy)
   - Tamanhos dos arquivos

2. **Scalers e Preprocessadores**:
   - alzheimer_binary_classifier_scaler.joblib
   - alzheimer_cdr_classifier_scaler.joblib
   - Tamanhos dos arquivos

3. **Dataset e Visualizações**:
   - alzheimer_complete_dataset.csv (Dataset Completo)
   - alzheimer_exploratory_analysis.png (Análise Visual)
   - Tamanhos dos arquivos

4. **GPU Performance**:
   - Placa: NVIDIA RTX A4000
   - Mixed Precision: Ativada
   - Memória Pico: 2.4MB
   - Speedup: 6-10x vs CPU
   - Tempo Total: 39s (ambos modelos)

**Tempo de execução**: aproximadamente 5 segundos

**Indicado para**: Verificação técnica, auditoria de modelos, avaliação de performance.

#### Opção 7: Status TensorBoard

**Comando**: Digite `7` no menu principal

**Descrição**: Monitoramento do TensorBoard para visualização de métricas de treinamento.

**Conteúdo Exibido**:

1. **Verificação de Status**:
   - Se TensorBoard está rodando ou não
   - URL de acesso (http://localhost:6006)

2. **Dados Disponíveis** (se rodando):
   - Scalars: Loss, Accuracy, Learning Rate
   - Graphs: Arquitetura dos modelos
   - Histograms: Distribuição de pesos
   - Images: Visualizações (se disponível)

3. **Estrutura dos Logs**:
   - Tamanho dos logs
   - Diretórios disponíveis

4. **Instruções** (se não rodando):
   - Comando para iniciar: `tensorboard --logdir=logs --host=0.0.0.0 --port=6006 &`
   - URL de acesso após inicialização

**Tempo de execução**: aproximadamente 5 segundos

**Indicado para**: Monitoramento de treinamento, análise de convergência, depuração de modelos.

#### Opção 8: Gerar Relatório Clínico

**Comando**: Digite `8` no menu principal

**Descrição**: Geração de relatório executivo para uso clínico e científico.

**Conteúdo Exibido**:

1. **Geração do Arquivo**:
   - Nome: alzheimer_clinical_report_YYYYMMDD_HHMM.txt
   - Confirmação de criação

2. **Conteúdo do Relatório**:
   - Data e hora de geração
   - Sistema utilizado
   - Resumo da população (405 sujeitos)
   - Características do MCI
   - Biomarcadores críticos
   - Recomendações clínicas
   - Performance do sistema IA
   - Fatores de risco para progressão
   - Impacto clínico esperado

3. **Opção de Visualização**:
   - Escolha para visualizar o relatório imediatamente
   - Exibição completa do conteúdo na tela

**Tempo de execução**: aproximadamente 10 segundos

**Indicado para**: Documentação clínica, relatórios científicos, apresentações executivas.

#### Opção 9: Informações do Sistema

**Comando**: Digite `9` no menu principal

**Descrição**: Informações técnicas do ambiente computacional e dependências.

**Conteúdo Exibido**:

1. **Ambiente**:
   - Sistema operacional
   - Usuário
   - Diretório atual
   - Data e hora

2. **Python**:
   - Versão do Python instalada

3. **Pacotes Principais**:
   - pandas: versão
   - numpy: versão
   - tensorflow: versão
   - sklearn: versão
   - Status de instalação (OK/ERRO)

4. **GPU**:
   - Nome da GPU detectada
   - Memória total e utilizada
   - Status de detecção

5. **Arquivos Principais**:
   - Contagem total de arquivos no projeto (.py, .sh, .csv, .h5, .joblib)

**Tempo de execução**: aproximadamente 5 segundos

**Indicado para**: Diagnóstico técnico, verificação de dependências, suporte técnico.

## Navegação e Uso

### Como Navegar

1. Execute `./alzheimer_analysis_suite.sh`
2. Digite o número da opção desejada (0-9)
3. Pressione ENTER
4. Aguarde a execução da análise
5. Pressione ENTER para retornar ao menu principal
6. Digite `0` para sair do sistema

### Recomendações de Uso

- **Para visão geral rápida**: Opções 1 ou 2
- **Para análise científica completa**: Opção 3
- **Para diagnóstico clínico**: Opções 4 ou 5
- **Para verificação técnica**: Opções 6, 7 ou 9
- **Para documentação**: Opção 8

### Tempo Total de Análise Completa

Executar todas as 9 opções sequencialmente levará aproximadamente 3-4 minutos, fornecendo uma análise completa e abrangente do sistema de diagnóstico precoce de Alzheimer.

## Requisitos do Sistema

- Sistema operacional: Linux
- Python 3.x
- Bibliotecas: pandas, numpy, tensorflow, sklearn
- GPU: NVIDIA com suporte CUDA (opcional, mas recomendado)
- Espaço em disco: mínimo 500MB para logs e outputs

## Suporte e Manutenção

O sistema foi desenvolvido para pesquisa em diagnóstico precoce de Alzheimer, focando especificamente na detecção de Comprometimento Cognitivo Leve (MCI) através de biomarcadores neuroimagem e avaliação clínica integrada. 
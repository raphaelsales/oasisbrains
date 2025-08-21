# Visualizações Multiclasse para Classificador CDR

Este documento descreve as novas funcionalidades de visualização implementadas para avaliar o desempenho do classificador multiclasse de CDR (Clinical Dementia Rating).

## Arquivos Implementados

### 1. `alzheimer_ai_pipeline.py` (Atualizado)
- **Novas funcões na classe `AlzheimerAnalysisReport`:**
  - `generate_multiclass_classification_report_plot()`: Gera gráfico do classification report
  - `generate_multiclass_confusion_matrix_plot()`: Gera matriz de confusão multiclasse
  - `generate_multiclass_evaluation_plots()`: Gera ambos os gráficos

### 2. `generate_multiclass_plots.py` (Novo)
- **Script independente para gerar visualizações**
- Pode usar dados existentes ou dados sintéticos
- Funciona como demonstração ou análise separada

## Visualizações Geradas

### 1. Classification Report (`classification_report_multiclasse.png`)
- **Formato**: Tabela colorida com heatmap
- **Métricas**: Precisão, Revocação, F1-Score e Suporte por classe
- **Classes**: CDR=0.0, CDR=0.5, CDR=1.0, CDR=2.0
- **Informações adicionais**: 
  - Acurácia global no título
  - Médias macro e weighted
  - Colormap RdYlGn (vermelho-amarelo-verde)

### 2. Matriz de Confusão (`matriz_confusao_multiclasse.png`)
- **Formato**: Heatmap com valores absolutos
- **Colormap**: Blues (tons de azul)
- **Informações incluídas**:
  - Acurácia global no título
  - Valores absolutos em cada célula
  - Estatísticas por classe (Precisão e Revocação) na lateral
  - Labels dos eixos: Real vs Predição

## Como Usar

### Opção 1: Pipeline Completo
```bash
python alzheimer_ai_pipeline.py
```
- Executa todo o pipeline de análise
- Treina modelos e gera visualizações automaticamente
- Inclui as novas visualizações multiclasse na ETAPA 4

### Opção 2: Apenas Visualizações
```bash
python generate_multiclass_plots.py
```
- Gera apenas as visualizações
- Usa dados existentes se disponíveis
- Caso contrário, usa dados sintéticos realistas

### Opção 3: Uso Programático
```python
from alzheimer_ai_pipeline import AlzheimerAnalysisReport
import numpy as np

# Dados de exemplo (y_true e y_pred como arrays de inteiros)
y_true = np.array([0, 1, 2, 0, 1, 3, 0, 2])  # CDR real
y_pred = np.array([0, 1, 1, 0, 0, 3, 0, 2])  # CDR predito

# Criar gerador de relatórios
report = AlzheimerAnalysisReport(features_df)

# Definir nomes das classes
class_names = ['CDR=0.0', 'CDR=0.5', 'CDR=1.0', 'CDR=2.0']

# Gerar visualizações
report_path, confusion_path = report.generate_multiclass_evaluation_plots(
    y_true, y_pred, class_names
)
```

## Entrada de Dados

### Formato Esperado
- **y_true**: Array numpy com valores reais (inteiros 0, 1, 2, 3)
- **y_pred**: Array numpy com predições (inteiros 0, 1, 2, 3)
- **class_names**: Lista opcional com nomes das classes

### Mapeamento CDR
- **0**: CDR=0.0 (Normal)
- **1**: CDR=0.5 (Muito Leve)
- **2**: CDR=1.0 (Leve)
- **3**: CDR=2.0 (Moderado)

## Características Técnicas

### Dependências
- matplotlib
- seaborn 
- scikit-learn
- numpy
- pandas

### Configurações de Visualização
- **DPI**: 300 (alta resolução)
- **Formato**: PNG com fundo branco
- **Tamanho**: Ajustado automaticamente
- **Estilo**: Seaborn com fallback para matplotlib padrão

### Tratamento de Dados
- **Conversão automática**: Float para inteiro quando necessário
- **Compatibilidade**: sklearn metrics (requer inteiros)
- **Validação**: Verificação de tipos e valores
- **Fallback**: Dados sintéticos se dados reais não disponíveis

## Exemplos de Saída

### Métricas Típicas
```
Classification Report:
              precision    recall  f1-score   support

     CDR=0.0       0.95      0.80      0.87        50
     CDR=0.5       0.57      0.86      0.69        14
     CDR=1.0       0.50      0.38      0.43        13
     CDR=2.0       0.25      0.50      0.33         4

    accuracy                           0.73        81
   macro avg       0.57      0.64      0.58        81
weighted avg       0.78      0.73      0.74        81
```

### Interpretação
- **CDR=0.0**: Melhor desempenho (normal vs demência)
- **CDR=0.5**: Desafio médio (estágio inicial)
- **CDR=1.0+**: Mais difícil (poucos exemplos, confusão entre estágios)

## Integração no Pipeline

As visualizações são automaticamente geradas na **ETAPA 4** do pipeline principal, após o treinamento dos modelos. Os arquivos são salvos no diretório de trabalho atual.

### Arquivos Gerados
- `classification_report_multiclasse.png`
- `matriz_confusao_multiclasse.png`
- Logs de execução com estatísticas detalhadas

## Personalização

### Modificar Cores
```python
# No método generate_classification_report_plot
im = ax.imshow(data_matrix[:, :3], cmap='RdYlGn', ...)  # Alterar cmap

# No método generate_confusion_matrix_plot  
sns.heatmap(cm, cmap='Blues', ...)  # Alterar cmap
```

### Ajustar Tamanhos
```python
# Tamanho da figura
fig, ax = plt.subplots(figsize=(12, 8))  # Alterar figsize

# DPI para resolução
plt.savefig(save_path, dpi=300, ...)  # Alterar dpi
```

## Resolução de Problemas

### Erro "continuous is not supported"
- **Causa**: Arrays com valores float em vez de int
- **Solução**: Converter para inteiros usando o mapeamento CDR

### Erro "No module named 'seaborn'"
- **Causa**: Dependência não instalada
- **Solução**: `pip install seaborn`

### Gráficos não aparecem
- **Causa**: Backend matplotlib em ambiente sem display
- **Solução**: Os arquivos PNG são sempre salvos, mesmo sem display

### Classes faltando
- **Causa**: Dados desbalanceados ou poucos exemplos
- **Solução**: Verificar distribuição de classes e aumentar dados de treino

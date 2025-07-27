# Guia de Execução - Pipelines de Análise de Alzheimer

## 📋 Visão Geral

Este projeto oferece **dois pipelines diferentes** para análise de neuroimagem focada em Alzheimer:

1. **Pipeline Original (MLP)**: Rápido, baseado em features extraídas
2. **Pipeline CNN 3D**: Avançado, trabalha com imagens completas

## 🎯 Qual Pipeline Escolher?

### Pipeline Original (MLP) - `alzheimer_ai_pipeline.py`
**Use quando:**
- ✅ Recursos computacionais limitados
- ✅ Precisa de resultados rápidos (10-30 minutos)
- ✅ Triagem clínica em massa
- ✅ Interpretabilidade é importante
- ✅ Dataset pequeno (<100 amostras)

### Pipeline CNN 3D - `alzheimer_cnn_pipeline.py`
**Use quando:**
- ✅ Tem GPU robusta (≥8GB VRAM)
- ✅ Foco em pesquisa científica
- ✅ Detecção precoce de MCI é prioridade
- ✅ Máxima acurácia é necessária
- ✅ Dataset grande (>500 amostras)

## 🔧 Requisitos do Sistema

### Requisitos Básicos (Ambos Pipelines)
```bash
# Python 3.8+
python --version

# Dependências principais
pip install pandas numpy nibabel scikit-learn matplotlib seaborn
pip install tensorflow keras joblib
```

### Requisitos Específicos por Pipeline

#### Pipeline Original (MLP)
```yaml
CPU: Qualquer processador moderno
RAM: 4-8 GB
GPU: Opcional (acelera treinamento)
Tempo: 10-30 minutos
```

#### Pipeline CNN 3D
```yaml
CPU: Intel i7+ ou AMD Ryzen 7+
RAM: 16-32 GB
GPU: NVIDIA RTX 3080+ (8GB+ VRAM)
CUDA: 11.0+
cuDNN: 8.0+
Tempo: 2-8 horas
```

### Instalação de Dependências GPU (CNN 3D)
```bash
# Para NVIDIA GPU com CUDA
pip install tensorflow[and-cuda]

# Verificar instalação
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

## 🚀 Como Executar

### 1. Preparação dos Dados

Certifique-se de que seus dados estão organizados assim:
```
oasis_data/
├── outputs_fastsurfer_definitivo_todos/
│   ├── OAS1_0001_MR1/
│   │   └── mri/
│   │       ├── T1.mgz
│   │       ├── aparc+aseg.mgz
│   │       └── ...
│   ├── OAS1_0002_MR1/
│   └── ...
```

### 2. Execução do Pipeline Original (MLP)

```bash
# Execução padrão (todos os sujeitos)
python alzheimer_ai_pipeline.py

# Execução rápida (primeiros 50 sujeitos)
# Edite a linha: max_subjects = 50
python alzheimer_ai_pipeline.py
```

**Saídas esperadas:**
- `alzheimer_complete_dataset.csv`
- `alzheimer_exploratory_analysis.png`
- `alzheimer_binary_classifier.h5`
- `alzheimer_cdr_classifier.h5`

### 3. Execução do Pipeline CNN 3D

```bash
# IMPORTANTE: Verificar GPU primeiro
python -c "import tensorflow as tf; print('GPU disponível:', len(tf.config.list_physical_devices('GPU')) > 0)"

# Execução (recomendado começar com dataset pequeno)
python alzheimer_cnn_pipeline.py
```

**Saídas esperadas:**
- `mci_subjects_metadata.csv`
- `mci_detection_performance_report.png`
- Logs detalhados no terminal

### 4. Comparação de Pipelines

```bash
# Execute análise comparativa
python pipeline_comparison.py
```

**Saídas:**
- `pipeline_comparison_report.txt`
- `pipeline_performance_comparison.png`

## 📊 Interpretação dos Resultados

### Métricas Importantes

#### Acurácia (Accuracy)
- **>85%**: Excelente para uso clínico
- **75-85%**: Boa para pesquisa
- **<75%**: Necessita otimização

#### AUC (Area Under Curve)
- **>90%**: Discriminação excelente
- **80-90%**: Discriminação boa
- **<80%**: Discriminação limitada

#### Interpretação Clínica
```python
# Normal vs Demência (Pipeline MLP)
if accuracy > 0.85 and auc > 0.90:
    print("Modelo adequado para triagem clínica")
    
# Normal vs MCI (Pipeline CNN)
if accuracy > 0.80 and auc > 0.85:
    print("Modelo adequado para detecção precoce")
```

## ⚡ Dicas de Otimização

### Para Pipeline MLP
```python
# Otimizar para datasets pequenos
max_subjects = 100  # Ajustar conforme necessário

# Balancear classes
# Verificar distribuição CDR no dataset
```

### Para Pipeline CNN 3D
```python
# Reduzir uso de memória
batch_size = 2  # Para GPUs com menos memória
target_shape = (64, 64, 64)  # Imagens menores

# Monitorar GPU
nvidia-smi -l 1  # Terminal separado
```

## 🛠️ Solução de Problemas

### Problemas Comuns

#### 1. Erro de Memória GPU
```bash
# Sintoma: "ResourceExhaustedError"
# Solução: Reduzir batch_size ou target_shape
batch_size = 1
target_shape = (48, 48, 48)
```

#### 2. TensorFlow não encontra GPU
```bash
# Verificar CUDA
nvidia-smi

# Reinstalar TensorFlow
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

#### 3. Dados insuficientes
```bash
# Sintoma: "Dados insuficientes para treinamento"
# Solução: Verificar estrutura de diretórios
ls -la oasis_data/outputs_fastsurfer_definitivo_todos/
```

#### 4. Erro de arquivo não encontrado
```python
# Verificar se arquivos MRI existem
subject_path = "caminho/para/sujeito"
os.path.exists(os.path.join(subject_path, 'mri', 'T1.mgz'))
```

### Logs de Debug

Ativar logs detalhados:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# No início dos pipelines
print("MODO DEBUG ATIVADO")
```

## 📈 Configurações Avançadas

### Pipeline MLP - Hiperparâmetros
```python
# Em DeepAlzheimerClassifier.create_deep_model()
layers.Dense(512, activation='relu'),  # Aumentar neurônios
learning_rate=0.0005,                  # Ajustar learning rate
patience=35,                           # Aumentar paciência
```

### Pipeline CNN 3D - Hiperparâmetros
```python
# Em MCI_CNN3D_Classifier.create_cnn3d_model()
layers.Conv3D(64, (3, 3, 3)),         # Mais filtros
learning_rate=0.00005,                 # LR mais baixo
dropout_rate=0.6,                      # Mais regularização
```

## 🎓 Melhores Práticas

### 1. Desenvolvimento Iterativo
```bash
# Começar pequeno
max_subjects = 20

# Expandir gradualmente
max_subjects = 50, 100, 200...
```

### 2. Validação Cruzada
```python
# Pipeline CNN usa StratifiedKFold
n_folds = 5  # Padrão
n_folds = 3  # Para datasets pequenos
```

### 3. Monitoramento
```python
# Usar TensorBoard (CNN)
tensorboard --logdir=./logs

# Salvar checkpoints
keras.callbacks.ModelCheckpoint('model_checkpoint.h5')
```

## 📚 Recursos Adicionais

### Literatura Relevante
- OASIS-1 Dataset: https://www.oasis-brains.org/
- FreeSurfer Documentation: https://surfer.nmr.mgh.harvard.edu/
- TensorFlow GPU Guide: https://www.tensorflow.org/install/gpu

### Suporte
- Verificar logs de erro completos
- Testar com subset pequeno primeiro
- Monitorar uso de recursos durante execução

## 🔍 Checklist de Execução

Antes de executar:
- [ ] Dados OASIS organizados corretamente
- [ ] Python 3.8+ instalado
- [ ] Dependências instaladas
- [ ] GPU configurada (para CNN)
- [ ] Espaço em disco suficiente (≥10GB)

Durante a execução:
- [ ] Monitorar uso de memória
- [ ] Verificar logs de erro
- [ ] Acompanhar métricas de treinamento

Após a execução:
- [ ] Verificar arquivos de saída
- [ ] Interpretar métricas de performance
- [ ] Salvar resultados importantes

---

**💡 Dica Final**: Comece sempre com o Pipeline MLP para validar seus dados e configuração, depois migre para CNN 3D se necessário maior precisão! 
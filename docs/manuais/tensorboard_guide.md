# 📊 GUIA TENSORBOARD - PIPELINE ALZHEIMER

## 🔍 Como Acessar
```bash
# TensorBoard já está rodando em:
http://localhost:6006

# Ou inicie manualmente:
tensorboard --logdir=logs --host=0.0.0.0 --port=6006
```

## 📈 INTERPRETANDO AS MÉTRICAS

### 🎯 SCALARS (Métricas Principais)

#### **📊 Loss (Perda)**
- **Training Loss**: Como o modelo está aprendendo
  - ⬇️ Deve diminuir ao longo das épocas
  - 📉 Se aumenta = overfitting
  
- **Validation Loss**: Generalização do modelo
  - ⬇️ Deve diminuir junto com training
  - ⚠️ Se aumenta enquanto training diminui = overfitting

#### **🎯 Accuracy (Precisão)**
- **Training Accuracy**: Precisão nos dados de treino
  - ⬆️ Deve aumentar ao longo das épocas
  - 🎯 Meta: > 80%

- **Validation Accuracy**: Precisão em dados não vistos
  - ⬆️ Deve aumentar (mais importante que training)
  - 🎯 Meta: > 75%

#### **📊 Learning Rate**
- 📈 Como a taxa de aprendizagem muda
- 🔧 ReduceLROnPlateau: diminui quando loss para de melhorar

### 🧠 GRAPHS (Arquitetura)

#### **🏗️ Estrutura do Modelo**
- **Input Layer**: 39 features neuroimagem
- **Hidden Layers**: 256 → 128 → 64 → 32 → 16 neurons
- **Output Layer**: 1 (binário) ou 4 (CDR) neurons
- **Activations**: ReLU (hidden), Sigmoid/Softmax (output)
- **Regularization**: Dropout + BatchNormalization

### ⚡ PROFILER (Performance GPU)

#### **🚀 GPU Performance**
- **Device Utilization**: % uso da GPU
  - 🎯 Meta: > 80% durante treinamento
  - ⚠️ Se < 50% = gargalo CPU/I-O

- **Memory Usage**: Uso de VRAM
  - 📊 Atual: ~2MB (muito eficiente!)
  - 💾 RTX A4000: 16GB disponível

- **Step Time**: Tempo por batch
  - ⚡ GPU: ~48-300ms por step
  - 🐌 CPU seria: ~1-3s por step

### 📊 HISTOGRAMS (Distribuição)

#### **⚖️ Weights Distribution**
- 📊 Como os pesos estão distribuídos
- ✅ Boa distribuição: gaussiana centrada em 0
- ❌ Problema: todos zeros ou muito grandes

#### **📈 Gradients**
- 🔄 Como os gradientes fluem na rede
- ✅ Bom: valores pequenos mas não zero
- ❌ Vanishing: muito próximos de zero
- ❌ Exploding: valores muito grandes

## 🎨 VISUALIZAÇÕES ESPECÍFICAS DO SEU PROJETO

### 🧠 Classificador Binário (Demented/Non-demented)
```
📊 Métricas Finais:
- Training Accuracy: 93.8%
- Validation Accuracy: 75.0%
- AUC Score: 0.736
- Épocas: 50
- Tempo: 18.2s
```

### 🎯 Classificador CDR (0, 0.5, 1, 2)
```
📊 Métricas Finais:
- Training Accuracy: 82.5%
- Validation Accuracy: 85.0%
- Classes: 4 (CDR levels)
- Épocas: 50
- Tempo: 18.3s
```

## 🔧 DICAS DE ANÁLISE

### ✅ Sinais de Bom Treinamento:
- 📉 Loss diminuindo suavemente
- 📈 Accuracy aumentando consistentemente
- 🎯 Diferença training/validation < 10%
- ⚡ GPU utilization > 80%

### ⚠️ Sinais de Problemas:
- 📈 Loss aumentando ou oscilando muito
- 🔄 Training accuracy muito maior que validation
- 🐌 GPU utilization < 50%
- 💥 Gradientes exploding/vanishing

### 🎛️ Ajustes Recomendados:
- **Overfitting**: Aumentar dropout, reduzir modelo
- **Underfitting**: Diminuir dropout, aumentar modelo
- **Slow convergence**: Aumentar learning rate
- **Instability**: Diminuir learning rate

## 🎉 RECURSOS AVANÇADOS

### 📸 Export de Dados:
```bash
# Salvar gráficos como imagem
Download → PNG

# Exportar dados como CSV
Download → CSV
```

### 🔄 Comparar Experimentos:
- Upload múltiplos runs
- Comparar side-by-side
- Analisar hyperparameters

### ⏱️ Real-time Monitoring:
- Auto-refresh a cada 30s
- Monitorar treinamentos longos
- Parar early se necessário 
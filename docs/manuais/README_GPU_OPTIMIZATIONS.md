# 🚀 Otimizações GPU/CPU - Pipeline Alzheimer

## 🔥 Otimizações Implementadas

### 1. **Configuração Automática de GPU**
- ✅ Detecção automática de GPUs disponíveis
- ✅ Configuração de crescimento de memória GPU
- ✅ Mixed Precision (float16) para acelerar treinamento
- ✅ Estratégias de distribuição otimizadas
- ✅ Monitoramento de uso de memória GPU

### 2. **Modelo Deep Learning Otimizado**
- ✅ Arquitetura mais profunda quando GPU disponível (256→128→64→32→16)
- ✅ Batch size adaptativo (64 para GPU, 32 para CPU)
- ✅ Dropout e BatchNormalization para regularização
- ✅ Otimizador Adam com configurações específicas para mixed precision

### 3. **Pipeline de Treinamento Avançado**
- ✅ Conversão para tensores TensorFlow para melhor performance
- ✅ Callbacks otimizados (EarlyStopping, ReduceLROnPlateau)
- ✅ TensorBoard para monitoramento (quando GPU disponível)
- ✅ Cronometragem precisa de treinamento
- ✅ Monitoramento contínuo de GPU

### 4. **Fallback Inteligente para CPU**
- ✅ Configurações otimizadas quando GPU não disponível
- ✅ Threading paralelo otimizado para CPU
- ✅ Batch sizes apropriados para CPU
- ✅ Relatórios de performance comparativos

## 🔧 Status do Sistema Atual

**Baseado nos testes executados:**

```
📦 TensorFlow: 2.19.0
🎯 GPUs detectadas: 0
🔥 CUDA build: True (TensorFlow compilado com CUDA)
🔥 GPU disponível: False (Bibliotecas GPU ausentes)
```

## 💡 Como Ativar GPU (se você tiver uma GPU NVIDIA)

### Pré-requisitos:
1. **GPU NVIDIA** com compute capability ≥ 3.5
2. **Drivers NVIDIA** atualizados
3. **CUDA** e **cuDNN** instalados

### Passos para Ativação:

#### 1. Verificar Hardware
```bash
# Verificar se há GPU NVIDIA
lspci | grep -i nvidia

# Verificar drivers
nvidia-smi
```

#### 2. Instalar CUDA e cuDNN
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

#### 3. Configurar Variáveis de Ambiente
```bash
# Adicionar ao ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### 4. Instalar TensorFlow com GPU
```bash
pip install tensorflow[gpu]==2.19.0
# ou
conda install tensorflow-gpu=2.19.0
```

#### 5. Testar Configuração
```bash
python3 teste_gpu_config.py
```

## 🚀 Executando o Pipeline

### Comando Principal:
```bash
python3 alzheimer_ai_pipeline.py
```

### Teste Rápido:
```bash
python3 teste_gpu_config.py
```

## 📊 Performance Esperada

### Com GPU (estimativa):
- ⚡ **3-5x mais rápido** que CPU
- 🧠 **Mixed Precision**: 1.5-2x adicional
- 📦 **Batch Size**: 64+ amostras
- ⏱️ **Tempo típico**: 30-60s por modelo

### Com CPU (atual):
- 🖥️ **Performance otimizada** para CPU
- 📦 **Batch Size**: 32 amostras
- ⏱️ **Tempo típico**: 2-5 minutos por modelo
- 🔧 **Threading**: Paralelo otimizado

## 🎯 Arquivos Gerados

O pipeline gera automaticamente:

```
📁 Resultados:
   - alzheimer_complete_dataset.csv      # Dataset completo
   - alzheimer_exploratory_analysis.png  # Visualizações
   - alzheimer_binary_classifier.h5      # Modelo binário
   - alzheimer_cdr_classifier.h5         # Modelo CDR
   - alzheimer_*_scaler.joblib           # Scalers

📊 Monitoramento (se GPU):
   - logs/                               # TensorBoard logs
```

## 🔍 Troubleshooting

### Erro: "Cannot dlopen some GPU libraries"
```bash
# Verificar instalação CUDA
nvcc --version
ldconfig -p | grep cuda

# Reinstalar bibliotecas
sudo apt-get install libcudnn8 libcudnn8-dev
```

### Performance lenta em CPU
```bash
# Otimizar threads
export OMP_NUM_THREADS=4
export TF_NUM_INTEROP_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
```

### Memória insuficiente
- Reduza `batch_size` em `alzheimer_ai_pipeline.py`
- Use datasets menores para teste
- Configure `tf.config.experimental.set_memory_growth()`

## 🎉 Recursos Implementados

✅ **Detecção automática GPU/CPU**  
✅ **Mixed Precision training**  
✅ **Batch size adaptativo**  
✅ **Monitoramento de memória**  
✅ **TensorBoard integration**  
✅ **Fallback inteligente CPU**  
✅ **Performance profiling**  
✅ **Relatórios detalhados**  

---

🚀 **Pipeline otimizado para máxima performance em qualquer hardware!** 
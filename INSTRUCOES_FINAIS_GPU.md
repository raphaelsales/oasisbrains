# 🚀 CONFIGURAÇÃO COMPLETA: NVIDIA RTX A4000 + CUDA 12.9 + Ubuntu 24.04

## ✅ STATUS ATUAL DA INSTALAÇÃO

### ✅ Instalado com Sucesso:
- **CUDA Toolkit 12.9** ✅
- **Driver NVIDIA 575.57.08** ✅ 
- **CuPy 13.5.1 (cupy-cuda12x)** ✅
- **Variáveis de ambiente configuradas** ✅

### 🔄 Próximo Passo OBRIGATÓRIO:
**REINICIAR O SISTEMA** para carregar os novos drivers NVIDIA 575.

```bash
sudo reboot
```

---

## 🧪 TESTES APÓS REBOOT

### 1. Teste Básico do Sistema
```bash
# Verificar CUDA
nvcc --version
# Deve mostrar: Cuda compilation tools, release 12.9, V12.9.86

# Verificar GPU
nvidia-smi
# Deve mostrar: NVIDIA RTX A4000 com informações de memória
```

### 2. Teste Completo GPU vs CPU
```bash
cd /app/alzheimer
python3 teste_gpu_completo.py
```

### 3. Análise de Hipocampo Otimizada
```bash
python3 analisar_hipocampo_gpu_otimizado.py
```

---

## 📋 COMO SABER SE ESTÁ USANDO GPU

### 🔍 Indicadores Visuais nos Scripts:

**GPU Funcionando:**
```
⚡ GPU: ✅ FUNCIONANDO
✅ GPU detectada: NVIDIA RTX A4000
⚡ Classificação Bayesiana - Executando na GPU
⚡ Tempo GPU: 0.0234 segundos
```

**CPU (Fallback):**
```
🖥️ GPU: ❌ FALHA
🖥️ Classificação Bayesiana - Executando na CPU
🖥️ Tempo CPU: 0.1456 segundos
```

### 📊 Comparação de Performance:
```
🚀 GPU é 6.22x MAIS RÁPIDO!
⚡ Operações GPU: 50 (tempo médio: 0.0234s)
🖥️ Operações CPU: 0 (tempo médio: 0.0000s)
```

---

## 🔧 VERIFICAÇÕES DE COMPATIBILIDADE

### Baseado em: [Saturn Cloud - GPU Check Guide](https://saturncloud.io/blog/how-to-check-whether-your-code-is-running-on-the-gpu-or-cpu/)

### Para PyTorch (se usar):
```python
import torch
print(f"GPU disponível: {torch.cuda.is_available()}")
print(f"Dispositivo: {torch.cuda.get_device_name(0)}")
```

### Para CuPy (nosso caso):
```python
import cupy as cp
print(f"CuPy versão: {cp.__version__}")
print(f"Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
```

---

## 🎯 SCRIPTS DISPONÍVEIS

### 1. `teste_gpu_completo.py`
- **Propósito:** Diagnóstico completo GPU vs CPU
- **Recursos:** Benchmarks, testes de memória, indicadores visuais
- **Uso:** Verificar se GPU está funcionando após reboot

### 2. `analisar_hipocampo_gpu_otimizado.py`
- **Propósito:** Análise científica com monitoramento GPU/CPU
- **Recursos:** Fallback automático, métricas de performance, comparação de tempos
- **Uso:** Análise de produção dos dados de hipocampo

### 3. `analisar_hipocampo_direto.py` (Original)
- **Propósito:** Versão básica com fallback simples
- **Status:** Funcional, mas sem monitoramento avançado

---

## 🛠️ SOLUÇÃO DE PROBLEMAS

### Se após reboot ainda não funcionar:

#### 1. Verificar Módulos
```bash
lsmod | grep nvidia
# Deve mostrar: nvidia_uvm, nvidia_drm, nvidia_modeset, nvidia
```

#### 2. Recarregar Drivers
```bash
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia nvidia_modeset nvidia_drm nvidia_uvm
```

#### 3. Verificar PATH
```bash
echo $PATH | grep cuda
echo $LD_LIBRARY_PATH | grep cuda
```

#### 4. Reinstalar CuPy (se necessário)
```bash
pip uninstall cupy-cuda12x -y
pip install cupy-cuda12x
```

---

## 📈 PERFORMANCE ESPERADA

### NVIDIA RTX A4000 Specifications:
- **CUDA Cores:** 6,144
- **Memory:** 16GB GDDR6
- **Compute Capability:** 8.6
- **Expected Speedup:** 5-10x sobre CPU para operações matriciais

### Benchmark Típico (matriz 5000x5000):
- **CPU:** ~0.31 segundos
- **GPU:** ~0.05 segundos  
- **Speedup:** ~6x

---

## 💡 DICAS DE USO

### Para Máxima Performance:
1. **Use matrizes grandes** (>1000x1000) para aproveitar paralelismo GPU
2. **Evite transferências frequentes** CPU ↔ GPU
3. **Use dtype=float32** em vez de float64 quando possível
4. **Monitore uso de memória** GPU com `nvidia-smi`

### Exemplo de Código Otimizado:
```python
import cupy as cp

# ✅ Bom: Operações na GPU
a_gpu = cp.random.random((5000, 5000), dtype=cp.float32)
b_gpu = cp.random.random((5000, 5000), dtype=cp.float32)
result_gpu = cp.dot(a_gpu, b_gpu)

# ❌ Evitar: Transferências desnecessárias
a_cpu = np.random.random((5000, 5000))
a_gpu = cp.asarray(a_cpu)  # CPU → GPU
result_cpu = cp.asnumpy(result_gpu)  # GPU → CPU
```

---

## 🎯 RESULTADO FINAL

Após o **reboot**, você terá:
- ✅ **GPU NVIDIA RTX A4000** totalmente funcional
- ✅ **CUDA 12.9** configurado corretamente
- ✅ **Scripts de análise** com aceleração GPU
- ✅ **Monitoramento em tempo real** de GPU vs CPU
- ✅ **Indicadores visuais claros** do dispositivo em uso

### 🚀 Performance de Produção:
- **401 sujeitos** processados com aceleração GPU
- **Speedup esperado:** 5-10x mais rápido que CPU
- **Monitoramento completo** de performance e uso de recursos

---

**Próximo passo:** `sudo reboot` e depois execute os scripts de teste! 🎉 
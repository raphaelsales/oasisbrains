#!/usr/bin/env python3
"""
<<<<<<< HEAD
TESTE COMPLETO DE GPU - ALZHEIMER PIPELINE
Verifica capacidade e performance da GPU para processamento
"""

import time
import numpy as np
from typing import Tuple

# Cores para output colorido
=======
🧪 TESTE COMPLETO DE GPU vs CPU
Baseado em: https://saturncloud.io/blog/how-to-check-whether-your-code-is-running-on-the-gpu-or-cpu/

Este script testa e compara performance GPU vs CPU com indicadores visuais claros.
Compatível com NVIDIA RTX A4000 + Ubuntu 24.04 + CUDA 12.9
"""

import time
import sys
import numpy as np
from typing import Tuple, Optional

# Cores para output
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title: str):
<<<<<<< HEAD
    """Imprime cabeçalho formatado"""
    print(f"\n{Colors.BOLD}{Colors.WHITE}{title}{Colors.END}")
    print("=" * len(title))

def print_status(device: str, status: str, details: str = ""):
    """Imprime status com cores"""
    if "GPU" in device:
        icon = "GPU"
    elif "CPU" in device:
        icon = "CPU"
    else:
        icon = "SYS"
    
    color = Colors.GREEN if "OK" in status or "FUNCIONANDO" in status or "DETECTADA" in status or "DISPONÍVEL" in status else Colors.RED
    print(f"  {icon} {device}: {color}{status}{Colors.END} {details}")

def test_gpu_availability():
    """Testa disponibilidade e capacidade da GPU"""
    print_header("TESTE DE GPU - CAPACIDADE E DISPONIBILIDADE")
    
    # Testar CuPy
    try:
        import cupy as cp
        print_status("CuPy", "IMPORTADO", "Biblioteca GPU disponível")
        
        # Teste básico
=======
    """Imprime cabeçalho estilizado"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.WHITE}🚀 {title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")

def print_status(device: str, status: str, details: str = ""):
    """Imprime status do dispositivo"""
    if "GPU" in device:
        icon = "⚡"
        color = Colors.GREEN
    else:
        icon = "🖥️"
        color = Colors.BLUE
    
    print(f"{color}{icon} {device}: {status}{Colors.END}")
    if details:
        print(f"   {Colors.WHITE}└─ {details}{Colors.END}")

def test_gpu_availability():
    """Testa disponibilidade e capacidades da GPU"""
    print_header("DIAGNÓSTICO DE GPU")
    
    try:
        import cupy as cp
        print_status("CuPy", "✅ IMPORTADO", "Biblioteca GPU disponível")
        
        # Teste básico de criação de array
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
        try:
            start_time = time.time()
            test_array = cp.array([1, 2, 3, 4, 5])
            result = cp.sum(test_array)
            end_time = time.time()
<<<<<<< HEAD
            print_status("GPU", "FUNCIONANDO", f"Teste básico OK (resultado: {result}, tempo: {end_time-start_time:.4f}s)")
=======
            
            print_status("GPU", "✅ FUNCIONANDO", f"Teste básico OK (resultado: {result}, tempo: {end_time-start_time:.4f}s)")
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
            
            # Informações do dispositivo
            try:
                device_id = cp.cuda.device.get_device_id()
                device_props = cp.cuda.runtime.getDeviceProperties(device_id)
                device_name = device_props['name'].decode()
                
                memory_info = cp.cuda.runtime.memGetInfo()
<<<<<<< HEAD
                free_mem = memory_info[0] / 1024**3  # GB
                total_mem = memory_info[1] / 1024**3  # GB
                
                print_status("GPU Info", "DETECTADA", f"{device_name}")
                print_status("GPU Memória", "DISPONÍVEL", f"{free_mem:.1f}GB livre de {total_mem:.1f}GB total")
                
                return True, cp
                
            except Exception as e:
                print_status("GPU Info", "ERRO", f"Não foi possível obter informações: {str(e)[:50]}...")
                return True, cp
                
        except Exception as e:
            print_status("GPU", "FALHA", f"Erro no teste básico: {str(e)[:50]}...")
            return False, None
            
    except ImportError:
        print_status("CuPy", "NÃO INSTALADO", "pip install cupy-cuda12x")
        return False, None

def performance_test_cpu(size: int = 5000) -> Tuple[float, float]:
    """Teste de performance na CPU"""
    print(f"\n{Colors.BLUE}TESTE CPU - Matriz {size}x{size}{Colors.END}")
    
    # Operações com NumPy (CPU)
    start_time = time.time()
    
    # Criar matrizes
=======
                total_mem = memory_info[1] / 1024**3
                free_mem = memory_info[0] / 1024**3
                
                print_status("GPU Info", "✅ DETECTADA", f"{device_name}")
                print_status("GPU Memória", "✅ DISPONÍVEL", f"{free_mem:.1f}GB livre de {total_mem:.1f}GB total")
                
                return True, cp, device_name
                
            except Exception as e:
                print_status("GPU Info", "❌ ERRO", f"Não foi possível obter informações: {str(e)[:50]}...")
                return False, None, "GPU genérica"
                
        except Exception as e:
            print_status("GPU", "❌ FALHA", f"Erro no teste básico: {str(e)[:50]}...")
            return False, None, "N/A"
            
    except ImportError:
        print_status("CuPy", "❌ NÃO INSTALADO", "pip install cupy-cuda12x")
        return False, None, "N/A"

def performance_test_cpu(size: int = 5000) -> Tuple[float, float]:
    """Teste de performance CPU"""
    print(f"\n{Colors.BLUE}🖥️  TESTE CPU - Matriz {size}x{size}{Colors.END}")
    
    start_time = time.time()
    
    # Operações NumPy (CPU)
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
    a = np.random.random((size, size)).astype(np.float32)
    b = np.random.random((size, size)).astype(np.float32)
    
    # Multiplicação de matrizes
    c = np.dot(a, b)
    
    # Operações estatísticas
    mean_val = np.mean(c)
    std_val = np.std(c)
<<<<<<< HEAD
    
    cpu_time = time.time() - start_time
    
    print(f"   Resultado: média={mean_val:.4f}, std={std_val:.4f}")
    print(f"   Tempo CPU: {Colors.BOLD}{cpu_time:.4f} segundos{Colors.END}")
=======
    sum_val = np.sum(c)
    
    end_time = time.time()
    cpu_time = end_time - start_time
    
    print(f"   📊 Resultado: média={mean_val:.4f}, std={std_val:.4f}")
    print(f"   ⏱️  Tempo CPU: {Colors.BOLD}{cpu_time:.4f} segundos{Colors.END}")
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
    
    return cpu_time, mean_val

def performance_test_gpu(cp, size: int = 5000) -> Tuple[float, float]:
<<<<<<< HEAD
    """Teste de performance na GPU"""
    print(f"\n{Colors.GREEN}TESTE GPU - Matriz {size}x{size}{Colors.END}")
    
    try:
        # Operações com CuPy (GPU)
        start_time = time.time()
        
        # Criar matrizes na GPU
        a = cp.random.random((size, size), dtype=cp.float32)
        b = cp.random.random((size, size), dtype=cp.float32)
        
        # Multiplicação de matrizes
        c = cp.dot(a, b)
        
        # Operações estatísticas
        mean_val = float(cp.mean(c))
        std_val = float(cp.std(c))
        
        # Sincronizar para medir tempo correto
        cp.cuda.Stream.null.synchronize()
        
        gpu_time = time.time() - start_time
        
        print(f"   Resultado: média={mean_val:.4f}, std={std_val:.4f}")
        print(f"   Tempo GPU: {Colors.BOLD}{gpu_time:.4f} segundos{Colors.END}")
        
        return gpu_time, mean_val
        
    except Exception as e:
        print_status("GPU Performance", "ERRO", str(e)[:50])
        return float('inf'), 0.0

def compare_performance(cpu_time: float, gpu_time: float):
    """Compara performance entre CPU e GPU"""
    print(f"\n{Colors.CYAN}COMPARAÇÃO DE PERFORMANCE{Colors.END}")
    print("-" * 30)
    
    print(f"{Colors.BLUE}CPU: {cpu_time:.4f} segundos{Colors.END}")
    print(f"{Colors.GREEN}GPU: {gpu_time:.4f} segundos{Colors.END}")
    
    if gpu_time > 0 and cpu_time > 0:
        speedup = cpu_time / gpu_time
        if speedup > 1:
            print(f"\n{Colors.BOLD}{Colors.GREEN}GPU é {speedup:.2f}x MAIS RÁPIDO!{Colors.END}")
        else:
            print(f"\n{Colors.YELLOW}CPU foi {1/speedup:.2f}x mais rápido (matriz pequena demais ou overhead GPU){Colors.END}")
    else:
        print(f"\n{Colors.RED}Não foi possível calcular speedup{Colors.END}")

def test_cupy_features(cp, device_name: str):
    """Testa funcionalidades específicas do CuPy"""
    print_header(f"TESTE DE FUNCIONALIDADES - {device_name}")
    
    tests = [
        ("Array Creation", lambda: cp.array([1, 2, 3, 4, 5])),
        ("Random Generation", lambda: cp.random.random((100, 100))),
        ("Mathematical Operations", lambda: cp.sin(cp.linspace(0, cp.pi, 1000))),
        ("Matrix Operations", lambda: cp.linalg.inv(cp.random.random((50, 50)))),
        ("FFT Operations", lambda: cp.fft.fft(cp.random.random(1024))),
=======
    """Teste de performance GPU"""
    print(f"\n{Colors.GREEN}⚡ TESTE GPU - Matriz {size}x{size}{Colors.END}")
    
    start_time = time.time()
    
    # Operações CuPy (GPU)
    a_gpu = cp.random.random((size, size), dtype=cp.float32)
    b_gpu = cp.random.random((size, size), dtype=cp.float32)
    
    # Multiplicação de matrizes
    c_gpu = cp.dot(a_gpu, b_gpu)
    
    # Operações estatísticas
    mean_val = float(cp.mean(c_gpu))
    std_val = float(cp.std(c_gpu))
    sum_val = float(cp.sum(c_gpu))
    
    # Sincronizar para garantir que todas as operações GPU terminaram
    cp.cuda.Stream.null.synchronize()
    
    end_time = time.time()
    gpu_time = end_time - start_time
    
    print(f"   📊 Resultado: média={mean_val:.4f}, std={std_val:.4f}")
    print(f"   ⚡ Tempo GPU: {Colors.BOLD}{gpu_time:.4f} segundos{Colors.END}")
    
    return gpu_time, mean_val

def compare_performance(cpu_time: float, gpu_time: float):
    """Compara performance entre CPU e GPU"""
    print_header("COMPARAÇÃO DE PERFORMANCE")
    
    print(f"{Colors.BLUE}🖥️  CPU: {cpu_time:.4f} segundos{Colors.END}")
    print(f"{Colors.GREEN}⚡ GPU: {gpu_time:.4f} segundos{Colors.END}")
    
    if gpu_time > 0:
        speedup = cpu_time / gpu_time
        if speedup > 1:
            print(f"\n{Colors.BOLD}{Colors.GREEN}🚀 GPU é {speedup:.2f}x MAIS RÁPIDO!{Colors.END}")
        else:
            print(f"\n{Colors.YELLOW}⚠️  CPU foi {1/speedup:.2f}x mais rápido (matriz pequena demais ou overhead GPU){Colors.END}")
    else:
        print(f"\n{Colors.RED}❌ Não foi possível calcular speedup{Colors.END}")

def test_cupy_features(cp, device_name: str):
    """Testa recursos específicos do CuPy"""
    print_header(f"TESTE DE RECURSOS - {device_name}")
    
    tests = [
        ("Criação de Array", lambda: cp.array([1, 2, 3, 4, 5])),
        ("Operações Matemáticas", lambda: cp.sqrt(cp.array([1, 4, 9, 16, 25]))),
        ("Random Numbers", lambda: cp.random.random(1000)),
        ("FFT", lambda: cp.fft.fft(cp.random.random(1024))),
        ("Linear Algebra", lambda: cp.linalg.inv(cp.eye(100))),
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
    ]
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
<<<<<<< HEAD
            print_status(test_name, "OK", f"{end_time-start_time:.4f}s")
        except Exception as e:
            print_status(test_name, "ERRO", str(e)[:50])

def memory_transfer_test(cp):
    """Testa transferência de memória CPU-GPU"""
    print_header("TESTE DE TRANSFERÊNCIA DE MEMÓRIA")
    
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        print(f"\nTamanho da matriz: {size}x{size}")
        
        # Criar dados na CPU
        cpu_data = np.random.random((size, size)).astype(np.float32)
        
        # CPU -> GPU
        start_time = time.time()
        gpu_data = cp.asarray(cpu_data)
        transfer_to_gpu_time = time.time() - start_time
        
        # GPU -> CPU
        start_time = time.time()
        result_cpu = cp.asnumpy(gpu_data)
        transfer_to_cpu_time = time.time() - start_time
        
        print(f"  CPU->GPU: {transfer_to_gpu_time:.4f}s")
        print(f"  GPU->CPU: {transfer_to_cpu_time:.4f}s")
        print(f"  Total: {transfer_to_gpu_time + transfer_to_cpu_time:.4f}s")

def main():
    """Função principal"""
    print_header("TESTE COMPLETO DE GPU - ALZHEIMER PIPELINE")
    print(f"{Colors.CYAN}Verificação completa de capacidade GPU para processamento neuroimagem{Colors.END}")
    
    # 1. Teste de disponibilidade da GPU
    gpu_available, cp = test_gpu_availability()
    
    if not gpu_available:
        print(f"\n{Colors.RED}GPU não disponível. Pipeline será executado na CPU.{Colors.END}")
        print(f"{Colors.YELLOW}Para melhor performance, considere:{Colors.END}")
        print("  - Instalar drivers NVIDIA")
        print("  - Instalar CUDA Toolkit")
        print("  - Instalar CuPy: pip install cupy-cuda12x")
        return
    
    # 2. Obter informações do dispositivo
    try:
        device_props = cp.cuda.runtime.getDeviceProperties(0)
        device_name = device_props['name'].decode()
    except:
        device_name = "GPU Desconhecida"
    
    # 3. Testes de performance
    print_header("TESTE DE PERFORMANCE - CPU vs GPU")
    
    # Teste diferentes tamanhos
    sizes = [1000, 3000, 5000]
    
    for size in sizes:
        print(f"\n{Colors.BOLD}MATRIZ {size}x{size}{Colors.END}")
        print("-" * 20)
        
        cpu_time, cpu_result = performance_test_cpu(size)
        gpu_time, gpu_result = performance_test_gpu(cp, size)
        
        compare_performance(cpu_time, gpu_time)
    
    # 4. Teste de funcionalidades
    test_cupy_features(cp, device_name)
    
    # 5. Teste de transferência de memória
    memory_transfer_test(cp)
    
    # 6. Resumo final
    print_header("RESUMO E RECOMENDAÇÕES")
    
    print(f"{Colors.GREEN}GPU detectada e funcional: {device_name}{Colors.END}")
    print(f"{Colors.GREEN}CuPy instalado e operacional{Colors.END}")
    print(f"{Colors.GREEN}Todas as operações básicas funcionando{Colors.END}")
    
    print(f"\n{Colors.CYAN}RECOMENDAÇÕES PARA O PIPELINE:{Colors.END}")
    print("  - Use force_gpu=True nos scripts de análise")
    print("  - Configure batch_size >= 64 para melhor performance")
    print("  - Use mixed precision para economizar memória")
    print("  - Monitore uso de memória GPU durante execução")
    
    print(f"\n{Colors.BOLD}PRÓXIMOS PASSOS:{Colors.END}")
    print("  1. Execute: python3 alzheimer_ai_pipeline.py")
    print("  2. Execute: python3 analisar_hipocampo_gpu_otimizado.py")
    print("  3. Monitor TensorBoard: tensorboard --logdir=logs")
=======
            
            print_status(test_name, "✅ OK", f"{end_time-start_time:.4f}s")
        except Exception as e:
            print_status(test_name, "❌ ERRO", str(e)[:50])

def memory_transfer_test(cp):
    """Testa transferência de memória GPU ↔ CPU"""
    print_header("TESTE DE TRANSFERÊNCIA DE MEMÓRIA")
    
    size = (2000, 2000)
    
    # CPU → GPU
    print(f"{Colors.CYAN}📤 Transferindo dados CPU → GPU...{Colors.END}")
    start_time = time.time()
    cpu_array = np.random.random(size).astype(np.float32)
    gpu_array = cp.asarray(cpu_array)
    cp.cuda.Stream.null.synchronize()
    transfer_to_gpu_time = time.time() - start_time
    print(f"   ⏱️  Tempo: {transfer_to_gpu_time:.4f} segundos")
    
    # GPU → CPU  
    print(f"{Colors.CYAN}📥 Transferindo dados GPU → CPU...{Colors.END}")
    start_time = time.time()
    result_cpu = cp.asnumpy(gpu_array)
    transfer_to_cpu_time = time.time() - start_time
    print(f"   ⏱️  Tempo: {transfer_to_cpu_time:.4f} segundos")
    
    # Verificar integridade
    if np.allclose(cpu_array, result_cpu):
        print_status("Integridade", "✅ OK", "Dados transferidos corretamente")
    else:
        print_status("Integridade", "❌ ERRO", "Dados corrompidos na transferência")

def main():
    """Função principal"""
    print(f"{Colors.BOLD}{Colors.PURPLE}")
    print("🧪 TESTE COMPLETO DE GPU vs CPU")
    print("=" * 60)
    print("NVIDIA RTX A4000 | Ubuntu 24.04 | CUDA 12.9")
    print("=" * 60)
    print(f"{Colors.END}")
    
    # Teste de disponibilidade da GPU
    gpu_available, cp, device_name = test_gpu_availability()
    
    if gpu_available:
        print(f"\n{Colors.GREEN}✅ GPU DISPONÍVEL E FUNCIONANDO!{Colors.END}")
        
        # Testes de recursos
        test_cupy_features(cp, device_name)
        
        # Teste de transferência de memória
        memory_transfer_test(cp)
        
        # Teste de performance
        print_header("BENCHMARK DE PERFORMANCE")
        
        sizes = [1000, 3000, 5000]
        for size in sizes:
            print(f"\n{Colors.BOLD}📏 Testando matrizes {size}x{size}...{Colors.END}")
            
            cpu_time, cpu_result = performance_test_cpu(size)
            gpu_time, gpu_result = performance_test_gpu(cp, size)
            
            compare_performance(cpu_time, gpu_time)
            
        print_header("RESUMO FINAL")
        print(f"{Colors.BOLD}{Colors.GREEN}✅ GPU está funcionando perfeitamente!{Colors.END}")
        print(f"{Colors.WHITE}📋 Para usar GPU em seus scripts Python:{Colors.END}")
        print(f"{Colors.CYAN}   import cupy as cp{Colors.END}")
        print(f"{Colors.CYAN}   array_gpu = cp.array([1, 2, 3])  # Array na GPU{Colors.END}")
        print(f"{Colors.CYAN}   result = cp.sum(array_gpu)       # Processamento na GPU{Colors.END}")
        
    else:
        print(f"\n{Colors.RED}❌ GPU NÃO ESTÁ FUNCIONANDO{Colors.END}")
        print(f"{Colors.YELLOW}💡 Soluções possíveis:{Colors.END}")
        print(f"   1. Reiniciar o sistema para carregar novos drivers")
        print(f"   2. Verificar se CUDA 12.9 está no PATH")
        print(f"   3. Reinstalar drivers: sudo apt install nvidia-open")
        
        # Teste só CPU
        print_header("TESTE APENAS CPU")
        cpu_time, cpu_result = performance_test_cpu(3000)
        print(f"{Colors.BLUE}🖥️  CPU funcionando normalmente{Colors.END}")
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)

if __name__ == "__main__":
    main() 
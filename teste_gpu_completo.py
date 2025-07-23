#!/usr/bin/env python3
"""
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
        try:
            start_time = time.time()
            test_array = cp.array([1, 2, 3, 4, 5])
            result = cp.sum(test_array)
            end_time = time.time()
            
            print_status("GPU", "✅ FUNCIONANDO", f"Teste básico OK (resultado: {result}, tempo: {end_time-start_time:.4f}s)")
            
            # Informações do dispositivo
            try:
                device_id = cp.cuda.device.get_device_id()
                device_props = cp.cuda.runtime.getDeviceProperties(device_id)
                device_name = device_props['name'].decode()
                
                memory_info = cp.cuda.runtime.memGetInfo()
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
    a = np.random.random((size, size)).astype(np.float32)
    b = np.random.random((size, size)).astype(np.float32)
    
    # Multiplicação de matrizes
    c = np.dot(a, b)
    
    # Operações estatísticas
    mean_val = np.mean(c)
    std_val = np.std(c)
    sum_val = np.sum(c)
    
    end_time = time.time()
    cpu_time = end_time - start_time
    
    print(f"   📊 Resultado: média={mean_val:.4f}, std={std_val:.4f}")
    print(f"   ⏱️  Tempo CPU: {Colors.BOLD}{cpu_time:.4f} segundos{Colors.END}")
    
    return cpu_time, mean_val

def performance_test_gpu(cp, size: int = 5000) -> Tuple[float, float]:
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
    ]
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
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

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Script de Teste para Configurações GPU/CPU do Pipeline Alzheimer
Útil para verificar configurações antes de executar o pipeline completo
"""

import tensorflow as tf
import numpy as np
import time

def test_gpu_configuration():
    """Testa configurações de GPU e CPU"""
    print("🧪 TESTE DE CONFIGURAÇÃO GPU/CPU")
    print("=" * 40)
    
    # Informações básicas
    print(f"📦 TensorFlow: {tf.__version__}")
    print(f"🐍 NumPy: {np.__version__}")
    
    # Verificar GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n🎯 GPUs físicas detectadas: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
            try:
                # Tentar configurar memória
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"   ✅ Memória configurada para {gpu.name}")
            except Exception as e:
                print(f"   ❌ Erro na configuração: {e}")
    else:
        print("   ⚠️  Nenhuma GPU detectada - usando CPU")
    
    # Verificar CUDA
    print(f"\n🔥 CUDA build: {tf.test.is_built_with_cuda()}")
    print(f"🔥 GPU disponível: {tf.test.is_gpu_available()}")
    
    # Teste de performance simples
    print("\n⚡ TESTE DE PERFORMANCE:")
    print("-" * 25)
    
    # Teste CPU
    with tf.device('/CPU:0'):
        start_time = time.time()
        x = tf.random.normal([1000, 1000])
        y = tf.matmul(x, x)
        result_cpu = tf.reduce_sum(y)
        cpu_time = time.time() - start_time
        print(f"🖥️  CPU: {cpu_time:.4f}s")
    
    # Teste GPU (se disponível)
    if gpus and tf.test.is_gpu_available():
        try:
            with tf.device('/GPU:0'):
                start_time = time.time()
                x = tf.random.normal([1000, 1000])
                y = tf.matmul(x, x)
                result_gpu = tf.reduce_sum(y)
                gpu_time = time.time() - start_time
                print(f"🚀 GPU: {gpu_time:.4f}s")
                
                if cpu_time > 0:
                    speedup = cpu_time / gpu_time
                    print(f"⚡ Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"❌ Erro no teste GPU: {e}")
    
    # Recomendações
    print("\n💡 RECOMENDAÇÕES:")
    print("-" * 15)
    
    if not gpus:
        print("🔧 Para configurar GPU:")
        print("   1. Instale drivers NVIDIA adequados")
        print("   2. Instale CUDA e cuDNN")
        print("   3. Instale tensorflow[gpu]")
        print("   4. Reinicie o sistema")
    elif not tf.test.is_gpu_available():
        print("🔧 GPU detectada mas não utilizável:")
        print("   1. Verifique drivers NVIDIA: nvidia-smi")
        print("   2. Verifique CUDA: nvcc --version")
        print("   3. Verifique compatibilidade TensorFlow-CUDA")
        print("   4. Reinicialize os drivers: sudo service nvidia-restart")
    else:
        print("✅ GPU configurada corretamente!")
        print("🚀 Pipeline otimizado para aceleração GPU")
    
    # Configurações recomendadas para CPU
    print("\n🖥️  OTIMIZAÇÕES CPU (caso não tenha GPU):")
    print("   - Use batch_size menor (16-32)")
    print("   - Reduza número de camadas do modelo")
    print("   - Use mixed precision pode ajudar mesmo em CPU")
    print("   - Configure threads: export OMP_NUM_THREADS=4")

if __name__ == "__main__":
    test_gpu_configuration()
    print("\n🚀 Para executar o pipeline completo:")
    print("   python3 alzheimer_ai_pipeline.py") 
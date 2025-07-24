#!/usr/bin/env python3
"""
Teste de Configuração GPU para TensorFlow/Keras
"""

import tensorflow as tf
import numpy as np
import time

def test_gpu_configuration():
    """Testa configuração da GPU com TensorFlow"""
    
    print("TESTE DE CONFIGURACAO GPU")
    print("=" * 30)
    
    # Versão do TensorFlow
    print(f"TensorFlow: {tf.__version__}")
    
    # GPUs físicas detectadas
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPUs físicas detectadas: {len(gpus)}")
    
    # Configurar GPUs
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"   Memória configurada para {gpu.name}")
        except RuntimeError as e:
            print(f"   Erro na configuração: {e}")
    else:
        print("   Nenhuma GPU detectada - usando CPU")
    
    # Verificar CUDA
    print(f"\nCUDA build: {tf.test.is_built_with_cuda()}")
    print(f"GPU disponível: {tf.test.is_gpu_available()}")
    
    # Teste de performance
    print("\nTESTE DE PERFORMANCE:")
    print("-" * 20)
    
    # Criar dados de teste
    with tf.device('/CPU:0'):
        start_time = time.time()
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        cpu_time = time.time() - start_time
        print(f"CPU: {cpu_time:.4f}s")
    
    # Teste GPU (se disponível)
    if tf.config.list_physical_devices('GPU'):
        try:
            with tf.device('/GPU:0'):
                start_time = time.time()
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
                gpu_time = time.time() - start_time
                print(f"GPU: {gpu_time:.4f}s")
                
                # Calcular speedup
                if cpu_time > 0:
                    speedup = cpu_time / gpu_time
                    print(f"Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"Erro no teste GPU: {e}")
    
    # Recomendações
    print("\nRECOMENDAÇÕES:")
    print("-" * 15)
    
    if not gpus:
        print("Para configurar GPU:")
        print("  1. Instale CUDA Toolkit 11.2+")
        print("  2. Instale cuDNN 8.1+")
        print("  3. pip install tensorflow[gpu]")
        print("  4. Reinicie o sistema")
    elif not tf.test.is_gpu_available():
        print("GPU detectada mas não utilizável:")
        print("  1. Verifique drivers NVIDIA")
        print("  2. Reinstale tensorflow: pip install tensorflow[gpu]")
        print("  3. Verifique compatibilidade CUDA/cuDNN")
        print("  4. Reinicie o sistema")
    else:
        print("GPU configurada corretamente!")
        print("Pipeline otimizado para aceleração GPU")
    
    # Instruções finais
    print("\nOTIMIZAÇÕES CPU (caso não tenha GPU):")
    print("  - export TF_NUM_INTEROP_THREADS=0")
    print("  - export TF_NUM_INTRAOP_THREADS=0")
    print("  - Usar batch_size menor (16-32)")
    print("  - Menos épocas de treinamento")
    print("  - Menos camadas no modelo")
    
    print("\nPara executar o pipeline completo:")
    print("  python3 alzheimer_ai_pipeline.py")

if __name__ == "__main__":
    test_gpu_configuration() 
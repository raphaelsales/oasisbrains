#!/usr/bin/env python3
"""
ANÁLISE DE HIPOCAMPO - OTIMIZADA PARA GPU

Análise avançada do hipocampo com otimizações específicas para GPU
Compara performance CPU vs GPU em operações de neuroimagem
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ===============================
# MÓDULOS GPU (CuPy) - IMPORTAÇÃO CONDICIONAL
# ===============================

# Classe para cores no terminal
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

class GPUMonitor:
    """Monitor de GPU otimizado para análise de hipocampo"""
    
    def __init__(self):
        self.gpu_available = False
        self.cupy_module = None
        self.device_name = "CPU"
        self.gpu_memory_total = 0
        self.gpu_memory_free = 0
        self.processing_device = "CPU"
        self._test_gpu_capability()
    
    def test_gpu_capability(self):
        """Verifica e configura capacidade da GPU"""
        print(f"{Colors.CYAN}SISTEMA DE MONITORAMENTO GPU/CPU{Colors.END}")
        print("-" * 40)
        
        try:
            import cupy as cp
            self.cupy_module = cp
            print(f"Teste CuPy importado: versão {cp.__version__}")
            
            # Teste básico
            test_array = cp.array([1, 2, 3])
            result = cp.sum(test_array)
            print(f"Teste básico GPU: OK (resultado: {result})")
            
            self.gpu_available = True
            self.processing_device = "GPU"
            
            try:
                # Informações da GPU
                mempool = cp.get_default_memory_pool()
                self.device_name = cp.cuda.Device().compute_capability
                
                # Informações de memória (em GB)
                self.gpu_memory_total = cp.cuda.Device().mem_info[1] / (1024**3)
                self.gpu_memory_free = cp.cuda.Device().mem_info[0] / (1024**3)
                
                print(f"GPU detectada: {self.device_name}")
                print(f"Memória: {self.gpu_memory_free:.1f}GB livre de {self.gpu_memory_total:.1f}GB")
                
            except Exception as e:
                print(f"Info GPU indisponível: {str(e)[:50]}...")
                
        except Exception as e:
            print(f"Teste GPU falhou: {str(e)[:50]}...")
            self.gpu_available = False
            
        except ImportError:
            print("CuPy não instalado")
            self.gpu_available = False
    
    def _test_gpu_capability(self):
        """Método interno para testar GPU"""
        self.test_gpu_capability()
    
    def show_processing_status(self, operation: str, using_gpu: bool):
        """Mostra status do processamento"""
        if using_gpu and self.gpu_available:
            print(f"{Colors.GREEN}{operation} - Executando na GPU{Colors.END}")
        else:
            print(f"{Colors.BLUE}{operation} - Executando na CPU{Colors.END}")

def load_image(file_path: str):
    """Carrega imagem neurológica com verificação de erro"""
    try:
        img = nib.load(file_path)
        return img.get_fdata(), img.header, img.affine
    except Exception as e:
        print(f"Erro ao carregar {file_path}: {e}")
        return None, None, None

def extract_hippocampus_mask(seg_data: Any, left_label: int = 17, right_label: int = 53) -> Any:
    """Extrai máscara do hipocampo bilateral"""
    return (seg_data == left_label) | (seg_data == right_label)

def regional_voxel_analysis(t1_data: Any, hippo_mask: Any) -> Dict[str, float]:
    """Análise voxel-wise da região do hipocampo"""
    hippo_voxels = t1_data[hippo_mask]
    
    if len(hippo_voxels) == 0:
        return {'volume': 0, 'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
    
    return {
        'volume': len(hippo_voxels),
        'mean': float(np.mean(hippo_voxels)),
        'median': float(np.median(hippo_voxels)),
        'std': float(np.std(hippo_voxels)),
        'min': float(np.min(hippo_voxels)),
        'max': float(np.max(hippo_voxels))
    }

def bayesian_classification_cpu(hippo_voxels: np.ndarray, mu: float, sigma: float, prior: float = 0.9) -> Tuple[np.ndarray, float]:
    """Classificação Bayesiana dos voxels (versão CPU)"""
    start_time = time.time()
    
    if len(hippo_voxels) == 0:
        return np.array([]), 0.0
    
    # Log-likelihood
    log_likelihood = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((hippo_voxels - mu) / sigma)**2
    
    # Posterior probability
    log_prior = np.log(prior)
    log_posterior = log_likelihood + log_prior
    
    # Convert to probabilities
    posterior_prob = np.exp(log_posterior - np.max(log_posterior))
    
    processing_time = time.time() - start_time
    return posterior_prob, processing_time

def bayesian_classification_gpu(hippo_voxels: np.ndarray, mu: float, sigma: float, prior: float = 0.9) -> Tuple[np.ndarray, float]:
    """Classificação Bayesiana dos voxels (versão GPU otimizada)"""
    monitor = GPUMonitor()
    
    if not monitor.gpu_available:
        print(f"{Colors.YELLOW}GPU não disponível, usando CPU...{Colors.END}")
        return bayesian_classification_cpu(hippo_voxels, mu, sigma, prior)
    
    try:
        cp = monitor.cupy_module
        start_time = time.time()
        
        if len(hippo_voxels) == 0:
            return np.array([]), 0.0
        
        # Transferir dados para GPU
        gpu_voxels = cp.asarray(hippo_voxels)
        
        # Operações GPU
        log_likelihood = -0.5 * cp.log(2 * cp.pi * sigma**2) - 0.5 * ((gpu_voxels - mu) / sigma)**2
        log_prior = cp.log(prior)
        log_posterior = log_likelihood + log_prior
        
        # Normalização
        posterior_prob = cp.exp(log_posterior - cp.max(log_posterior))
        
        # Transferir resultado de volta para CPU
        result = cp.asnumpy(posterior_prob)
        
        processing_time = time.time() - start_time
        return result, processing_time
        
    except Exception as e:
        print(f"{Colors.RED}Erro na GPU: {str(e)[:50]}...{Colors.END}")
        print(f"{Colors.BLUE}Fallback para CPU...{Colors.END}")
        return bayesian_classification_cpu(hippo_voxels, mu, sigma, prior)

def analyze_subject_hippocampus(subject_path: str, force_gpu: bool = False, show_timing: bool = False) -> Dict[str, Any]:
    """
    Análise completa do hipocampo para um sujeito específico
    
    Args:
        subject_path: Caminho para o diretório do sujeito
        force_gpu: Se True, força uso da GPU (quando disponível)
        show_timing: Se True, mostra tempos de processamento
    """
    
    monitor = GPUMonitor()
    subject_id = os.path.basename(subject_path)
    
    # Arquivos necessários
    seg_file = os.path.join(subject_path, 'mri', 'aparc+aseg.mgz')
    t1_file = os.path.join(subject_path, 'mri', 'T1.mgz')
    
    results = {
        'subject_id': subject_id,
        'success': False,
        'error': None,
        'processing_device': 'CPU',
        'processing_time': 0.0
    }
    
    if not os.path.exists(seg_file) or not os.path.exists(t1_file):
        results['error'] = 'Arquivos necessários não encontrados'
        return results
    
    try:
        total_start = time.time()
        
        # Carregar dados
        t1_data, t1_header, t1_affine = load_image(t1_file)
        seg_data, seg_header, seg_affine = load_image(seg_file)
        
        if t1_data is None or seg_data is None:
            results['error'] = 'Falha ao carregar imagens'
            return results
        
        # Extrair máscara do hipocampo
        hippo_mask = extract_hippocampus_mask(seg_data)
        
        # Análise básica
        basic_stats = regional_voxel_analysis(t1_data, hippo_mask)
        results.update(basic_stats)
        
        # Análise Bayesiana (escolher método baseado na disponibilidade e preferência)
        hippo_voxels = t1_data[hippo_mask]
        
        # Parâmetros para classificação
        mu = np.mean(hippo_voxels) if len(hippo_voxels) > 0 else 100
        sigma = np.std(hippo_voxels) if len(hippo_voxels) > 0 else 20
        
        # Escolher processamento
        use_gpu = force_gpu and monitor.gpu_available
        device = "GPU" if use_gpu else "CPU"
        
        if show_timing:
            print(f"   Tempo {device}: {processing_time:.4f} segundos")
        
        monitor.show_processing_status("Classificação Bayesiana", use_gpu)
        
        if use_gpu:
            posterior_prob, processing_time = bayesian_classification_gpu(hippo_voxels, mu, sigma)
            results['processing_device'] = 'GPU'
        else:
            posterior_prob, processing_time = bayesian_classification_cpu(hippo_voxels, mu, sigma)
            results['processing_device'] = 'CPU'
        
        # Estatísticas da classificação
        if len(posterior_prob) > 0:
            results['posterior_mean'] = float(np.mean(posterior_prob))
            results['posterior_std'] = float(np.std(posterior_prob))
            results['high_confidence_voxels'] = int(np.sum(posterior_prob > 0.8))
            results['total_voxels'] = len(posterior_prob)
        
        # Volume do hipocampo (considerando tamanho do voxel)
        voxel_volume = 1.0  # Assumir 1mm³ por voxel (pode ser refinado)
        results['hippocampus_volume_mm3'] = results['volume'] * voxel_volume
        
        results['processing_time'] = processing_time
        results['total_time'] = time.time() - total_start
        results['success'] = True
        
    except Exception as e:
        results['error'] = str(e)
        print(f"Erro ao processar {subject_id}: {e}")
    
    return results

def analyze_all_subjects(data_dir: str, max_subjects: int = None, force_gpu: bool = False, show_timing: bool = False) -> pd.DataFrame:
    """
    Analisa hipocampo para todos os sujeitos disponíveis
    
    Args:
        data_dir: Diretório com dados dos sujeitos
        max_subjects: Máximo de sujeitos a processar (None = todos)
        force_gpu: Se True, força uso da GPU quando disponível
        show_timing: Se True, mostra informações de timing
    """
    import glob
    
    print(f"Analisando hipocampo em: {data_dir}")
    
    # Encontrar todos os sujeitos
    subject_pattern = os.path.join(data_dir, "OAS1_*_MR1")
    subject_dirs = glob.glob(subject_pattern)
    
    if max_subjects:
        subject_dirs = subject_dirs[:max_subjects]
    
    print(f"Total de sujeitos encontrados: {len(subject_dirs)}")
    
    results = []
    successes = 0
    failures = 0
    
    # Estatísticas de performance
    gpu_times = []
    cpu_times = []
    gpu_operations = 0
    cpu_operations = 0
    
    for i, subject_dir in enumerate(subject_dirs, 1):
        subject_id = os.path.basename(subject_dir)
        print(f"Processando {subject_id} ({i}/{len(subject_dirs)})")
        
        try:
            result = analyze_subject_hippocampus(subject_dir, force_gpu, show_timing)
            results.append(result)
            
            if result['success']:
                successes += 1
                
                # Coletar estatísticas de performance
                if result['processing_device'] == 'GPU':
                    gpu_times.append(result['processing_time'])
                    gpu_operations += 1
                else:
                    cpu_times.append(result['processing_time'])
                    cpu_operations += 1
            else:
                failures += 1
                
        except Exception as e:
            print(f"Erro ao processar {subject_id}: {e}")
            failures += 1
    
    # Resumo de performance
    print(f"\n{Colors.CYAN}RESUMO DE PERFORMANCE:{Colors.END}")
    print(f"Sucessos: {successes}")
    print(f"Falhas: {failures}")
    
    if gpu_times:
        avg_gpu_time = np.mean(gpu_times)
        print(f"Operações GPU: {gpu_operations} (tempo médio: {avg_gpu_time:.4f}s)")
    
    if cpu_times:
        avg_cpu_time = np.mean(cpu_times)
        print(f"Operações CPU: {cpu_operations} (tempo médio: {avg_cpu_time:.4f}s)")
    
    if gpu_times and cpu_times:
        speedup = np.mean(cpu_times) / np.mean(gpu_times)
        print(f"Aceleração GPU: {speedup:.2f}x mais rápido")
    
    return pd.DataFrame(results)

def main():
    """Função principal"""
    print("ANÁLISE HIPOCAMPO - OTIMIZADA PARA GPU")
    print("=" * 50)
    
    # Configurações
    data_dir = "/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos"
    max_subjects = 50  # Para teste rápido
    force_gpu = True  # Forçar uso da GPU quando disponível
    show_timing = True  # Mostrar informações de timing
    
    # Verificar diretório
    if not os.path.exists(data_dir):
        print(f"Diretório não encontrado: {data_dir}")
        return
    
    # Monitor inicial
    monitor = GPUMonitor()
    
    print(f"\nConfiguração:")
    print(f"  Diretório: {data_dir}")
    print(f"  Máx. sujeitos: {max_subjects}")
    print(f"  Forçar GPU: {force_gpu}")
    print(f"  GPU disponível: {monitor.gpu_available}")
    print(f"  Dispositivo: {monitor.processing_device}")
    
    print("\nIniciando análise...")
    start_time = time.time()
    
    # Executar análise
    results_df = analyze_all_subjects(
        data_dir=data_dir,
        max_subjects=max_subjects,
        force_gpu=force_gpu,
        show_timing=show_timing
    )
    
    total_time = time.time() - start_time
    
    # Salvar resultados
    output_file = "hippocampus_gpu_analysis_results.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\n{Colors.GREEN}Análise concluída em {total_time:.2f} segundos{Colors.END}")
    print(f"Resultados salvos em: {output_file}")
    print(f"Total de registros: {len(results_df)}")
    
    # Estatísticas básicas
    if len(results_df) > 0:
        successful = results_df['success'].sum()
        total_vol = results_df['volume'].sum()
        gpu_count = (results_df['processing_device'] == 'GPU').sum()
        cpu_count = (results_df['processing_device'] == 'CPU').sum()
        
        print(f"\nESTATÍSTICAS BÁSICAS:")
        print(f"  Sucessos: {successful}/{len(results_df)}")
        print(f"  Volume total processado: {total_vol:,.0f} voxels")
        print(f"  Volume médio por sujeito: {total_vol/len(results_df):,.0f} voxels")
        
        print(f"\nDISTRIBUIÇÃO DE PROCESSAMENTO:")
        print(f"   GPU: {gpu_count} operações")
        print(f"   CPU: {cpu_count} operações")
    else:
        print(f"\n{Colors.RED}Nenhum resultado foi gerado{Colors.END}")

if __name__ == "__main__":
    main() 
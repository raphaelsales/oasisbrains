#!/usr/bin/env python3
"""
<<<<<<< HEAD
ANÁLISE DE HIPOCAMPO - OTIMIZADA PARA GPU

Análise avançada do hipocampo com otimizações específicas para GPU
Compara performance CPU vs GPU em operações de neuroimagem
"""

import os
import sys
=======
🧠 ANÁLISE DE HIPOCAMPO - OTIMIZADA PARA GPU
Compatível com NVIDIA RTX A4000 + Ubuntu 24.04 + CUDA 12.9

Funcionalidades:
1. Sistema de monitoramento GPU/CPU em tempo real
2. Análise direta dos dados FastSurfer processados
3. Comparação de performance GPU vs CPU
4. Indicadores visuais de onde está executando
5. Fallback automático CPU quando GPU falha
"""

import os
import glob
import csv
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
import time
import numpy as np
import pandas as pd
import nibabel as nib
<<<<<<< HEAD
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ===============================
# MÓDULOS GPU (CuPy) - IMPORTAÇÃO CONDICIONAL
# ===============================

# Classe para cores no terminal
=======
from typing import Tuple, Dict, Any, Optional

# Cores para indicadores visuais
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

class GPUMonitor:
<<<<<<< HEAD
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
=======
    """Sistema de monitoramento GPU/CPU"""
    def __init__(self):
        self.gpu_available = False
        self.processing_device = "CPU"
        self.cp_module = None
        self.device_name = "N/A"
        self.gpu_memory_total = 0
        self.gpu_memory_free = 0
        self.test_gpu_capability()
    
    def test_gpu_capability(self):
        """Testa capacidade da GPU de forma detalhada"""
        print(f"{Colors.CYAN}🔍 SISTEMA DE MONITORAMENTO GPU/CPU{Colors.END}")
        print("=" * 50)
        
        try:
            import cupy as cp
            self.cp_module = cp
            print(f"✅ CuPy importado: versão {cp.__version__}")
            
            # Teste básico
            try:
                test_array = cp.array([1, 2, 3, 4, 5])
                result = cp.sum(test_array)
                print(f"✅ Teste básico GPU: OK (resultado: {result})")
                
                # Info do dispositivo
                try:
                    device_id = cp.cuda.device.get_device_id()
                    device_props = cp.cuda.runtime.getDeviceProperties(device_id)
                    self.device_name = device_props['name'].decode()
                    
                    memory_info = cp.cuda.runtime.memGetInfo()
                    self.gpu_memory_total = memory_info[1] / 1024**3
                    self.gpu_memory_free = memory_info[0] / 1024**3
                    
                    print(f"✅ GPU detectada: {self.device_name}")
                    print(f"✅ Memória: {self.gpu_memory_free:.1f}GB livre de {self.gpu_memory_total:.1f}GB")
                    
                    self.gpu_available = True
                    self.processing_device = f"GPU ({self.device_name})"
                    
                except Exception as e:
                    print(f"⚠️ Info GPU indisponível: {str(e)[:50]}...")
                    self.gpu_available = False
                    
            except Exception as e:
                print(f"❌ Teste GPU falhou: {str(e)[:50]}...")
                self.gpu_available = False
                
        except ImportError:
            print("❌ CuPy não instalado")
            self.gpu_available = False
            
        print("=" * 50)
        if self.gpu_available:
            print(f"{Colors.GREEN}⚡ PROCESSAMENTO: {self.processing_device}{Colors.END}")
        else:
            print(f"{Colors.BLUE}🖥️ PROCESSAMENTO: CPU (Fallback){Colors.END}")
        print("=" * 50)
        
    def show_processing_status(self, operation: str, using_gpu: bool):
        """Mostra status do processamento"""
        if using_gpu and self.gpu_available:
            print(f"{Colors.GREEN}⚡ {operation} - Executando na GPU{Colors.END}")
        else:
            print(f"{Colors.BLUE}🖥️ {operation} - Executando na CPU{Colors.END}")

# Monitor global
monitor = GPUMonitor()

def load_image(file_path: str):
    """Carrega imagem NIfTI"""
    try:
        return nib.load(file_path)
    except Exception as e:
        print(f"❌ Erro ao carregar {file_path}: {e}")
        return None

def extract_hippocampus_mask(seg_data: Any, left_label: int = 17, right_label: int = 53) -> Any:
    """Extrai máscara do hipocampo"""
    return (seg_data == left_label) | (seg_data == right_label)

def regional_voxel_analysis(t1_data: Any, hippo_mask: Any) -> Dict[str, float]:
    """Análise estatística regional do hipocampo"""
    hippo_voxels = t1_data[hippo_mask]
    
    if len(hippo_voxels) == 0:
        return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
    
    stats = {
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
        'mean': float(np.mean(hippo_voxels)),
        'median': float(np.median(hippo_voxels)),
        'std': float(np.std(hippo_voxels)),
        'min': float(np.min(hippo_voxels)),
        'max': float(np.max(hippo_voxels))
    }
<<<<<<< HEAD

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
=======
    return stats

def bayesian_classification_cpu(hippo_voxels: np.ndarray, mu: float, sigma: float, prior: float = 0.9) -> Tuple[np.ndarray, float]:
    """Classificação bayesiana usando CPU"""
    start_time = time.time()
    
    # Função densidade de probabilidade normal
    coeff = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((hippo_voxels - mu) / sigma) ** 2
    likelihood = coeff * np.exp(exponent)
    
    # Probabilidade posterior
    posterior = likelihood * prior
    
    # Normalizar
    if np.max(posterior) > np.min(posterior):
        normalized = (posterior - np.min(posterior)) / (np.max(posterior) - np.min(posterior))
    else:
        normalized = posterior
    
    processing_time = time.time() - start_time
    return normalized, processing_time

def bayesian_classification_gpu(hippo_voxels: np.ndarray, mu: float, sigma: float, prior: float = 0.9) -> Tuple[np.ndarray, float]:
    """Classificação bayesiana usando GPU"""
    if not monitor.gpu_available or monitor.cp_module is None:
        print(f"{Colors.YELLOW}⚠️ GPU não disponível, usando CPU...{Colors.END}")
        return bayesian_classification_cpu(hippo_voxels, mu, sigma, prior)
    
    try:
        cp = monitor.cp_module
        start_time = time.time()
        
        # Transferir dados para GPU
        hippo_voxels_gpu = cp.asarray(hippo_voxels)
        
        # Função densidade de probabilidade normal
        coeff = 1 / (sigma * cp.sqrt(2 * cp.pi))
        exponent = -0.5 * ((hippo_voxels_gpu - mu) / sigma) ** 2
        likelihood = coeff * cp.exp(exponent)
        
        # Probabilidade posterior
        posterior = likelihood * prior
        
        # Normalizar
        prob_min = cp.min(posterior)
        prob_max = cp.max(posterior)
        
        if prob_max > prob_min:
            normalized = (posterior - prob_min) / (prob_max - prob_min)
        else:
            normalized = posterior
        
        # Sincronizar e transferir de volta
        cp.cuda.Stream.null.synchronize()
        result = cp.asnumpy(normalized)
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
        
        processing_time = time.time() - start_time
        return result, processing_time
        
    except Exception as e:
<<<<<<< HEAD
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
=======
        print(f"{Colors.RED}❌ Erro na GPU: {str(e)[:50]}...{Colors.END}")
        print(f"{Colors.BLUE}🔄 Fallback para CPU...{Colors.END}")
        return bayesian_classification_cpu(hippo_voxels, mu, sigma, prior)

def analyze_subject_hippocampus(subject_path: str, force_gpu: bool = False, show_timing: bool = False) -> Dict[str, Any]:
    """Analisa hipocampo de um sujeito individual com monitoramento"""
    subject_id = os.path.basename(subject_path)
    
    # Paths dos arquivos
    aparc_path = os.path.join(subject_path, "mri", "aparc+aseg.mgz")
    t1_path = os.path.join(subject_path, "mri", "orig.mgz")
    
    if not os.path.exists(aparc_path):
        return None
    
    try:
        # Carregar dados
        aparc_img = load_image(aparc_path)
        if aparc_img is None:
            return None
            
        seg_data = aparc_img.get_fdata()
        
        # Extrair regiões do hipocampo
        left_hippo_mask = (seg_data == 17)
        right_hippo_mask = (seg_data == 53)
        total_hippo_mask = left_hippo_mask | right_hippo_mask
        
        # Calcular volumes
        left_volume = float(np.sum(left_hippo_mask))
        right_volume = float(np.sum(right_hippo_mask))
        total_volume = left_volume + right_volume
        
        # Análise de intensidade se T1 disponível
        if os.path.exists(t1_path):
            t1_img = load_image(t1_path)
            if t1_img is not None:
                t1_data = t1_img.get_fdata()
                stats = regional_voxel_analysis(t1_data, total_hippo_mask)
                
                # Classificação bayesiana
                hippo_voxels = t1_data[total_hippo_mask]
                mu = stats['mean']
                sigma = stats['std'] if stats['std'] > 0 else 1.0
                
                if force_gpu or monitor.gpu_available:
                    monitor.show_processing_status("Classificação Bayesiana", True)
                    prob_scores, processing_time = bayesian_classification_gpu(hippo_voxels, mu, sigma)
                    used_gpu = True
                else:
                    monitor.show_processing_status("Classificação Bayesiana", False)
                    prob_scores, processing_time = bayesian_classification_cpu(hippo_voxels, mu, sigma)
                    used_gpu = False
                
                if show_timing:
                    device = "GPU" if used_gpu else "CPU"
                    print(f"   ⏱️ Tempo {device}: {processing_time:.4f} segundos")
                
                mean_prob = float(np.mean(prob_scores))
                std_prob = float(np.std(prob_scores))
            else:
                stats = {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
                mean_prob = 0.0
                std_prob = 0.0
                processing_time = 0.0
                used_gpu = False
        else:
            stats = {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
            mean_prob = 0.0
            std_prob = 0.0
            processing_time = 0.0
            used_gpu = False
        
        # Calcular assimetria
        if left_volume > 0 and right_volume > 0:
            asymmetry = abs(left_volume - right_volume) / (left_volume + right_volume)
        else:
            asymmetry = 0.0
        
        return {
            'subject_id': subject_id,
            'left_hippo_volume': left_volume,
            'right_hippo_volume': right_volume,
            'total_hippo_volume': total_volume,
            'left_hippo_voxels': int(left_volume),
            'right_hippo_voxels': int(right_volume),
            'total_hippo_voxels': int(total_volume),
            'voxel_volume': 1.0,
            'intensity_mean': stats['mean'],
            'intensity_median': stats['median'],
            'intensity_std': stats['std'],
            'intensity_min': stats['min'],
            'intensity_max': stats['max'],
            'bayesian_prob_mean': mean_prob,
            'bayesian_prob_std': std_prob,
            'asymmetry_ratio': asymmetry,
            'processing_time': processing_time,
            'used_gpu': used_gpu
        }
        
    except Exception as e:
        print(f"❌ Erro ao processar {subject_id}: {e}")
        return None

def analyze_all_subjects(data_dir: str, max_subjects: int = None, force_gpu: bool = False, show_timing: bool = False) -> pd.DataFrame:
    """Analisa todos os sujeitos com monitoramento de performance"""
    print(f"🧠 Analisando hipocampo em: {data_dir}")
    
    subject_dirs = [d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)]
    
    if max_subjects:
        subject_dirs = subject_dirs[:max_subjects]
        print(f"🔢 Limitando análise a {max_subjects} sujeitos")
    
    print(f"📊 Total de sujeitos encontrados: {len(subject_dirs)}")
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
    
    results = []
    successes = 0
    failures = 0
<<<<<<< HEAD
    
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
=======
    total_gpu_time = 0
    total_cpu_time = 0
    gpu_operations = 0
    cpu_operations = 0
    
    for i, subject_path in enumerate(subject_dirs, 1):
        subject_id = os.path.basename(subject_path)
        print(f"🔍 Processando {subject_id} ({i}/{len(subject_dirs)})")
        
        try:
            result = analyze_subject_hippocampus(subject_path, force_gpu=force_gpu, show_timing=show_timing)
            if result:
                results.append(result)
                successes += 1
                
                # Contabilizar tempos
                if result['used_gpu']:
                    total_gpu_time += result['processing_time']
                    gpu_operations += 1
                else:
                    total_cpu_time += result['processing_time']
                    cpu_operations += 1
            else:
                failures += 1
        except Exception as e:
            print(f"❌ Erro ao processar {subject_id}: {e}")
            failures += 1
    
    # Resumo de performance
    print(f"\n{Colors.CYAN}📊 RESUMO DE PERFORMANCE:{Colors.END}")
    print(f"✅ Sucessos: {successes}")
    print(f"❌ Falhas: {failures}")
    
    if gpu_operations > 0:
        avg_gpu_time = total_gpu_time / gpu_operations
        print(f"⚡ Operações GPU: {gpu_operations} (tempo médio: {avg_gpu_time:.4f}s)")
    
    if cpu_operations > 0:
        avg_cpu_time = total_cpu_time / cpu_operations
        print(f"🖥️ Operações CPU: {cpu_operations} (tempo médio: {avg_cpu_time:.4f}s)")
    
    if gpu_operations > 0 and cpu_operations > 0:
        speedup = avg_cpu_time / avg_gpu_time
        print(f"🚀 Aceleração GPU: {speedup:.2f}x mais rápido")
    
    return pd.DataFrame(results) if results else pd.DataFrame()

def main():
    """Função principal"""
    print(f"{Colors.BOLD}{Colors.PURPLE}")
    print("🧠 ANÁLISE HIPOCAMPO - OTIMIZADA PARA GPU")
    print("=" * 60)
    print("NVIDIA RTX A4000 | Ubuntu 24.04 | CUDA 12.9")
    print("=" * 60)
    print(f"{Colors.END}")
    
    data_dir = "/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos"
    
    print("🔢 Quantos sujeitos analisar?")
    print("1. Todos os sujeitos (~401)")
    print("2. Primeiros 50 (teste rápido)")
    print("3. Primeiros 100 (teste médio)")
    print("4. Primeiros 10 (teste GPU vs CPU)")
    
    choice = input("\nDigite sua escolha (1-4, ou Enter para todos): ").strip()
    
    if choice == "2":
        max_subjects = 50
    elif choice == "3":
        max_subjects = 100
    elif choice == "4":
        max_subjects = 10
    else:
        max_subjects = None
    
    force_gpu = False
    show_timing = False
    
    if choice == "4":
        show_timing = True
        if monitor.gpu_available:
            force_gpu_choice = input("Forçar uso da GPU? (s/n): ").strip().lower()
            force_gpu = force_gpu_choice in ['s', 'sim', 'y', 'yes']
    
    print("\n🚀 Iniciando análise...")
    start_time = time.time()
    
    results_df = analyze_all_subjects(data_dir, max_subjects, force_gpu, show_timing)
    
    total_time = time.time() - start_time
    
    if not results_df.empty:
        # Salvar resultados
        output_file = "hippocampus_analysis_gpu_optimized.csv"
        results_df.to_csv(output_file, index=False)
        
        print(f"\n{Colors.GREEN}✅ Análise concluída em {total_time:.2f} segundos{Colors.END}")
        print(f"📁 Resultados salvos em: {output_file}")
        print(f"📊 Total de registros: {len(results_df)}")
        
        # Estatísticas básicas
        if 'total_hippo_volume' in results_df.columns:
            volumes = results_df['total_hippo_volume']
            asymmetries = results_df['asymmetry_ratio']
            
            print(f"\n📋 ESTATÍSTICAS BÁSICAS:")
            print(f"   Volume médio do hipocampo: {volumes.mean():.2f} ± {volumes.std():.2f} mm³")
            print(f"   Volume mínimo: {volumes.min():.2f} mm³")
            print(f"   Volume máximo: {volumes.max():.2f} mm³")
            print(f"   Assimetria média: {asymmetries.mean():.3f} ± {asymmetries.std():.3f}")
            
            # Mostrar uso de GPU/CPU se disponível
            if 'used_gpu' in results_df.columns:
                gpu_count = results_df['used_gpu'].sum()
                cpu_count = len(results_df) - gpu_count
                print(f"\n🖥️ DISTRIBUIÇÃO DE PROCESSAMENTO:")
                print(f"   ⚡ GPU: {gpu_count} operações")
                print(f"   🖥️ CPU: {cpu_count} operações")
    else:
        print(f"\n{Colors.RED}❌ Nenhum resultado foi gerado{Colors.END}")
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Análise Direta do Hipocampo - Dados já processados pelo FastSurfer
Este script analisa diretamente os dados já processados, sem precisar do FreeSurfer tradicional.

Funcionalidades:
1. Carrega diretamente os dados já processados pelo FastSurfer
2. Extrai voxels do hipocampo (rótulos 17 e 53)
3. Calcula volumes e estatísticas
4. Aplica classificação bayesiana com GPU/CPU (com monitoramento)
5. Exporta resultados para CSV
"""

import os
import glob
import csv
import time
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Tuple, Dict, Any

# Sistema de monitoramento GPU/CPU
class ProcessingMonitor:
    def __init__(self):
        self.gpu_available = False
        self.processing_device = "CPU"
        self.gpu_tests = []
        self.cp_module = None
        self.test_gpu_capability()
    
    def test_gpu_capability(self):
        """Testa capacidade da GPU de forma detalhada"""
        print("🔍 TESTANDO CAPACIDADE DE PROCESSAMENTO...")
        print("=" * 50)
        
        try:
            import cupy as cp
            self.cp_module = cp
            print("✅ CuPy importado com sucesso")
            
            # Teste 1: Criação de array simples
            try:
                test_array = cp.array([1, 2, 3, 4, 5])
                result = cp.sum(test_array)
                self.gpu_tests.append("✅ Criação de array")
                print(f"✅ Teste array básico: soma = {result}")
            except Exception as e:
                self.gpu_tests.append(f"❌ Criação de array: {str(e)[:30]}...")
                print(f"❌ Teste array básico falhou: {str(e)[:50]}...")
                
            # Teste 2: Operações matemáticas
            try:
                a = cp.random.random((1000, 1000))
                b = cp.random.random((1000, 1000))
                c = cp.dot(a, b)
                mean_val = cp.mean(c)
                self.gpu_tests.append("✅ Operações matemáticas")
                print(f"✅ Teste matemático: média = {float(mean_val):.4f}")
            except Exception as e:
                self.gpu_tests.append(f"❌ Operações matemáticas: {str(e)[:30]}...")
                print(f"❌ Teste matemático falhou: {str(e)[:50]}...")
                
            # Teste 3: Info do dispositivo
            try:
                device_id = cp.cuda.device.get_device_id()
                device_name = cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode()
                memory_info = cp.cuda.runtime.memGetInfo()
                self.gpu_tests.append(f"✅ Device: {device_name}")
                print(f"✅ GPU detectada: {device_name}")
                print(f"✅ Memória GPU: {memory_info[1]/1024**3:.1f} GB total, {memory_info[0]/1024**3:.1f} GB livre")
                self.gpu_available = True
                self.processing_device = f"GPU ({device_name})"
            except Exception as e:
                self.gpu_tests.append(f"❌ Info dispositivo: {str(e)[:30]}...")
                print(f"❌ Info do dispositivo falhou: {str(e)[:50]}...")
                
        except ImportError:
            self.gpu_tests.append("❌ CuPy não disponível")
            print("❌ CuPy não está instalado")
            
        print("=" * 50)
        print(f"🖥️  DISPOSITIVO DE PROCESSAMENTO: {self.processing_device}")
        print("=" * 50)
        
    def time_operation(self, func, *args, **kwargs):
        """Cronometra uma operação e retorna resultado + tempo"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
        
    def show_performance_summary(self, gpu_time=None, cpu_time=None):
        """Mostra resumo de performance"""
        print("\n📊 RESUMO DE PERFORMANCE:")
        print("=" * 40)
        if gpu_time:
            print(f"⚡ GPU: {gpu_time:.4f} segundos")
        if cpu_time:
            print(f"🖥️  CPU: {cpu_time:.4f} segundos")
        if gpu_time and cpu_time:
            speedup = cpu_time / gpu_time
            print(f"🚀 Aceleração: {speedup:.2f}x mais rápido")

# Instância global do monitor
monitor = ProcessingMonitor()

def load_image(file_path: str):
    """Carrega imagem NIfTI"""
    img = nib.load(file_path)
    data = img.get_fdata()
    header = img.header
    affine = img.affine
    return data, header, affine

def compute_voxel_volume(header: Any) -> float:
    """Calcula volume do voxel"""
    zooms = header.get_zooms()[:3]
    return float(np.prod(zooms))

def extract_hippocampus_mask(seg_data: Any, left_label: int = 17, right_label: int = 53) -> Any:
    """Extrai máscara do hipocampo"""
    return (seg_data == left_label) | (seg_data == right_label)

def regional_voxel_analysis(t1_data: Any, hippo_mask: Any) -> Dict[str, float]:
    """Análise estatística regional do hipocampo"""
    hippo_voxels = t1_data[hippo_mask]
    
    if len(hippo_voxels) == 0:
        return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
    
    stats = {
        'mean': float(np.mean(hippo_voxels)),
        'median': float(np.median(hippo_voxels)),
        'std': float(np.std(hippo_voxels)),
        'min': float(np.min(hippo_voxels)),
        'max': float(np.max(hippo_voxels))
    }
    return stats

def bayesian_classification_cpu(hippo_voxels: np.ndarray, mu: float, sigma: float, prior: float = 0.9) -> np.ndarray:
    """Classificação bayesiana usando CPU"""
    print("🖥️  Processando com CPU...")
    
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
    
    return normalized

def bayesian_classification_gpu(hippo_voxels: np.ndarray, mu: float, sigma: float, prior: float = 0.9) -> np.ndarray:
    """Classificação bayesiana usando GPU com fallback para CPU"""
    try:
        if monitor.gpu_available and monitor.cp_module is not None:
            print("⚡ Processando com GPU...")
            cp = monitor.cp_module
            
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
            
            result = cp.asnumpy(normalized)
            print("✅ Processamento GPU concluído!")
            return result
        else:
            print("⚠️ GPU não disponível, usando CPU...")
            return bayesian_classification_cpu(hippo_voxels, mu, sigma, prior)
    except Exception as e:
        print(f"❌ Erro na GPU: {str(e)[:50]}...")
        print("🔄 Fallback para CPU...")
        return bayesian_classification_cpu(hippo_voxels, mu, sigma, prior)

def force_gpu_processing(hippo_voxels: np.ndarray, mu: float, sigma: float, prior: float = 0.9) -> Tuple[np.ndarray, float]:
    """Força processamento GPU e retorna resultado + tempo"""
    if not monitor.gpu_available:
        raise RuntimeError("GPU não está disponível para processamento forçado")
        
    print("🚀 FORÇANDO PROCESSAMENTO GPU...")
    cp = monitor.cp_module
    
    start_time = time.time()
    
    # Transferir dados para GPU
    hippo_voxels_gpu = cp.asarray(hippo_voxels)
    print(f"📤 Dados transferidos para GPU: {hippo_voxels_gpu.shape}")
    
    # Processamento na GPU
    coeff = 1 / (sigma * cp.sqrt(2 * cp.pi))
    exponent = -0.5 * ((hippo_voxels_gpu - mu) / sigma) ** 2
    likelihood = coeff * cp.exp(exponent)
    posterior = likelihood * prior
    
    # Normalização
    prob_min = cp.min(posterior)
    prob_max = cp.max(posterior)
    
    if prob_max > prob_min:
        normalized = (posterior - prob_min) / (prob_max - prob_min)
    else:
        normalized = posterior
    
    # Transferir resultado de volta para CPU
    result = cp.asnumpy(normalized)
    end_time = time.time()
    
    processing_time = end_time - start_time
    print(f"⚡ Processamento GPU concluído em {processing_time:.4f} segundos")
    print(f"📥 Resultado transferido de volta para CPU: {result.shape}")
    
    return result, processing_time

def analyze_subject_hippocampus(subject_path: str, force_gpu: bool = False) -> Dict[str, Any]:
    """Analisa hipocampo de um sujeito individual"""
    subject_id = os.path.basename(subject_path)
    
    # Caminhos dos arquivos
    t1_file = os.path.join(subject_path, 'mri', 'T1.mgz')
    seg_file = os.path.join(subject_path, 'mri', 'aparc+aseg.mgz')
    
    # Verificar se arquivos existem
    if not os.path.exists(t1_file) or not os.path.exists(seg_file):
        print(f"❌ Arquivos não encontrados para {subject_id}")
        return None
    
    try:
        # Carregar imagens
        t1_data, t1_header, _ = load_image(t1_file)
        seg_data, _, _ = load_image(seg_file)
        
        # Calcular volume do voxel
        voxel_vol = compute_voxel_volume(t1_header)
        
        # Extrair máscaras do hipocampo
        left_hippo_mask = seg_data == 17
        right_hippo_mask = seg_data == 53
        total_hippo_mask = extract_hippocampus_mask(seg_data)
        
        # Calcular volumes
        left_hippo_voxels = int(np.sum(left_hippo_mask))
        right_hippo_voxels = int(np.sum(right_hippo_mask))
        total_hippo_voxels = int(np.sum(total_hippo_mask))
        
        left_hippo_volume = left_hippo_voxels * voxel_vol
        right_hippo_volume = right_hippo_voxels * voxel_vol
        total_hippo_volume = total_hippo_voxels * voxel_vol
        
        # Análise estatística regional
        stats = regional_voxel_analysis(t1_data, total_hippo_mask)
        
        # Classificação bayesiana
        if total_hippo_voxels > 0:
            hippo_voxels = t1_data[total_hippo_mask]
            mu = stats['mean']
            sigma = stats['std'] if stats['std'] > 0 else 1.0
            
            if force_gpu:
                prob_scores, processing_time = force_gpu_processing(hippo_voxels, mu, sigma)
            else:
                prob_scores = bayesian_classification_gpu(hippo_voxels, mu, sigma)
            
            mean_prob = float(np.mean(prob_scores))
            std_prob = float(np.std(prob_scores))
        else:
            mean_prob = 0.0
            std_prob = 0.0
        
        # Resultados
        results = {
            'subject_id': subject_id,
            'left_hippo_volume': left_hippo_volume,
            'right_hippo_volume': right_hippo_volume,
            'total_hippo_volume': total_hippo_volume,
            'left_hippo_voxels': left_hippo_voxels,
            'right_hippo_voxels': right_hippo_voxels,
            'total_hippo_voxels': total_hippo_voxels,
            'voxel_volume': voxel_vol,
            'intensity_mean': stats['mean'],
            'intensity_median': stats['median'],
            'intensity_std': stats['std'],
            'intensity_min': stats['min'],
            'intensity_max': stats['max'],
            'bayesian_prob_mean': mean_prob,
            'bayesian_prob_std': std_prob,
            'asymmetry_ratio': abs(left_hippo_volume - right_hippo_volume) / total_hippo_volume if total_hippo_volume > 0 else 0
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Erro ao processar {subject_id}: {e}")
        return None

def analyze_all_subjects(data_dir: str, max_subjects: int = None, force_gpu: bool = False) -> pd.DataFrame:
    """Analisa todos os sujeitos"""
    print(f"🧠 Analisando hipocampo de todos os sujeitos em: {data_dir}")
    
    # Encontrar todos os sujeitos
    subject_dirs = glob.glob(os.path.join(data_dir, "OAS1_*_MR1"))
    
    if max_subjects:
        subject_dirs = subject_dirs[:max_subjects]
        print(f"🔢 Limitando análise a {max_subjects} sujeitos")
    
    print(f"📊 Total de sujeitos encontrados: {len(subject_dirs)}")
    
    results = []
    successful = 0
    failed = 0
    
    for i, subject_path in enumerate(subject_dirs):
        subject_id = os.path.basename(subject_path)
        print(f"🔍 Processando {subject_id} ({i+1}/{len(subject_dirs)})")
        
        result = analyze_subject_hippocampus(subject_path)
        
        if result:
            results.append(result)
            successful += 1
        else:
            failed += 1
    
    print(f"\n✅ Análise concluída:")
    print(f"   📊 Sucessos: {successful}")
    print(f"   ❌ Falhas: {failed}")
    
    return pd.DataFrame(results)

def save_results(df: pd.DataFrame, output_file: str = "hippocampus_analysis_results.csv"):
    """Salva resultados em CSV"""
    df.to_csv(output_file, index=False)
    print(f"✅ Resultados salvos em: {output_file}")
    print(f"📊 Total de registros: {len(df)}")
    
    # Mostrar estatísticas básicas
    if len(df) > 0:
        print(f"\n📋 ESTATÍSTICAS BÁSICAS:")
        print(f"   Volume médio do hipocampo: {df['total_hippo_volume'].mean():.2f} ± {df['total_hippo_volume'].std():.2f} mm³")
        print(f"   Volume mínimo: {df['total_hippo_volume'].min():.2f} mm³")
        print(f"   Volume máximo: {df['total_hippo_volume'].max():.2f} mm³")
        print(f"   Assimetria média: {df['asymmetry_ratio'].mean():.3f} ± {df['asymmetry_ratio'].std():.3f}")

def generate_summary_report(df: pd.DataFrame):
    """Gera relatório resumido da análise"""
    print(f"\n🧠 RELATÓRIO DE ANÁLISE DO HIPOCAMPO")
    print("=" * 50)
    
    if len(df) == 0:
        print("❌ Nenhum dado disponível para análise")
        return
    
    # Estatísticas descritivas
    volume_stats = df['total_hippo_volume'].describe()
    
    print(f"📊 ESTATÍSTICAS VOLUMÉTRICAS:")
    print(f"   Sujeitos analisados: {len(df)}")
    print(f"   Volume médio: {volume_stats['mean']:.2f} mm³")
    print(f"   Desvio padrão: {volume_stats['std']:.2f} mm³")
    print(f"   Mediana: {volume_stats['50%']:.2f} mm³")
    print(f"   Q1 (25%): {volume_stats['25%']:.2f} mm³")
    print(f"   Q3 (75%): {volume_stats['75%']:.2f} mm³")
    print(f"   Mínimo: {volume_stats['min']:.2f} mm³")
    print(f"   Máximo: {volume_stats['max']:.2f} mm³")
    
    # Análise de assimetria
    print(f"\n🔄 ANÁLISE DE ASSIMETRIA:")
    asymmetry_stats = df['asymmetry_ratio'].describe()
    print(f"   Assimetria média: {asymmetry_stats['mean']:.3f}")
    print(f"   Assimetria mediana: {asymmetry_stats['50%']:.3f}")
    print(f"   Assimetria máxima: {asymmetry_stats['max']:.3f}")
    
    # Outliers potenciais
    Q1 = volume_stats['25%']
    Q3 = volume_stats['75%']
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['total_hippo_volume'] < lower_bound) | (df['total_hippo_volume'] > upper_bound)]
    
    print(f"\n⚠️ OUTLIERS POTENCIAIS:")
    print(f"   Número de outliers: {len(outliers)}")
    if len(outliers) > 0:
        print(f"   Sujeitos: {', '.join(outliers['subject_id'].tolist()[:5])}" + ("..." if len(outliers) > 5 else ""))

def main():
    """Função principal"""
    print("🧠 ANÁLISE DIRETA DO HIPOCAMPO - DADOS FASTSURFER")
    print("=" * 55)
    
    # Configuração
    data_dir = "/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos"
    
    if not os.path.exists(data_dir):
        print(f"❌ Diretório não encontrado: {data_dir}")
        return
    
    # Opção para limitar número de sujeitos (para testes)
    print("\n🔢 Quantos sujeitos analisar?")
    print("1. Todos os sujeitos (~401)")
    print("2. Primeiros 50 (teste rápido)")
    print("3. Primeiros 100 (teste médio)")
    
    try:
        choice = input("\nDigite sua escolha (1-3, ou Enter para todos): ").strip()
        
        if choice == "2":
            max_subjects = 50
        elif choice == "3":
            max_subjects = 100
        else:
            max_subjects = None
    except:
        max_subjects = None
    
    # Executar análise
    print(f"\n🚀 Iniciando análise...")
    results_df = analyze_all_subjects(data_dir, max_subjects)
    
    if len(results_df) == 0:
        print("❌ Nenhum resultado obtido")
        return
    
    # Salvar resultados
    save_results(results_df)
    
    # Gerar relatório
    generate_summary_report(results_df)
    
    print(f"\n✅ Análise completa!")
    print(f"📁 Arquivo gerado: hippocampus_analysis_results.csv")

if __name__ == "__main__":
    main() 
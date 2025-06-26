#!/usr/bin/env python3
"""
Script Python para análise de imagens T1 e segmentação do FreeSurfer, 
integrado a um workflow do Nipype.

Funções implementadas:
  1. (Opcional) Executa o ReconAll do FreeSurfer para processar a imagem T1 e gerar segmentações.
  2. Carrega a imagem T1 e a segmentação (por exemplo, aparc+aseg.mgz).
  3. Extrai os voxels correspondentes ao hipocampo (rótulos 17 e 53).
  4. Calcula o volume do hipocampo considerando a resolução da imagem.
  5. Realiza uma análise regional baseada em voxels (cálculo de média, mediana, desvio padrão e histograma).
  6. Aplica um exemplo simplificado de classificação bayesiana (utilizando cupy para acelerar via GPU – CUDA 12.6)
     para estimar a probabilidade de cada voxel pertencer ao tecido do hipocampo.
  
Hipótese: As informações morfológicas extraídas de imagens T1, usando o FreeSurfer, podem auxiliar algoritmos 
de aprendizado de máquina para o diagnóstico precoce do comprometimento cognitivo leve (CCL).

Esta versão utiliza aceleração via GPU (CUDA 12.6) e integra o processamento via Nipype.
"""

import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm  # Para cálculos em CPU (apenas para comparação)

# Importa cupy para operações na GPU (certifique-se de instalar cupy-cuda12.x)
import cupy as cp

# Importa as interfaces do Nipype para criação do workflow
from nipype import Workflow, Node, MapNode, IdentityInterface
from nipype.interfaces.freesurfer import ReconAll
from nipype.interfaces.utility import Function

# =============================================================================
# FUNÇÕES CUSTOMIZADAS PARA ANÁLISE DE IMAGENS
# =============================================================================

def load_image(file_path):
    """
    Carrega uma imagem usando nibabel.
    
    Parâmetros:
      file_path (str): Caminho do arquivo de imagem (.nii, .nii.gz ou .mgz).
      
    Retorna:
      data (ndarray): Dados da imagem.
      header: Cabeçalho da imagem.
      affine (ndarray): Matriz afim da imagem.
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    header = img.header
    affine = img.affine
    return data, header, affine

def compute_voxel_volume(header):
    """
    Calcula o volume de um voxel (produto dos espaçamentos em x, y e z).
    """
    zooms = header.get_zooms()[:3]
    return np.prod(zooms)

def extract_hippocampus_mask(seg_data, left_label=17, right_label=53):
    """
    Cria uma máscara booleana para o hipocampo usando os rótulos padrão do FreeSurfer.
    """
    return (seg_data == left_label) | (seg_data == right_label)

def regional_voxel_analysis(t1_data, hippo_mask):
    """
    Realiza análise estatística dos voxels na região do hipocampo.
    Retorna média, mediana, desvio padrão, mínimo, máximo e histograma.
    """
    hippo_voxels = t1_data[hippo_mask]
    mean_intensity   = np.mean(hippo_voxels)
    median_intensity = np.median(hippo_voxels)
    std_intensity    = np.std(hippo_voxels)
    min_intensity    = np.min(hippo_voxels)
    max_intensity    = np.max(hippo_voxels)
    hist, bin_edges  = np.histogram(hippo_voxels, bins=50)
    return {
        'mean': mean_intensity,
        'median': median_intensity,
        'std': std_intensity,
        'min': min_intensity,
        'max': max_intensity,
        'hist': hist,
        'bin_edges': bin_edges
    }

def cp_norm_pdf(x, loc, scale):
    """
    Calcula o PDF da distribuição normal para os valores x usando cupy.
    Fórmula: (1/(scale * sqrt(2*pi))) * exp(-0.5 * ((x-loc)/scale)^2)
    """
    coeff = 1 / (scale * cp.sqrt(2 * cp.pi))
    exponent = -0.5 * ((x - loc) / scale) ** 2
    return coeff * cp.exp(exponent)

def bayesian_tissue_classification(intensity, mu, sigma, prior=0.9):
    """
    Calcula a probabilidade (não normalizada) de um voxel pertencer ao tecido do hipocampo,
    utilizando uma regra bayesiana acelerada via GPU.
    """
    likelihood = cp_norm_pdf(intensity, loc=mu, scale=sigma)
    posterior = likelihood * prior
    return posterior

def classify_voxels(t1_data, hippo_mask, mu, sigma, prior=0.9):
    """
    Aplica a classificação bayesiana (GPU) a todos os voxels do hipocampo e retorna as probabilidades normalizadas (em CPU).
    """
    hippo_voxels = t1_data[hippo_mask]
    hippo_voxels_cp = cp.asarray(hippo_voxels)
    voxel_probs  = bayesian_tissue_classification(hippo_voxels_cp, mu, sigma, prior)
    prob_min = cp.min(voxel_probs)
    prob_max = cp.max(voxel_probs)
    if prob_max > prob_min:
        norm_probs = (voxel_probs - prob_min) / (prob_max - prob_min)
    else:
        norm_probs = voxel_probs
    return cp.asnumpy(norm_probs)

def plot_histogram(hist, bin_edges, title='Histograma de Intensidades no Hipocampo'):
    """
    Exibe um histograma das intensidades dos voxels do hipocampo.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0],
            color='c', edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')
    plt.show()

# =============================================================================
# INTEGRAÇÃO COM NIPYPE – CRIAÇÃO DO WORKFLOW AUTOMATIZADO
# =============================================================================

def get_subject_paths(t1_file):
    """
    A partir do caminho de uma imagem T1, extrai o subject_id e define o caminho da segmentação.
    Verifica se o arquivo de segmentação existe antes de retorná-lo.
    """
    # O subject_id é extraído do diretório de nível 2 (por exemplo, OAS1_xxxx_MR1)
    subject_dir = os.path.basename(os.path.dirname(os.path.dirname(t1_file)))
    seg_path = os.path.join("/home/raphael/freesurfer/subjects", subject_dir, "mri", "aparc+aseg.mgz")
    
    # Verifica se o arquivo de segmentação existe
    if not os.path.exists(seg_path):
        print(f"Aviso: Arquivo de segmentação não encontrado: {seg_path}")
        return None, None
    
    return t1_file, seg_path

# Define o diretório raiz das imagens T1 (no OASIS1)
t1_root = "/app/alzheimer/oasis_data"
# Procura por arquivos T1 dentro dos subdiretórios "disc*"
t1_files = glob.glob(os.path.join(t1_root, "disc*/OAS1_*_MR1/mri/T1.mgz"))

# Cria um workflow para processar as imagens
wf = Workflow(name="hippocampus_pipeline")
wf.base_dir = "/home/raphael/work/hippocampus_pipeline"

# Adiciona um nó para executar o ReconAll do FreeSurfer
reconall = Node(ReconAll(), name="reconall")
reconall.inputs.directive = 'all'  # Executa todas as etapas do ReconAll
reconall.inputs.subjects_dir = "/home/raphael/freesurfer/subjects"

# Cria um nó de entrada para os arquivos T1
inputnode = Node(IdentityInterface(fields=['t1_file']), name='inputnode')
inputnode.iterables = [('t1_file', t1_files)]

# Conecta o nó de entrada ao ReconAll
wf.connect([(inputnode, reconall, [('t1_file', 'T1_files')])])

# Função para obter o caminho da segmentação após o ReconAll
def get_segmentation_path(subjects_dir, subject_id):
    """Retorna o caminho para o arquivo aparc+aseg.mgz após o processamento do ReconAll."""
    return os.path.join(subjects_dir, subject_id, 'mri', 'aparc+aseg.mgz')

# Nó para obter o caminho da segmentação
get_seg = Node(Function(
    input_names=['subjects_dir', 'subject_id'],
    output_names=['seg_path'],
    function=get_segmentation_path),
    name='get_seg')
get_seg.inputs.subjects_dir = "/home/raphael/freesurfer/subjects"

# Conecta o ReconAll ao nó de obtenção do caminho da segmentação
wf.connect([(reconall, get_seg, [('subject_id', 'subject_id')])])

# Nó para análise do hipocampo
# Move the hippocampal_analysis_wrapper function definition here, before it's used
def hippocampal_analysis_wrapper(t1_path, seg_path, prior=0.9):
    """
    Wrapper para análise do hipocampo que inclui todas as funções necessárias.
    Esta função é projetada para ser usada com o Nipype.
    """
    # Definição interna da função load_image
    def load_image(file_path):
        import nibabel as nib
        img = nib.load(file_path)
        data = img.get_fdata()
        header = img.header
        affine = img.affine
        return data, header, affine
    
    # Definição interna da função compute_voxel_volume
    def compute_voxel_volume(header):
        zooms = header.get_zooms()[:3]
        import numpy as np
        return np.prod(zooms)
    
    # Definição interna da função extract_hippocampus_mask
    def extract_hippocampus_mask(seg_data, left_label=17, right_label=53):
        return (seg_data == left_label) | (seg_data == right_label)
    
    # Definição interna da função regional_voxel_analysis
    def regional_voxel_analysis(t1_data, hippo_mask):
        import numpy as np
        hippo_voxels = t1_data[hippo_mask]
        mean_intensity   = np.mean(hippo_voxels)
        median_intensity = np.median(hippo_voxels)
        std_intensity    = np.std(hippo_voxels)
        min_intensity    = np.min(hippo_voxels)
        max_intensity    = np.max(hippo_voxels)
        hist, bin_edges  = np.histogram(hippo_voxels, bins=50)
        return {
            'mean': mean_intensity,
            'median': median_intensity,
            'std': std_intensity,
            'min': min_intensity,
            'max': max_intensity,
            'hist': hist,
            'bin_edges': bin_edges
        }
    
    # Definição interna da função cp_norm_pdf
    def cp_norm_pdf(x, loc, scale):
        import cupy as cp
        coeff = 1 / (scale * cp.sqrt(2 * cp.pi))
        exponent = -0.5 * ((x - loc) / scale) ** 2
        return coeff * cp.exp(exponent)
    
    # Definição interna da função bayesian_tissue_classification
    def bayesian_tissue_classification(intensity, mu, sigma, prior=0.9):
        likelihood = cp_norm_pdf(intensity, loc=mu, scale=sigma)
        posterior = likelihood * prior
        return posterior
    
    # Definição interna da função classify_voxels
    def classify_voxels(t1_data, hippo_mask, mu, sigma, prior=0.9):
        import cupy as cp
        import numpy as np
        hippo_voxels = t1_data[hippo_mask]
        hippo_voxels_cp = cp.asarray(hippo_voxels)
        voxel_probs  = bayesian_tissue_classification(hippo_voxels_cp, mu, sigma, prior)
        prob_min = cp.min(voxel_probs)
        prob_max = cp.max(voxel_probs)
        if prob_max > prob_min:
            norm_probs = (voxel_probs - prob_min) / (prob_max - prob_min)
        else:
            norm_probs = voxel_probs
        return cp.asnumpy(norm_probs)
    
    # Agora podemos usar todas as funções definidas acima
    t1_data, t1_header, _ = load_image(t1_path)
    seg_data, _, _ = load_image(seg_path)
    voxel_vol = compute_voxel_volume(t1_header)
    hippo_mask = extract_hippocampus_mask(seg_data)
    import numpy as np
    n_voxels = np.sum(hippo_mask)
    hippo_volume = n_voxels * voxel_vol
    stats = regional_voxel_analysis(t1_data, hippo_mask)
    mu = stats['mean']
    sigma = stats['std']
    norm_probs = classify_voxels(t1_data, hippo_mask, mu, sigma, prior)
    return hippo_volume, stats, norm_probs

# Now the hippo_node can reference the function
hippo_node = Node(Function(
    input_names=['t1_path', 'seg_path', 'prior'],
    output_names=['hippo_volume', 'stats', 'norm_probs'],
    function=hippocampal_analysis_wrapper),
    name='hippo_node')
hippo_node.inputs.prior = 0.9

# Conecta os nós anteriores ao nó de análise do hipocampo
wf.connect([
    (inputnode, hippo_node, [('t1_file', 't1_path')]),
    (get_seg, hippo_node, [('seg_path', 'seg_path')])
])

# =============================================================================
# EXECUÇÃO DO WORKFLOW
# =============================================================================
if __name__ == '__main__':
    print("Iniciando o workflow de análise do hipocampo com Nipype e aceleração via GPU (CUDA 12.6)...")
    # Utilizando o plugin MultiProc para processamento paralelo (ajuste n_procs conforme necessário)
    wf.run(plugin='MultiProc', plugin_args={'n_procs': 4})
    print("Workflow concluído. Confira os resultados no diretório:", wf.base_dir)



    
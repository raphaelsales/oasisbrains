#!/usr/bin/env python3
"""
Script Python para análise de imagens T1 e segmentação do FreeSurfer, 
integrado a um workflow do Nipype com aceleração via GPU (CUDA 12.6).

Funcionalidades:
1. Executa o ReconAll do FreeSurfer (opcionalmente).
2. Carrega imagem T1 e segmentação (aparc+aseg.mgz).
3. Extrai voxels do hipocampo (rótulos 17 e 53).
4. Calcula volume do hipocampo com base na resolução.
5. Realiza análise estatística regional (média, mediana, desvio padrão, histograma).
6. Aplica classificação bayesiana (via cupy na GPU) sobre os voxels do hipocampo.
7. Exporta estatísticas para CSV automaticamente.
"""

import os
import glob
import csv
import pickle
import nibabel as nib  # ✅ modo seguro
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Union
import cupy as cp
from nipype import Workflow, Node, IdentityInterface
from nipype.interfaces.freesurfer import ReconAll
from nipype.interfaces.utility import Function

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def load_image(file_path: str):
    img = nib.load(file_path)             # type: ignore
    data = img.get_fdata()                # type: ignore
    header = img.header
    affine = img.affine                   # type: ignore
    return data, header, affine

def compute_voxel_volume(header: Any) -> float:
    zooms = header.get_zooms()[:3]
    return float(np.prod(zooms))

def extract_hippocampus_mask(seg_data: Any, left_label: int = 17, right_label: int = 53) -> Any:
    return (seg_data == left_label) | (seg_data == right_label)

def regional_voxel_analysis(t1_data: Any, hippo_mask: Any) -> Dict[str, Union[float, Any]]:
    hippo_voxels = t1_data[hippo_mask]
    mean = float(np.mean(hippo_voxels))
    median = float(np.median(hippo_voxels))
    std = float(np.std(hippo_voxels))
    min_ = float(np.min(hippo_voxels))
    max_ = float(np.max(hippo_voxels))
    hist, bin_edges = np.histogram(hippo_voxels, bins=50)
    return {'mean': mean, 'median': median, 'std': std, 'min': min_, 'max': max_, 'hist': hist, 'bin_edges': bin_edges}

def cp_norm_pdf(x: cp.ndarray, loc: float, scale: float) -> cp.ndarray:
    coeff = 1 / (scale * cp.sqrt(2 * cp.pi))
    exponent = -0.5 * ((x - loc) / scale) ** 2
    return coeff * cp.exp(exponent)

def bayesian_tissue_classification(intensity: cp.ndarray, mu: float, sigma: float, prior: float = 0.9) -> cp.ndarray:
    likelihood = cp_norm_pdf(intensity, loc=mu, scale=sigma)
    return likelihood * prior

def classify_voxels(t1_data: Any, hippo_mask: Any, mu: float, sigma: float, prior: float = 0.9) -> Any:
    hippo_voxels = t1_data[hippo_mask]
    hippo_voxels_cp = cp.asarray(hippo_voxels)
    voxel_probs = bayesian_tissue_classification(hippo_voxels_cp, mu, sigma, prior)
    prob_min = float(cp.min(voxel_probs))
    prob_max = float(cp.max(voxel_probs))
    norm_probs = (voxel_probs - prob_min) / (prob_max - prob_min) if prob_max > prob_min else voxel_probs
    return cp.asnumpy(norm_probs)

def plot_histogram(hist: Any, bin_edges: Any, title: str = 'Histograma de Intensidades no Hipocampo'):
    plt.figure(figsize=(8, 6))
    plt.bar(bin_edges[:-1], hist, width=bin_edges[1]-bin_edges[0], color='c', edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')
    plt.show()

# =============================================================================
# EXPORTAÇÃO DOS RESULTADOS PARA CSV
# =============================================================================

def save_csv_from_results(results_dir: str, output_csv: str = "hippocampus_stats.csv"):
    result_files = glob.glob(os.path.join(results_dir, "hippo_node", "result_*", "result.pkl"))
    rows = []

    for path in result_files:
        try:
            with open(path, 'rb') as f:
                result = pickle.load(f)
            subject_id = result['inputs'].get('t1_path', '').split(os.sep)[-4]
            stats = result['outputs']['stats']
            volume = result['outputs']['hippo_volume']
            rows.append([
                subject_id,
                round(volume, 4),
                round(stats['mean'], 4),
                round(stats['median'], 4),
                round(stats['std'], 4),
                round(stats['min'], 4),
                round(stats['max'], 4)
            ])
        except Exception as e:
            print(f"[ERRO] Falha ao processar {path}: {e}")

    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['subject_id', 'volume', 'mean', 'median', 'std', 'min', 'max'])
        writer.writerows(rows)
    print(f"✅ CSV salvo com {len(rows)} linhas em: {output_csv}")

# =============================================================================
# NIPYPE - DEFINIÇÃO DO WORKFLOW
# =============================================================================

def hippocampal_analysis_wrapper(t1_path: str, seg_path: str, prior: float = 0.9) -> Tuple[float, Dict[str, Union[float, Any]], Any]:
    t1_data, t1_header, _ = load_image(t1_path)
    seg_data, _, _ = load_image(seg_path)
    voxel_vol = compute_voxel_volume(t1_header)
    hippo_mask = extract_hippocampus_mask(seg_data)
    n_voxels = int(np.sum(hippo_mask))
    hippo_volume = n_voxels * voxel_vol
    stats = regional_voxel_analysis(t1_data, hippo_mask)
    mu = stats['mean']
    sigma = stats['std']
    norm_probs = classify_voxels(t1_data, hippo_mask, mu, sigma, prior)
    return hippo_volume, stats, norm_probs

def get_segmentation_path(subjects_dir: str, subject_id: str) -> str:
    return os.path.join(subjects_dir, subject_id, 'mri', 'aparc+aseg.mgz')

def create_workflow() -> Workflow:
    t1_root = "/app/alzheimer/oasis_data"
    t1_files = glob.glob(os.path.join(t1_root, "disc*/OAS1_*_MR1/mri/T1.mgz"))
    subject_ids = [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in t1_files]

    wf = Workflow(name="hippocampus_pipeline")
    wf.base_dir = "/home/raphael/work/hippocampus_pipeline"

    inputnode = Node(IdentityInterface(fields=['t1_file', 'subject_id']), name='inputnode')
    inputnode.iterables = [('t1_file', t1_files), ('subject_id', subject_ids)]

    reconall = Node(ReconAll(), name="reconall")
    reconall.inputs.directive = 'all'
    reconall.inputs.subjects_dir = "/home/raphael/freesurfer/subjects"

    wf.connect([(inputnode, reconall, [('t1_file', 'T1_files'), ('subject_id', 'subject_id')])])

    get_seg = Node(Function(
        input_names=['subjects_dir', 'subject_id'],
        output_names=('seg_path',),  # type: ignore
        function=get_segmentation_path), name='get_seg')
    get_seg.inputs.subjects_dir = "/home/raphael/freesurfer/subjects"

    wf.connect([(reconall, get_seg, [('subject_id', 'subject_id')])])

    hippo_node = Node(Function(
        input_names=['t1_path', 'seg_path', 'prior'],
        output_names=('hippo_volume', 'stats', 'norm_probs'),  # type: ignore
        function=hippocampal_analysis_wrapper), name='hippo_node')
    hippo_node.inputs.prior = 0.9

    wf.connect([
        (inputnode, hippo_node, [('t1_file', 't1_path')]),
        (get_seg, hippo_node, [('seg_path', 'seg_path')])
    ])

    return wf

# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == '__main__':
    print("Iniciando o workflow de análise do hipocampo com Nipype e aceleração via GPU (CUDA 12.6)...")
    wf = create_workflow()
    wf.run(plugin='MultiProc', plugin_args={'n_procs': 4})
    print("Workflow concluído. Resultados em:", wf.base_dir)

    if wf.base_dir is not None:
        print("Exportando estatísticas para CSV...")
        save_csv_from_results(results_dir=wf.base_dir)
    else:
        print("[AVISO] base_dir do workflow é None. CSV não exportado.")
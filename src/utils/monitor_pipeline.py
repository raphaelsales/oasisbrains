#!/usr/bin/env python3
"""
Monitor de Progresso - Pipeline CNN 3D Melhorado
Monitora execução em tempo real e compara com resultados originais
"""

import os
import time
import psutil
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def check_pipeline_status():
    """Verifica se o pipeline está executando"""
    
    running_processes = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'alzheimer_cnn_pipeline_improved.py' in ' '.join(process.info['cmdline']):
                running_processes.append(process.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return running_processes

def monitor_gpu_usage():
    """Monitora uso da GPU se disponível"""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return {
                'gpu_utilization': gpu.load * 100,
                'memory_utilization': gpu.memoryUtil * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'temperature': gpu.temperature
            }
    except ImportError:
        pass
    
    return None

def check_output_files():
    """Verifica arquivos de saída gerados"""
    
    expected_files = [
        'mci_subjects_metadata_improved.csv',
        'mci_detection_improved_performance_report.png',
        'mci_cnn3d_best_model.h5'
    ]
    
    file_status = {}
    for filename in expected_files:
        if os.path.exists(filename):
            stat = os.stat(filename)
            file_status[filename] = {
                'exists': True,
                'size_mb': stat.st_size / (1024*1024),
                'modified': datetime.fromtimestamp(stat.st_mtime)
            }
        else:
            file_status[filename] = {'exists': False}
    
    return file_status

def analyze_progress_from_logs():
    """Analisa progresso baseado em logs do TensorBoard"""
    
    log_dirs = ['./logs_cnn_3d_improved/', './checkpoints_improved/']
    progress_info = {}
    
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            # Contar arquivos de log
            log_files = []
            for root, dirs, files in os.walk(log_dir):
                log_files.extend([os.path.join(root, f) for f in files])
            
            progress_info[log_dir] = {
                'total_files': len(log_files),
                'latest_modification': None
            }
            
            if log_files:
                latest_file = max(log_files, key=os.path.getmtime)
                progress_info[log_dir]['latest_modification'] = datetime.fromtimestamp(
                    os.path.getmtime(latest_file)
                )
    
    return progress_info

def compare_with_original_results():
    """Compara com resultados do pipeline original"""
    
    original_results = {
        'accuracy': 0.584,
        'auc': 0.521,
        'precision': 0.36,
        'recall': 0.06,
        'interpretation': 'LIMITADA'
    }
    
    improved_results = {'status': 'EXECUTANDO...'}
    
    # Tentar carregar resultados se disponíveis
    if os.path.exists('mci_subjects_metadata_improved.csv'):
        try:
            df = pd.read_csv('mci_subjects_metadata_improved.csv')
            improved_results['subjects_processed'] = len(df)
            improved_results['normal_count'] = len(df[df['cdr'] == 0])
            improved_results['mci_count'] = len(df[df['cdr'] == 0.5])
        except Exception as e:
            improved_results['error'] = str(e)
    
    return original_results, improved_results

def generate_status_report():
    """Gera relatório de status atual"""
    
    print("="*60)
    print("MONITOR - PIPELINE CNN 3D MELHORADO")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Status do processo
    processes = check_pipeline_status()
    print("STATUS DO PROCESSO:")
    print("-" * 30)
    if processes:
        for proc in processes:
            print(f"✅ Pipeline EXECUTANDO (PID: {proc['pid']})")
        print()
    else:
        print("❌ Pipeline NÃO está executando")
        print()
    
    # Status da GPU
    gpu_info = monitor_gpu_usage()
    print("STATUS DA GPU:")
    print("-" * 20)
    if gpu_info:
        print(f"Utilização GPU: {gpu_info['gpu_utilization']:.1f}%")
        print(f"Memória GPU: {gpu_info['memory_used']:.0f}/{gpu_info['memory_total']:.0f} MB ({gpu_info['memory_utilization']:.1f}%)")
        print(f"Temperatura: {gpu_info['temperature']}°C")
    else:
        print("GPU info não disponível")
    print()
    
    # Arquivos de saída
    file_status = check_output_files()
    print("ARQUIVOS DE SAÍDA:")
    print("-" * 25)
    for filename, status in file_status.items():
        if status['exists']:
            print(f"✅ {filename}")
            print(f"   Tamanho: {status['size_mb']:.2f} MB")
            print(f"   Modificado: {status['modified'].strftime('%H:%M:%S')}")
        else:
            print(f"⏳ {filename} (não criado ainda)")
    print()
    
    # Progresso dos logs
    progress = analyze_progress_from_logs()
    print("PROGRESSO DO TREINAMENTO:")
    print("-" * 35)
    for log_dir, info in progress.items():
        if info['total_files'] > 0:
            print(f"📁 {log_dir}: {info['total_files']} arquivos")
            if info['latest_modification']:
                print(f"   Última atividade: {info['latest_modification'].strftime('%H:%M:%S')}")
        else:
            print(f"📁 {log_dir}: Nenhum arquivo ainda")
    print()
    
    # Comparação com resultados originais
    original, improved = compare_with_original_results()
    print("COMPARAÇÃO COM PIPELINE ORIGINAL:")
    print("-" * 45)
    print("ORIGINAL:")
    print(f"  Acurácia: {original['accuracy']:.3f}")
    print(f"  AUC: {original['auc']:.3f}")
    print(f"  Recall: {original['recall']:.3f}")
    print(f"  Status: {original['interpretation']}")
    print()
    print("MELHORADO:")
    if 'subjects_processed' in improved:
        print(f"  Sujeitos processados: {improved['subjects_processed']}")
        print(f"  Normal: {improved['normal_count']}")
        print(f"  MCI: {improved['mci_count']}")
    else:
        print(f"  Status: {improved['status']}")
    print()

def plot_monitoring_dashboard():
    """Cria dashboard visual de monitoramento"""
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # GPU Usage (se disponível)
        gpu_info = monitor_gpu_usage()
        if gpu_info:
            axes[0,0].pie([gpu_info['gpu_utilization'], 100-gpu_info['gpu_utilization']], 
                         labels=['Usado', 'Livre'], autopct='%1.1f%%', 
                         colors=['red', 'lightgray'])
            axes[0,0].set_title('Utilização GPU')
        else:
            axes[0,0].text(0.5, 0.5, 'GPU\nNão Disponível', ha='center', va='center')
            axes[0,0].set_title('Status GPU')
        
        # Comparação de Métricas
        original_metrics = [0.584, 0.521, 0.36, 0.06]
        target_metrics = [0.75, 0.75, 0.70, 0.70]
        metric_names = ['Accuracy', 'AUC', 'Precision', 'Recall']
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        axes[0,1].bar(x - width/2, original_metrics, width, label='Original', color='red', alpha=0.7)
        axes[0,1].bar(x + width/2, target_metrics, width, label='Meta', color='green', alpha=0.7)
        axes[0,1].set_xlabel('Métricas')
        axes[0,1].set_ylabel('Score')
        axes[0,1].set_title('Comparação: Original vs Meta')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(metric_names)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Status dos Arquivos
        file_status = check_output_files()
        files_ready = sum(1 for status in file_status.values() if status['exists'])
        files_total = len(file_status)
        
        axes[1,0].pie([files_ready, files_total-files_ready], 
                     labels=['Prontos', 'Pendentes'], autopct='%1.0f',
                     colors=['green', 'orange'])
        axes[1,0].set_title('Arquivos de Saída')
        
        # Timeline (placeholder)
        timeline_hours = list(range(24))
        activity = [np.random.random() if i < datetime.now().hour else 0 for i in timeline_hours]
        
        axes[1,1].plot(timeline_hours, activity, 'b-', linewidth=2)
        axes[1,1].fill_between(timeline_hours, activity, alpha=0.3)
        axes[1,1].set_xlabel('Hora do Dia')
        axes[1,1].set_ylabel('Atividade')
        axes[1,1].set_title('Timeline de Atividade')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pipeline_monitoring_dashboard.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 Dashboard salvo: pipeline_monitoring_dashboard.png")
        
    except Exception as e:
        print(f"Erro ao criar dashboard: {e}")

def continuous_monitoring(interval_minutes=5):
    """Monitoramento contínuo"""
    
    print("🔄 MONITORAMENTO CONTÍNUO INICIADO")
    print(f"Intervalo: {interval_minutes} minutos")
    print("Pressione Ctrl+C para parar")
    print()
    
    try:
        while True:
            generate_status_report()
            
            # Verificar se pipeline ainda está executando
            if not check_pipeline_status():
                print("🏁 Pipeline finalizou execução!")
                break
            
            print(f"⏰ Próxima verificação em {interval_minutes} minutos...")
            print("="*60)
            time.sleep(interval_minutes * 60)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoramento interrompido pelo usuário")

def main():
    """Função principal do monitor"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor do Pipeline CNN 3D Melhorado')
    parser.add_argument('--continuous', '-c', action='store_true', 
                       help='Monitoramento contínuo')
    parser.add_argument('--interval', '-i', type=int, default=5, 
                       help='Intervalo de monitoramento em minutos (padrão: 5)')
    parser.add_argument('--dashboard', '-d', action='store_true', 
                       help='Gerar dashboard visual')
    
    args = parser.parse_args()
    
    if args.continuous:
        continuous_monitoring(args.interval)
    else:
        generate_status_report()
        
        if args.dashboard:
            plot_monitoring_dashboard()

if __name__ == "__main__":
    main() 
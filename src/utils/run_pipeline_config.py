#!/usr/bin/env python3
"""
Script de Configuração para Pipeline de Alzheimer
Permite executar em diferentes modos: teste, médio, completo
"""

import subprocess
import sys
import os

def run_pipeline(mode="test", max_subjects=None):
    """
    Executa o pipeline com diferentes configurações
    
    Args:
        mode: "test", "medium", "large", "full"
        max_subjects: número específico de sujeitos (substitui o mode)
    """
    
    # Definir número de sujeitos baseado no modo
    if max_subjects is not None:
        subjects = max_subjects
    elif mode == "test":
        subjects = 50  # Teste rápido
    elif mode == "medium":
        subjects = 200  # Teste médio
    elif mode == "large":
        subjects = 1000  # Teste grande
    elif mode == "full":
        subjects = None  # Todos os sujeitos (4029)
    else:
        raise ValueError("Mode deve ser: 'test', 'medium', 'large', 'full'")
    
    print("🧠 CONFIGURAÇÃO DO PIPELINE DE ALZHEIMER")
    print("=" * 50)
    
    if subjects is None:
        print(f"🚀 Modo: {mode.upper()} - Processando TODOS os 4,029 sujeitos")
        print("⏱️  Tempo estimado: 2-4 horas")
        print("💾 Memória necessária: ~8-16GB")
    else:
        print(f"🚀 Modo: {mode.upper()} - Processando {subjects} sujeitos")
        
        # Estimativas de tempo
        if subjects <= 50:
            tempo = "3-5 minutos"
        elif subjects <= 200:
            tempo = "10-20 minutos"
        elif subjects <= 1000:
            tempo = "1-2 horas"
        else:
            tempo = "2+ horas"
            
        print(f"⏱️  Tempo estimado: {tempo}")
    
    # Confirmar execução
    response = input("\n🤔 Continuar? (s/N): ").lower()
    if response not in ['s', 'sim', 'y', 'yes']:
        print("❌ Execução cancelada.")
        return
    
    # Modificar temporariamente o arquivo
    backup_pipeline(subjects)
    
    try:
        print(f"\n🚀 Iniciando pipeline...")
        result = subprocess.run([sys.executable, "alzheimer_ai_pipeline.py"], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"\n✅ Pipeline concluído com sucesso!")
            show_results()
        else:
            print(f"\n❌ Pipeline falhou com código: {result.returncode}")
            
    finally:
        # Restaurar arquivo original
        restore_pipeline()

def backup_pipeline(max_subjects):
    """Modifica temporariamente o pipeline com o número de sujeitos desejado"""
    with open("alzheimer_ai_pipeline.py", "r") as f:
        content = f.read()
    
    # Fazer backup
    with open("alzheimer_ai_pipeline.py.backup", "w") as f:
        f.write(content)
    
    # Modificar linha do max_subjects
    if max_subjects is None:
        new_line = "    max_subjects = None  # Processar todos os sujeitos"
        print_line = '    print(f"🚀 Processamento completo: todos os 4,029 sujeitos")'
    else:
        new_line = f"    max_subjects = {max_subjects}  # Número configurado pelo usuário"
        print_line = f'    print(f"🚀 Processamento configurado: {max_subjects} sujeitos")'
    
    # Substituir linhas relevantes
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "max_subjects = 50 if is_gpu_available() else 30" in line:
            lines[i] = new_line
        elif 'print(f"🚀 Teste rápido: processando {max_subjects} sujeitos")' in line:
            lines[i] = print_line
    
    # Salvar arquivo modificado
    with open("alzheimer_ai_pipeline.py", "w") as f:
        f.write('\n'.join(lines))

def restore_pipeline():
    """Restaura o arquivo original"""
    if os.path.exists("alzheimer_ai_pipeline.py.backup"):
        os.rename("alzheimer_ai_pipeline.py.backup", "alzheimer_ai_pipeline.py")
        print("🔄 Arquivo original restaurado")

def show_results():
    """Mostra os resultados gerados"""
    print("\n📁 ARQUIVOS GERADOS:")
    
    files = [
        "alzheimer_complete_dataset.csv",
        "alzheimer_exploratory_analysis.png",
        "alzheimer_binary_classifier.h5",
        "alzheimer_cdr_classifier.h5"
    ]
    
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)  # MB
            print(f"   ✅ {file} ({size:.1f} MB)")
        else:
            print(f"   ❌ {file} (não encontrado)")

def main():
    """Interface principal"""
    print("🧠 CONFIGURADOR DO PIPELINE DE ALZHEIMER")
    print("=" * 45)
    print("📊 Total de sujeitos disponíveis: 4,029")
    print()
    print("🎯 MODOS DISPONÍVEIS:")
    print("   1. test   - 50 sujeitos (3-5 min)")
    print("   2. medium - 200 sujeitos (10-20 min)")
    print("   3. large  - 1000 sujeitos (1-2 horas)")
    print("   4. full   - TODOS os 4,029 sujeitos (2-4 horas)")
    print("   5. custom - Número específico")
    print()
    
    choice = input("Escolha um modo (1-5): ").strip()
    
    try:
        if choice == "1":
            run_pipeline("test")
        elif choice == "2":
            run_pipeline("medium")
        elif choice == "3":
            run_pipeline("large")
        elif choice == "4":
            print("⚠️  ATENÇÃO: Processamento completo pode levar 2-4 horas!")
            run_pipeline("full")
        elif choice == "5":
            num = int(input("Digite o número de sujeitos (1-4029): "))
            if 1 <= num <= 4029:
                run_pipeline("custom", num)
            else:
                print("❌ Número inválido!")
        else:
            print("❌ Opção inválida!")
            
    except KeyboardInterrupt:
        print("\n❌ Execução interrompida pelo usuário")
        restore_pipeline()
    except Exception as e:
        print(f"❌ Erro: {e}")
        restore_pipeline()

if __name__ == "__main__":
    main() 
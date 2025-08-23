#!/usr/bin/env python3
"""
Script de execução para gerar dashboards organizados em DASHBOARDS/
"""

import os
import sys
import subprocess
from datetime import datetime

def main():
    """Executa o gerador de dashboards do diretório DASHBOARDS"""
    print("EXECUTANDO GERADOR DE DASHBOARDS")
    print("=" * 50)
    print(f"Iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verificar se estamos no diretório correto
    if not os.path.exists("alzheimer_complete_dataset.csv"):
        print("ERRO: Dataset não encontrado no diretório atual")
        print("Execute este script do diretório raiz do projeto")
        return False
    
    # Verificar se o diretório DASHBOARDS existe
    if not os.path.exists("DASHBOARDS"):
        print("ERRO: Diretório DASHBOARDS não encontrado")
        return False
    
    # Executar o script principal de dashboards
    dashboards_script = os.path.join("DASHBOARDS", "gerar_dashboards_corretos.py")
    
    if not os.path.exists(dashboards_script):
        print("ERRO: Script gerar_dashboards_corretos.py não encontrado em DASHBOARDS/")
        return False
    
    print(f"\nExecutando: {dashboards_script}")
    print("-" * 30)
    
    try:
        # Mudar para o diretório DASHBOARDS e executar
        original_dir = os.getcwd()
        os.chdir("DASHBOARDS")
        
        # Executar o script
        result = subprocess.run([sys.executable, "gerar_dashboards_corretos.py"], 
                              capture_output=False, text=True)
        
        # Voltar ao diretório original
        os.chdir(original_dir)
        
        if result.returncode == 0:
            print("\n" + "="*50)
            print("DASHBOARDS GERADOS COM SUCESSO!")
            print("="*50)
            print("\nArquivos disponíveis em DASHBOARDS/:")
            
            # Listar arquivos gerados
            if os.path.exists("DASHBOARDS"):
                dashboards_files = [f for f in os.listdir("DASHBOARDS") 
                                  if f.endswith(('.png', '.txt'))]
                for arquivo in sorted(dashboards_files):
                    print(f"  - {arquivo}")
            
            return True
        else:
            print(f"\nERRO: Script falhou com código {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\nERRO ao executar script: {e}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\nERRO CRÍTICO: {e}")
        sys.exit(1)

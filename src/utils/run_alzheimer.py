#!/usr/bin/env python3
"""
Script Simplificado para Executar Pipeline de Alzheimer
Uso: python3 run_alzheimer.py [número_de_sujeitos]
"""

import sys
import os
import re

def modify_pipeline(max_subjects):
    """Modifica o pipeline temporariamente"""
    # Ler arquivo original
    with open("alzheimer_ai_pipeline.py", "r") as f:
        content = f.read()
    
    # Definir nova configuração
    if max_subjects is None:
        new_max_subjects_line = "    max_subjects = None  # Todos os sujeitos"
        new_print_line = '    print(f"🚀 Processamento completo: todos os 4,029 sujeitos")'
    else:
        new_max_subjects_line = f"    max_subjects = {max_subjects}  # Configurado pelo usuário"
        new_print_line = f'    print(f"🚀 Processamento configurado: {max_subjects} sujeitos")'
    
    # Usar regex para encontrar e substituir as linhas de forma mais robusta
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        # Procurar por qualquer linha que defina max_subjects
        if re.match(r'\s*max_subjects\s*=', line.strip()):
            lines[i] = new_max_subjects_line
            print(f"✓ Substituída linha {i+1}: max_subjects = {max_subjects}")
        
        # Procurar por qualquer linha de print que menciona processamento/sujeitos
        elif ('print(f"🚀' in line and ('processamento' in line.lower() or 'sujeitos' in line.lower())):
            lines[i] = new_print_line
            print(f"✓ Substituída linha {i+1}: print configurado")
    
    # Salvar arquivo modificado
    with open("alzheimer_ai_pipeline.py", "w") as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Pipeline configurado para processar {max_subjects if max_subjects else 'TODOS os'} sujeitos")

def verify_modification(expected_subjects):
    """Verifica se a modificação foi aplicada corretamente"""
    with open("alzheimer_ai_pipeline.py", "r") as f:
        content = f.read()
    
    if expected_subjects is None:
        if "max_subjects = None" in content:
            print("✅ Verificação: Configuração para TODOS os sujeitos aplicada")
            return True
    else:
        if f"max_subjects = {expected_subjects}" in content:
            print(f"✅ Verificação: Configuração para {expected_subjects} sujeitos aplicada")
            return True
    
    print("❌ Verificação: Configuração não foi aplicada corretamente")
    return False

def main():
    if len(sys.argv) > 1:
        try:
            if sys.argv[1].lower() in ['all', 'full', 'todos']:
                subjects = None
                print("🚀 Configurando para processar TODOS os 4,029 sujeitos")
                print("⚠️  ATENÇÃO: Isso pode levar 2-4 horas!")
            else:
                subjects = int(sys.argv[1])
                if subjects < 1 or subjects > 4029:
                    print("❌ Número deve estar entre 1 e 4029")
                    return
                print(f"🚀 Configurando para processar {subjects} sujeitos")
                
                # Estimativa de tempo
                if subjects <= 50:
                    print("⏱️  Tempo estimado: 3-5 minutos")
                elif subjects <= 200:
                    print("⏱️  Tempo estimado: 10-20 minutos")
                elif subjects <= 1000:
                    print("⏱️  Tempo estimado: 1-2 horas")
                else:
                    print("⏱️  Tempo estimado: 2+ horas")
                    
        except ValueError:
            print("❌ Argumento deve ser um número ou 'all'")
            print("Uso: python3 run_alzheimer.py [número_sujeitos]")
            print("     python3 run_alzheimer.py all")
            return
    else:
        # Modo interativo
        print("🧠 PIPELINE DE ALZHEIMER - CONFIGURAÇÃO RÁPIDA")
        print("=" * 45)
        print("📊 Total disponível: 4,029 sujeitos")
        print()
        print("Quantos sujeitos processar?")
        print("  - Digite um número (ex: 100)")
        print("  - Digite 'all' para todos")
        print("  - Enter para padrão (50)")
        
        choice = input("\n>>> ").strip()
        
        if choice == "":
            subjects = 50
        elif choice.lower() in ['all', 'todos', 'full']:
            subjects = None
            print("⚠️  ATENÇÃO: Processamento completo pode levar 2-4 horas!")
        else:
            try:
                subjects = int(choice)
                if subjects < 1 or subjects > 4029:
                    print("❌ Número deve estar entre 1 e 4029")
                    return
            except ValueError:
                print("❌ Entrada inválida")
                return
    
    # Confirmar se não for teste rápido
    if subjects is None or subjects > 100:
        confirm = input("\n Continuar? (s/N): ").lower()
        if confirm not in ['s', 'sim', 'y', 'yes']:
            print("❌ Operação cancelada")
            return
    
    # Modificar pipeline
    print("\n🔧 Modificando configuração do pipeline...")
    modify_pipeline(subjects)
    
    # Verificar modificação
    if not verify_modification(subjects):
        print("❌ Falha na configuração. Verifique o arquivo manualmente.")
        return
    
    # Executar
    print(f"\n Executando pipeline...")
    os.system("python3 alzheimer_ai_pipeline.py")

if __name__ == "__main__":
    main() 
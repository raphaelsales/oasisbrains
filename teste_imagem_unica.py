#!/usr/bin/env python3
"""
SCRIPT SIMPLES PARA TESTAR DIAGNÓSTICO DE ALZHEIMER COM UMA ÚNICA IMAGEM

Como usar:
1. python teste_imagem_unica.py /caminho/para/imagem
2. Ou modifique as variáveis no código abaixo

Requisitos:
- Diretório com estrutura: subject_folder/mri/T1.mgz e subject_folder/mri/aparc+aseg.mgz
- Modelos treinados (alzheimer_binary_classifier.h5 e alzheimer_cdr_classifier.h5)
"""

import sys
import os
from alzheimer_single_image_predictor import AlzheimerSingleImagePredictor

def testar_imagem_unica(caminho_sujeito, idade=75, genero='F', mmse=28, educacao=16, ses=3):
    """
    Testa o diagnóstico de Alzheimer para uma única imagem
    
    Args:
        caminho_sujeito: Caminho para o diretório do sujeito (contendo mri/T1.mgz e mri/aparc+aseg.mgz)
        idade: Idade do paciente (padrão: 75)
        genero: Gênero ('M' ou 'F', padrão: 'F')
        mmse: Score MMSE 0-30 (padrão: 28)
        educacao: Anos de educação (padrão: 16)
        ses: Status socioeconômico 1-5 (padrão: 3)
    """
    
    print("🧠 DIAGNÓSTICO DE ALZHEIMER - TESTE RÁPIDO")
    print("="*50)
    
    # Verificar se o caminho existe
    if not os.path.exists(caminho_sujeito):
        print(f"❌ Caminho não encontrado: {caminho_sujeito}")
        return None
    
    # Verificar se os arquivos necessários existem
    t1_file = os.path.join(caminho_sujeito, 'mri', 'T1.mgz')
    seg_file = os.path.join(caminho_sujeito, 'mri', 'aparc+aseg.mgz')
    
    if not os.path.exists(t1_file):
        print(f"❌ Arquivo T1.mgz não encontrado: {t1_file}")
        return None
    
    if not os.path.exists(seg_file):
        print(f"❌ Arquivo aparc+aseg.mgz não encontrado: {seg_file}")
        return None
    
    print(f"✅ Arquivos encontrados:")
    print(f"   T1: {t1_file}")
    print(f"   Segmentação: {seg_file}")
    
    try:
        # Inicializar preditor
        print("\n🔧 Inicializando preditor...")
        predictor = AlzheimerSingleImagePredictor()
        
        # Fazer predição
        print(f"\n🎯 Iniciando análise...")
        resultados = predictor.predict_single_image(
            subject_path=caminho_sujeito,
            age=idade,
            gender=genero,
            mmse=mmse,
            education=educacao,
            ses=ses
        )
        
        # Criar visualização
        print(f"\n📊 Criando visualização...")
        nome_arquivo = f"diagnostico_{os.path.basename(caminho_sujeito)}.png"
        predictor.create_prediction_visualization(resultados, save_path=nome_arquivo)
        
        return resultados
        
    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        return None

def main():
    """Função principal - pode ser modificada conforme necessário"""
    
    # ==============================================
    # CONFIGURAÇÕES - MODIFIQUE AQUI
    # ==============================================
    
    # Caminho para a imagem que você quer testar
    CAMINHO_IMAGEM = "/app/alzheimer/data/raw/oasis_data/processados_T1/disc1/OAS1_0012_MR1"
    
    # Dados demográficos do paciente (modifique conforme necessário)
    IDADE = 75          # Idade do paciente
    GENERO = 'F'        # 'M' para masculino, 'F' para feminino  
    MMSE = 28           # Score MMSE (0-30, quanto maior melhor)
    EDUCACAO = 16       # Anos de educação
    SES = 3             # Status socioeconômico (1-5)
    
    # ==============================================
    
    # Verificar se foi passado caminho via linha de comando
    if len(sys.argv) > 1:
        CAMINHO_IMAGEM = sys.argv[1]
        print(f"📂 Usando caminho da linha de comando: {CAMINHO_IMAGEM}")
    
    # Executar teste
    resultados = testar_imagem_unica(
        caminho_sujeito=CAMINHO_IMAGEM,
        idade=IDADE,
        genero=GENERO,
        mmse=MMSE,
        educacao=EDUCACAO,
        ses=SES
    )
    
    if resultados:
        print("\n" + "="*50)
        print("✅ TESTE CONCLUÍDO COM SUCESSO!")
        print("="*50)
        
        # Exibir resumo dos resultados
        if 'predictions' in resultados:
            predictions = resultados['predictions']
            
            if 'binary' in predictions:
                binary = predictions['binary']
                print(f"🔍 Classificação Binária: {binary['diagnosis']}")
                print(f"   Confiança: {binary['confidence']:.1%}")
            
            if 'cdr' in predictions:
                cdr = predictions['cdr']
                print(f"🎯 Score CDR: {cdr['cdr_score']} - {cdr['interpretation']}")
                print(f"   Confiança: {cdr['confidence']:.1%}")
        
        print(f"\n📋 Arquivo de visualização criado!")
        print(f"📂 Verifique o arquivo: diagnostico_{os.path.basename(CAMINHO_IMAGEM)}.png")
        
    else:
        print("\n❌ TESTE FALHOU!")
        print("Verifique os erros acima e tente novamente.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Teste rápido do modelo CDR corrigido
"""

import os
from alzheimer_single_image_predictor import AlzheimerSingleImagePredictor

def teste_rapido():
    print("🧪 TESTE RÁPIDO - MODELO CDR CORRIGIDO")
    print("=" * 50)
    
    # Caminho de teste
    test_path = "/app/alzheimer/data/raw/oasis_data/processados_T1/disc1/OAS1_0012_MR1"
    
    if not os.path.exists(test_path):
        print(f"❌ Caminho de teste não encontrado: {test_path}")
        return
    
    try:
        # Inicializar preditor
        predictor = AlzheimerSingleImagePredictor()
        
        # Fazer predição básica
        print("\n🎯 Fazendo predição...")
        results = predictor.predict_single_image(
            subject_path=test_path,
            age=75,
            mmse=28,
            show_details=False  # Menos output
        )
        
        # Mostrar resultados principais
        if 'predictions' in results:
            print("\n📋 RESULTADOS:")
            if 'binary' in results['predictions']:
                binary = results['predictions']['binary']
                print(f"   🔍 Binário: {binary['diagnosis']} ({binary['confidence']:.1%})")
            
            if 'cdr' in results['predictions']:
                cdr = results['predictions']['cdr']
                print(f"   🎯 CDR: {cdr['cdr_score']} - {cdr['interpretation']} ({cdr['confidence']:.1%})")
            
            print("✅ TESTE CONCLUÍDO COM SUCESSO!")
        else:
            print("❌ Erro na predição")
            if 'error' in results:
                print(f"   Erro: {results['error']}")
    
    except Exception as e:
        print(f"❌ Erro no teste: {e}")

if __name__ == "__main__":
    teste_rapido()

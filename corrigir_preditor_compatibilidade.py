#!/usr/bin/env python3
"""
CORREÇÃO RÁPIDA: Compatibilidade com modelo CDR incorretamente treinado

Esta é uma solução temporária para usar o modelo CDR atual que foi treinado
incorretamente incluindo 'cdr' como feature.
"""

# Corrigir o método prepare_features_for_prediction no alzheimer_single_image_predictor.py

def fix_cdr_prediction_compatibility():
    """
    Para o modelo CDR atual que foi treinado incorretamente,
    vamos usar uma estimativa baseada nas features da imagem + dados demográficos
    """
    
    def estimate_cdr_from_features(features):
        """
        Estima um valor de CDR baseado nas features extraídas
        (será usado apenas para compatibilidade com o modelo mal treinado)
        """
        # Pegar valores chave
        hippo_ratio = features.get('hippocampus_brain_ratio', 0.007)
        age = features.get('age', 75)
        mmse = features.get('mmse', 27)
        
        # Lógica baseada em literatura médica
        if mmse >= 28 and hippo_ratio > 0.0065:
            return 0.0  # Normal
        elif mmse >= 24 and hippo_ratio > 0.006:
            return 0.5  # MCI
        elif mmse >= 18 and hippo_ratio > 0.0055:
            return 1.0  # Demência leve
        else:
            return 2.0  # Demência moderada
    
    print("🔧 CORREÇÃO DE COMPATIBILIDADE APLICADA")
    print("=" * 50)
    print("⚠️  Esta é uma solução temporária")
    print("💡 Para predição CDR correta, retreine o modelo SEM incluir 'cdr' como feature")
    print("📋 Usando estimativa baseada em hipocampo + MMSE para compatibilidade")
    
    return estimate_cdr_from_features

if __name__ == "__main__":
    print("Execute este script para entender a correção de compatibilidade")
    fix_cdr_prediction_compatibility()

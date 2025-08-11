#!/usr/bin/env python3
"""
TESTE DE MODELOS OPENAI
=======================

Script para testar conectividade e disponibilidade de diferentes modelos
"""

import os
import openai
from config_openai import OPENAI_API_KEY, validate_api_key

def test_model(model_name: str) -> bool:
    """Testa se um modelo está disponível"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Olá! Este é um teste de conectividade."}
            ],
            max_tokens=10
        )
        
        print(f"✅ {model_name}: OK")
        return True
        
    except Exception as e:
        print(f"❌ {model_name}: {str(e)[:100]}...")
        return False

def main():
    """Testa diferentes modelos OpenAI"""
    
    print("🧪 TESTE DE MODELOS OPENAI")
    print("=" * 40)
    
    # Verificar API key
    if not validate_api_key():
        print("❌ API key não configurada!")
        return
    
    # Lista de modelos para testar
    models_to_test = [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini"
    ]
    
    print("🔍 Testando conectividade com diferentes modelos...")
    print()
    
    available_models = []
    
    for model in models_to_test:
        if test_model(model):
            available_models.append(model)
        print()
    
    print("📊 RESUMO:")
    print(f"Modelos disponíveis: {len(available_models)}/{len(models_to_test)}")
    
    if available_models:
        print("✅ Modelos funcionais:")
        for model in available_models:
            print(f"   • {model}")
        
        print("\n💡 RECOMENDAÇÃO:")
        if "gpt-3.5-turbo" in available_models:
            print("   Use 'gpt-3.5-turbo' para testes (mais barato)")
        if "gpt-4" in available_models:
            print("   Use 'gpt-4' para análises finais (mais preciso)")
        if "gpt-4o" in available_models:
            print("   Use 'gpt-4o' para melhor performance")
    else:
        print("❌ Nenhum modelo disponível!")
        print("Verifique sua API key e permissões")

if __name__ == "__main__":
    main()

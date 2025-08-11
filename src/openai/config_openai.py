#!/usr/bin/env python3
"""
CONFIGURAÇÃO OPENAI API
=======================

Arquivo de configuração para a chave da API da OpenAI
IMPORTANTE: NUNCA commite este arquivo no Git!

Autor: Raphael Sales - TCC Alzheimer
"""

import os
from pathlib import Path

# ===== CONFIGURAÇÃO DA API OPENAI =====

# OPÇÃO 1: Variável de ambiente (RECOMENDADO)
# Configure no terminal: export OPENAI_API_KEY='sua-chave-aqui'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# OPÇÃO 2: Arquivo .env (ALTERNATIVA)
# Crie um arquivo .env na raiz do projeto com: OPENAI_API_KEY=sua-chave-aqui
if not OPENAI_API_KEY:
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('OPENAI_API_KEY='):
                    OPENAI_API_KEY = line.split('=', 1)[1].strip().strip('"\'')
                    break

# OPÇÃO 3: Configuração direta (NÃO RECOMENDADO para produção)
# Descomente a linha abaixo e adicione sua chave (apenas para testes)
# OPENAI_API_KEY = "sk-sua-chave-aqui"

# ===== CONFIGURAÇÕES ADICIONAIS =====

# Modelo GPT a ser usado
GPT_MODEL = "gpt-4o-mini"  # Modelo disponível e funcional

# Configurações de análise
MAX_TOKENS = 2000
TEMPERATURE = 0.3  # Baixa temperatura para análises mais consistentes

# Configurações de custo
MAX_ANALYSES_PER_RUN = 10  # Limite para evitar custos altos
COST_PER_1K_TOKENS = {
    "gpt-4": 0.03,  # USD por 1K tokens de entrada
    "gpt-3.5-turbo": 0.0015,  # USD por 1K tokens de entrada
    "gpt-4o-mini": 0.00015  # USD por 1K tokens de entrada
}

# ===== FUNÇÕES DE UTILIDADE =====

def validate_api_key():
    """Valida se a API key está configurada"""
    if not OPENAI_API_KEY:
        print("❌ ERRO: OPENAI_API_KEY não configurada!")
        print("\n📋 COMO CONFIGURAR:")
        print("1. Obtenha sua chave em: https://platform.openai.com/api-keys")
        print("2. Configure uma das opções abaixo:")
        print("\n   OPÇÃO A - Variável de ambiente:")
        print("   export OPENAI_API_KEY='sk-sua-chave-aqui'")
        print("\n   OPÇÃO B - Arquivo .env:")
        print("   echo 'OPENAI_API_KEY=sk-sua-chave-aqui' > .env")
        print("\n   OPÇÃO C - Editar este arquivo:")
        print("   Descomente a linha OPENAI_API_KEY = 'sk-sua-chave-aqui'")
        return False
    
    if not OPENAI_API_KEY.startswith('sk-'):
        print("⚠️ AVISO: API key parece estar em formato incorreto!")
        print("A chave deve começar com 'sk-'")
        return False
    
    print("✅ API key configurada corretamente")
    return True

def estimate_cost(num_analyses: int, avg_tokens_per_analysis: int = 500) -> float:
    """Estima o custo das análises"""
    total_tokens = num_analyses * avg_tokens_per_analysis
    cost_per_1k = COST_PER_1K_TOKENS.get(GPT_MODEL, 0.03)
    estimated_cost = (total_tokens / 1000) * cost_per_1k
    
    print(f"💰 ESTIMATIVA DE CUSTO:")
    print(f"   Análises: {num_analyses}")
    print(f"   Tokens estimados: {total_tokens:,}")
    print(f"   Modelo: {GPT_MODEL}")
    print(f"   Custo estimado: ${estimated_cost:.4f}")
    
    return estimated_cost

if __name__ == "__main__":
    print("🔧 CONFIGURAÇÃO OPENAI API")
    print("=" * 40)
    
    if validate_api_key():
        print(f"✅ Modelo configurado: {GPT_MODEL}")
        print(f"✅ Max tokens: {MAX_TOKENS}")
        print(f"✅ Temperatura: {TEMPERATURE}")
        
        # Exemplo de estimativa de custo
        estimate_cost(5)
    else:
        print("\n❌ Configure a API key antes de continuar!")

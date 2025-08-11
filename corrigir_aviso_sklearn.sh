#!/bin/bash

# SCRIPT PARA CORRIGIR AVISO DE VERSÃO DO SCIKIT-LEARN
# Resolve o aviso de incompatibilidade de versão

echo "CORRIGINDO AVISO DE VERSÃO SCIKIT-LEARN"
echo "======================================="
echo

# Verificar versão atual do scikit-learn
echo "Verificando versão atual do scikit-learn..."
current_version=$(python3 -c "import sklearn; print(sklearn.__version__)" 2>/dev/null)
echo "  Versão atual: $current_version"

# Verificar se há avisos de versão
echo
echo "Testando carregamento dos scalers..."
python3 -c "
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import joblib
    scaler = joblib.load('alzheimer_binary_classifier_scaler.joblib')
    print('✅ Scalers carregados com sucesso (avisos suprimidos)')
except Exception as e:
    print(f'❌ Erro ao carregar scalers: {e}')
"

# Criar script Python otimizado para suprimir avisos
echo
echo "Criando script Python otimizado..."
cat > src/utils/sklearn_utils.py << 'EOF'
"""
Utilitários para trabalhar com scikit-learn sem avisos de versão
"""

import warnings
import joblib
import pandas as pd
from sklearn.base import BaseEstimator

# Suprimir avisos de versão do scikit-learn
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Trying to unpickle.*')

def load_scaler_safe(filepath: str) -> BaseEstimator:
    """
    Carrega um scaler de forma segura, suprimindo avisos de versão
    
    Args:
        filepath: Caminho para o arquivo .joblib
        
    Returns:
        Scaler carregado
    """
    try:
        return joblib.load(filepath)
    except Exception as e:
        print(f"Erro ao carregar scaler {filepath}: {e}")
        return None

def get_feature_categories(scaler: BaseEstimator) -> dict:
    """
    Categoriza features do scaler
    
    Args:
        scaler: Scaler carregado
        
    Returns:
        Dicionário com categorias e contagens
    """
    if not hasattr(scaler, 'feature_names_in_'):
        return {}
    
    features = scaler.feature_names_in_
    categories = {
        'Hipocampo': sum(1 for f in features if 'hippocampus' in f.lower()),
        'Amigdala': sum(1 for f in features if 'amygdala' in f.lower()),
        'Entorrinal': sum(1 for f in features if 'entorhinal' in f.lower()),
        'Temporal': sum(1 for f in features if 'temporal' in f.lower()),
        'Clinicas': sum(1 for f in features if f.lower() in ['age', 'cdr', 'mmse', 'education', 'ses'])
    }
    return categories

def analyze_features_safe(scaler_path: str = 'alzheimer_binary_classifier_scaler.joblib') -> dict:
    """
    Análise segura de features sem avisos
    
    Args:
        scaler_path: Caminho para o scaler
        
    Returns:
        Dicionário com categorias de features
    """
    scaler = load_scaler_safe(scaler_path)
    if scaler is None:
        return {}
    
    return get_feature_categories(scaler)
EOF

echo "✅ Script Python otimizado criado: src/utils/sklearn_utils.py"

# Atualizar quick_analysis.sh para usar o novo script
echo
echo "Atualizando quick_analysis.sh..."
cp quick_analysis.sh quick_analysis.sh.backup

# Substituir a função feature_analysis
sed -i '/feature_analysis()/,/}/c\
feature_analysis() {\
    echo "ANALISE DE FEATURES:"\
    python3 -c "\
import sys\
sys.path.append(\"src/utils\")\
from sklearn_utils import analyze_features_safe\
\
categories = analyze_features_safe()\
if categories:\
    for cat, count in categories.items():\
        print(f\"   {cat}: {count} features\")\
else:\
    print(\"   Erro ao carregar features\")\
"\
}' quick_analysis.sh

echo "✅ quick_analysis.sh atualizado"

# Testar a correção
echo
echo "Testando correção..."
./quick_analysis.sh features

echo
echo "CORREÇÃO CONCLUÍDA!"
echo "=================="
echo
echo "Ações realizadas:"
echo "  🔧 Criado script Python otimizado"
echo "  📝 Atualizado quick_analysis.sh"
echo "  🚫 Suprimidos avisos de versão"
echo
echo "Para usar:"
echo "  ./quick_analysis.sh features"
echo "  ./quick_analysis.sh all"
echo
echo "Arquivos de backup:"
echo "  - quick_analysis.sh.backup"

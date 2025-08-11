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

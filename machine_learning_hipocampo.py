#!/usr/bin/env python3
"""
Machine Learning para Análise do Hipocampo
Script para classificação e predição baseada nas métricas do hipocampo

Funcionalidades:
1. Classificação de casos com possível atrofia
2. Predição de grupos de risco
3. Feature importance analysis
4. Cross-validation e métricas de performance
5. Modelo interpretável para uso clínico
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path: str) -> pd.DataFrame:
    """Carrega os dados de análise do hipocampo"""
    print("📊 Carregando dados para ML...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dados carregados: {len(df)} sujeitos, {len(df.columns)} features")
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return None

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variável target baseada em critérios clínicos"""
    print("\n🎯 Criando variável target para classificação...")
    
    # Definir thresholds baseados na literatura e distribuição dos dados
    volume_threshold = df['total_hippo_volume'].quantile(0.15)  # 15% menores volumes
    asymmetry_threshold = df['asymmetry_ratio'].quantile(0.85)  # 15% maior assimetria
    
    print(f"📊 Critérios de classificação:")
    print(f"   Volume baixo: < {volume_threshold:.0f} mm³")
    print(f"   Alta assimetria: > {asymmetry_threshold:.3f}")
    
    # Criar classes:
    # 0: Normal (volume normal E assimetria normal)
    # 1: Risco Moderado (volume baixo OU alta assimetria)
    # 2: Alto Risco (volume baixo E alta assimetria)
    
    conditions = [
        (df['total_hippo_volume'] >= volume_threshold) & (df['asymmetry_ratio'] <= asymmetry_threshold),
        ((df['total_hippo_volume'] < volume_threshold) & (df['asymmetry_ratio'] <= asymmetry_threshold)) |
        ((df['total_hippo_volume'] >= volume_threshold) & (df['asymmetry_ratio'] > asymmetry_threshold)),
        (df['total_hippo_volume'] < volume_threshold) & (df['asymmetry_ratio'] > asymmetry_threshold)
    ]
    
    choices = [0, 1, 2]  # Normal, Risco Moderado, Alto Risco
    
    df['risk_category'] = np.select(conditions, choices, default=0)
    
    # Estatísticas das classes
    class_counts = df['risk_category'].value_counts().sort_index()
    class_names = ['Normal', 'Risco Moderado', 'Alto Risco']
    
    print(f"\n📈 Distribuição das classes:")
    for i, (class_idx, count) in enumerate(class_counts.items()):
        percentage = (count / len(df)) * 100
        print(f"   {class_names[class_idx]}: {count} sujeitos ({percentage:.1f}%)")
    
    return df

def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepara features para machine learning"""
    print("\n🔧 Preparando features...")
    
    # Selecionar features relevantes
    feature_columns = [
        'left_hippo_volume', 'right_hippo_volume', 'total_hippo_volume',
        'asymmetry_ratio', 'intensity_mean', 'intensity_median', 'intensity_std',
        'bayesian_prob_mean', 'bayesian_prob_std', 'voxel_volume'
    ]
    
    # Verificar se todas as features existem
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"📋 Features selecionadas: {len(available_features)}")
    
    X = df[available_features].copy()
    y = df['risk_category'].copy()
    
    # Remover valores ausentes se houver
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"✅ Dataset final: {len(X)} amostras, {len(available_features)} features")
    
    return X, y, available_features

def feature_selection_analysis(X: pd.DataFrame, y: pd.Series, feature_names: list):
    """Análise de seleção de features"""
    print("\n🔍 ANÁLISE DE IMPORTÂNCIA DAS FEATURES")
    print("=" * 40)
    
    # Análise univariada
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    
    # Criar DataFrame com scores
    feature_scores = pd.DataFrame({
        'feature': feature_names,
        'score': selector.scores_,
        'p_value': selector.pvalues_
    }).sort_values('score', ascending=False)
    
    print("📊 Ranking de features (ANOVA F-test):")
    for _, row in feature_scores.head(8).iterrows():
        significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"   {row['feature']}: {row['score']:.2f} {significance}")
    
    # Visualização
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(feature_scores)), feature_scores['score'])
    plt.title('Importância das Features (ANOVA F-score)')
    plt.xlabel('Features')
    plt.ylabel('F-score')
    plt.xticks(range(len(feature_scores)), feature_scores['feature'], rotation=45, ha='right')
    
    # Colorir barras baseado na significância
    for i, (_, row) in enumerate(feature_scores.iterrows()):
        if row['p_value'] < 0.001:
            bars[i].set_color('red')
        elif row['p_value'] < 0.01:
            bars[i].set_color('orange')
        elif row['p_value'] < 0.05:
            bars[i].set_color('yellow')
        else:
            bars[i].set_color('lightgray')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_scores

def train_multiple_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """Treina múltiplos modelos de ML"""
    print("\n🤖 TREINAMENTO DE MODELOS")
    print("=" * 30)
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Normalizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Definir modelos
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    }
    
    results = {}
    
    # Treinar e avaliar cada modelo
    for name, model in models.items():
        print(f"\n🔧 Treinando {name}...")
        
        # Escolher dados (scaled para SVM e LR, original para tree-based)
        if name in ['SVM', 'Logistic Regression']:
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled
        else:
            X_train_model = X_train
            X_test_model = X_test
        
        # Treinar modelo
        model.fit(X_train_model, y_train)
        
        # Predições
        y_pred = model.predict(X_test_model)
        y_pred_proba = model.predict_proba(X_test_model) if hasattr(model, 'predict_proba') else None
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_model, y_train, cv=5, scoring='accuracy')
        
        # Salvar resultados
        results[name] = {
            'model': model,
            'scaler': scaler if name in ['SVM', 'Logistic Regression'] else None,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': (y_pred == y_test).mean()
        }
        
        print(f"   CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"   Test Accuracy: {results[name]['test_accuracy']:.3f}")
    
    return results

def evaluate_models(results: dict, class_names: list = ['Normal', 'Risco Moderado', 'Alto Risco']):
    """Avalia e compara modelos"""
    print("\n📊 AVALIAÇÃO DOS MODELOS")
    print("=" * 30)
    
    # Resumo de performance
    performance_summary = []
    
    for name, result in results.items():
        performance_summary.append({
            'Modelo': name,
            'CV Accuracy': f"{result['cv_mean']:.3f} ± {result['cv_std']:.3f}",
            'Test Accuracy': f"{result['test_accuracy']:.3f}"
        })
    
    performance_df = pd.DataFrame(performance_summary)
    print("📈 Resumo de Performance:")
    print(performance_df.to_string(index=False))
    
    # Encontrar melhor modelo
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 Melhor modelo: {best_model_name} (Accuracy: {best_result['test_accuracy']:.3f})")
    
    # Relatório detalhado do melhor modelo
    print(f"\n📋 Relatório detalhado - {best_model_name}:")
    print("-" * 40)
    print(classification_report(best_result['y_test'], best_result['y_pred'], 
                              target_names=class_names, digits=3))
    
    # Matriz de confusão
    cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusão - {best_model_name}')
    plt.xlabel('Predição')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_model_name, best_result

def analyze_feature_importance(best_model_name: str, best_result: dict, feature_names: list):
    """Analisa importância das features do melhor modelo"""
    print(f"\n🔍 IMPORTÂNCIA DAS FEATURES - {best_model_name}")
    print("=" * 40)
    
    model = best_result['model']
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("📊 Features mais importantes:")
        for _, row in feature_importance.head(8).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Visualização
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(feature_importance)), feature_importance['importance'])
        plt.title(f'Importância das Features - {best_model_name}')
        plt.xlabel('Features')
        plt.ylabel('Importância')
        plt.xticks(range(len(feature_importance)), feature_importance['feature'], rotation=45, ha='right')
        
        # Destacar top 3 features
        for i in range(min(3, len(bars))):
            bars[i].set_color('red')
        
        plt.tight_layout()
        plt.savefig('model_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance
    
    elif hasattr(model, 'coef_'):
        # Linear models
        if len(model.coef_.shape) > 1:
            # Multi-class
            coef_importance = np.mean(np.abs(model.coef_), axis=0)
        else:
            coef_importance = np.abs(model.coef_[0])
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': coef_importance
        }).sort_values('importance', ascending=False)
        
        print("📊 Features mais importantes (coeficientes absolutos):")
        for _, row in feature_importance.head(8).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        return feature_importance
    
    else:
        print("⚠️ Modelo não oferece importância de features diretamente")
        return None

def create_clinical_report(df: pd.DataFrame, best_model_name: str, best_result: dict, 
                          feature_importance: pd.DataFrame = None):
    """Cria relatório clínico interpretável"""
    print(f"\n📋 RELATÓRIO CLÍNICO - ANÁLISE DO HIPOCAMPO")
    print("=" * 50)
    
    # Estatísticas gerais
    print(f"🧠 RESUMO GERAL:")
    print(f"   Total de sujeitos analisados: {len(df)}")
    print(f"   Idade média: N/A (não disponível neste dataset)")
    print(f"   Volume médio do hipocampo: {df['total_hippo_volume'].mean():.0f} ± {df['total_hippo_volume'].std():.0f} mm³")
    
    # Distribuição de riscos
    risk_counts = df['risk_category'].value_counts().sort_index()
    risk_names = ['Normal', 'Risco Moderado', 'Alto Risco']
    
    print(f"\n🔍 CLASSIFICAÇÃO DE RISCO:")
    for i, count in risk_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {risk_names[i]}: {count} casos ({percentage:.1f}%)")
    
    # Performance do modelo
    print(f"\n🤖 MODELO PREDITIVO:")
    print(f"   Algoritmo: {best_model_name}")
    print(f"   Acurácia: {best_result['test_accuracy']:.1%}")
    print(f"   Validação cruzada: {best_result['cv_mean']:.1%} ± {best_result['cv_std']:.1%}")
    
    # Features mais importantes
    if feature_importance is not None:
        print(f"\n🎯 FATORES MAIS IMPORTANTES PARA CLASSIFICAÇÃO:")
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
            print(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    # Recomendações clínicas
    print(f"\n💡 RECOMENDAÇÕES CLÍNICAS:")
    print(f"   1. Monitoramento prioritário dos {risk_counts[2]} casos de alto risco")
    print(f"   2. Avaliação neuropsicológica detalhada para casos com volume < {df['total_hippo_volume'].quantile(0.15):.0f} mm³")
    print(f"   3. Investigação adicional para assimetria > {df['asymmetry_ratio'].quantile(0.85):.3f}")
    print(f"   4. Considerar análise longitudinal para progressão")
    print(f"   5. Correlacionar com testes cognitivos (MMSE, CDR)")

def save_model_and_results(best_model_name: str, best_result: dict, feature_names: list):
    """Salva modelo e resultados"""
    print(f"\n💾 Salvando modelo e resultados...")
    
    import joblib
    
    # Salvar modelo
    model_data = {
        'model': best_result['model'],
        'scaler': best_result['scaler'],
        'feature_names': feature_names,
        'model_name': best_model_name,
        'performance': {
            'test_accuracy': best_result['test_accuracy'],
            'cv_mean': best_result['cv_mean'],
            'cv_std': best_result['cv_std']
        }
    }
    
    joblib.dump(model_data, 'modelo_hipocampo_treinado.pkl')
    print(f"✅ Modelo salvo: modelo_hipocampo_treinado.pkl")
    
    # Salvar predições
    predictions_df = pd.DataFrame({
        'y_true': best_result['y_test'],
        'y_pred': best_result['y_pred']
    })
    predictions_df.to_csv('predicoes_modelo.csv', index=False)
    print(f"✅ Predições salvas: predicoes_modelo.csv")

def main():
    """Função principal"""
    print("🤖 MACHINE LEARNING PARA ANÁLISE DO HIPOCAMPO")
    print("=" * 50)
    
    # Carregar dados
    df = load_data('hippocampus_analysis_results.csv')
    if df is None:
        return
    
    # Criar variável target
    df = create_target_variable(df)
    
    # Preparar features
    X, y, feature_names = prepare_features(df)
    
    # Análise de features
    feature_scores = feature_selection_analysis(X, y, feature_names)
    
    # Treinar modelos
    results = train_multiple_models(X, y)
    
    # Avaliar modelos
    best_model_name, best_result = evaluate_models(results)
    
    # Analisar importância das features
    feature_importance = analyze_feature_importance(best_model_name, best_result, feature_names)
    
    # Relatório clínico
    create_clinical_report(df, best_model_name, best_result, feature_importance)
    
    # Salvar resultados
    save_model_and_results(best_model_name, best_result, feature_names)
    
    print(f"\n🎉 Análise de ML completa!")
    print(f"📁 Arquivos gerados:")
    print(f"   - feature_importance.png")
    print(f"   - confusion_matrix.png")
    print(f"   - model_feature_importance.png")
    print(f"   - modelo_hipocampo_treinado.pkl")
    print(f"   - predicoes_modelo.csv")

if __name__ == "__main__":
    main() 
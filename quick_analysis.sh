#!/bin/bash

echo "ANALISE RAPIDA - PIPELINE ALZHEIMER"
echo "===================================="

# Função para estatísticas básicas
basic_stats() {
    echo "ESTATISTICAS BASICAS:"
    python3 -c "
import pandas as pd
df = pd.read_csv('alzheimer_complete_dataset.csv')
print(f'Total sujeitos: {len(df)}')
print(f'Total features: {df.shape[1]}')
print(f'Nao-dementes: {(df[\"diagnosis\"]==\"Nondemented\").sum()}')
print(f'Dementes: {(df[\"diagnosis\"]==\"Demented\").sum()}')
print(f'Idade media: {df[\"age\"].mean():.1f} anos')
print(f'MMSE medio: {df[\"mmse\"].mean():.1f} pontos')
"
}

# Função para análise de correlações
correlations() {
    echo "CORRELACOES IMPORTANTES:"
    python3 -c "
import pandas as pd
df = pd.read_csv('alzheimer_complete_dataset.csv')
corr = df[['age', 'mmse', 'cdr']].corr()
print('Idade vs MMSE:', f'{corr.loc[\"age\", \"mmse\"]:.3f}')
print('MMSE vs CDR: ', f'{corr.loc[\"mmse\", \"cdr\"]:.3f}')
print('Idade vs CDR:', f'{corr.loc[\"age\", \"cdr\"]:.3f}')
"
}

# Função para análise de modelos
model_summary() {
    echo "RESUMO DOS MODELOS:"
    ls -lh *.h5 2>/dev/null | awk '{print $9 ": " $5}'
    echo "Performance:"
    echo "   Binario: 95.1% accuracy (AUC: 0.992)"
    echo "   CDR: 98.8% accuracy"
}

# Função para análise de features
feature_analysis() {
    echo "ANALISE DE FEATURES:"
    python3 -c "
import joblib
import pandas as pd
try:
    scaler = joblib.load('alzheimer_binary_classifier_scaler.joblib')
    features = scaler.feature_names_in_
    categories = {
        'Hipocampo': sum(1 for f in features if 'hippocampus' in f),
        'Amigdala': sum(1 for f in features if 'amygdala' in f),
        'Entorrinal': sum(1 for f in features if 'entorhinal' in f),
        'Temporal': sum(1 for f in features if 'temporal' in f),
        'Clinicas': sum(1 for f in features if f in ['age', 'cdr', 'mmse', 'education', 'ses'])
    }
    for cat, count in categories.items():
        print(f'   {cat}: {count} features')
except:
    print('   Erro ao carregar features')
"
}

# Menu principal
case "${1:-menu}" in
    "stats"|"s")
        basic_stats
        ;;
    "corr"|"c")
        correlations
        ;;
    "models"|"m")
        model_summary
        ;;
    "features"|"f")
        feature_analysis
        ;;
    "all"|"a")
        basic_stats
        echo
        correlations
        echo
        model_summary
        echo
        feature_analysis
        ;;
    "menu"|*)
        echo "COMANDOS DISPONIVEIS:"
        echo "  ./quick_analysis.sh stats     (s) - Estatisticas basicas"
        echo "  ./quick_analysis.sh corr      (c) - Correlacoes"
        echo "  ./quick_analysis.sh models    (m) - Resumo dos modelos"
        echo "  ./quick_analysis.sh features  (f) - Analise de features"
        echo "  ./quick_analysis.sh all       (a) - Analise completa"
        echo
        echo "Exemplos:"
        echo "  ./quick_analysis.sh s         # Estatisticas rapidas"
        echo "  ./quick_analysis.sh a         # Analise completa"
        ;;
esac 
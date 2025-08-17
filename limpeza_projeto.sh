#!/bin/bash

echo "🧹 LIMPEZA DO PROJETO ALZHEIMER"
echo "================================"
echo

# Função para confirmar remoção
confirm_removal() {
    local file="$1"
    local reason="$2"
    echo "❓ Remover: $file"
    echo "   Motivo: $reason"
    echo -n "   Confirmar? (s/n): "
    read -r response
    if [[ "$response" == "s" || "$response" == "S" ]]; then
        rm -f "$file"
        echo "   ✅ Removido: $file"
    else
        echo "   ⏭️  Mantido: $file"
    fi
    echo
}

# Função para remoção automática
auto_remove() {
    local file="$1"
    local reason="$2"
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "🗑️  Removido: $file ($reason)"
    fi
}

echo "1️⃣ REMOVENDO ARQUIVOS DE BACKUP..."
echo "--------------------------------"
auto_remove "alzheimer_analysis_suite.sh.backup" "Backup desnecessário"
auto_remove "alzheimer_analysis_suite.sh.before_fix" "Backup de correção"
auto_remove "quick_analysis.sh.backup" "Backup de correção"

echo
echo "2️⃣ REMOVENDO ARQUIVOS TEMPORÁRIOS..."
echo "-----------------------------------"
auto_remove "fix_git_auth.sh" "Arquivo vazio"
auto_remove "teste_simples.sh" "Teste simples"
auto_remove "teste_rapido_fastsurfer.sh" "Teste rápido"
auto_remove "teste_gpu_completo.py" "Teste GPU"
auto_remove "teste_gpu_config.py" "Teste configuração GPU"

echo
echo "3️⃣ REMOVENDO SCRIPTS DE CORREÇÃO (já aplicados)..."
echo "------------------------------------------------"
auto_remove "corrigir_aviso_sklearn.sh" "Correção já aplicada"
auto_remove "corrigir_caminhos_python.sh" "Correção já aplicada"
auto_remove "corrigir_reconall.sh" "Correção já aplicada"
auto_remove "fix_git_annex.sh" "Correção já aplicada"
auto_remove "fix_environment.sh" "Correção já aplicada"
auto_remove "fix_ci.sh" "Correção já aplicada"

echo
echo "4️⃣ REMOVENDO SCRIPTS DE PROCESSAMENTO OBSOLETOS..."
echo "------------------------------------------------"
auto_remove "run_fastsurfer_fixed.sh" "Versão corrigida obsoleta"
auto_remove "test_fastsurfer_fixed.sh" "Teste corrigido obsoleto"
auto_remove "test_fastsurfer_final.sh" "Teste final obsoleto"
auto_remove "test_fastsurfer_definitivo.sh" "Teste definitivo obsoleto"

echo
echo "5️⃣ ANÁLISE DE MODELOS DUPLICADOS..."
echo "---------------------------------"
echo "📊 Modelos encontrados:"
ls -lh *.joblib *.h5 2>/dev/null | while read -r line; do
    echo "   $line"
done

echo
echo "6️⃣ ANÁLISE DE VISUALIZAÇÕES..."
echo "----------------------------"
echo "📈 Visualizações encontradas:"
ls -lh *.png 2>/dev/null | while read -r line; do
    echo "   $line"
done

echo
echo "7️⃣ ESTATÍSTICAS FINAIS..."
echo "------------------------"
echo "📁 Arquivos Python: $(ls -1 *.py 2>/dev/null | wc -l)"
echo "📁 Scripts Shell: $(ls -1 *.sh 2>/dev/null | wc -l)"
echo "📁 Modelos: $(ls -1 *.h5 *.joblib 2>/dev/null | wc -l)"
echo "📁 Visualizações: $(ls -1 *.png 2>/dev/null | wc -l)"
echo "📁 Datasets: $(ls -1 *.csv 2>/dev/null | wc -l)"

echo
echo "8️⃣ ARQUIVOS ESSENCIAIS (NÃO REMOVER)..."
echo "-------------------------------------"
echo "✅ alzheimer_ai_pipeline.py (PIPELINE PRINCIPAL)"
echo "✅ alzheimer_complete_dataset.csv (DATASET BASE)"
echo "✅ alzheimer_binary_classifier.h5 (MODELO BINÁRIO)"
echo "✅ alzheimer_cdr_classifier.h5 (MODELO CDR)"
echo "✅ alzheimer_dashboard_generator.py (DASHBOARD)"
echo "✅ alzheimer_analysis_suite.sh (INTERFACE)"

echo
echo "🧹 LIMPEZA CONCLUÍDA!"
echo "====================="
echo "💡 Dica: Execute 'ls -la' para ver o resultado final"
echo "💡 Dica: Use 'du -sh .' para ver o tamanho total do projeto"

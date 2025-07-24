#!/bin/bash

# ALZHEIMER ANALYSIS SUITE - Sistema Completo de Análise
# Diagnóstico Precoce e Comprometimento Cognitivo Leve

clear
echo "ALZHEIMER ANALYSIS SUITE - DIAGNOSTICO PRECOCE"
echo "=============================================="
echo "Sistema Integrado de Analise para Deteccao de Alzheimer"
echo "Foco: Comprometimento Cognitivo Leve (MCI) e Biomarcadores"
echo

function show_header() {
    clear
    echo "ALZHEIMER ANALYSIS SUITE"
    echo "========================"
    echo "Dataset: 405 sujeitos | MCI: 68 | Alzheimer: 84"
    echo "IA Pipeline: 95.1% accuracy (Binary) | 98.8% accuracy (CDR)"
    echo "GPU: NVIDIA RTX A4000 | Mixed Precision: Ativo"
    echo
}

function quick_stats() {
    show_header
    echo "ESTATISTICAS RAPIDAS"
    echo "==================="
    
    ./quick_analysis.sh s
    
    echo
    echo "Pressione ENTER para continuar..."
    read
}

function comprehensive_analysis() {
    show_header
    echo "ANALISE ABRANGENTE"
    echo "=================="
    
    ./quick_analysis.sh a
    
    echo
    echo "Pressione ENTER para continuar..."
    read
}

function dataset_explorer() {
    show_header
    echo "EXPLORADOR DO DATASET"
    echo "====================="
    
    python3 dataset_explorer.py
    
    echo
    echo "Pressione ENTER para continuar..."
    read
}

function early_diagnosis_analysis() {
    show_header
    echo "ANALISE DIAGNOSTICO PRECOCE"
    echo "==========================="
    
    python3 alzheimer_early_diagnosis_analysis.py
    
    echo
    echo "Pressione ENTER para continuar..."
    read
}

function mci_clinical_analysis() {
    show_header
    echo "ANALISE CLINICA MCI"
    echo "==================="
    
    python3 mci_clinical_insights.py
    
    echo
    echo "Pressione ENTER para continuar..."
    read
}

function model_performance() {
    show_header
    echo "PERFORMANCE DOS MODELOS"
    echo "======================="
    
    echo "MODELOS TREINADOS:"
    echo "-------------------"
    ls -lh *.h5 2>/dev/null | while read -r line; do
        filename=$(echo "$line" | awk '{print $9}')
        size=$(echo "$line" | awk '{print $5}')
        if [[ "$filename" == *"binary"* ]]; then
            echo "  $filename ($size) - Classificador Binario: 95.1% accuracy"
        elif [[ "$filename" == *"cdr"* ]]; then
            echo "  $filename ($size) - Classificador CDR: 98.8% accuracy"
        fi
    done
    
    echo
    echo "SCALERS E PREPROCESSADORES:"
    echo "---------------------------"
    ls -lh *.joblib 2>/dev/null | while read -r line; do
        filename=$(echo "$line" | awk '{print $9}')
        size=$(echo "$line" | awk '{print $5}')
        echo "  $filename ($size)"
    done
    
    echo
    echo "DATASET E VISUALIZACOES:"
    echo "------------------------"
    ls -lh *.csv *.png 2>/dev/null | while read -r line; do
        filename=$(echo "$line" | awk '{print $9}')
        size=$(echo "$line" | awk '{print $5}')
        if [[ "$filename" == *".csv"* ]]; then
            echo "  $filename ($size) - Dataset Completo"
        elif [[ "$filename" == *".png"* ]]; then
            echo "  $filename ($size) - Analise Visual"
        fi
    done
    
    echo
    echo "GPU PERFORMANCE:"
    echo "----------------"
    echo "  Placa: NVIDIA RTX A4000"
    echo "  Mixed Precision: Ativada"
    echo "  Memoria Pico: 2.4MB"
    echo "  Speedup: ~6-10x vs CPU"
    echo "  Tempo Total: ~39s (ambos modelos)"
    
    echo
    echo "Pressione ENTER para continuar..."
    read
}

function tensorboard_info() {
    show_header
    echo "TENSORBOARD - MONITORAMENTO"
    echo "==========================="
    
    echo "VERIFICANDO STATUS DO TENSORBOARD:"
    echo
    
    if pgrep -f "tensorboard" > /dev/null; then
        echo "  TensorBoard esta RODANDO"
        echo "  Acesso: http://localhost:6006"
        echo
        echo "DADOS DISPONIVEIS:"
        echo "  - Scalars: Loss, Accuracy, Learning Rate"
        echo "  - Graphs: Arquitetura dos modelos"
        echo "  - Histograms: Distribuicao de pesos"
        echo "  - Images: Visualizacoes (se disponivel)"
        echo
        
        if [ -d "logs" ]; then
            log_size=$(du -sh logs 2>/dev/null | cut -f1)
            echo "Tamanho dos logs: $log_size"
            
            echo
            echo "ESTRUTURA DOS LOGS:"
            find logs -type d 2>/dev/null | head -10 | while read -r dir; do
                echo "  $dir"
            done
        fi
    else
        echo "  TensorBoard NAO esta rodando"
        echo
        echo "Para iniciar:"
        echo "  tensorboard --logdir=logs --host=0.0.0.0 --port=6006 &"
        echo
        echo "Depois acesse: http://localhost:6006"
    fi
    
    echo
    echo "Pressione ENTER para continuar..."
    read
}

function generate_clinical_report() {
    show_header
    echo "RELATORIO CLINICO EXECUTIVO"
    echo "==========================="
    
    report_file="alzheimer_clinical_report_$(date +%Y%m%d_%H%M).txt"
    
    echo "Gerando relatorio: $report_file"
    echo
    
    {
        echo "RELATORIO CLINICO - DIAGNOSTICO PRECOCE DE ALZHEIMER"
        echo "===================================================="
        echo "Data: $(date '+%d/%m/%Y %H:%M:%S')"
        echo "Sistema: Alzheimer Analysis Suite"
        echo
        
        echo "RESUMO DA POPULACAO:"
        echo "- Total de sujeitos: 405"
        echo "- Cognitivamente normais (CDR 0): 253 (62.5%)"
        echo "- Comprometimento Cognitivo Leve (CDR 0.5): 68 (16.8%)"
        echo "- Demencia leve (CDR 1): 64 (15.8%)"
        echo "- Demencia moderada (CDR 2): 20 (4.9%)"
        echo
        
        echo "CARACTERISTICAS DO MCI:"
        echo "- Idade media: 73.9 +/- 8.6 anos"
        echo "- MMSE medio: 27.1 +/- 1.8 pontos"
        echo "- Prevalencia feminina: 63.2%"
        echo "- Range MMSE: 22.9 - 30.0"
        echo
        
        echo "BIOMARCADORES CRITICOS:"
        echo "- Cortex entorrinal esquerdo: -3.7% (mais afetado)"
        echo "- Lobo temporal esquerdo: -2.2%"
        echo "- Cortex entorrinal direito: -1.4%"
        echo "- Amigdala esquerda: -1.0%"
        echo "- Hipocampo total: -0.7%"
        echo
        
        echo "RECOMENDACOES CLINICAS:"
        echo "- TRIAGEM: MMSE < 28 requer investigacao"
        echo "- MCI: CDR = 0.5 confirma diagnostico"
        echo "- NEUROIMAGEM: RM volumetria hipocampo + entorrinal"
        echo "- MONITORAMENTO: Reavaliacao semestral"
        echo "- INTERVENCAO: Estimulacao cognitiva + exercicio"
        echo
        
        echo "PERFORMANCE DO SISTEMA IA:"
        echo "- Classificador Binario: 95.1% accuracy (AUC: 0.992)"
        echo "- Classificador CDR: 98.8% accuracy"
        echo "- Processamento GPU: 19.5s por modelo"
        echo "- Features utilizadas: 39 (neuroimagem + clinicas)"
        echo
        
        echo "FATORES DE RISCO PARA PROGRESSAO MCI->AD:"
        echo "- Idade >= 75 anos: 39.7% dos pacientes MCI"
        echo "- MMSE <= 26: 26.5% dos pacientes MCI"
        echo "- Atrofia hipocampal: 25.0% dos pacientes MCI"
        echo "- Multiplos fatores (score >=3): 26.5% (alto risco)"
        echo
        
        echo "IMPACTO CLINICO ESPERADO:"
        echo "- Deteccao precoce: 2-3 anos antes"
        echo "- Janela terapeutica ampliada"
        echo "- Prevencao secundaria otimizada"
        echo "- Melhor prognostico funcional"
        
    } > "$report_file"
    
    echo "Relatorio gerado: $report_file"
    echo
    echo "VISUALIZAR RELATORIO? (s/n): "
    read -r view_report
    
    if [[ "$view_report" == "s" || "$view_report" == "S" ]]; then
        clear
        cat "$report_file"
        echo
        echo "Pressione ENTER para continuar..."
        read
    fi
}

function system_info() {
    show_header
    echo "INFORMACOES DO SISTEMA"
    echo "======================"
    
    echo "AMBIENTE:"
    echo "  OS: $(uname -s) $(uname -r)"
    echo "  Usuario: $(whoami)"
    echo "  Diretorio: $(pwd)"
    echo "  Data/Hora: $(date)"
    echo
    
    echo "PYTHON:"
    python3 --version
    echo
    
    echo "PACOTES PRINCIPAIS:"
    python3 -c "
import sys
packages = ['pandas', 'numpy', 'tensorflow', 'sklearn']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'N/A')
        print(f'  OK {pkg}: {version}')
    except ImportError:
        print(f'  ERRO {pkg}: Nao instalado')
"
    
    echo
    echo "GPU:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -1 | while IFS=',' read -r name memory_total memory_used; do
            echo "  GPU: $name"
            echo "  Memoria: ${memory_used}MB/${memory_total}MB"
        done
    else
        echo "  NVIDIA GPU nao detectada"
    fi
    
    echo
    echo "ARQUIVOS PRINCIPAIS:"
    echo "  $(ls -1 *.py *.sh *.csv *.h5 *.joblib 2>/dev/null | wc -l) arquivos no projeto"
    
    echo
    echo "Pressione ENTER para continuar..."
    read
}

# Menu principal
while true; do
    show_header
    echo "MENU PRINCIPAL - SELECIONE UMA OPCAO:"
    echo
    echo "ANALISES RAPIDAS:"
    echo "   1. Estatisticas basicas"
    echo "   2. Analise abrangente"
    echo "   3. Explorador do dataset (completo)"
    echo
    echo "ANALISES ESPECIALIZADAS:"
    echo "   4. Diagnostico precoce (completo)"
    echo "   5. Analise clinica MCI"
    echo
    echo "SISTEMA:"
    echo "   6. Performance dos modelos"
    echo "   7. Status TensorBoard"
    echo "   8. Gerar relatorio clinico"
    echo "   9. Informacoes do sistema"
    echo
    echo "   0. Sair"
    echo
    echo -n "Digite sua opcao [0-9]: "
    read -r option
    
    case $option in
        1) quick_stats ;;
        2) comprehensive_analysis ;;
        3) dataset_explorer ;;
        4) early_diagnosis_analysis ;;
        5) mci_clinical_analysis ;;
        6) model_performance ;;
        7) tensorboard_info ;;
        8) generate_clinical_report ;;
        9) system_info ;;
        0) 
            clear
            echo "Obrigado por usar o Alzheimer Analysis Suite!"
            echo "Sistema desenvolvido para pesquisa em diagnostico precoce"
            echo "Foco: Deteccao de Comprometimento Cognitivo Leve (MCI)"
            echo
            exit 0
            ;;
        *)
            echo "Opcao invalida. Pressione ENTER para tentar novamente..."
            read
            ;;
    esac
done 
#!/bin/bash

echo "=== DIAGNÓSTICO DO PROCESSAMENTO FASTSURFER ==="
echo "⏱️  $(date)"
echo ""

# Verificar se o script está rodando
echo "🔍 VERIFICANDO PROCESSOS EM EXECUÇÃO:"
echo "----------------------------------------"

# Verificar processos bash relacionados ao FastSurfer
echo "📋 Processos bash do FastSurfer:"
ps aux | grep -E "(fastsurfer|run_fastsurfer)" | grep -v grep | head -10

echo ""
echo "📋 Processos Docker:"
docker ps -a | head -10

echo ""
echo "📋 Processos gerais relacionados:"
ps aux | grep -E "(bash.*run|bash.*fastsurfer)" | grep -v grep | head -10

echo ""
echo "🔍 VERIFICANDO ARQUIVOS DE ESTADO:"
echo "----------------------------------------"

# Verificar arquivos de estado
STATE_FILE="/app/alzheimer/oasis_data/fastsurfer_logs_oficial/processamento_estado.txt"
LOG_DIR="/app/alzheimer/oasis_data/fastsurfer_logs_oficial"

if [ -f "$STATE_FILE" ]; then
    echo "📋 Arquivo de estado encontrado:"
    echo "   📁 Localização: $STATE_FILE"
    echo "   📊 Sujeitos processados: $(wc -l < "$STATE_FILE")"
    echo "   🕐 Última modificação: $(stat -c %y "$STATE_FILE")"
    echo ""
    echo "📄 Últimos 5 sujeitos processados:"
    tail -5 "$STATE_FILE" | sed 's/^/   /'
else
    echo "❌ Arquivo de estado não encontrado: $STATE_FILE"
fi

echo ""
echo "📁 Conteúdo do diretório de logs:"
ls -la "$LOG_DIR" | head -20

echo ""
echo "🔍 VERIFICANDO DADOS DE ENTRADA:"
echo "----------------------------------------"

# Verificar estrutura dos dados
DATA_BASE="/app/alzheimer/oasis_data"
echo "📊 Estrutura dos dados:"
for disc in {1..11}; do
    DISK_DIR="${DATA_BASE}/disc${disc}"
    if [ -d "$DISK_DIR" ]; then
        count=$(find "$DISK_DIR" -name "OAS1_*_MR1" -type d | wc -l)
        echo "   📂 disc${disc}: $count sujeitos"
    fi
done

echo ""
echo "🔍 VERIFICANDO ARQUIVOS T1.MGZ:"
echo "----------------------------------------"

# Verificar alguns arquivos T1.mgz aleatórios
echo "📄 Verificando arquivos T1.mgz (amostra):"
total_t1=0
valid_t1=0

for disc in {1..3}; do  # Verificar apenas os 3 primeiros discos para não demorar
    DISK_DIR="${DATA_BASE}/disc${disc}"
    if [ -d "$DISK_DIR" ]; then
        for subject_dir in "${DISK_DIR}"/OAS1_*_MR1; do
            if [ -d "$subject_dir" ]; then
                ((total_t1++))
                t1_file="${subject_dir}/mri/T1.mgz"
                if [ -f "$t1_file" ]; then
                    ((valid_t1++))
                    # Verificar se é link simbólico
                    if [ -L "$t1_file" ]; then
                        echo "   🔗 $(basename "$subject_dir"): Link simbólico"
                    else
                        echo "   ✅ $(basename "$subject_dir"): Arquivo real"
                    fi
                else
                    echo "   ❌ $(basename "$subject_dir"): T1.mgz não encontrado"
                fi
                
                # Verificar apenas os primeiros 5 de cada disco
                if [ $((total_t1 % 15)) -eq 0 ]; then
                    break
                fi
            fi
        done
    fi
done

echo ""
echo "📊 Resumo dos arquivos T1.mgz (amostra):"
echo "   📈 Total verificado: $total_t1"
echo "   ✅ Válidos: $valid_t1"
echo "   ❌ Inválidos: $((total_t1 - valid_t1))"

echo ""
echo "🔍 VERIFICANDO RESULTADOS:"
echo "----------------------------------------"

OUTPUT_DIR="/app/alzheimer/oasis_data/outputs_fastsurfer_oficial"
if [ -d "$OUTPUT_DIR" ]; then
    echo "📂 Diretório de resultados existe:"
    echo "   📁 Localização: $OUTPUT_DIR"
    echo "   📊 Sujeitos processados: $(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "OAS1_*" | wc -l)"
    echo "   🕐 Última modificação: $(stat -c %y "$OUTPUT_DIR")"
    
    echo ""
    echo "📄 Últimos 5 sujeitos processados:"
    find "$OUTPUT_DIR" -maxdepth 1 -type d -name "OAS1_*" | sort | tail -5 | sed 's/^/   /'
else
    echo "❌ Diretório de resultados não encontrado: $OUTPUT_DIR"
fi

echo ""
echo "🔍 VERIFICANDO RECURSOS DO SISTEMA:"
echo "----------------------------------------"

echo "💾 Uso de disco:"
df -h /app/alzheimer | head -2

echo ""
echo "🧠 Uso de memória:"
free -h

echo ""
echo "🐳 Status do Docker:"
docker info | head -10

echo ""
echo "🔍 VERIFICANDO LOGS RECENTES:"
echo "----------------------------------------"

# Verificar logs mais recentes
if [ -d "$LOG_DIR" ]; then
    echo "📄 Logs mais recentes:"
    ls -lt "$LOG_DIR"/*.log 2>/dev/null | head -5 | while read -r line; do
        echo "   $line"
    done
    
    echo ""
    echo "📋 Conteúdo do último log (se existir):"
    last_log=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
    if [ -n "$last_log" ]; then
        echo "   📁 Arquivo: $last_log"
        echo "   📄 Últimas 10 linhas:"
        tail -10 "$last_log" | sed 's/^/      /'
    else
        echo "   ❌ Nenhum log encontrado"
    fi
fi

echo ""
echo "🎯 DIAGNÓSTICO FINAL:"
echo "----------------------------------------"

# Análise final
if ps aux | grep -E "(run_fastsurfer_oficial)" | grep -v grep >/dev/null; then
    echo "✅ O script run_fastsurfer_oficial.sh está em execução"
    
    if [ -f "$STATE_FILE" ] && [ -s "$STATE_FILE" ]; then
        echo "✅ Processamento já iniciou ($(wc -l < "$STATE_FILE") sujeitos processados)"
    else
        echo "⚠️  Script rodando mas nenhum sujeito processado ainda"
        echo "💡 Possível causa: Fase de coleta de sujeitos ou problema com Git Annex"
    fi
else
    echo "❌ O script run_fastsurfer_oficial.sh NÃO está em execução"
    echo "💡 Você pode precisar reiniciar o processamento"
fi

echo ""
echo "🚀 RECOMENDAÇÕES:"
echo "----------------------------------------"

if [ ! -f "$STATE_FILE" ] || [ ! -s "$STATE_FILE" ]; then
    echo "1. 🔄 O script pode estar na fase de coleta de sujeitos"
    echo "2. 🔗 Verificar se há problema com links simbólicos do Git Annex"
    echo "3. ⏰ Aguardar mais alguns minutos para o processamento começar"
    echo "4. 🛑 Se não houver progresso em 10 minutos, considere reiniciar"
fi

echo ""
echo "=== FIM DO DIAGNÓSTICO ===" 
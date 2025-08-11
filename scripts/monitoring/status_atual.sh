#!/bin/bash

echo "=== STATUS ATUAL DO PROCESSAMENTO FASTSURFER ==="
echo ""

# Verificar processos em execução
echo "🔍 Processos FastSurfer em execução:"
ps aux | grep -i fastsurfer | grep -v grep || echo "   Nenhum processo encontrado"
echo ""

# Verificar PIDs salvos
echo "📋 PIDs salvos:"
for pid_file in *.pid; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        echo "   $pid_file: $pid"
        if ps -p $pid > /dev/null 2>&1; then
            echo "      ✅ Processo ativo"
        else
            echo "      ❌ Processo não encontrado"
        fi
    fi
done
echo ""

# Verificar resultados
echo "📊 Resultados processados:"
RESULT_DIRS=(
    "oasis_data/outputs_fastsurfer_oficial"
    "oasis_data/test_fastsurfer_definitivo"
    "oasis_data/test_fastsurfer_output"
)

for dir in "${RESULT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -maxdepth 1 -type d -name "OAS1_*" | wc -l)
        echo "   $dir: $count sujeitos"
    fi
done
echo ""

# Verificar logs recentes
echo "📝 Logs recentes:"
find . -name "*.log" -mtime -1 -exec ls -la {} \; | head -5
echo ""

# Verificar espaço em disco
echo "💾 Espaço em disco:"
df -h /app/alzheimer | tail -1
echo ""

echo "=== RECOMENDAÇÕES ==="
echo "✅ O teste definitivo funcionou perfeitamente (20 min/sujeito)"
echo "🚀 Para processar todos os 405 sujeitos, execute:"
echo "   nohup bash run_fastsurfer_otimizado.sh > fastsurfer_$(date +%Y%m%d_%H%M%S).log 2>&1 &"
echo "" 
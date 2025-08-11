#!/bin/bash

echo "=== Correção do Processamento ReconAll ==="

# Configurações
DATA_BASE="/app/alzheimer/oasis_data"
LOG_DIR="${DATA_BASE}/processing_logs"

echo "1. Analisando logs atuais..."

# Verificar processos em andamento
echo "Processos recon-all em execução:"
ps aux | grep recon-all | grep -v grep || echo "Nenhum processo recon-all encontrado"

echo ""
echo "2. Analisando logs de erro..."

# Analisar logs mais recentes
for log_file in "${LOG_DIR}"/*.log; do
    if [ -f "$log_file" ]; then
        subject=$(basename "$log_file" .log)
        echo "Analisando: $subject"
        
        # Verificar se terminou com sucesso
        if tail -20 "$log_file" | grep -q "recon-all finished"; then
            echo "  ✅ $subject: SUCESSO"
        elif tail -20 "$log_file" | grep -q "ERROR\|Error\|FATAL\|Fatal"; then
            echo "  ❌ $subject: ERRO"
            echo "  Últimos erros:"
            tail -20 "$log_file" | grep -i "error\|fatal" | head -3
        else
            echo "  ⏳ $subject: EM ANDAMENTO ou INCOMPLETO"
            # Verificar há quanto tempo não há atividade
            last_log_time=$(stat -c %Y "$log_file")
            current_time=$(date +%s)
            time_diff=$((current_time - last_log_time))
            
            if [ $time_diff -gt 3600 ]; then  # Mais de 1 hora sem atividade
                echo "     ⚠️  Sem atividade há $(($time_diff/3600)) horas - pode estar travado"
            fi
        fi
        echo ""
    fi
done

echo "3. Recomendações de correção:"
echo ""

# PROBLEMA 1: Lentidão do FreeSurfer tradicional
echo "🔧 PROBLEMA IDENTIFICADO: FreeSurfer tradicional é muito lento"
echo "   SOLUÇÃO: Migrar para FastSurfer"
echo ""

# Criar script otimizado usando FastSurfer
cat > run_fastsurfer_optimized.sh << 'EOF'
#!/bin/bash

# Script otimizado usando FastSurfer
DATA_BASE="/app/alzheimer/oasis_data"
OUTPUT_DIR="${DATA_BASE}/outputs_fastsurfer"
LOG_DIR="${DATA_BASE}/fastsurfer_logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Função para processar um sujeito com FastSurfer
process_with_fastsurfer() {
    local subject_dir=$1
    local subject=$(basename "$subject_dir")
    local t1_file="${subject_dir}/mri/T1.mgz"
    
    echo "Processando $subject com FastSurfer..."
    
    if [ ! -f "$t1_file" ]; then
        echo "❌ Arquivo T1.mgz não encontrado para $subject"
        return 1
    fi
    
    # Executar FastSurfer
    docker run --gpus all --rm \
        -v "${subject_dir}:/input" \
        -v "${OUTPUT_DIR}:/output" \
<<<<<<< HEAD
        -v "$HOME/license.txt:/license.txt" \
=======
        -v "/home/comaisserveria/license.txt:/license.txt" \
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
        deepmi/fastsurfer:latest \
        --fs_license /license.txt \
        --t1 /input/mri/T1.mgz \
        --sid "${subject}" \
        --sd /output \
        --parallel \
        --threads 4 \
        2>&1 | tee "${LOG_DIR}/${subject}_fastsurfer.log"
    
    if [ $? -eq 0 ]; then
        echo "✅ $subject processado com sucesso"
        return 0
    else
        echo "❌ Erro ao processar $subject"
        return 1
    fi
}

# Processar todos os sujeitos pendentes
echo "Iniciando processamento com FastSurfer..."
for disc in {1..11}; do
    DISK_DIR="${DATA_BASE}/disc${disc}"
    [ -d "$DISK_DIR" ] || continue
    
    for subject_dir in "${DISK_DIR}"/OAS1_*_MR1; do
        [ -d "$subject_dir" ] || continue
        process_with_fastsurfer "$subject_dir"
    done
done
EOF

chmod +x run_fastsurfer_optimized.sh

echo "✅ Script otimizado criado: run_fastsurfer_optimized.sh"
echo ""

# PROBLEMA 2: Configuração de ambiente inconsistente
echo "🔧 PROBLEMA: Configuração de ambiente inconsistente"
echo "   SOLUÇÃO: Padronizar variáveis de ambiente"
echo ""

# Criar script de configuração
cat > fix_environment.sh << 'EOF'
#!/bin/bash

echo "Corrigindo configuração do ambiente..."

# Definir FreeSurfer Home baseado na instalação atual
<<<<<<< HEAD
if [ -d "$HOME/freesurfer/freesurfer" ]; then
    export FREESURFER_HOME="$HOME/freesurfer/freesurfer"
=======
if [ -d "/home/comaisserveria/freesurfer/freesurfer" ]; then
    export FREESURFER_HOME="/home/comaisserveria/freesurfer/freesurfer"
>>>>>>> 3f8bd3ee87 (Add new processing scripts and documentation)
elif [ -d "/usr/local/freesurfer" ]; then
    export FREESURFER_HOME="/usr/local/freesurfer"
else
    echo "❌ FreeSurfer não encontrado!"
    exit 1
fi

export PATH="$FREESURFER_HOME/bin:$PATH"
export SUBJECTS_DIR="/app/alzheimer/oasis_data/subjects"

# Criar diretório de sujeitos se não existir
mkdir -p "$SUBJECTS_DIR"

echo "✅ Ambiente configurado:"
echo "   FREESURFER_HOME: $FREESURFER_HOME"
echo "   SUBJECTS_DIR: $SUBJECTS_DIR"
echo "   PATH: Atualizado"

# Salvar configuração
echo "export FREESURFER_HOME=\"$FREESURFER_HOME\"" > freesurfer_env.sh
echo "export PATH=\"\$FREESURFER_HOME/bin:\$PATH\"" >> freesurfer_env.sh
echo "export SUBJECTS_DIR=\"/app/alzheimer/oasis_data/subjects\"" >> freesurfer_env.sh

echo "✅ Configuração salva em: freesurfer_env.sh"
echo "   Para usar: source freesurfer_env.sh"
EOF

chmod +x fix_environment.sh

echo "✅ Script de correção de ambiente criado: fix_environment.sh"
echo ""

# PROBLEMA 3: Processos travados
echo "🔧 PROBLEMA: Possíveis processos travados"
echo "   SOLUÇÃO: Verificar e finalizar processos problemáticos"
echo ""

cat > kill_stuck_processes.sh << 'EOF'
#!/bin/bash

echo "Verificando processos do FreeSurfer..."

# Listar processos relacionados ao FreeSurfer
FREESURFER_PIDS=$(ps aux | grep -E "(recon-all|mri_|fs_)" | grep -v grep | awk '{print $2}')

if [ -n "$FREESURFER_PIDS" ]; then
    echo "Processos encontrados:"
    ps aux | grep -E "(recon-all|mri_|fs_)" | grep -v grep
    echo ""
    echo "Para finalizar todos os processos, execute:"
    echo "kill -9 $FREESURFER_PIDS"
    echo ""
    echo "⚠️  CUIDADO: Isso irá interromper todos os processamentos em andamento!"
else
    echo "✅ Nenhum processo do FreeSurfer em execução"
fi
EOF

chmod +x kill_stuck_processes.sh

echo "✅ Script para finalizar processos criado: kill_stuck_processes.sh"
echo ""

echo "=== RESUMO DAS SOLUÇÕES ==="
echo ""
echo "1. 🚀 USAR FASTSURFER (RECOMENDADO):"
echo "   ./run_fastsurfer_optimized.sh"
echo ""
echo "2. 🔧 CORRIGIR AMBIENTE:"
echo "   ./fix_environment.sh"
echo ""
echo "3. 🛑 FINALIZAR PROCESSOS TRAVADOS:"
echo "   ./kill_stuck_processes.sh"
echo ""
echo "4. 📊 VERIFICAR STATUS:"
echo "   ./setup_freesurfer.sh"
echo ""
echo "=== PRÓXIMOS PASSOS ==="
echo "1. Execute: ./fix_environment.sh"
echo "2. Se necessário: ./kill_stuck_processes.sh"
echo "3. Use FastSurfer: ./run_fastsurfer_optimized.sh"
echo ""
echo "✅ FastSurfer é ~10x mais rápido que o FreeSurfer tradicional!" 
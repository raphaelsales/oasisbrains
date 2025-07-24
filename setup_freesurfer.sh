#!/bin/bash

echo "=== Configuração do FreeSurfer ==="

# 1. VERIFICAR INSTALAÇÃO ATUAL
echo "1. Verificando instalação atual..."
echo "Procurando FreeSurfer..."

# Locais comuns onde o FreeSurfer pode estar
FREESURFER_PATHS=(
    "/usr/local/freesurfer"
    "$HOME/freesurfer/freesurfer"
    "/opt/freesurfer"
    "/usr/share/freesurfer"
)

FOUND_FREESURFER=""
for path in "${FREESURFER_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/bin/recon-all" ]; then
        echo "✅ FreeSurfer encontrado em: $path"
        FOUND_FREESURFER="$path"
        break
    fi
done

if [ -z "$FOUND_FREESURFER" ]; then
    echo "❌ FreeSurfer não encontrado nos locais padrão"
    echo "Procurando em todo o sistema..."
    find /usr /opt /home -name "recon-all" -type f 2>/dev/null | head -5
    exit 1
fi

# 2. CONFIGURAR VARIÁVEIS DE AMBIENTE
echo "2. Configurando variáveis de ambiente..."
export FREESURFER_HOME="$FOUND_FREESURFER"
export PATH="$FREESURFER_HOME/bin:$PATH"
export SUBJECTS_DIR="$FREESURFER_HOME/subjects"

# 3. VERIFICAR LICENÇA
echo "3. Verificando licença..."
LICENSE_LOCATIONS=(
    "$FREESURFER_HOME/license.txt"
    "$HOME/license.txt"
    "./license.txt"
)

FOUND_LICENSE=""
for license in "${LICENSE_LOCATIONS[@]}"; do
    if [ -f "$license" ]; then
        echo "✅ Licença encontrada em: $license"
        FOUND_LICENSE="$license"
        break
    fi
done

if [ -z "$FOUND_LICENSE" ]; then
    echo "⚠️  Licença não encontrada. Você precisa registrar no FreeSurfer e baixar o license.txt"
    echo "Registre-se em: https://surfer.nmr.mgh.harvard.edu/registration.html"
fi

# 4. TESTAR INSTALAÇÃO
echo "4. Testando instalação..."
echo "Versão do FreeSurfer:"
"$FREESURFER_HOME/bin/freesurfer" --version 2>/dev/null || echo "Comando 'freesurfer --version' falhou"

echo "Testando recon-all:"
"$FREESURFER_HOME/bin/recon-all" -version 2>/dev/null || echo "Comando 'recon-all -version' falhou"

# 5. GERAR ARQUIVO DE CONFIGURAÇÃO
echo "5. Gerando arquivo de configuração..."
cat > freesurfer_config.sh << EOF
#!/bin/bash
# Configuração do FreeSurfer
export FREESURFER_HOME="$FOUND_FREESURFER"
export PATH="\$FREESURFER_HOME/bin:\$PATH"
export SUBJECTS_DIR="\$FREESURFER_HOME/subjects"

# Adicionar ao ~/.bashrc se ainda não estiver lá
if ! grep -q "FREESURFER_HOME" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# FreeSurfer Configuration" >> ~/.bashrc
    echo "export FREESURFER_HOME=\"$FOUND_FREESURFER\"" >> ~/.bashrc
    echo "export PATH=\"\\\$FREESURFER_HOME/bin:\\\$PATH\"" >> ~/.bashrc
    echo "export SUBJECTS_DIR=\"\\\$FREESURFER_HOME/subjects\"" >> ~/.bashrc
fi
EOF

chmod +x freesurfer_config.sh

# 6. DIAGNÓSTICO DE PERFORMANCE
echo "6. Diagnóstico de performance..."
echo "Verificando recursos do sistema:"
echo "CPU cores: $(nproc)"
echo "Memória RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "Espaço em disco: $(df -h . | tail -1 | awk '{print $4}')"

# 7. RECOMENDAÇÕES
echo ""
echo "=== RECOMENDAÇÕES ==="
echo "✅ Para usar FastSurfer (mais rápido):"
echo "   docker run --gpus all --rm deepmi/fastsurfer:latest"
echo ""
echo "✅ Para usar FreeSurfer tradicional:"
echo "   source ./freesurfer_config.sh"
echo "   recon-all -s subject_name -i input.mgz -all"
echo ""
echo "✅ Para acelerar o processamento:"
echo "   - Use FastSurfer em vez do FreeSurfer tradicional"
echo "   - Use GPU se disponível"
echo "   - Processe múltiplos sujeitos em paralelo (mas limite para evitar sobrecarga)"
echo ""
echo "⚠️  PROBLEMAS IDENTIFICADOS:"
if [ -z "$FOUND_LICENSE" ]; then
    echo "   - Licença do FreeSurfer não encontrada"
fi
echo "   - FreeSurfer tradicional é muito lento (várias horas por sujeito)"
echo "   - FastSurfer pode reduzir o tempo para ~30-60 minutos por sujeito"
echo ""
echo "=== CONFIGURAÇÃO CONCLUÍDA ===" 
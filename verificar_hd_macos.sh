#!/bin/bash

echo "=== VERIFICAÇÃO DO HD EXTERNO PARA macOS ==="
echo

# Verificar se o HD está montado
if mountpoint -q /mnt/hd_externo; then
    echo "✅ HD externo está montado"
else
    echo "❌ HD externo não está montado"
    exit 1
fi

echo
echo "=== ESTRUTURA DOS DADOS ==="
echo "Diretório raiz do HD:"
ls -la /mnt/hd_externo/

echo
echo "Diretório OASIS:"
ls -la /mnt/hd_externo/oasis_data/

echo
echo "=== VERIFICAÇÃO DOS ARQUIVOS T1.mgz ==="
echo "Total de arquivos T1.mgz encontrados:"
find /mnt/hd_externo/oasis_data/disc*_real -name "T1.mgz" | wc -l

echo
echo "Primeiros 5 arquivos T1.mgz:"
find /mnt/hd_externo/oasis_data/disc*_real -name "T1.mgz" | head -5

echo
echo "=== VERIFICAÇÃO DE PERMISSÕES ==="
echo "Permissões de um arquivo T1.mgz:"
ls -la /mnt/hd_externo/oasis_data/disc1_real/OAS1_0001_MR1/mri/T1.mgz

echo
echo "=== VERIFICAÇÃO DE INTEGRIDADE ==="
echo "Testando leitura de um arquivo T1.mgz:"
file /mnt/hd_externo/oasis_data/disc1_real/OAS1_0001_MR1/mri/T1.mgz

echo
echo "=== INSTRUÇÕES PARA macOS ==="
echo "1. Conecte o HD no macOS"
echo "2. Abra o Terminal"
echo "3. Execute: diskutil list"
echo "4. Procure o HD (geralmente /dev/disk2 ou similar)"
echo "5. Execute: find /Volumes/HD_NAME -name 'T1.mgz' | head -5"
echo
echo "Se não encontrar os arquivos, pode ser um problema de:"
echo "- Sistema de arquivos não suportado"
echo "- HD não montado corretamente"
echo "- Arquivos ocultos" 
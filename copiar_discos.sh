#!/bin/bash

# Script para copiar todos os discos OASIS para o HD externo
# Usando cp -rL para copiar os arquivos reais (não links simbólicos)

echo "Iniciando cópia dos 11 discos para o HD externo..."
echo "Destino: /mnt/hd_externo/oasis_data/"

# Criar diretório de destino se não existir
sudo mkdir -p /mnt/hd_externo/oasis_data

# Copiar cada disco
for i in {1..11}; do
    echo "Copiando disc$i..."
    sudo cp -rL oasis_data/disc$i /mnt/hd_externo/oasis_data/disc${i}_real
    echo "Disc$i copiado com sucesso!"
done

echo "Cópia concluída! Verificando tamanhos..."

# Verificar tamanhos dos diretórios copiados
for i in {1..11}; do
    size=$(du -sh /mnt/hd_externo/oasis_data/disc${i}_real | cut -f1)
    echo "Disc${i}_real: $size"
done

echo "Todos os discos foram copiados para /mnt/hd_externo/oasis_data/" 
#!/bin/bash

# Script para diagnosticar e corrigir problemas do Git Annex com FastSurfer
# Uso: ./fix_git_annex.sh

echo "Verificando status do Git Annex..."

# Verificar se estamos em um repositório Git Annex
if ! git annex version &>/dev/null; then
    echo "Este não é um repositório Git Annex"
    exit 1
fi

echo "Repositório Git Annex detectado"

# Verificar status do annex
echo
echo "STATUS ATUAL DO GIT ANNEX:"
git annex info 2>/dev/null | head -10

echo
echo "VERIFICANDO ARQUIVOS T1.MGZ:"

# Encontrar arquivos T1.mgz (devem ser links simbólicos no Git Annex)
T1_FILES=$(find . -name "T1.mgz" -type l 2>/dev/null)

if [ -z "$T1_FILES" ]; then
    echo "Nenhum arquivo T1.mgz (link simbólico) encontrado"
    exit 1
fi

echo "Arquivos T1.mgz encontrados:"
echo "$T1_FILES" | head -5

echo
echo "ANALISANDO PRIMEIRO ARQUIVO:"
FIRST_T1=$(echo "$T1_FILES" | head -1)
echo "Arquivo: $FIRST_T1"

# Verificar se o arquivo está no annex
echo "Status no Git Annex:"
if git annex find "$FIRST_T1" &>/dev/null; then
    echo "Arquivo está rastreado pelo Git Annex"
else
    echo "Arquivo não está no Git Annex ou há problema"
fi

# Verificar onde o arquivo está
git annex whereis "$FIRST_T1" 2>/dev/null || echo "Arquivo não rastreado pelo Git Annex"

echo
echo "OPÇÕES DE CORREÇÃO:"
echo "1. unlock - Converte links simbólicos em arquivos regulares"
echo "2. get - Baixa arquivos do repositório remoto (se disponível)"
echo "3. lock - Converte arquivos regulares de volta para links simbólicos"
echo "0. Sair"

read -p "Digite sua opção (0-3): " opcao

case $opcao in
    1)
        echo "Realizando unlock dos arquivos T1.mgz..."
        echo "Isso pode demorar alguns minutos..."
        
        unlocked_count=0
        while IFS= read -r t1_file; do
            echo "Processando: $t1_file"
            if git annex unlock "$t1_file" 2>/dev/null; then
                if [ ! -L "$t1_file" ]; then
                    echo "  Unlock realizado com sucesso"
                    unlocked_count=$((unlocked_count + 1))
                    
                    echo "  Arquivo convertido para regular"
                    echo "  Tamanho: $(du -h "$t1_file" | cut -f1)"
                else
                    echo "  Ainda é link simbólico"
                fi
            else
                echo "  Falha no unlock"
            fi
        done <<< "$T1_FILES"
        
        echo "UNLOCK CONCLUÍDO!"
        echo "Agora os arquivos T1.mgz são arquivos regulares"
        echo "Teste o FastSurfer novamente: ./test_fastsurfer_annex.sh"
        ;;
    
    2)
        echo "Tentando baixar arquivos do repositório remoto..."
        echo "Isso pode demorar bastante dependendo do tamanho..."
        
        while IFS= read -r t1_file; do
            echo "Baixando: $t1_file"
            if git annex get "$t1_file" 2>/dev/null; then
                echo "  Download realizado com sucesso"
            else
                echo "  Falha no download (pode não estar disponível remotamente)"
            fi
        done <<< "$T1_FILES"
        
        echo "DOWNLOAD CONCLUÍDO!"
        echo "Arquivos baixados (se disponíveis)"
        echo "Teste o FastSurfer novamente: ./test_fastsurfer_annex.sh"
        ;;
    
    3)
        echo "Realizando lock dos arquivos T1.mgz..."
        echo "Convertendo arquivos regulares para links simbólicos..."
        
        while IFS= read -r t1_file; do
            echo "Processando: $t1_file"
            if git annex lock "$t1_file" 2>/dev/null; then
                echo "  Lock realizado com sucesso"
            else
                echo "  Falha no lock"
            fi
        done <<< "$T1_FILES"
        
        echo "LOCK CONCLUÍDO!"
        ;;
    
    0)
        echo "Opção inválida"
        exit 1
        ;;
esac

echo
echo "STATUS FINAL:"
echo "Verificando primeiro arquivo após correção:"
echo "Arquivo: $FIRST_T1"

if [ -L "$FIRST_T1" ]; then
    echo "Tipo: Link simbólico"
    echo "Aponta para: $(readlink "$FIRST_T1")"
else
    echo "Tipo: Arquivo regular"
    echo "Tamanho: $(du -h "$FIRST_T1" | cut -f1)"
fi

echo
echo "PRÓXIMOS PASSOS:"
echo "1. Execute: ./test_fastsurfer_annex.sh"
echo "2. Se ainda der erro, tente: git annex fsck"
echo "3. Para mais diagnósticos: git annex info" 
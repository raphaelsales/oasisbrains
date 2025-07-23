#!/bin/bash

echo "=== Correção Git Annex - Unlock dos Arquivos T1 ==="

# Configurações
DATA_BASE="/app/alzheimer/oasis_data"

echo "🔍 Verificando status do Git Annex..."

# Verificar se estamos em um repositório Git Annex
if [ ! -d "/app/alzheimer/.git/annex" ]; then
    echo "❌ Este não é um repositório Git Annex"
    exit 1
fi

echo "✅ Repositório Git Annex detectado"

# Navegar para o diretório raiz
cd /app/alzheimer

echo ""
echo "📊 STATUS ATUAL DO GIT ANNEX:"
git annex status | head -10

echo ""
echo "🔍 VERIFICANDO ARQUIVOS T1.MGZ:"

# Encontrar todos os arquivos T1.mgz
T1_FILES=$(find "$DATA_BASE" -name "T1.mgz" -type l | head -5)

if [ -z "$T1_FILES" ]; then
    echo "❌ Nenhum arquivo T1.mgz (link simbólico) encontrado"
    exit 1
fi

echo "✅ Arquivos T1.mgz encontrados:"
echo "$T1_FILES"

echo ""
echo "📋 ANALISANDO PRIMEIRO ARQUIVO:"
FIRST_T1=$(echo "$T1_FILES" | head -1)
echo "📁 Arquivo: $FIRST_T1"

# Verificar informações do arquivo
echo "🔗 Link aponta para: $(readlink "$FIRST_T1")"
echo "📄 Arquivo real: $(readlink -f "$FIRST_T1")"

# Verificar onde está o arquivo no Git Annex
echo ""
echo "🌐 LOCALIZAÇÃO NO GIT ANNEX:"
git annex whereis "$FIRST_T1" 2>/dev/null || echo "⚠️  Arquivo não rastreado pelo Git Annex"

echo ""
echo "🔧 OPÇÕES DE CORREÇÃO:"
echo ""
echo "1. 🔓 UNLOCK (Converter links em arquivos regulares)"
echo "2. 📥 GET (Baixar arquivos do repositório remoto)"
echo "3. 🔒 LOCK (Converter de volta para links)"

echo ""
read -p "Escolha uma opção (1-3): " opcao

case $opcao in
    1)
        echo ""
        echo "🔓 FAZENDO UNLOCK DOS ARQUIVOS T1.MGZ..."
        
        # Contador
        count=0
        total=$(echo "$T1_FILES" | wc -l)
        
        echo "$T1_FILES" | while read -r t1_file; do
            count=$((count + 1))
            echo "[$count/$total] Processando: $t1_file"
            
            # Fazer unlock do arquivo
            if git annex unlock "$t1_file"; then
                echo "  ✅ Unlock realizado com sucesso"
                
                # Verificar se agora é arquivo regular
                if [ ! -L "$t1_file" ]; then
                    echo "  ✅ Arquivo convertido para regular"
                    echo "  📊 Tamanho: $(du -h "$t1_file" | cut -f1)"
                else
                    echo "  ⚠️  Ainda é link simbólico"
                fi
            else
                echo "  ❌ Falha no unlock"
            fi
            echo ""
        done
        
        echo "🎉 UNLOCK CONCLUÍDO!"
        echo "💡 Agora os arquivos T1.mgz são arquivos regulares"
        echo "🚀 Teste o FastSurfer novamente: ./test_fastsurfer_annex.sh"
        ;;
        
    2)
        echo ""
        echo "📥 BAIXANDO ARQUIVOS DO REPOSITÓRIO REMOTO..."
        
        # Tentar baixar os arquivos
        echo "$T1_FILES" | while read -r t1_file; do
            echo "📥 Baixando: $t1_file"
            
            if git annex get "$t1_file"; then
                echo "  ✅ Download realizado com sucesso"
            else
                echo "  ❌ Falha no download (pode não estar disponível remotamente)"
            fi
        done
        
        echo "🎉 DOWNLOAD CONCLUÍDO!"
        echo "💡 Arquivos baixados (se disponíveis)"
        echo "🚀 Teste o FastSurfer novamente: ./test_fastsurfer_annex.sh"
        ;;
        
    3)
        echo ""
        echo "🔒 FAZENDO LOCK DOS ARQUIVOS (converter para links)..."
        
        echo "$T1_FILES" | while read -r t1_file; do
            echo "🔒 Processando: $t1_file"
            
            if git annex lock "$t1_file"; then
                echo "  ✅ Lock realizado com sucesso"
            else
                echo "  ❌ Falha no lock"
            fi
        done
        
        echo "🎉 LOCK CONCLUÍDO!"
        ;;
        
    *)
        echo "❌ Opção inválida"
        exit 1
        ;;
esac

echo ""
echo "📊 STATUS FINAL:"
echo "🔍 Verificando primeiro arquivo após correção:"
echo "📁 Arquivo: $FIRST_T1"

if [ -L "$FIRST_T1" ]; then
    echo "🔗 Ainda é link simbólico"
    echo "🎯 Aponta para: $(readlink "$FIRST_T1")"
else
    echo "📄 Agora é arquivo regular"
    echo "📊 Tamanho: $(du -h "$FIRST_T1" | cut -f1)"
fi

echo ""
echo "🚀 PRÓXIMOS PASSOS:"
echo "1. Testar FastSurfer: ./test_fastsurfer_annex.sh"
echo "2. Se funcionar, processar todos: ./run_fastsurfer_oficial.sh"
echo ""
echo "=== FIM DA CORREÇÃO GIT ANNEX ===" 
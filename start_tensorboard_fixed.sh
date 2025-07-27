#!/bin/bash

echo "🚀 INICIANDO TENSORBOARD CORRIGIDO PARA CNN 3D"
echo "=============================================="
echo ""
echo "🔧 Configurações aplicadas:"
echo "   - Grafo desabilitado (evita erro de GraphDef)"
echo "   - Histogramas habilitados"
echo "   - Métricas em tempo real"
echo ""
echo "📊 Logs: ./logs_cnn_3d/"
echo "🌐 URL: http://localhost:6006"
echo "⏹️  Parar: Ctrl+C"
echo ""

# Criar diretório se não existir
mkdir -p logs_cnn_3d

echo "INICIANDO TENSORBOARD..."

# Configurações otimizadas para evitar erros
tensorboard --logdir=./logs_cnn_3d \
           --port=6006 \
           --host=0.0.0.0 \
           --reload_interval=5 \
           --max_reload_threads=1 \
           --samples_per_plugin="" 
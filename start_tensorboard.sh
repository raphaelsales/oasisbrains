#!/bin/bash

echo "🚀 INICIANDO TENSORBOARD PARA MONITORAMENTO CNN 3D"
echo "=================================================="
echo ""
echo "📊 Logs disponíveis em: ./logs_cnn_3d/"
echo "🌐 Acesse: http://localhost:6006"
echo "⏹️  Para parar: Ctrl+C"
echo ""
echo "INICIANDO..."

tensorboard --logdir=./logs_cnn_3d --port=6006 --host=0.0.0.0 
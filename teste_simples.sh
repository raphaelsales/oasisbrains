#!/bin/bash

echo "=== TESTE SIMPLES DOCKER ==="

# Configurar usuário
USER_ID=$(id -u)
GROUP_ID=$(id -g)
echo "User ID: $USER_ID, Group ID: $GROUP_ID"

# Testar Docker básico
echo "Testando Docker básico..."
docker run --rm hello-world

echo ""
echo "Testando Docker com usuário específico..."
docker run --rm --user "$USER_ID:$GROUP_ID" alpine:latest whoami

echo ""
echo "=== FIM DO TESTE ===" 
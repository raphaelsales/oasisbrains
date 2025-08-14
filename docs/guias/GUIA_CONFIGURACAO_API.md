# 🔑 GUIA RÁPIDO: CONFIGURAÇÃO DA API OPENAI

## 🚀 CONFIGURAÇÃO EM 3 PASSOS

### PASSO 1: Obter API Key
1. Acesse: https://platform.openai.com/api-keys
2. Faça login na sua conta OpenAI
3. Clique em "Create new secret key"
4. Copie a chave (começa com `sk-`)

### PASSO 2: Configurar (ESCOLHA UMA OPÇÃO)

#### OPÇÃO A: Script Automático (RECOMENDADO)
```bash
./setup_api_key.sh
```
- Execute o script
- Cole sua API key quando solicitado
- Pronto! ✅

#### OPÇÃO B: Arquivo .env Manual
```bash
# 1. Copiar exemplo
cp env_example.txt .env

# 2. Editar arquivo
nano .env

# 3. Substituir 'sua-chave-aqui' pela sua chave real
OPENAI_API_KEY=sk-sua-chave-real-aqui
```

#### OPÇÃO C: Variável de Ambiente
```bash
export OPENAI_API_KEY='sk-sua-chave-aqui'
```

### PASSO 3: Testar Configuração
```bash
python3 config_openai.py
```

---

## 🔒 SEGURANÇA

### ✅ O QUE ESTÁ PROTEGIDO
- Arquivo `.env` está no `.gitignore`
- Scripts não salvam a chave em texto plano
- Validação de formato da chave

### ⚠️ BOAS PRÁTICAS
- Nunca commite arquivos com API keys
- Use limites de uso na plataforma OpenAI
- Monitore custos em: https://platform.openai.com/usage
- Use GPT-3.5-turbo para testes (mais barato)

---

## 💰 ESTIMATIVA DE CUSTOS

### GPT-4 (Recomendado para análises finais)
- **Entrada**: $0.03 por 1K tokens
- **Saída**: $0.06 por 1K tokens
- **5 análises**: ~$0.15-0.30

### GPT-3.5-turbo (Para testes)
- **Entrada**: $0.0015 por 1K tokens  
- **Saída**: $0.002 por 1K tokens
- **5 análises**: ~$0.01-0.02

---

## 🎯 USO

### Análise Individual
```bash
python3 openai_fastsurfer_analyzer.py
```

### Configuração Avançada
```bash
# Editar configurações
nano config_openai.py

# Alterar modelo, tokens, etc.
GPT_MODEL = "gpt-3.5-turbo"  # Para testes
MAX_TOKENS = 1000            # Reduzir custo
```

---

## 🆘 SOLUÇÃO DE PROBLEMAS

### Erro: "API key não configurada"
```bash
./setup_api_key.sh
```

### Erro: "Formato inválido"
- Verifique se a chave começa com `sk-`
- Verifique se tem pelo menos 32 caracteres

### Erro: "Rate limit exceeded"
- Aguarde alguns minutos
- Reduza o número de análises
- Use GPT-3.5-turbo

### Erro: "Insufficient credits"
- Adicione créditos na plataforma OpenAI
- Use GPT-3.5-turbo para testes

---

## 📞 SUPORTE

- **Documentação OpenAI**: https://platform.openai.com/docs
- **Monitor de Uso**: https://platform.openai.com/usage
- **Limites de Rate**: https://platform.openai.com/docs/guides/rate-limits

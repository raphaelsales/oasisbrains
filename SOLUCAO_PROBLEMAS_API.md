# 🆘 SOLUÇÃO DE PROBLEMAS: CONFIGURAÇÃO DA API KEY

## ❌ **PROBLEMA: "Não está aceitando colar a chave da API"**

### 🔧 **SOLUÇÕES DISPONÍVEIS:**

---

## **OPÇÃO 1: Script Manual (MAIS CONFIÁVEL)**
```bash
./setup_api_key_manual.sh
```
- Abre o editor nano
- Você cola a chave diretamente no arquivo
- Salva e sai (Ctrl+X, Y, Enter)

---

## **OPÇÃO 2: Script Direto**
```bash
./setup_api_key_direct.sh
```
- Permite colar o comando completo
- Exemplo: `echo 'OPENAI_API_KEY=sk-abc123...' > .env`

---

## **OPÇÃO 3: Método Manual Simples**
```bash
# 1. Criar arquivo .env
nano .env

# 2. Adicionar conteúdo:
OPENAI_API_KEY=sk-sua-chave-aqui
GPT_MODEL=gpt-4
MAX_TOKENS=2000
TEMPERATURE=0.3

# 3. Salvar (Ctrl+X, Y, Enter)
```

---

## **OPÇÃO 4: Comando Direto no Terminal**
```bash
# Cole este comando (substitua pela sua chave):
echo 'OPENAI_API_KEY=sk-sua-chave-real-aqui' > .env
```

---

## **OPÇÃO 5: Variável de Ambiente**
```bash
# Configure temporariamente:
export OPENAI_API_KEY='sk-sua-chave-aqui'

# Teste:
python3 config_openai.py
```

---

## **OPÇÃO 6: Editar Arquivo de Configuração**
```bash
# Editar config_openai.py
nano config_openai.py

# Descomente e edite a linha:
OPENAI_API_KEY = "sk-sua-chave-aqui"
```

---

## 🧪 **TESTAR CONFIGURAÇÃO:**

Após qualquer método, teste com:
```bash
python3 config_openai.py
```

---

## ⚠️ **PROBLEMAS COMUNS:**

### **Problema: "Formato inválido"**
- Verifique se a chave começa com `sk-`
- Verifique se tem pelo menos 32 caracteres
- Remova espaços extras

### **Problema: "Arquivo não encontrado"**
- Verifique se o arquivo `.env` foi criado: `ls -la .env`
- Verifique se está no diretório correto: `pwd`

### **Problema: "Permissão negada"**
- Execute: `chmod +x setup_api_key_*.sh`

### **Problema: "Editor não encontrado"**
- Instale nano: `sudo apt install nano`
- Ou use vim: `sudo apt install vim`

---

## 🔒 **VERIFICAÇÃO DE SEGURANÇA:**

```bash
# Verificar se .env está no .gitignore
grep -n ".env" .gitignore

# Verificar conteúdo do .env (sem mostrar a chave)
cat .env | sed 's/sk-[a-zA-Z0-9]*/sk-***HIDDEN***/'
```

---

## 📞 **SUPORTE ADICIONAL:**

Se nenhuma opção funcionar:
1. Use o método manual com nano
2. Verifique se a chave está correta
3. Teste com uma chave de exemplo primeiro

---

## 🎯 **COMANDOS RÁPIDOS:**

```bash
# Criar .env rapidamente:
echo 'OPENAI_API_KEY=sk-sua-chave-aqui' > .env

# Testar:
python3 config_openai.py

# Executar análise:
python3 openai_fastsurfer_analyzer.py
```

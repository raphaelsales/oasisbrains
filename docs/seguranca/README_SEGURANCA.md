# 🔒 GUIA DE SEGURANÇA - ALZHEIMER PROJECT

## ⚠️ DADOS SENSÍVEIS - NUNCA COMMITAR

### 🚫 **ARQUIVOS PROIBIDOS NO GIT:**
- `freesurfer_license.txt` - Licença real do FreeSurfer
- `env_config.sh` - Configurações pessoais
- `output.log` - Logs com dados do usuário
- Qualquer arquivo com emails, paths pessoais ou chaves

### ✅ **ARQUIVOS SEGUROS PARA COMMIT:**
- `freesurfer_license_template.txt` - Template genérico
- `env_template.sh` - Template de configuração
- Scripts com variáveis `$HOME` em vez de paths absolutos

## 🛠️ CONFIGURAÇÃO SEGURA

### 1. **Setup Inicial:**
```bash
# Copiar templates
cp freesurfer_license_template.txt freesurfer_license.txt
cp env_template.sh env_config.sh

# Editar com seus dados reais
nano freesurfer_license.txt  # Inserir licença real
nano env_config.sh          # Configurar ambiente
```

### 2. **Usar Variáveis de Ambiente:**
```bash
# Carregar configuração
source env_config.sh

# Usar em scripts
LICENSE_FILE="$FREESURFER_LICENSE"
FREESURFER_HOME="$FREESURFER_HOME"
```

### 3. **Verificar Antes de Commit:**
```bash
# Verificar arquivos staged
git status

# Verificar conteúdo sensível
grep -r "comaisserveria" .
grep -r "@" . --include="*.sh"
grep -r "/home/" . --include="*.sh"
```

## 🔍 DADOS SENSÍVEIS REMOVIDOS

### ✅ **Correções Aplicadas:**
- `test_fastsurfer_fixed.sh` - Email e licença removidos
- `run_fastsurfer_fixed.sh` - Dados pessoais limpos
- `fix_environment.sh` - Paths absolutos → `$HOME`
- `processar_T1_discos.sh` - Path de licença corrigido
- `setup_freesurfer.sh` - Configurações seguras
- `bash.sh` - Paths pessoais removidos
- `output.log` - **DELETADO** (continha dados sensíveis)

### 🛡️ **Proteções Adicionadas:**
- `.gitignore` atualizado com regras abrangentes
- Templates seguros criados
- Variáveis de ambiente configuradas

## 📋 CHECKLIST PRÉ-COMMIT

Antes de fazer `git commit`, verifique:

- [ ] Nenhum email pessoal nos arquivos
- [ ] Nenhum path `/home/usuario/` absoluto
- [ ] Licenças reais não commitadas
- [ ] Logs sensíveis removidos
- [ ] `.gitignore` atualizado
- [ ] Templates usados em vez de dados reais

## 🚨 EM CASO DE EXPOSIÇÃO ACIDENTAL

Se dados sensíveis foram commitados:

```bash
# 1. Remover do último commit
git reset --soft HEAD~1
git reset HEAD arquivo_sensível.sh
rm arquivo_sensível.sh

# 2. Limpar histórico (CUIDADO!)
git filter-branch --force --index-filter \
"git rm --cached --ignore-unmatch arquivo_sensível.sh" \
--prune-empty --tag-name-filter cat -- --all

# 3. Forçar push limpo
git push origin --force --all
```

## 💡 BOAS PRÁTICAS

1. **Sempre use templates** para arquivos com dados pessoais
2. **Variáveis de ambiente** para configurações
3. **Paths relativos** ou `$HOME` em vez de absolutos
4. **Revisar diffs** antes de commits
5. **Logs em .gitignore** por padrão
6. **Separar dados públicos vs privados**

---
**🔐 Segurança é responsabilidade de todos!** 
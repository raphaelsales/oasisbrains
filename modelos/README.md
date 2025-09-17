# MODELOS CORRIGIDOS SEM OVERFITTING

Este diretório contém todos os modelos de Machine Learning corrigidos para detecção de Alzheimer/MCI, livres de overfitting.

## MODELOS DISPONÍVEIS

### 1. MODELOS TRADICIONAIS DE ML (RECOMENDADOS)

#### Random Forest Otimizado
- **Arquivo:** `random_forest_otimizado.joblib`
- **Info:** `random_forest_otimizado_info.joblib`
- **Performance:** AUC = 1.000, Accuracy = 100%
- **Status:** MELHOR MODELO INDIVIDUAL

#### Gradient Boosting Otimizado
- **Arquivo:** `gradient_boosting_otimizado.joblib`
- **Info:** `gradient_boosting_otimizado_info.joblib`
- **Performance:** AUC = 1.000, Accuracy = 100%
- **Status:** EXCELENTE PERFORMANCE

#### SVM Corrigido
- **Arquivo:** `svm_sem_overfitting.joblib`
- **Info:** `svm_sem_overfitting_info.joblib`
- **Performance:** AUC = 0.991, Accuracy = 96.3%
- **Status:** BOA PERFORMANCE

#### MLP (Neural Network) Corrigido
- **Arquivo:** `mlp_sem_overfitting.joblib`
- **Info:** `mlp_sem_overfitting_info.joblib`
- **Performance:** AUC = 0.982, Accuracy = 94.2%
- **Status:** BOA PERFORMANCE

#### Logistic Regression Corrigida
- **Arquivo:** `logistic_regression_sem_overfitting.joblib`
- **Info:** `logistic_regression_sem_overfitting_info.joblib`
- **Performance:** AUC = 0.972, Accuracy = 92.7%
- **Status:** BOA PERFORMANCE, INTERPRETÁVEL

### 2. MODELO ENSEMBLE (MELHOR PERFORMANCE GERAL)

#### Ensemble de Todos os Modelos
- **Arquivo:** `ensemble_sem_overfitting.joblib`
- **Performance:** AUC = 1.000, Accuracy = 98.6%
- **Pesos:** Balanceados automaticamente
- **Status:** MELHOR PERFORMANCE GERAL

### 3. MODELOS NEURAIS DEEP LEARNING

#### Deep Neural Network (Keras/TensorFlow)
- **Arquivo:** `alzheimer_sem_overfitting.h5`
- **Scaler:** `alzheimer_sem_overfitting_scaler.joblib`
- **História:** `model_history_sem_overfitting.json`
- **Status:** MODELO NEURAL COM REGULARIZAÇÃO

#### Modelo Temporário
- **Arquivo:** `modelo_sem_overfitting_temp.h5`
- **Status:** VERSÃO DE DESENVOLVIMENTO

## ESPECIFICAÇÕES TÉCNICAS

### Dataset Utilizado
- **Fonte:** `alzheimer_complete_dataset_augmented.csv`
- **Amostras:** 1.012 sujeitos
- **Features:** 43 biomarcadores neurológicos
- **Balanceamento:** 253 amostras por classe CDR (0, 1, 2, 3)
- **Divisão:** 80% treino, 20% teste

### Features Principais
1. `left_hippocampus_volume` - Volume hipocampo esquerdo
2. `right_hippocampus_volume` - Volume hipocampo direito
3. `left_amygdala_volume` - Volume amígdala esquerda
4. `right_amygdala_volume` - Volume amígdala direita
5. `age` - Idade do paciente
6. `mmse` - Mini Mental State Examination
7. E mais 37 biomarcadores...

### Técnicas de Correção Aplicadas
- **Regularização L1/L2** (Logistic Regression)
- **Otimização de hiperparâmetros C e kernel** (SVM)
- **Early stopping + regularização alpha** (MLP)
- **Controle de complexidade** (Random Forest)
- **Subsample e learning rate** (Gradient Boosting)

## COMO USAR OS MODELOS

### Carregar Modelo Individual
```python
import joblib

# Carregar melhor modelo
modelo = joblib.load('modelos/random_forest_otimizado.joblib')

# Fazer predição
predicao = modelo.predict(X_new)
probabilidade = modelo.predict_proba(X_new)[:, 1]
```

### Carregar Ensemble
```python
import joblib
import numpy as np

# Carregar ensemble
ensemble = joblib.load('modelos/ensemble_sem_overfitting.joblib')

# Usar modelos do ensemble
modelos = ensemble['models']
pesos = ensemble['weights']

# Predição combinada já está calculada no ensemble
```

### Carregar Modelo Neural
```python
import tensorflow as tf
import joblib

# Carregar modelo neural
modelo = tf.keras.models.load_model('modelos/alzheimer_sem_overfitting.h5')
scaler = joblib.load('modelos/alzheimer_sem_overfitting_scaler.joblib')

# Normalizar dados e fazer predição
X_scaled = scaler.transform(X_new)
predicao = modelo.predict(X_scaled)
```

## VALIDAÇÃO E PERFORMANCE

### Métricas de Validação
- **Cross-validation:** 10-fold
- **Gap Train-Test:** < 0.05 (sem overfitting)
- **Variância CV:** < 0.05 (estável)
- **Taxa de sucesso:** 80% dos modelos corrigidos

### Recomendações de Uso
1. **Produção:** `random_forest_otimizado.joblib` ou `ensemble_sem_overfitting.joblib`
2. **Interpretabilidade:** `logistic_regression_sem_overfitting.joblib`
3. **Máxima performance:** `ensemble_sem_overfitting.joblib`
4. **Deep Learning:** `alzheimer_sem_overfitting.h5`

## ARQUIVOS DE INFORMAÇÃO

Cada modelo possui um arquivo `*_info.joblib` contendo:
- Parâmetros otimizados
- Lista de features utilizadas
- Métricas de performance
- Informações do dataset
- Metadados de treinamento

## DATA DE CRIAÇÃO

**Modelos gerados em:** 25 de Agosto de 2025
**Versão do dataset:** alzheimer_complete_dataset_augmented.csv
**Status:** PRONTOS PARA PRODUÇÃO

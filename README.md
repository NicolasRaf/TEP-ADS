# Análise Comparativa de Modelos de Classificação e Técnicas de Tuning (TEP-ADS)

Este repositório contém um conjunto de scripts paratreinar e avaliar diferentes algoritmos de classificação de machine learning, bem como comparar a eficácia de várias técnicas de otimização de hiperparâmetros.

O projeto está dividido em duas partes principais:

1. **`Modelos_de_classificação`**: Aplicação e avaliação de seis modelos de classificação diferentes em três conjuntos de dados distintos (Adult, Covertype e Credit Card Fraud).
2. **`Tuning`**: Comparação de performance entre modelos sem otimização e modelos otimizados com GridSearchCV, RandomizedSearchCV e BayesSearchCV.

-----

## 1\. Modelos de Classificação

Esta seção avalia o desempenho de seis algoritmos de classificação populares em três conjuntos de dados.

### Modelos Avaliados

Para cada dataset, os seguintes modelos são treinados e avaliados:

* Regressão Logística
* SVM (Linear)
* SVM (RBF)
* k-NN (k=5)
* MLP (Rede Neural)
* Naive Bayes (Gaussiano)

### Conjuntos de Dados (Datasets)

1. **Adult (Census Income)**

      * **Script**: `Modelos_de_classificação/scripts/adult.py`
      * **Objetivo**: Prever se o rendimento de um indivíduo excede $50K/ano.
      * **Métricas**: Acurácia, Precisão, Recall, F1-Score e Matriz de Confusão.
      * **Gráficos**: Gera e salva fronteiras de decisão 2D (Idade vs. Horas por Semana) para cada modelo no diretório `Modelos_de_classificação/grafics/adult/`.

2. **Covertype (Forest Cover Type)**

      * **Script**: `Modelos_de_classificação/scripts/covertype.py`
      * **Objetivo**: Prever o tipo de cobertura florestal (binarizado para Classe 1 vs. Outras).
      * **Métricas**: Acurácia, Precisão, Recall, F1-Score e Matriz de Confusão.
      * **Gráficos**: Gera e salva fronteiras de decisão 2D (Elevação vs. Aspecto) para cada modelo no diretório `Modelos_de_classificação/grafics/covertype/`.

3. **Credit Card Fraud**

      * **Script**: `Modelos_de_classificação/scripts/creditcard.py`
      * **Objetivo**: Detectar transações fraudulentas (Classe 1).
      * **Métricas**: Acurácia, Precisão, Recall, F1-Score, ROC-AUC e Matriz de Confusão.
      * **Gráficos**: Gera e salva fronteiras de decisão 2D (Feature V1 vs. V2) para cada modelo no diretório `Modelos_de_classificação/grafics/creditcard/`.

-----

## 2\. Comparação de Técnicas de Tuning

Esta seção compara a performance de modelos de classificação com e sem otimização de hiperparâmetros.

* **Script**: `Tuning/main.py`

### Metodologia

1. **Datasets**:

      * Wine
      * Digits
      * Breast Cancer

2. **Modelos**:

      * Logistic Regression
      * SVM
      * Random Forest Classifier

3. **Técnicas de Tuning Comparadas**:

      * Sem Tuning (Baseline)
      * GridSearchCV
      * RandomizedSearchCV
      * BayesSearchCV (via `skopt`)

### Saída (Output)

O script imprime no console quatro tabelas comparativas formatadas em Markdown, detalhando os resultados de Acurácia, F1-Score, Precisão e Recall para cada combinação de dataset, modelo e técnica de tuning.

-----

## Como Executar

### Pré-requisitos

Certifique-se de ter o Python 3 instalado. Você precisará instalar as seguintes bibliotecas:

```bash
pip install pandas numpy scikit-learn matplotlib scikit-optimize
```

### Arquivos de Dados

* Os scripts `adult.py` e `covertype.py` baixam seus dados automaticamente usando o `scikit-learn`.
* Para o script `creditcard.py`, você deve baixar o dataset e salvá-lo como `creditcard.csv` dentro da pasta `Modelos_de_classificação/`.

### Executando os Scripts

1. **Modelos de Classificação**:
    Navegue até o diretório dos scripts e execute-os individualmente.

    ```bash
    cd Modelos_de_classificação/scripts
    python adult.py
    python covertype.py
    python creditcard.py
    ```

    Os gráficos serão salvos automaticamente nas pastas `grafics/` correspondentes.

2. **Comparação de Tuning**:
    Navegue até o diretório `Tuning` e execute o script `main.py`.

    ```bash
    cd Tuning
    python main.py
    ```

    As tabelas de resultado serão impressas no terminal.

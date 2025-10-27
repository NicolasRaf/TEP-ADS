import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine, load_digits, load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skopt import BayesSearchCV

# Bloco 1: Configuração Principal
# =================================

# Ignorar avisos para manter a saída limpa
warnings.filterwarnings('ignore')

# Dicionário para carregar as bases de dados de forma organizada
datasets = {
    "Wine": load_wine(),
    "Digits": load_digits(),
    "Breast Cancer": load_breast_cancer()
}

# Dicionário com instâncias dos modelos a serem utilizados
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier()
}

# Dicionário contendo os hiperparâmetros a serem testados para cada modelo
param_grids = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10],  
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    "SVM": {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 1]
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
}

# Lista para armazenar os resultados de todos os experimentos
results = []

# Bloco 2: Função de Avaliação
# ==============================

def evaluate_model(y_true, y_pred):
    """Calcula e retorna as métricas de avaliação para um modelo."""
    accuracy = accuracy_score(y_true, y_pred)
    # Usar average='macro' para problemas multiclasse para tratar todas as classes igualmente
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return accuracy, precision, f1, recall


# Bloco 3: Loop Principal de Processamento
# ==========================================

# Loop principal que itera sobre cada base de dados
for d_name, dataset in datasets.items():
    print(f"================= Processando Base de Dados: {d_name} =================")
    X, y = dataset.data, dataset.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Loop aninhado que itera sobre cada modelo
    for m_name, model in models.items():
        print(f"  --> Modelo: {m_name}")
        model_results = {"Base": d_name, "Modelo": m_name}
        params = param_grids[m_name]

        # 1. Modelo sem tuning
        model.fit(X_train, y_train)
        y_pred_default = model.predict(X_test)
        acc, prec, f1, rec = evaluate_model(y_test, y_pred_default)
        model_results["Sem Tuning (Acc)"] = acc
        model_results["Sem Tuning (F1)"] = f1
        model_results["Sem Tuning (Prec)"] = prec 
        model_results["Sem Tuning (Rec)"] = rec   

        # 2. GridSearchCV
        print("    Executando GridSearchCV...")
        grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1, scoring='f1_macro', verbose=0)
        grid_search.fit(X_train, y_train)
        y_pred_grid = grid_search.predict(X_test)
        acc, prec, f1, rec = evaluate_model(y_test, y_pred_grid)
        model_results["GridSearchCV (Acc)"] = acc
        model_results["GridSearchCV (F1)"] = f1
        model_results["GridSearchCV (Prec)"] = prec 
        model_results["GridSearchCV (Rec)"] = rec   

        # 3. RandomizedSearchCV
        print("    Executando RandomizedSearchCV...")
        random_search = RandomizedSearchCV(model, params, n_iter=20, cv=5, n_jobs=-1, random_state=42, scoring='f1_macro', verbose=0)
        random_search.fit(X_train, y_train)
        y_pred_random = random_search.predict(X_test)
        acc, prec, f1, rec = evaluate_model(y_test, y_pred_random)
        model_results["RandomizedSearchCV (Acc)"] = acc
        model_results["RandomizedSearchCV (F1)"] = f1
        model_results["RandomizedSearchCV (Prec)"] = prec 
        model_results["RandomizedSearchCV (Rec)"] = rec   

        # 4. BayesSearchCV
        print("    Executando BayesSearchCV...")
        bayes_search = BayesSearchCV(model, params, n_iter=20, cv=5, n_jobs=-1, random_state=42, scoring='f1_macro', verbose=0)
        bayes_search.fit(X_train, y_train)
        y_pred_bayes = bayes_search.predict(X_test)
        acc, prec, f1, rec = evaluate_model(y_test, y_pred_bayes)
        model_results["BayesSearchCV (Acc)"] = acc
        model_results["BayesSearchCV (F1)"] = f1
        model_results["BayesSearchCV (Prec)"] = prec 
        model_results["BayesSearchCV (Rec)"] = rec   
        
        results.append(model_results)
        print(f"  - {m_name} concluído para a base {d_name}.\n")


# Bloco 4: Exibição dos Resultados (Formatado com Todas as Métricas)
# =================================================================

# Converte a lista de resultados em um DataFrame do Pandas
df_results = pd.DataFrame(results)

# ---  Definindo colunas para as métricas ---
acc_cols = ["Base", "Modelo", "Sem Tuning (Acc)", "GridSearchCV (Acc)", "RandomizedSearchCV (Acc)", "BayesSearchCV (Acc)"]
f1_cols = ["Base", "Modelo", "Sem Tuning (F1)", "GridSearchCV (F1)", "RandomizedSearchCV (F1)", "BayesSearchCV (F1)"]
prec_cols = ["Base", "Modelo", "Sem Tuning (Prec)", "GridSearchCV (Prec)", "RandomizedSearchCV (Prec)", "BayesSearchCV (Prec)"]
rec_cols = ["Base", "Modelo", "Sem Tuning (Rec)", "GridSearchCV (Rec)", "RandomizedSearchCV (Rec)", "BayesSearchCV (Rec)"]     


# --- Impressão das 4 tabelas formatadas em Markdown ---

print("\n\n--- Tabela Comparativa de Resultados (Acurácia Média) ---")
print(df_results[acc_cols].round(3).to_markdown(index=False))

print("\n\n--- Tabela Comparativa de Resultados (F1-Score Médio) ---")
print(df_results[f1_cols].round(3).to_markdown(index=False))

print("\n\n--- Tabela Comparativa de Resultados (Precision Médio) ---")
print(df_results[prec_cols].round(3).to_markdown(index=False))

print("\n\n--- Tabela Comparativa de Resultados (Recall Médio) ---")
print(df_results[rec_cols].round(3).to_markdown(index=False))
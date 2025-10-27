import time
import numpy as np
import pandas as pd
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

RANDOM_STATE = 42
TEST_SIZE = 0.30

# ===============================================================
# 1) Carregar e preparar a base Adult
# ===============================================================
print("="*60)
print("INICIANDO ANÁLISE DO DATASET: ADULT (CENSUS INCOME)")
print("="*60)

adult = fetch_openml("adult", version=2, as_frame=True, parser='auto')
df = adult.frame
X = df.drop(columns=["class"])
y = (df["class"] == ">50K").astype(int)

cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["category", "object"]).columns.tolist()

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

# ===============================================================
# 2) Split treino/teste
# ===============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Tamanhos -> treino: {X_train.shape[0]} | teste: {X_test.shape[0]}\n")

# ===============================================================
# 3) Criar um subconjunto de treino para modelos lentos
# ===============================================================
subsample_size = 15000
if len(y_train) > subsample_size:
    print(f"Criando subconjunto de {subsample_size} amostras para modelos lentos...")
    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train, y_train, train_size=subsample_size, stratify=y_train, random_state=RANDOM_STATE
    )
    print(f"Tamanho do subconjunto -> treino: {X_train_sub.shape[0]}\n")
else:
    print("O conjunto de treino já é pequeno, não será criado um subconjunto.\n")
    X_train_sub, y_train_sub = X_train, y_train

# ===============================================================
# 4) Definir todos os modelos a serem testados
# ===============================================================
modelos = [
    ("Regressão Logística", Pipeline([("preprocess", preprocess), ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))]), "full"),
    ("SVM (Linear)", Pipeline([("preprocess", preprocess), ("clf", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE))]), "full"),
    ("k-NN (k=5)", Pipeline([("preprocess", preprocess), ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))]), "full"),
    ("MLP (Rede Neural)", Pipeline([("preprocess", preprocess), ("clf", MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=RANDOM_STATE, early_stopping=True))]), "full"),
    ("Naive Bayes (Gaussiano)", Pipeline([("preprocess", preprocess), ("to_dense", FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)), ("clf", GaussianNB())]), "full"),
    ("SVM (RBF)", Pipeline([("preprocess", preprocess), ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]), "subsample")
]

# ===============================================================
# 5) Funções utilitárias para avaliação
# ===============================================================

# --- Função de avaliação com impressão no formato desejado ---
def avaliar_modelo(nome, modelo, X_tr, y_tr, X_te, y_te):
    print(f"--- Treinando e Avaliando: {nome} (usando {X_tr.shape[0]} amostras) ---")
    start_time = time.time()
    modelo.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_te)
    end_time = time.time()

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    cm = confusion_matrix(y_te, y_pred)

    print(f"Tempo de execução: {end_time - start_time:.2f} segundos")
    print("\n=== Desempenho no TESTE ===")
    print(f"Acurácia : {acc:.3f}")
    print(f"Precisão : {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1       : {f1:.3f}")
    
    print("\nMatriz de Confusão (linhas=Real, colunas=Previsto):")
    print("           Prev 0   Prev 1")
    print(f"Real 0 |  {cm[0,0]:>7}   {cm[0,1]:>7}   <- Salário <=50K")
    print(f"Real 1 |  {cm[1,0]:>7}   {cm[1,1]:>7}   <- Salário >50K\n")
    print("-"*(len(nome) + 48) + "\n")
    return modelo

# --- Função para realizar predições individuais ---
def predicoes_individuais(nome, modelo, exemplos_df):
    print(f"--- Predições Individuais — {nome} ---")
    
    if hasattr(modelo.named_steps['clf'], "predict_proba"):
        probas = modelo.predict_proba(exemplos_df)
        preds  = modelo.predict(exemplos_df)
        
        for i, (p, pred) in enumerate(zip(probas, preds)):
            entrada_str = exemplos_df.iloc[i].to_dict()
            print(f"Entrada {i+1} -> prev={'Salário >50K (1)' if pred==1 else 'Salário <=50K (0)'} "
                  f"| P(<=50K)={p[0]:.3f} | P(>50K)={p[1]:.3f}")
    else: 
        preds = modelo.predict(exemplos_df)
        for i, pred in enumerate(preds):
            print(f"Entrada {i+1} -> prev={'Salário >50K (1)' if pred==1 else 'Salário <=50K (0)'} (sem prob.)")
    print()


# ===============================================================
# 6) Rodar todos os modelos e guardar os modelos treinados
# ===============================================================
modelos_treinados = []
for nome, mdl, tipo_treino in modelos:
    if tipo_treino == "full":
        modelo_treinado = avaliar_modelo(nome, mdl, X_train, y_train, X_test, y_test)
    else: # tipo_treino == "subsample"
        modelo_treinado = avaliar_modelo(nome, mdl, X_train_sub, y_train_sub, X_test, y_test)
    modelos_treinados.append((nome, modelo_treinado, tipo_treino))


# ===============================================================
# 7) Realizar predições individuais com os modelos treinados
# ===============================================================
print("\n" + "="*60)
print("INICIANDO PREDIÇÕES INDIVIDUAIS EM NOVOS DADOS")
print("="*60 + "\n")

exemplos = pd.DataFrame({
    'age': [35, 50, 22],
    'workclass': ['Private', 'Self-emp-not-inc', 'State-gov'],
    'fnlwgt': [150000, 200000, 50000],
    'education': ['Bachelors', 'HS-grad', 'Some-college'],
    'education-num': [13, 9, 10],
    'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married'],
    'occupation': ['Exec-managerial', 'Craft-repair', 'Adm-clerical'],
    'relationship': ['Husband', 'Not-in-family', 'Own-child'],
    'race': ['White', 'White', 'Black'],
    'sex': ['Male', 'Female', 'Male'],
    'capital-gain': [0, 0, 0],
    'capital-loss': [0, 0, 0],
    'hours-per-week': [50, 40, 35],
    'native-country': ['United-States', 'United-States', 'United-States']
})

for nome, mdl_treinado, _ in modelos_treinados:
    predicoes_individuais(nome, mdl_treinado, exemplos)

# ===============================================================
#  Funções para Plotar Fronteiras de Decisão
# ===============================================================
GRAFICOS_DIR = "grafics/adult"

def plotar_fronteiras(nome, modelo, X, y, feature_indices, feature_names):
    """
    Plota as fronteiras de decisão de um modelo usando duas features.
    """
    print(f"--- Gerando Gráfico de Fronteiras para: {nome} ---")
    
    X_plot = X.iloc[:, feature_indices] if hasattr(X, 'iloc') else X[:, feature_indices]
    
    from sklearn.base import clone
    plot_model = clone(modelo)
    plot_model.fit(X_plot, y)
    
    x_min, x_max = X_plot.iloc[:, 0].min() - 1, X_plot.iloc[:, 0].max() + 1
    y_min, y_max = X_plot.iloc[:, 1].min() - 1, X_plot.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    Z = plot_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ['#FF0000', '#0000FF']
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    
    subset_indices = np.random.choice(len(X_plot), size=min(1000, len(X_plot)), replace=False)
    X_subset = X_plot.iloc[subset_indices] if hasattr(X_plot, 'iloc') else X_plot[subset_indices]
    y_subset = y.iloc[subset_indices] if hasattr(y, 'iloc') else y[subset_indices]
    
    scatter = plt.scatter(X_subset.iloc[:, 0], X_subset.iloc[:, 1], c=y_subset, cmap=ListedColormap(cmap_bold),
                          edgecolor='k', s=20)

    plt.title(f"Fronteira de Decisão - {nome}")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend(handles=scatter.legend_elements()[0], labels=['Classe 0', 'Classe 1'])
    
    nome_seguro = nome.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
    nome_arquivo = f"adult_{nome_seguro}.png"
    caminho_salvar = os.path.join(GRAFICOS_DIR, nome_arquivo)
    
    try:
        plt.savefig(caminho_salvar)
        print(f"Gráfico salvo em: {caminho_salvar}")
    except Exception as e:
        print(f"Erro ao salvar gráfico {nome_arquivo}: {e}")

    # plt.show() 
    plt.close() 

    print("-"*(len(nome) + 47) + "\n")

# ===============================================================
# 8) Gerar Gráficos de Fronteiras de Decisão
# ===============================================================
os.makedirs(GRAFICOS_DIR, exist_ok=True)

print("\n" + "="*60)
print("INICIANDO GERAÇÃO DOS GRÁFICOS DE FRONTEIRAS")
print("="*60 + "\n")

plot_features_indices = [X_train.columns.get_loc('age'), X_train.columns.get_loc('hours-per-week')]
plot_features_names = ['Idade (Age)', 'Horas por Semana (Hours-per-week)']

for nome, mdl_treinado, tipo_treino in modelos_treinados:
    plot_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', mdl_treinado.named_steps['clf'])
    ])

    if tipo_treino == "subsample":
        print(f"*** Usando SUB-CONJUNTO ({X_train_sub.shape[0]} amostras) para plotar {nome} ***")
        plotar_fronteiras(nome, plot_pipeline, X_train_sub, y_train_sub, plot_features_indices, plot_features_names)
    else:
        plotar_fronteiras(nome, plot_pipeline, X_train, y_train, plot_features_indices, plot_features_names)
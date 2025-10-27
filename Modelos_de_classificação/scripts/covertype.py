import time
import numpy as np
import os  
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

RANDOM_STATE = 42
TEST_SIZE = 0.30

# ===============================================================
# 1) Carregar e preparar a base Covertype
# ===============================================================
print("="*60)
print("INICIANDO ANÁLISE DETALHADA DO DATASET: COVERTYPE")
print("="*60)

# Binarizar: classe 1 (Spruce/Fir) vs outras
cov = fetch_covtype(as_frame=False)
X = cov.data
y = (cov.target == 1).astype(int)
print(f"Dimensões dos dados: {X.shape} | Positivos (Spruce/Fir): {y.sum()}\n")

# ===============================================================
# 2) Split treino/teste (conjunto completo)
# ===============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Tamanhos -> treino: {X_train.shape[0]} | teste: {X_test.shape[0]}\n")

# ===============================================================
# 3) Criar um subconjunto de treino para modelos lentos
# ===============================================================
subsample_size = 20000
if len(y_train) > subsample_size:
    print(f"Criando subconjunto de {subsample_size} amostras para modelos lentos...")
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=subsample_size, random_state=RANDOM_STATE)
    train_indices, _ = next(splitter.split(X_train, y_train))
    X_train_sub, y_train_sub = X_train[train_indices], y_train[train_indices]
    print(f"Tamanho do subconjunto -> treino: {X_train_sub.shape[0]}\n")
else:
    print("O conjunto de treino já é pequeno, não será criado um subconjunto.\n")
    X_train_sub, y_train_sub = X_train, y_train

# ===============================================================
# 4) Definir todos os modelos a serem testados
# ===============================================================
modelos = [
    ("Regressão Logística", Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=RANDOM_STATE, solver='saga', n_jobs=-1))]), "full"),
    ("SVM (Linear)", Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(random_state=RANDOM_STATE, dual=False, max_iter=2000))]), "full"),
    ("k-NN (k=5)", Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))]), "full"),
    ("MLP (Rede Neural)", Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=RANDOM_STATE, early_stopping=True))]), "full"),
    ("Naive Bayes (Gaussiano)", Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]), "full"),
    ("SVM (RBF)", Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]), "subsample")
]

# ===============================================================
# 5) Funções utilitárias para avaliação e predição
# ===============================================================

# --- Função de avaliação com impressão no formato detalhado ---
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
    print(f"Real 0 |  {cm[0,0]:>7}   {cm[0,1]:>7}   <- Outras Classes")
    print(f"Real 1 |  {cm[1,0]:>7}   {cm[1,1]:>7}   <- Spruce/Fir\n")
    print("-"*(len(nome) + 48) + "\n")
    return modelo

# --- Função para realizar predições individuais ---
def predicoes_individuais(nome, modelo, exemplos):
    print(f"--- Predições Individuais — {nome} ---")
    if hasattr(modelo, "predict_proba"):
        probas = modelo.predict_proba(exemplos)
        preds  = modelo.predict(exemplos)
        for i, (p, pred) in enumerate(zip(probas, preds)):
            print(f"Exemplo {i+1} -> prev={'Spruce/Fir (1)' if pred==1 else 'Outra Classe (0)'} "
                  f"| P(Outra)={p[0]:.3f} | P(Spruce/Fir)={p[1]:.3f}")
    else:
        preds = modelo.predict(exemplos)
        for i, pred in enumerate(preds):
            print(f"Exemplo {i+1} -> prev={'Spruce/Fir (1)' if pred==1 else 'Outra Classe (0)'} (sem prob.)")
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
print("INICIANDO PREDIÇÕES INDIVIDUAIS EM DADOS DE TESTE")
print("="*60 + "\n")

# Usando as 3 primeiras amostras do conjunto de teste como exemplos
exemplos_para_predicao = X_test[:3]

# Loop original (aqui não precisa do tipo_treino)
for nome, mdl_treinado, _ in modelos_treinados:
    predicoes_individuais(nome, mdl_treinado, exemplos_para_predicao)


# ===============================================================
#  Funções para Plotar Fronteiras de Decisão
# ===============================================================

def plotar_fronteiras(nome, modelo, X, y, feature_indices, feature_names):
    """
    Plota as fronteiras de decisão de um modelo usando duas features.
    """
    print(f"--- Gerando Gráfico de Fronteiras para: {nome} ---")
    
    # Selecionar apenas as duas features (X é NumPy array)
    X_plot = X[:, feature_indices]
    
    # Clona o pipeline original para não interferir nos resultados principais
    from sklearn.base import clone
    plot_model = clone(modelo)
    plot_model.fit(X_plot, y)
    
    # Criar um meshgrid para o plot (usando sintaxe NumPy)
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    h = 5.0
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Fazer predições no meshgrid
    Z = plot_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Cores para o plot
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ['#FF0000', '#0000FF']
    
    # Plotar
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    
    # Plotar os pontos de dados (usando um subconjunto) (sintaxe NumPy)
    subset_indices = np.random.choice(len(X_plot), size=min(1000, len(X_plot)), replace=False)
    X_subset = X_plot[subset_indices]
    y_subset = y[subset_indices]
    
    scatter = plt.scatter(X_subset[:, 0], X_subset[:, 1], c=y_subset, cmap=ListedColormap(cmap_bold),
                          edgecolor='k', s=20)

    plt.title(f"Fronteira de Decisão - {nome}")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend(handles=scatter.legend_elements()[0], labels=['Classe 0', 'Classe 1'])

    nome_seguro = nome.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
    nome_arquivo = f"covertype_{nome_seguro}.png"
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

GRAFICOS_DIR = "grafics/covertype" 
os.makedirs(GRAFICOS_DIR, exist_ok=True)

print("\n" + "="*60)
print("INICIANDO GERAÇÃO DOS GRÁFICOS DE FRONTEIRAS")
print("="*60 + "\n")

plot_features_indices = [0, 1]
plot_features_names = ['Elevação (Feature 0)', 'Aspecto (Feature 1)']

for nome, mdl_treinado, tipo_treino in modelos_treinados:
    
    if tipo_treino == "subsample":
        print(f"*** Usando SUB-CONJUNTO ({X_train_sub.shape[0]} amostras) para plotar {nome} ***")
        plotar_fronteiras(nome, mdl_treinado, X_train_sub, y_train_sub, plot_features_indices, plot_features_names)
    else:
        plotar_fronteiras(nome, mdl_treinado, X_train, y_train, plot_features_indices, plot_features_names)
"""
main_ga.py — Integrante 3
"""

import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from data.loader import carregar_fashion_mnist, preprocessar
from results.exporter import salvar_txt, salvar_npy

# Configuração 
DATA_DIR = "."
OUTPUT_TXT = "results/output/wrapper_resultados.txt"
OUTPUT_NPY = "results/output/wrapper_resultados.npy"

# ==========================
# PARÂMETROS DO GA
# ==========================

POP_SIZE = 20
GENERATIONS = 20
MUTATION_RATE = 0.01


# ==========================
# FITNESS
# ==========================

def fitness(individual, X, y):

    selected = np.where(individual == 1)[0]

    if len(selected) == 0:
        return 0

    X_sel = X[:, selected]

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_sel,
        y,
        test_size=0.2,
        random_state=1
    )

    clf = DecisionTreeClassifier(random_state=1)
    clf.fit(X_fit, y_fit)

    pred = clf.predict(X_val)

    acc = accuracy_score(y_val, pred)

    feature_ratio = len(selected) / 784

    return 0.9 * acc + 0.1 * (1 - feature_ratio)


# ==========================
# GA
# ==========================

def executar_ga(X, y):

    population = np.random.randint(
        0,
        2,
        size=(POP_SIZE, 784)
    )

    best = None
    best_fit = -1

    for _ in range(GENERATIONS):

        fitnesses = np.array(
            [fitness(ind, X, y) for ind in population]
        )

        idx = np.argmax(fitnesses)

        if fitnesses[idx] > best_fit:
            best_fit = fitnesses[idx]
            best = population[idx].copy()

        nova_pop = [best.copy()]

        while len(nova_pop) < POP_SIZE:

            pai1 = population[np.random.randint(POP_SIZE)]
            pai2 = population[np.random.randint(POP_SIZE)]

            ponto = np.random.randint(1, 784)

            filho = np.concatenate(
                [pai1[:ponto], pai2[ponto:]]
            )

            mutacao = np.random.rand(784) < MUTATION_RATE

            filho[mutacao] = 1 - filho[mutacao]

            nova_pop.append(filho)

        population = np.array(nova_pop)

    return best


def main():

    print("=" * 50)
    print("Algoritmo Genético")
    print("=" * 50)

    X_train, y_train, X_test, y_test = carregar_fashion_mnist(DATA_DIR)

    X_train_norm, X_test_norm = preprocessar(
        X_train,
        X_test
    )

    # usa amostra para reduzir custo
    X_ga = X_train_norm[:5000]
    y_ga = y_train[:5000]

    inicio_busca = time.time()

    best = executar_ga(X_ga, y_ga)

    tempo_busca = time.time() - inicio_busca

    selected = np.where(best == 1)[0]

    X_train_sel = X_train_norm[:, selected]
    X_test_sel = X_test_norm[:, selected]

    clf = DecisionTreeClassifier(random_state=1)

    inicio_treino = time.time()

    clf.fit(X_train_sel, y_train)

    tempo_treino = time.time() - inicio_treino

    pred = clf.predict(X_test_sel)

    acuracia = accuracy_score(y_test, pred)

    relatorio = classification_report(
        y_test,
        pred,
        target_names=CLASSES
    )

    dados = {
        "abordagem": "GA",
        "acuracia": acuracia,
        "num_features": len(selected),
        "porcentagem_features": len(selected) / 784 * 100,
        "tempo_treino_s": tempo_treino,
        "tempo_busca_s": tempo_busca,
        "relatorio": relatorio,
    }

    salvar_txt(dados, OUTPUT_TXT)
    salvar_npy(dados, OUTPUT_NPY)

    print(f"Acurácia: {acuracia:.4f}")
    print(f"Features: {len(selected)}")
    print(f"Tempo busca: {tempo_busca:.2f}s")


if __name__ == "__main__":
    main()

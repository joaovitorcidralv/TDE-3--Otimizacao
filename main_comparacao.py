"""
main_comparacao.py

Integrante 4
Responsável por comparar Baseline, Wrapper e Algoritmo Genético.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ARQ_BASELINE = "results/output/baseline_resultados.npy"
ARQ_WRAPPER = "results/output/wrapper_resultados.npy"
ARQ_GA = "results/output/ga_resultados.npy"


def carregar_resultado(caminho):
    return np.load(caminho, allow_pickle=True).item()


def main():

    print("=" * 60)
    print("COMPARAÇÃO DAS ABORDAGENS")
    print("=" * 60)

    baseline = carregar_resultado(ARQ_BASELINE)
    wrapper = carregar_resultado(ARQ_WRAPPER)
    ga = carregar_resultado(ARQ_GA)

    tabela = pd.DataFrame({
        "Métrica": [
            "Acurácia",
            "% Features",
            "Tempo Treino (s)",
            "Tempo Busca (s)"
        ],

        "Baseline": [
            baseline["acuracia"],
            baseline["porcentagem_features"],
            baseline["tempo_treino_s"],
            baseline["tempo_busca_s"]
        ],

        "Wrapper": [
            wrapper["acuracia"],
            wrapper["porcentagem_features"],
            wrapper["tempo_treino_s"],
            wrapper["tempo_busca_s"]
        ],

        "GA": [
            ga["acuracia"],
            ga["porcentagem_features"],
            ga["tempo_treino_s"],
            ga["tempo_busca_s"]
        ]
    })

    print("\nTabela Comparativa:\n")
    print(tabela)

    tabela.to_csv(
        "results/output/tabela_comparativa.csv",
        index=False
    )

    gerar_graficos(baseline, wrapper, ga)

    print("\nTabela salva em:")
    print("results/output/tabela_comparativa.csv")


def gerar_graficos(baseline, wrapper, ga):

    abordagens = ["Baseline", "Wrapper", "GA"]

    # -----------------------
    # Gráfico de Acurácia
    # -----------------------

    plt.figure(figsize=(8,5))

    plt.bar(
        abordagens,
        [
            baseline["acuracia"],
            wrapper["acuracia"],
            ga["acuracia"]
        ]
    )

    plt.title("Comparação de Acurácia")
    plt.ylabel("Acurácia")

    plt.savefig(
        "results/output/grafico_acuracia.png"
    )

    plt.close()

    # -----------------------
    # Gráfico de Features
    # -----------------------

    plt.figure(figsize=(8,5))

    plt.bar(
        abordagens,
        [
            baseline["porcentagem_features"],
            wrapper["porcentagem_features"],
            ga["porcentagem_features"]
        ]
    )

    plt.title("Porcentagem de Features")
    plt.ylabel("% Features")

    plt.savefig(
        "results/output/grafico_features.png"
    )

    plt.close()

    # -----------------------
    # Gráfico de Tempo Busca
    # -----------------------

    plt.figure(figsize=(8,5))

    plt.bar(
        abordagens,
        [
            baseline["tempo_busca_s"],
            wrapper["tempo_busca_s"],
            ga["tempo_busca_s"]
        ]
    )

    plt.title("Tempo de Busca")
    plt.ylabel("Segundos")

    plt.savefig(
        "results/output/grafico_tempo_busca.png"
    )

    # -----------------------
    # Gráfico de Tempo de Treino
    # -----------------------

    plt.figure(figsize=(8,5))
    
    plt.bar(
        abordagens,
        [
            baseline["tempo_treino_s"],
            wrapper["tempo_treino_s"],
            ga["tempo_treino_s"]
        ]
    )

    plt.title("Tempo de Treinamento")
    plt.ylabel("Segundos")

    plt.savefig(
        "results/output/grafico_tempo_treinamento.png"
        )


    plt.close()

    print("\nGráficos gerados com sucesso!")


if __name__ == "__main__":
    main()

"""
main_wrapper.py — Integrante 2
Este script realiza a seleção de atributos utilizando o método Wrapper (Forward Selection),
treina o modelo com as características selecionadas e avalia o resultado.
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

# Para evitar que o Forward Selection demore dias rodando nos 784 atributos,
# definimos um limite máximo de features e uma paciência para parada precoce.
MAX_FEATURES_TO_SELECT = 50  
PATIENCE = 5  # Quantas iterações sem melhora significativa na acurácia para interromper a busca


def forward_selection(X_train, y_train, max_features=30, patience=3):
    """
    Versão Otimizada do Forward Selection usando subamostragem para acelerar a busca.
    """
    print("\nIniciando a busca Wrapper (Forward Selection)...")
    
    TAMANHO_AMOSTRA_BUSCA = 3000 
    if X_train.shape[0] > TAMANHO_AMOSTRA_BUSCA:
        # Pega as primeiras X amostras 
        X_busca = X_train[:TAMANHO_AMOSTRA_BUSCA]
        y_busca = y_train[:TAMANHO_AMOSTRA_BUSCA]
    else:
        X_busca = X_train
        y_busca = y_train

    # Divisão interna de validação sobre a amostra (80% treino busca / 20% val busca)
    n_samples = X_busca.shape[0]
    split_idx = int(n_samples * 0.8)
    
    X_tr_inner, X_val_inner = X_busca[:split_idx], X_busca[split_idx:]
    y_tr_inner, y_val_inner = y_busca[:split_idx], y_busca[split_idx:]
    # ─────────────────────────────────────────────────────────────────────────────
    
    n_features = X_train.shape[1]
    features_selecionadas = []
    features_restantes = list(range(n_features))
    
    melhor_acuracia_global = 0.0
    sem_melhora_cont = 0
    
    for i in range(max_features):
        melhor_acuracia_it = -1.0
        melhor_feature_it = None
        
        t_inicio_it = time.time()
        
        # Testa cada feature restante
        for feature in features_restantes:
            features_teste = features_selecionadas + [feature]
            
            # max_depth limitado agiliza muito o treino interno da busca
            clf = DecisionTreeClassifier(max_depth=10, random_state=1)
            clf.fit(X_tr_inner[:, features_teste], y_tr_inner)
            
            preds = clf.predict(X_val_inner[:, features_teste])
            acuracia = accuracy_score(y_val_inner, preds)
            
            if acuracia > melhor_acuracia_it:
                melhor_acuracia_it = acuracia
                melhor_feature_it = feature
                
        if melhor_feature_it is not None:
            features_selecionadas.append(melhor_feature_it)
            features_restantes.remove(melhor_feature_it)
            
            t_fim_it = time.time() - t_inicio_it
            print(f"Iteração {i+1}: Add Feature {melhor_feature_it} | Acurácia: {melhor_acuracia_it:.4f} | Tempo: {t_fim_it:.2f}s")
            
            if melhor_acuracia_it > melhor_acuracia_global + 0.001:
                melhor_acuracia_global = melhor_acuracia_it
                sem_melhora_cont = 0
            else:
                sem_melhora_cont += 1
                
            if sem_melhora_cont >= patience:
                print(f"Parada precoce ativada: {patience} iterações sem melhora.")
                break
        else:
            break
            
    return features_selecionadas


def main():
    print("=" * 50)
    print("Método Wrapper — Forward Selection")
    print("=" * 50)

    # 1. Carregamento e Pré-processamento
    X_train, y_train, X_test, y_test = carregar_fashion_mnist(DATA_DIR)
    X_train_norm, X_test_norm = preprocessar(X_train, X_test)

    # 2. Executar a busca utilizando apenas os dados de treinamento e medir o tempo
    tempo_inicio_busca = time.time()
    features_selecionadas = forward_selection(
        X_train_norm, 
        y_train, 
        max_features=MAX_FEATURES_TO_SELECT, 
        patience=PATIENCE
    )
    tempo_busca = time.time() - tempo_inicio_busca

    num_features_sel = len(features_selecionadas)
    porcentagem_features = (num_features_sel / X_train_norm.shape[1]) * 100

    print("\n" + "-"*40)
    print(f"Busca finalizada!")
    print(f"Tempo de busca: {tempo_busca:.2f} segundos")
    print(f"Quantidade de features selecionadas: {num_features_sel} de {X_train_norm.shape[1]} ({porcentagem_features:.2f}%)")
    print(f"Índices das features: {features_selecionadas}")
    print("-"*40)

    # 3. Treinar a Decision Tree final utilizando APENAS as features selecionadas
    print("\nTreinando o modelo final com as features selecionadas...")
    X_train_sel = X_train_norm[:, features_selecionadas]
    X_test_sel = X_test_norm[:, features_selecionadas]

    clf_wrapper = DecisionTreeClassifier(random_state=1)
    
    tempo_inicio_treino = time.time()
    clf_wrapper.fit(X_train_sel, y_train)
    tempo_treino = time.time() - tempo_inicio_treino

    # 4. Avaliar a acurácia no conjunto de teste completo
    print("Avaliando o modelo no conjunto de teste...")
    y_pred = clf_wrapper.predict(X_test_sel)
    acuracia_teste = accuracy_score(y_test, y_pred)
    relatorio = classification_report(y_test, y_pred)

    print(f"\nAcurácia no conjunto de teste: {acuracia_teste * 100:.2f}%")

    # 5. Exportação dos Resultados 
    dados = {
        "abordagem":             "Wrapper (Forward Selection)",
        "acuracia":              acuracia_teste,
        "num_features":          num_features_sel,
        "porcentagem_features":  porcentagem_features,
        "tempo_treino_s":        tempo_treino,
        "tempo_busca_s":         tempo_busca,
        "relatorio":             relatorio,
        "features_selecionadas": features_selecionadas
    }

    salvar_txt(dados, OUTPUT_TXT)
    salvar_npy(dados, OUTPUT_NPY)
    print(f"\nResultados salvos com sucesso em '{OUTPUT_TXT}'")


if __name__ == "__main__":
    main()
"""Classificador Árvore de Decisão (com busca de hiperparâmetros)."""
from pickle import dump

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from breast_cancer_preprocessamento import (PASTA, SEMENTE, preparar_dados,
                                            salvar_artefatos)

ARQUIVO_MODELO = PASTA / 'breast_cancer_tree.model'


def treinar(dados):
    """Treina a árvore e devolve (modelo, acurácia no teste, melhores parâmetros)."""
    criterion = ['gini', 'entropy', 'log_loss']
    splitter = ['best', 'random']
    min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=5)]
    max_depth = [None, 3, 5, 7, 10]
    grade = {
        'criterion': criterion,
        'splitter': splitter,
        'min_samples_split': min_samples_split,
        'max_depth': max_depth,
    }

    busca = RandomizedSearchCV(
        estimator=DecisionTreeClassifier(random_state=SEMENTE),
        param_distributions=grade,
        n_iter=20,
        cv=5,
        random_state=SEMENTE,
        n_jobs=-1,
    )
    busca.fit(dados['x_treino'], dados['y_treino'])

    modelo = DecisionTreeClassifier(random_state=SEMENTE, **busca.best_params_)
    modelo.fit(dados['x_treino'], dados['y_treino'])

    classes_preditas = modelo.predict(dados['x_teste'])
    acuracia = accuracy_score(dados['y_teste'], classes_preditas)
    return modelo, acuracia, busca.best_params_


if __name__ == '__main__':
    dados = preparar_dados()
    cancer_tree, acuracia_tree, parametros = treinar(dados)

    salvar_artefatos(dados)
    with open(ARQUIVO_MODELO, 'wb') as f:
        dump(cancer_tree, f)

    print('Melhores parâmetros:', parametros)
    print('Acurácia (teste):', acuracia_tree)
    print(classification_report(dados['y_teste'],
                                cancer_tree.predict(dados['x_teste'])))

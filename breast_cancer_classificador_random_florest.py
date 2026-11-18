"""Classificador Random Forest (com busca de hiperparâmetros)."""
from pickle import dump

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV

from breast_cancer_preprocessamento import (PASTA, SEMENTE, preparar_dados,
                                            salvar_artefatos)

ARQUIVO_MODELO = PASTA / 'breast_cancer_rf.model'


def treinar(dados):
    """Treina a floresta e devolve (modelo, acurácia no teste, melhores parâmetros)."""
    n_estimators = [int(x) for x in np.linspace(start=10, stop=300, num=10)]
    criterion = ['gini', 'entropy']
    min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=5)]
    max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=3)]
    max_features = ['sqrt', 'log2', None]
    grade = {
        'n_estimators': n_estimators,
        'criterion': criterion,
        'min_samples_split': min_samples_split,
        'max_depth': max_depth,
        'max_features': max_features,
    }

    busca = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=SEMENTE),
        param_distributions=grade,
        n_iter=20,
        cv=5,
        random_state=SEMENTE,
        n_jobs=-1,
    )
    busca.fit(dados['x_treino'], dados['y_treino'])

    modelo = RandomForestClassifier(random_state=SEMENTE, **busca.best_params_)
    modelo.fit(dados['x_treino'], dados['y_treino'])

    classes_preditas = modelo.predict(dados['x_teste'])
    acuracia = accuracy_score(dados['y_teste'], classes_preditas)
    return modelo, acuracia, busca.best_params_


if __name__ == '__main__':
    dados = preparar_dados()
    cancer_rf, acuracia_rf, parametros = treinar(dados)

    salvar_artefatos(dados)
    with open(ARQUIVO_MODELO, 'wb') as f:
        dump(cancer_rf, f)

    print('Melhores parâmetros:', parametros)
    print('Acurácia (teste):', acuracia_rf)
    print(classification_report(dados['y_teste'],
                                cancer_rf.predict(dados['x_teste'])))

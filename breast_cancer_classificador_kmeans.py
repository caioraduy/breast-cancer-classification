"""K-Means não supervisionado, convertido em classificador por voto majoritário.

Os clusters são calculados usando apenas X de treinamento. Após o ajuste,
cada cluster recebe o rótulo mais comum entre os exemplos de treino nele.
Esses rótulos são usados exclusivamente para avaliar/predizer a classe.
"""
from collections import Counter
from pickle import dump

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from breast_cancer_preprocessamento import PASTA, SEMENTE, preparar_dados, salvar_artefatos

ARQUIVO_MODELO = PASTA / 'breast_cancer_kmeans.model'


class KMeansClassificador:
    def __init__(self, n_clusters=2, n_init=10, random_state=SEMENTE):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X, y):
        self.modelo_ = KMeans(n_clusters=self.n_clusters, n_init=self.n_init,
                              random_state=self.random_state)
        grupos = self.modelo_.fit_predict(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.classe_padrao_ = Counter(y).most_common(1)[0][0]
        self.rotulos_ = {}
        self.proporcoes_ = {}
        for grupo in range(self.n_clusters):
            valores = y[grupos == grupo]
            if len(valores):
                contagens = Counter(valores)
                self.rotulos_[grupo] = contagens.most_common(1)[0][0]
                self.proporcoes_[grupo] = np.array(
                    [contagens.get(classe, 0) / len(valores) for classe in self.classes_]
                )
            else:
                self.rotulos_[grupo] = self.classe_padrao_
                self.proporcoes_[grupo] = np.array(
                    [float(c == self.classe_padrao_) for c in self.classes_]
                )
        return self

    def predict(self, X):
        return np.asarray([self.rotulos_[int(g)] for g in self.modelo_.predict(X)])

    def predict_proba(self, X):
        # Frequência das classes por cluster; NÃO são probabilidades clínicas calibradas.
        return np.vstack([self.proporcoes_[int(g)] for g in self.modelo_.predict(X)])


def treinar(dados):
    X = dados['x_treino']
    y = np.asarray(dados['y_treino'])
    # Seleção de hiperparâmetros no treinamento, sem usar o teste.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    melhor = None
    melhor_media = -1.0
    parametros = {}
    for p in ParameterGrid({'n_clusters': [2, 3, 4, 5, 6], 'n_init': [10, 20]}):
        scores = []
        for idx_t, idx_v in cv.split(X, y):
            modelo = KMeansClassificador(**p).fit(X.iloc[idx_t], y[idx_t])
            scores.append(accuracy_score(y[idx_v], modelo.predict(X.iloc[idx_v])))
        media = float(np.mean(scores))
        if media > melhor_media:
            melhor_media, parametros = media, p
    melhor = KMeansClassificador(**parametros).fit(X, y)
    acuracia = float(accuracy_score(dados['y_teste'], melhor.predict(dados['x_teste'])))
    return melhor, acuracia, {**parametros, 'acuracia_media_cv': melhor_media}


if __name__ == '__main__':
    dados = preparar_dados()
    modelo, acuracia, parametros = treinar(dados)
    salvar_artefatos(dados)
    with open(ARQUIVO_MODELO, 'wb') as arquivo:
        dump(modelo, arquivo)
    print('Parâmetros:', parametros)
    print('Acurácia no teste:', round(acuracia, 4))
    print(classification_report(dados['y_teste'], modelo.predict(dados['x_teste'])))

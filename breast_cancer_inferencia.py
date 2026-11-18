"""MÓDULO DE INFERÊNCIA

Recebe os dados de uma nova paciente e determina o indicativo de diagnóstico
(atributo Class), usando o modelo de melhor acurácia salvo pelo treinamento.

Uso:  python breast_cancer_inferencia.py
(rode antes: python breast_cancer_treinamento.py)
"""
from pickle import load

import pandas as pd

from breast_cancer_preprocessamento import (ARQUIVO_COLUNAS,
                                            ARQUIVO_NORMALIZADOR,
                                            COLUNAS_CATEGORICAS, PASTA)

ARQUIVO_MELHOR_MODELO = PASTA / 'breast_cancer_melhor.model'

SIGNIFICADO = {
    'recurrence-events': 'com indicativo de recorrência',
    'no-recurrence-events': 'sem indicativo de recorrência',
}


def carregar(caminho):
    if not caminho.exists():
        raise FileNotFoundError(
            f"'{caminho.name}' não encontrado. "
            "Execute antes: python breast_cancer_treinamento.py")
    with open(caminho, 'rb') as f:
        return load(f)


def montar_vetor(paciente, colunas):
    """Converte o dicionário da paciente no vetor (one-hot) esperado pelo modelo."""
    if not 1 <= int(paciente['deg-malig']) <= 3:
        raise ValueError("'deg-malig' deve ser 1, 2 ou 3.")
    linha = {'deg-malig': int(paciente['deg-malig'])}
    for atributo in COLUNAS_CATEGORICAS:
        valor = str(paciente[atributo]).replace('_', '-')
        coluna = f'{atributo}_{valor}'
        if coluna not in colunas:
            validos = [c.split('_', 1)[1] for c in colunas
                       if c.startswith(atributo + '_')]
            raise ValueError(
                f"Valor inválido para '{atributo}': '{valor}'. "
                f"Valores aceitos: {validos}")
        linha[coluna] = 1
    return pd.DataFrame([linha]).reindex(columns=colunas, fill_value=0)


def prever(paciente):
    """Devolve (classe prevista, probabilidades por classe)."""
    colunas = carregar(ARQUIVO_COLUNAS)
    normalizador = carregar(ARQUIVO_NORMALIZADOR)
    modelo = carregar(ARQUIVO_MELHOR_MODELO)

    vetor = montar_vetor(paciente, colunas)
    vetor_normalizado = pd.DataFrame(normalizador.transform(vetor),
                                     columns=colunas)
    classe = modelo.predict(vetor_normalizado)[0]
    probabilidades = dict(zip(modelo.classes_,
                              modelo.predict_proba(vetor_normalizado)[0]))
    return classe, probabilidades


if __name__ == '__main__':
    nova_paciente = {
        'age': '50-59',
        'menopause': 'ge40',
        'tumor-size': '30-34',
        'inv-nodes': '0-2',
        'node-caps': 'no',
        'deg-malig': 2,
        'breast': 'left',
        'breast-quad': 'left-low',
        'irradiat': 'no',
    }

    classe, probabilidades = prever(nova_paciente)
    print('Classe prevista (Class):', classe)
    print('Interpretação:', SIGNIFICADO.get(classe, classe))
    for rotulo, p in probabilidades.items():
        print(f'  {rotulo} (frequência estimada pelo modelo): {p:.2%}')

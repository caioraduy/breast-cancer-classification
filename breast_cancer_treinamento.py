"""MÓDULO DE TREINAMENTO

Executa todo o procedimento: pré-processamento, treino da Árvore de Decisão e
da Random Forest e do K-Means, comparação das acurácias (no conjunto de teste) e gravação do
modelo de melhor acurácia em 'breast_cancer_melhor.model', que é o usado pelo
módulo de inferência.

Uso:  python breast_cancer_treinamento.py
"""
import json
from pickle import dump

import breast_cancer_classificador_kmeans as kmeans
import breast_cancer_classificador_random_florest as floresta
import breast_cancer_classificador_tree as arvore
from breast_cancer_preprocessamento import PASTA, preparar_dados, salvar_artefatos

ARQUIVO_MELHOR_MODELO = PASTA / 'breast_cancer_melhor.model'
ARQUIVO_RESULTADOS = PASTA / 'breast_cancer_resultados.json'


def main():
    dados = preparar_dados()
    salvar_artefatos(dados)

    candidatos = {
        'Árvore de Decisão': (arvore, arvore.ARQUIVO_MODELO),
        'Random Forest': (floresta, floresta.ARQUIVO_MODELO),
        'K-Means': (kmeans, kmeans.ARQUIVO_MODELO),
    }

    resultados = {}
    modelos = {}
    for nome, (modulo, arquivo) in candidatos.items():
        print(f'Treinando {nome}...')
        modelo, acuracia, parametros = modulo.treinar(dados)
        with open(arquivo, 'wb') as f:
            dump(modelo, f)
        modelos[nome] = modelo
        resultados[nome] = {'acuracia_teste': acuracia,
                            'parametros': parametros}
        print(f'  acurácia (teste): {acuracia:.4f}')

    melhor_nome = max(resultados, key=lambda n: resultados[n]['acuracia_teste'])
    with open(ARQUIVO_MELHOR_MODELO, 'wb') as f:
        dump(modelos[melhor_nome], f)

    resumo = {'melhor_modelo': melhor_nome, 'resultados': resultados}
    with open(ARQUIVO_RESULTADOS, 'w', encoding='utf-8') as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nMelhor modelo: {melhor_nome} "
          f"(acurácia {resultados[melhor_nome]['acuracia_teste']:.4f})")
    print(f'Salvo em: {ARQUIVO_MELHOR_MODELO.name}')


if __name__ == '__main__':
    main()

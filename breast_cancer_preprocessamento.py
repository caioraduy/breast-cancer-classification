"""Pré-processamento compartilhado entre os módulos de treinamento e inferência.

Etapas (as mesmas vistas em sala):
1. leitura do CSV;
2. codificação das variáveis categóricas (one-hot / get_dummies);
3. separação em treino (80%) e teste (20%), estratificada pela classe;
4. balanceamento das classes com SMOTE (somente no conjunto de treino);
5. normalização com MinMaxScaler (ajustado somente no treino).
"""
from pathlib import Path
from pickle import dump

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

PASTA = Path(__file__).resolve().parent
ARQUIVO_DADOS = PASTA / 'breast-cancer.csv'
ARQUIVO_DADOS_UCI = PASTA / 'breast-cancer.data'
COLUNAS_UCI = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes',
               'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
ARQUIVO_NORMALIZADOR = PASTA / 'breast_cancer_normalizador.model'
ARQUIVO_COLUNAS = PASTA / 'breast_cancer_colunas.model'

SEMENTE = 42
COLUNA_CLASSE = 'Class'
COLUNAS_CATEGORICAS = ['age', 'menopause', 'tumor-size', 'inv-nodes',
                       'node-caps', 'breast', 'breast-quad', 'irradiat']


def carregar_dados(caminho=ARQUIVO_DADOS):
    """Lê CSV com cabeçalho ou o arquivo .data original da UCI (sem cabeçalho)."""
    caminho = Path(caminho)
    if not caminho.exists() and caminho == ARQUIVO_DADOS and ARQUIVO_DADOS_UCI.exists():
        caminho = ARQUIVO_DADOS_UCI
    if not caminho.exists():
        raise FileNotFoundError(
            "Baixe breast-cancer.data na UCI e coloque na pasta do projeto, "
            "ou forneça breast-cancer.csv com as 10 colunas descritas no README.")
    if caminho.suffix == '.data':
        dados = pd.read_csv(caminho, header=None, names=COLUNAS_UCI)
    else:
        dados = pd.read_csv(caminho)
    faltantes = set(COLUNAS_UCI) - set(dados.columns)
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(faltantes)}")
    dados = dados[COLUNAS_UCI].replace('?', pd.NA)
    dados['deg-malig'] = pd.to_numeric(dados['deg-malig'], errors='coerce')
    # A base tem alguns valores ausentes em node-caps / breast-quad.
    for coluna in COLUNAS_CATEGORICAS:
        moda = dados[coluna].mode(dropna=True)
        if moda.empty:
            raise ValueError(f"Atributo sem valores para imputar: {coluna}")
        dados[coluna] = dados[coluna].fillna(moda.iloc[0]).astype(str).str.replace('_', '-', regex=False)
    dados['deg-malig'] = dados['deg-malig'].fillna(dados['deg-malig'].median())
    return dados


def codificar_atributos(atributos):
    """Mantém a parte numérica (deg-malig) e aplica one-hot nas categóricas."""
    atributos_num = atributos.drop(columns=COLUNAS_CATEGORICAS)
    atributos_cat = atributos[COLUNAS_CATEGORICAS]
    atributos_cat_codificados = pd.get_dummies(
        atributos_cat, prefix=COLUNAS_CATEGORICAS, dummy_na=False, dtype=int)
    return atributos_num.join(atributos_cat_codificados)


def preparar_dados():
    """Devolve um dicionário com treino/teste já balanceados e normalizados."""
    dados = carregar_dados()
    classes = dados[COLUNA_CLASSE]
    atributos = codificar_atributos(dados.drop(columns=[COLUNA_CLASSE]))
    colunas = list(atributos.columns)

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        atributos, classes, test_size=0.2, stratify=classes,
        random_state=SEMENTE)

    x_treino_bal, y_treino_bal = SMOTE(random_state=SEMENTE).fit_resample(
        x_treino, y_treino)

    normalizador = MinMaxScaler().fit(x_treino_bal)
    x_treino_norm = pd.DataFrame(normalizador.transform(x_treino_bal),
                                 columns=colunas)
    x_teste_norm = pd.DataFrame(normalizador.transform(x_teste),
                                columns=colunas)

    return {
        'x_treino': x_treino_norm, 'y_treino': y_treino_bal,
        'x_teste': x_teste_norm, 'y_teste': y_teste,
        'normalizador': normalizador, 'colunas': colunas,
    }


def salvar_artefatos(dados):
    """Salva o normalizador e a lista de colunas usados pela inferência."""
    with open(ARQUIVO_NORMALIZADOR, 'wb') as f:
        dump(dados['normalizador'], f)
    with open(ARQUIVO_COLUNAS, 'wb') as f:
        dump(dados['colunas'], f)

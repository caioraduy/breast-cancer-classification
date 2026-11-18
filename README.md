# Breast Cancer (UCI dataset 14) — Classificação e K-Means

Trabalho de aprendizado de máquina com **módulo de treinamento** e **módulo de inferência** para prever o atributo `Class`: `no-recurrence-events` ou `recurrence-events`.

> **Importante:** este conjunto de dados não prevê se uma paciente tem câncer. As duas classes representam a **ocorrência ou não de recorrência** no conjunto de dados histórico. O resultado é apenas acadêmico e não constitui diagnóstico médico.

## Base de dados

Fonte: https://archive.ics.uci.edu/dataset/14/breast+cancer

A base possui 286 registros, 9 atributos preditores e valores ausentes. Faça o download do arquivo **`breast-cancer.data`** na página da UCI e coloque-o na pasta do projeto. O programa aceita também `breast-cancer.csv` com cabeçalho e as colunas:

`Class, age, menopause, tumor-size, inv-nodes, node-caps, deg-malig, breast, breast-quad, irradiat`

O código não inclui dados reais no repositório. A variável `breast-quad` usa valores como `left-low` e `right-up` (com hífen).

## Arquivos

| Arquivo | Função |
| --- | --- |
| `breast_cancer_preprocessamento.py` | Leitura, tratamento de faltantes, one-hot, split 80/20, SMOTE e MinMaxScaler |
| `breast_cancer_classificador_tree.py` | Árvore de decisão com RandomizedSearchCV |
| `breast_cancer_classificador_random_florest.py` | Random Forest com RandomizedSearchCV |
| `breast_cancer_classificador_kmeans.py` | **K-Means**, seleção do número de clusters via validação cruzada e voto majoritário por cluster |
| `breast_cancer_treinamento.py` | Treina os três, compara a acurácia de teste e salva o de maior acurácia |
| `breast_cancer_inferencia.py` | Carrega o modelo selecionado e recebe uma paciente para prever `Class` |

## Como executar

```bash
pip install -r requirements.txt
python breast_cancer_treinamento.py
python breast_cancer_inferencia.py
```

O treinamento cria `breast_cancer_resultados.json` e arquivos `.model` (ignorados pelo Git). O resultado inclui o modelo vencedor e a acurácia no conjunto de teste.

### Informar uma nova paciente

Edite o dicionário `nova_paciente` em `breast_cancer_inferencia.py`, ou use a função em outro script:

```python
from breast_cancer_inferencia import prever

paciente = {
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
classe, proporcoes = prever(paciente)
print(classe, proporcoes)
```

## Como o K-Means participa

O **K-Means é não supervisionado** e não prevê `Class` diretamente. Para compará-lo com os outros modelos, cada cluster recebe a classe mais frequente entre as amostras de treino atribuídas a ele. O número de clusters (2 a 6) é escolhido pela acurácia média em 5 partes do conjunto de treino, e só depois o modelo é testado. `predict_proba` do adaptador devolve proporções das classes por cluster, **não probabilidades clínicas calibradas**.

Os modelos de Árvore e Random Forest usam busca de hiperparâmetros. A avaliação principal é a acurácia de teste estratificado, com 20% dos registros. O conjunto é pequeno e as classes são desbalanceadas; portanto, acurácia isolada não basta para uso clínico.

**Limitação metodológica do projeto original:** SMOTE e a imputação/codificação inicial são aplicados antes da validação cruzada dos hiperparâmetros; para estimar generalização com maior rigor, esses passos deveriam estar dentro de um pipeline ajustado separadamente em cada dobra. O teste é separado antes do SMOTE e seu desempenho é registrado apenas após o treinamento.

## Publicar no GitHub

Crie um repositório vazio no GitHub e execute na pasta deste projeto:

```bash
git init
git add .
git commit -m "Adiciona treinamento, inferência e K-Means"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/breast-cancer-uci.git
git push -u origin main
```

Substitua `SEU_USUARIO` pelo seu nome de usuário. O link para entrega será o URL real do repositório após o push.

Fonte: Zwitter, M.; Soklic, M. (1988). *Breast Cancer*. UCI Machine Learning Repository. https://doi.org/10.24432/C51P4M

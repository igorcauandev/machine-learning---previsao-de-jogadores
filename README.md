# Machine-Learning-Previsao-De-Jogadores

Projeto desenvolvido para aplicar técnicas de Machine Learning Supervisionado no contexto do futebol, utilizando algoritmos de regressão para realizar previsões sobre jogadores.

## Sobre o Projeto

O sistema realiza duas tarefas principais:

### 1. Previsão de Nota Geral

A partir dos atributos de um jogador:

- Velocidade
- Chute
- Passe
- Resistência
- Drible

O modelo prevê sua nota geral.

### 2. Previsão de Gols na Temporada

A partir dos atributos:

- Posição
- Nota Geral
- Chute
- Jogos
- Assistências

O modelo prevê a quantidade de gols marcados na temporada.

---

## Modelos Utilizados

O projeto compara dois algoritmos de Machine Learning:

### Regressão Linear

Modelo estatístico que busca encontrar uma relação matemática entre as variáveis de entrada e a variável de saída.

### Random Forest Regressor

Modelo baseado em múltiplas árvores de decisão, capaz de capturar relações mais complexas entre os dados.

---

## Métricas de Avaliação

Os modelos são avaliados utilizando:

### MAE (Mean Absolute Error)

Mede o erro médio entre os valores reais e os valores previstos.

### R² (Coeficiente de Determinação)

Indica o quanto o modelo consegue explicar os dados.

- Próximo de 1 → Excelente
- Próximo de 0 → Fraco

---

## Estrutura do Projeto
projeto

├── main.py

├── jogadores_notas.csv

├── jogadores_gols.json

└── README.md

---

## Tecnologias Utilizadas

- Python 3
- NumPy
- Pandas
- Scikit-Learn

---

## Instalação

Instale as dependências do projeto:

```bash
pip install numpy pandas scikit-learn
```

Ou:

```bash
pip install -r requirements.txt
```

---

## Execução

Execute o projeto com:

```bash
python main.py
```

---

## Funcionalidades

- Leitura de arquivos CSV e JSON
- Divisão automática dos dados em treino e teste
- Treinamento de Regressão Linear
- Treinamento de Random Forest
- Comparação automática dos modelos
- Cálculo de métricas de desempenho
- Análise de importância das variáveis
- Previsão para novos jogadores

---

## Exemplo de Saída
[Regressão Linear]

MAE: 1.42

R² : 0.89
[Random Forest]

MAE: 1.10

R² : 0.93
Melhor modelo: Random Forest

---

## Objetivo Acadêmico

Este projeto foi desenvolvido como atividade da disciplina de Aprendizado de Máquina, com o objetivo de aplicar conceitos de:

- Regressão
- Treinamento e teste de modelos
- Avaliação de desempenho
- Comparação entre algoritmos
- Predição baseada em dados

---

## Autor

Desenvolvido por **Igor Dev** para fins acadêmicos.

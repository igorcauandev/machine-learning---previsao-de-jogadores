# Machine-Learning-Previsao-De-Jogadores
Este projeto foi desenvolvido para a disciplina de Aprendizado de Máquina e tem como objetivo aplicar técnicas de Machine Learning Supervisionado no contexto do futebol.

O sistema utiliza dois algoritmos de regressão:

Regressão Linear
Random Forest Regressor

Os modelos são treinados e comparados automaticamente para identificar qual apresenta o melhor desempenho.

Objetivos
Parte 1 – Previsão de Nota Geral

A partir dos atributos de um jogador:

Velocidade
Chute
Passe
Resistência
Drible

o modelo prevê sua nota geral.

Parte 2 – Previsão de Gols

A partir dos atributos:

Posição
Nota Geral
Chute
Jogos
Assistências

o modelo prevê a quantidade de gols que o jogador pode marcar em uma temporada.

Tecnologias Utilizadas
Python 3
NumPy
Pandas
Scikit-Learn
Estrutura dos Arquivos
projeto/
│
├── jogadores_notas.csv
├── jogadores_gols.json
├── main.py
└── README.md
Instalação

Instale as dependências:

pip install numpy pandas scikit-learn
Execução

Execute o programa com:

python main.py
Métricas Utilizadas

O desempenho dos modelos é avaliado através de:

MAE (Mean Absolute Error)
R² (Coeficiente de Determinação)
Funcionalidades
Leitura de arquivos CSV e JSON
Divisão automática entre treino e teste
Treinamento de Regressão Linear
Treinamento de Random Forest
Comparação automática dos modelos
Exibição da importância das variáveis
Previsão para novos jogadores
Autor

Projeto desenvolvido para fins acadêmicos na disciplina de Aprendizado de Máquina.

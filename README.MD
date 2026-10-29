Entendendo o Desafio
Agora é a sua hora de praticar, aprender fazendo e treinar modelos que detectam fraude em transações reais de cartão de crédito, um problema em que a fraude é rara e a acurácia engana.

Nas Aulas, a Expert Isadora percorre o pipeline inteiro num notebook em Python, da coleta dos dados à explicação do modelo, comparando algoritmos pelas métricas que importam em fraude. O link do dataset aparece na Aula 1: confira lá antes de começar.

O Que Criar
Um notebook que carrega o dataset, trata o desbalanceamento, treina e compara modelos e explica o que levou uma transação a ser marcada como fraude. A base traz as colunas Time, Amount e Class, além das variáveis V1 a V28, transformadas por PCA para proteger a privacidade.

O ponto central é o desbalanceamento: 99,8% das transações são normais e só 0,17% são fraude. Um modelo que responde "não é fraude" para tudo acerta quase sempre e não serve para nada, por isso o foco sai da acurácia e vai para recall, precisão e F1.

Como Fazer
Não existe repositório base. Você monta o seu notebook do zero, seguindo o caminho da Expert:

Explore. Carregue a base com o pandas, entenda as colunas e meça a proporção de fraudes;
Prepare. Crie variáveis novas, como o log do valor, padronize com o StandardScaler e separe treino e teste com stratify, para manter a proporção de fraudes nos dois;
Treine. Comece pela regressão logística como baseline e compare com Random Forest e XGBoost, usando o peso das classes e o ajuste do limiar de decisão;
Avalie. Leia o relatório de classificação, a curva ROC e a curva de precisão e recall, com o recall da fraude como métrica principal;
Explique. Veja a importância das variáveis e use o SHAP para mostrar o que pesou em cada decisão.
As ferramentas vêm do scikit-learn, com o NumPy nas transformações. Undersampling e oversampling entram como alternativas para equilibrar as classes, e para escolher entre os dois vale a regra da Expert: teste.

Ideias para Evoluir
Testar undersampling e oversampling e comparar o recall de cada um;
Levar o Random Forest para o pipeline com ajuste de limiar, como a Expert sugere;
Ampliar a busca de hiperparâmetros do GridSearchCV;
Criar variáveis de comportamento ao longo do tempo, como transações em sequência;
Testar outros modelos de classificação e comparar todos pelo recall.
Se você ainda está começando, tudo bem. Um modelo simples, com as métricas certas e bem explicadas, vale mais que cinco algoritmos comparados só pela acurácia.

Uma Ajuda durante o Caminho
Quando as métricas parecerem boas demais, o DIO Agent ajuda a interpretar:

Estou treinando modelos para detectar fraude em cartão,
com uma base em que só 0,17% das transações são fraude.

Me ajude a entender por que a acurácia engana aqui e
como ler recall, precisão e F1 para a classe de fraude.

Importante: não quero uma resposta pronta para copiar.
Quero entender o processo e construir a minha.
O Que Entregar
Suba o notebook para um repositório público na sua conta do GitHub. No README.md, explique:

O problema, e por que o desbalanceamento muda a forma de avaliar;
O que você fez na preparação dos dados;
A comparação entre os modelos, com recall, precisão e F1 da classe de fraude;
O limiar de decisão escolhido e o que o SHAP mostrou;
O que você mudou em relação ao que a Expert fez.
Deixe as saídas salvas no notebook, com as tabelas e as curvas. Evidência de que o modelo roda é o que mais pesa em um portfólio.

Antes de submeter, confira:

O notebook roda do início ao fim, sem célula quebrada;
O dataset é carregado pelo link, como no vídeo, e não fica no repositório;
Todo arquivo citado no README existe e tem conteúdo;
O link enviado é o do repositório, não o de um arquivo dentro dele;
O repositório está na sua conta e público;
O nome do repositório ficou legível, em minúsculas e sem acento.
Resultado Esperado
Ao final, você terá um projeto de detecção de fraude com dados reais, do tratamento à explicação do modelo. O repositório mostra o que mais conta nessa área: escolher a métrica certa para um problema desbalanceado e justificar cada decisão do modelo.

Bons estudos e bom projeto 🚀
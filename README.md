# PyBCOPS

Implementação em Python do algoritmo BCOPS proposto em Guan, L. & Tibshirani, R. (2022). Prediction and outlier detection in classiﬁcation problems. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 84:524–546.

Com o algoritmo BCOPS, nosso objetivo é detectar observações do conjunto de teste cuja classe verdadeira não estava presente durante o treinamento. Essa biblioteca visa a simplificação do uso do algoritmo BCOPS e foi projetada para ser utilizada em conjunto com o pacote scikit-learn. Entretanto, pequenas mudanças nas funções train() e prediction() podem adaptar o código para funcionar com outros classificadores.

* O Notebook Python exemplo1.ipynb contém uma demonstração do uso do algoritmo com dados sintéticos.
* O Notebook Python exemplo_mnist.ipynb contém uma demonstração do uso do algoritmo com a base de dados MNIST. 

PyBCOPS requer as bibliotecas NumPy e Pandas. Matplotlib, Seaborn, scikit-learn e Keras (para a base MNIST) também são necessárias para rodar os exemplos.

Baseado na implementação em R disponível em https://github.com/LeyingGuan/BCOPS.

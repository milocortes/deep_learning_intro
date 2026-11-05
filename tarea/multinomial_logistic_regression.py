import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Regresión multinomial

    En una **regresión multinomial** (también llamada **regresión softmax** o **clasificador maxent**) el target $y$ puede corresponder a más de dos clases. Queremos conocer la probabilidad que $y$ pertenezca a cada una de las clases potenciales $c \in C$, $p(y=c|x)$.

    El clasificador multinomial usa una generalización de la función sigmoide, llamada función *softmax*, para calcular la probabilidad $c \in C$, $p(y=c|x)$. La función softmax toma un vector $z = [z_1, z_2, \dots, z_k]$ de $k$ valores arbitrarios y los mapea a una distribución de probabilidad, con cada valor correspondiente en el rango $[0,1]$, y la suma total de los valores igual a 1. Al igual que la función sigmoide, la función softmax es una función exponencial.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Para un vector $z$ de dimensionalidad $k$, la función softmax se define como:

    \begin{equation}
    \operatorname{softmax}\left(z_i\right)=\frac{\exp \left(z_i\right)}{\sum_{j=1}^k \exp \left(z_j\right)} 1 \leq i \leq k
    \end{equation}

    La función softmax de un vector de entrada $z = [z_1, \dots, z_k]$ es a su vez un vector:

    \begin{equation}
    \operatorname{softmax}(z)=\left[\frac{\exp \left(z_1\right)}{\sum_{i=1}^k \exp \left(z_i\right)}, \frac{\exp \left(z_2\right)}{\sum_{i=1}^k \exp \left(z_i\right)}, \ldots, \frac{\exp \left(z_k\right)}{\sum_{i=1}^k \exp \left(z_i\right)}\right]
    \end{equation}

    Se usa el denominador $\frac{\exp \left(z_k\right)}{\sum_{i=1}^k \exp \left(z_i\right)}$ para normalizar todos los valores en probabilidades.

    Por ejemplo, para un vector:

    \begin{equation}
    z=[0.6,1.1,-1.5,1.2,3.2,-1.1]
    \end{equation}

    La evaluación con la función softmax es :

    \begin{equation}
    \text{softmax}(x) = \text { [0.055, 0.090, 0.006, 0.099, 0.74, 0.010] }
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Al igual que la función sigmoide, el input de la función softmax será el producto punto entre un vector de pesos $w$ y un vector de entrada $x$ más el sesgo $b$. Pero ahora, tendremos que separar los vectores de peso y sesgo para cada $K$ clases :

    \begin{equation}
    p(y=c \mid x)=\frac{\exp \left(w_c \cdot x+b_c\right)}{\sum_{j=1}^k \exp \left(w_j \cdot x+b_j\right)}
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Aprendizaje en la regresión multinomial

    La función de pérdida de la regresión multinomial generaliza la función de pérdida de la regresión logística de 2 a $K$ clases. Recordemos la función de pérdida entropía cruzada para la regresión logística:

    \begin{equation}
    L_{\mathrm{CE}}(\hat{y}, y)=-\log p(y \mid x)=-[y \log \hat{y}+(1-y) \log (1-\hat{y})]
    \end{equation}

    La función de pérdida de la regresión multinomial generaliza los dos términos de la entropía cruzada a $K$ términos. La función de pérdida para un individuo $x$ es la suma de los logs de las $K$ clases, cada uno ponderado por $y_k$, la probabilidad de la clase verdadera:

    \begin{equation}
    \begin{aligned}
    L_{\mathrm{CE}}(\hat{y}, y) & =-\sum_{k=1}^K y_k \log \hat{y}_k \\
    & =-\sum_{k=1}^K y_k \log \hat{p}(y=k \mid x)
    \end{aligned}
    \end{equation}

    Dado que sólo una clase (llamémosla $i$) es la correcta, el vector $y$ toma el valor de 1 sólo para este valor de $k$, $i.e.$, tiene $y_i=1$ y $y_j=0  \quad \forall j \neq i$. Un vector de este tipo con un valor $1$ y el resto $0$ se le llama **one-hot vector**. El término en la suma en la ecuación anterior será 0 excepto para el término correspondiente a la clase verdadera :

    \begin{equation}
    \begin{aligned}
    L_{\mathrm{CE}}(\hat{y}, y) & =-\sum_{k=1}^K \mathbb{1}\{y=k\} \log \hat{p}(y=k \mid x) \\
    & =-\sum_{k=1}^K \mathbb{1}\{y=k\} \log \frac{\exp \left(w_k \cdot x+b_k\right)}{\sum_{j=1}^K \exp \left(w_j \cdot x+b_j\right)}
    \end{aligned}
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    De manera que la función de pérdida entropía cruzada para la regresión multinomial se simplifica al log de la probabilidad estimada de la clase correcta:

    \begin{equation}
    \begin{aligned}
    L_{\mathrm{CE}}(\hat{y}, y) & =-\log \hat{y}_k, \quad(\text { donde } k \text { es la clase correcta) } \\
    & =-\log \frac{\exp \left(w_k \cdot x+b_k\right)}{\sum_{j=1}^K \exp \left(w_j \cdot x+b_j\right)} \quad(\text { donde } k \text { es la clase correcta })(5.36)
    \end{aligned}
    \end{equation}

    El gradiente es muy parecido al gradiente de la regresión logística. Este es la diferencia entre el valor de la clase verdadera $k$ (que es 1) y la probabilidad del clasificador para esa clase $k$, ponderado por el valor del input $x_i$ correspondiente al $i$-ésimo elemento del peso para la clase $k$:

    \begin{equation}
    \begin{aligned}
    \frac{\partial L_{\mathrm{CE}}}{\partial w_{k, i}} & =-(\mathbb{1}\{y=k\}-p(y=k \mid x)) x_i \\
    & =-\left(\mathbb{1}\{y=k\}-\frac{\exp \left(w_k \cdot x+b_k\right)}{\sum_{j=1}^K \exp \left(w_j \cdot x+b_j\right)}\right) x_i
    \end{aligned}
    \end{equation}
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import jax.numpy as np
    import jax
    from jax import random
    import seaborn as sns
    sns.set(
        style='white',
        context='talk'
    )
    red, blue, green = sns.color_palette('Set1', 3)
    from sklearn.metrics import confusion_matrix


    return jax, np


@app.cell
def _(jax, np):
    import keras
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    X_train, X_test = (X_train/255).astype(np.float32), (X_test/255).astype(np.float32)

    Y_train = jax.nn.one_hot(Y_train, 10).astype(np.float32)
    Y_test = jax.nn.one_hot(Y_test, 10).astype(np.float32)
    Y_train
    return X_test, Y_test


@app.cell
def _(jax, np):
    def NLL(W, X, Y, ϵ=1e-10):
        Z = X @ W # (n, 784) @ (784, 10) -> (n,10)
        Yhat = jax.nn.softmax(Z)    
        return -(Y * np.log(Yhat)).sum() / X.shape[0] # 2D sum over i and k

    def accuracy(W, X, Y):
        Z = X @ W
        # no need for softmax here, it's monotone so won't change the result
        return (Z.argmax(axis=1) == Y.argmax(axis=1)).mean()

    return NLL, accuracy


@app.cell
def _(W, X_test, Y_test, accuracy):
    accuracy(W, X_test, Y_test)
    return


@app.cell
def _(NLL, X_test, Y_test, accuracy, np):
    W = np.zeros((X_test.shape[1], Y_test.shape[1])) 
    print('NLL:', NLL(W, X_test, Y_test))
    print('Accuracy', accuracy(W, X_test, Y_test))
    return (W,)


if __name__ == "__main__":
    app.run()

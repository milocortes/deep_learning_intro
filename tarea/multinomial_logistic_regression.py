import marimo

__generated_with = "0.25.2"
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementación en JAX
    """)
    return


@app.cell
def _():
    import jax
    import jax.numpy as jnp

    def predict(weights, bias, X):
        return jax.nn.softmax(jax.lax.batch_matmul(X, weights) + bias)

    def cross_entropy(logprobs, targets):
        target_class = jnp.argmax(targets, axis=1)
        nll = jnp.take_along_axis(
            logprobs, 
            jnp.expand_dims(target_class, axis=1), 
            axis=1
        )
        ce = -jnp.mean(nll)
        return ce

    def loss(weights, bias, X, y):
        y_hat = predict(weights, bias, X)
        return cross_entropy(y_hat, y)

    grad = jax.grad(loss, argnums=(0, 1))

    @jax.jit
    def update_step(weights, bias, predictors, targets, learning_rate):
        dw, db = grad(weights, bias, predictors, targets)
        new_weights = weights - learning_rate * dw
        new_bias = bias - learning_rate * db
        return new_weights, new_bias

    return jax, jnp, loss, predict, update_step


@app.cell
def _(jnp, loss, update_step):
    def train_model(X, y, n_classes, key, n_epochs, n_epoch_report):
        n_features = X.shape[1]
        w = jnp.ones( [n_features, n_classes])
        b = jnp.zeros([n_classes])

        learning_rate = 0.1

        for i in range(n_epochs):
            w, b = update_step(w, b, X, y, learning_rate)
            if i % n_epoch_report == 0:
                current_loss = loss(w, b, X, y)
                print(f"Epoch {i}: Loss = {current_loss:.4f}")

        print(f"Learned Weights: {w}")
        print(f"Learned Bias: {b}")

        return w, b 


    return (train_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Iris Dataset
    This data sets consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica) petal and sepal length, stored in a 150x4 numpy.ndarray

    The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.
    """)
    return


@app.cell
def _(jax):
    from sklearn import datasets


    iris = datasets.load_iris()
    xs = iris.data
    n_classes = max(iris.target) + 1
    ys = jax.nn.one_hot(iris.target, n_classes)
    return iris, n_classes, xs, ys


@app.cell
def _(iris):
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
    ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
    _ = ax.legend(
        scatter.legend_elements()[0], iris.target_names, loc="upper right", title="Classes"
    )
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrenamiento del modelo con JAX
    """)
    return


@app.cell
def _(jax, n_classes, predict, train_model, xs, ys):
    key = jax.random.PRNGKey(0)

    w,b = train_model(xs, ys, n_classes, key, 1200, 100)

    y_hat = predict(w, b, xs )
    return key, y_hat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Performance Metrics
    """)
    return


@app.cell
def _(iris, y_hat, ys):
    from sklearn.metrics import classification_report

    ### Decodificamos one-hot
    ys_label = ys.argmax(axis=1)
    y_hat_label = y_hat.argmax(axis=1)

    ### Métricas de desempeño
    cr = classification_report(ys_label , y_hat_label , target_names=iris.target_names)
    print(cr)
    return classification_report, cr, ys_label


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrenamiento del modelo con [statsmodels](https://www.statsmodels.org/stable/index.html)
    """)
    return


@app.cell
def _():
    import statsmodels.formula.api as smf
    import seaborn as sns
    import numpy as np 

    # Cargamos el conjunto de datos de sns
    df = sns.load_dataset('iris')

    df["species"] = df["species"].replace({'setosa' : 0.0, 'versicolor' : 1.0, 'virginica' : 2.0}).astype(float)

    # Ajustamos el modelo de regresión logística multinomial (mnlogit)
    # La variable dependiente es 'species' (3 categorías)
    # Las variables independientes son las medidas del sépalo y pétalo
    formula = 'species ~ sepal_length + sepal_width + petal_length + petal_width'
    multinomial_model = smf.mnlogit(formula, data=df).fit(method="bfgs")

    # Resumen estadístico
    print(multinomial_model.summary())
    return df, multinomial_model, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Performance Metrics
    """)
    return


@app.cell
def _(df, multinomial_model, np):
    # 1. Predecir probabilidades
    pred_probs = multinomial_model.predict(df.iloc[:, 0:4]).to_numpy()

    # 2. Obtener la clase con mayor probabilidad
    pred_classes = np.argmax(pred_probs, axis=1)
    return (pred_classes,)


@app.cell
def _(classification_report, iris, pred_classes, ys_label):
    ### Métricas de desempeño
    cr_sm = classification_report(ys_label , pred_classes , target_names=iris.target_names)
    print(cr_sm)
    return


@app.cell
def _(cr):
    print(cr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tarea
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | Puntos a evaluar

    * Implementa la función `train_model` con el algoritmo de descenso por gradiente estocástico (SGD) con mini-batch con tamaños $m=1$ y $m=20$.
    * Reentrena los pesos con estas nuevas implementaciones y verifica las métricas de desempeño.
    ///
    """)
    return


@app.cell
def _(np):
    import keras
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    X_train, X_test = (X_train/255).astype(np.float32), (X_test/255).astype(np.float32)


    return X_test, X_train, Y_test, Y_train


@app.cell
def _(X_train, plt):
    def display_image(im, ax=None):
        if ax is None: fig, ax = plt.subplots()
        ax.imshow(im, cmap='gray_r')
        ax.set_xticks([])
        ax.set_yticks([])
    
    display_image(X_train[2, :, :])
    plt.show()

    return


@app.cell
def _(Y_train):
    Y_train
    return


@app.cell
def _(Y_test, Y_train, jax, np):
    Y_train_oh = jax.nn.one_hot(Y_train, 10).astype(np.float32)
    Y_test_oh = jax.nn.one_hot(Y_test, 10).astype(np.float32)

    Y_test_oh
    return Y_test_oh, Y_train_oh


@app.cell
def _(X_test, X_train):
    X_train_flatten = X_train.reshape(-1, 28*28)
    X_test_flatten = X_test.reshape(-1, 28*28)
    return X_test_flatten, X_train_flatten


@app.cell
def _(X_train_flatten, Y_train_oh, key, train_model):
    n_classes_mnist = 10
    w_mnist,b_mnist = train_model(X_train_flatten, Y_train_oh, n_classes_mnist, key, 2000, 100)

    #y_hat = predict(w, b, xs )

    return b_mnist, w_mnist


@app.cell
def _(Y_test_oh):
    Y_test_oh.argmax(axis=1)
    return


@app.cell
def _(X_test_flatten, b_mnist, predict, w_mnist):
    predict(w_mnist, b_mnist, X_test_flatten ).argmax(axis=1)
    return


@app.cell
def _(
    X_test_flatten,
    Y_test_oh,
    b_mnist,
    classification_report,
    predict,
    w_mnist,
):
    print(
        classification_report(predict(w_mnist, b_mnist, X_test_flatten ).argmax(axis=1) , Y_test_oh.argmax(axis=1)  )
    )
    return


if __name__ == "__main__":
    app.run()

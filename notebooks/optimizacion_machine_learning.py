import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimización para Machine Learning
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Machine learning es una de las aplicaciones modernas más importantes de optimización. Cada modelo de machine learning es entrenado usando optimización.

    Desarrollar un modelo de aprendizaje de máquina involucra 5 etapas clave :

    - **El problema** : formular cuáles son las entradas y salidas a modelar,
    - **Los datos** : recolectar y seleccionar datos de entrenamiento para informar el modelo,
    - **El modelo** : elegir una arquitectura para representar el modelo,
    - **La función de pérdida** : diseñar una función de pérdida para evaluar el desempeño del modelo,
    - **La optimización** : implementar un algortimo de optimización para entrenar el modelo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Usamos optimización para ajustar los parámetros del modelo con el objetivo que las salidas de este se ajusten lo mejor posible a los datos al minimizar la función de pérdida. Fundamentalmente, este es un problema inverso que resolvemos usando optimización. Si los datos son el combustible de machine learning, la optimización es el motor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La organización de estos pasos es sólo aproximada, y hay traslapes entre cada etapa. Por ejemplo, elegir el problema y los datos son dos decisiones muy relacionadas. De forma similar, diseñar una función de pérdida personalizada e implementar el algoritmo de optimización están estrechamente vinculados.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Generalmente el modelo de machine learning puede ser escrito como :

    \begin{equation}
    \mathbf{y}=\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x}),
    \end{equation}

    donde son $\mathbf{x}$ los insumos del modelo, $\mathbf{y}$ son las salidas, también conocidas como *targets*,  y $\boldsymbol{\theta}$ son los parámetros que se deben optimizar. La función objetivo, que llamamos *función de pérdida* (*loss function*), $L$, por lo general incluye varios términos que combinan diversos objetivos contrapuestos.

    Para un modelo de regresión, el término de pérdida principal es el Error Cuadrático Médio (MSE) del ajuste del modelo a los datos.

    \begin{equation}
    L=\sum_{j=1}^N\left\|\mathbf{y}_j-\mathbf{f}_{\boldsymbol{\theta}}\left(\mathbf{x}_j\right)\right\|^2 .
    \end{equation}

    Este término de error es sumado sobre los $N$ datos $\left\{\mathbf{x}_j, \mathbf{y}_j\right\}_{j=1}^N$..
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regresion por Mínimos Cuadrados

    La regresión por minimos cuadrados es uno de los conceptos fundacionales de matemáticas aplicadas y estadística, con aplicaciones al análisis de datos, problemas inversos, teoría de control y machine learning.

    El problema de regresión lineal por mínimos cuadrados involucra resolver para los parámetros desconocidos $\mathbf{x}$ de un modelo que es escrito como un sistema lineal de ecuaciones:

    \begin{equation}
    \mathbf{A x}=\mathbf{b} .
    \end{equation}

    Para una matriz invertible $\mathbf{A}$ esto equivale a un problema de inversión de matrices, pero cuando $\mathbf{A}$ no es cuadrada, este requiere una solución vía optimización para encontrar el *mejor* ajuste.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dado un conjunto de datos de las variables predictoras y sus resultados asociados, ordenados como filas de una matriz $\mathbf{A} \in \mathbb{R}^{m \times n}$ y un vector $\mathbf{b} \in \mathbb{R}^m$, respectivamente, la regresión busca encontrar la relación entre las columnas de $\mathbf{A}$ que es más consistente con resultados asociados en $\mathbf{b}$.

    Esta relación es cuantificada por el sistema linear de ecuaciones :

    \begin{equation}
    \mathbf{A x} \approx \mathbf{b},
    \end{equation}

    que indica que el vector $\mathbf{b}$ puede ser aproximado como una combinación lineal de las columnas de $\mathbf{A}$. Esta combinación lineal está dada por el vector, que está por determinarse.

    El vector de salida $\mathbf{b}$ está frecuentemente contaminado con ruido de medición, el cual tipicamente es modelado como la suma de un vector $\boldsymbol{\epsilon}$ de ruido blanco gausiano independiente e identicamente distribuido $\mathbf{b}=\mathbf{b}_{\text {true }}+\boldsymbol{\epsilon}$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Es posible resolver el problema lineal de mínimos cuadrados en (2) analiticamente al calcular el gradiente de la función objetivo e igualarlo a cero. Podemos expandir la función objetivo como :

    \begin{equation}
    \begin{aligned}
    \|\mathbf{A x}-\mathbf{b}\|_2^2 & =(\mathbf{A x}-\mathbf{b})^T(\mathbf{A x}-\mathbf{b}) \\
    & =\left(\mathbf{x}^T \mathbf{A}^T-\mathbf{b}^T\right)(\mathbf{A x}-\mathbf{b}) \\
    & =\mathbf{x}^T \mathbf{A}^T \mathbf{A} \mathbf{x}-\mathbf{x}^T \mathbf{A}^T \mathbf{b}-\mathbf{b}^T \mathbf{A} \mathbf{x}+\mathbf{b}^T \mathbf{b} .
    \end{aligned}
    \end{equation}

    Calcular el gradiente e igualarlo a cero produce:

    \begin{equation}
    \begin{aligned}
    \nabla\|\mathbf{A x}-\mathbf{b}\|_2^2 & =2 \mathbf{A}^T \mathbf{A} \mathbf{x}-\mathbf{A}^T \mathbf{b}-\mathbf{A}^T \mathbf{b}=0 \\
    & \Longrightarrow \mathbf{A}^T \mathbf{A} \mathbf{x}=\mathbf{A}^T \mathbf{b} \\
    & \Longrightarrow \mathbf{x}=\left(\mathbf{A}^T \mathbf{A}\right)^{-1} \mathbf{A}^T \mathbf{b} .
    \end{aligned}
    \end{equation}

    La matriz $\mathbf{A}^{\dagger}=\left(\mathbf{A}^T \mathbf{A}\right)^{-1} \mathbf{A}^T$ es conocida como la pseudo-inversa de Moore-Penrose, y esta está definida incluso para matrices no cuadradas $\mathbf{A}$. Sin embargo, esto es sólo válido cuando $\left(\mathbf{A}^T \mathbf{A}\right)$ es invertible. La matriz $\mathbf{A}^T \mathbf{A}$ es invertible cuando $\mathbf{A}$ tiene $n$ columnas linealmente independientes y $m \geq n$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Descenso del Gradiente Estocástico y ADAM

    Descenso de gradiente estocástico (SGD-Stochastic gradient descent) es una variante del algoritmo de descenso de gradiente que escala a problemas de optimización de altas dimensiones al aproximar el gradiente en cada paso, haciéndolo adecuado para el entrenamiento de modelos sofisticados, como el entrenamiento de redes neuronales de gran escala. ADAM (adaptive momentum estimation) es un algoritmo de optimización que combina SGD con inercia (momentum) y tasa de aprendizaje adaptativa, via AdaGrad y RMSProp, generando actualizaciones suaves y más responsivas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    SGD tiene más sentido en el contexto de machine learning, donde estamos optimizando los parámetros $\boldsymbol{\theta}$ de una función $\mathrm{f}_\theta(\mathrm{x})$, tal como en un problema de regresión o una red neuronal, para ajustarse mejor a los datos observados.

    Dados pares de datos de entrada y salida $\left\{\mathbf{x}_j, \mathbf{y}_j\right\}_{j=1}^N$, buscamos encontrar los parámetros $\boldsymbol{\theta}$ de manera que $\mathbf{f}_{\boldsymbol{\theta}}\left(\mathbf{x}_j\right)$ mejor aproxime $\mathbf{y}_j$ promediado sobre los datos.

    \begin{equation}
    \min _{\boldsymbol{\theta}} \sum_{j=1}^N\left\|\mathbf{f}_{\boldsymbol{\theta}}\left(\mathbf{x}_j\right)-\mathbf{y}_j\right\|^2
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    En machine learning se denomina al objetivo una *función de pérdida* $L(\mathbf{x}, \mathbf{y} ; \boldsymbol{\theta})=\left\|\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x})-\mathbf{y}\right\|^2$ la cual minimizamos sobre $\theta$. Es posible calcular el gradiente de la función de pérdida con respecto a $\boldsymbol{\theta}$. Es posible calcular el gradiente de la función de pérdida con respecto a $\boldsymbol{\theta}$:

    \begin{equation}
    \nabla L=2 \nabla \mathbf{f}_{\boldsymbol{\theta}}^T \cdot\left(\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x})-\mathbf{y}\right)
    \end{equation}


    NOTA: el gradiente de puede ser calculado analiticamente o aproximado usando el algoritmo de backpropagation si se trata de una red neuronal. Backpropagation es esencialmente la regla de la cadena aplicada a capas de una red neuronal usando diferenciación automática. Diferenciación automática y backpropagation son la columna vertebral del entrenamiento de machine learning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    SGD aproxima este gradiente en sólo un subconjunto de los datos, llamado *batch*, en lugar de tomar el conjunto completo de datos. SGD evalúa la función de pérdida y los gradientes en batches aleatorios de datos, añadiendo aleatoriedad al proceso. Esto tiene varios beneficios clave como :
    - Aceleraciones computacionales,
    - Robustez frente al ruido y al estancamiento en mínimos locales,
    - Mejor exploración del landscape de optimización, etc.

    Por lo tanto, SGD introduce un nuevo hiperparámetro : el tamaño del batch.

    La regla de actualización del algoritmo puede ser escrita como :

    \begin{equation}
    \boldsymbol{\theta}_{k+1}=\boldsymbol{\theta}_k-\gamma \nabla_{\boldsymbol{\theta}} L
    \end{equation}

    En principio, nada parece que haya cambiado del algoritmo de descenso de gradiente. Toda la aleatoriedad está escondida dentro del gradiente $\nabla_{\boldsymbol{\theta}} L$. Esta es una característica, no un bug, y significa que podemos extender de forma sencilla SGD con otras reglas de actualización basadas en gradientes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SGD para resolver un problema de Regresión Lineal

    Estamos listos para probar el SGD para resolver un problema de regresión lineal simple.

    Primero, generamos datos a partir de un modelo lineal con dos estados dimensionales de $\boldsymbol{x}$ y agregamos una cantidad relativamente grande de ruido.
    """)
    return


@app.cell
def _(np):
    N = 2000 # number of samples
    A = np.random.randn(N, 2) # design matrix
    true_x = np.array([2.0, -3.0]) # true slopes
    true_b = 5.0
    # True y = A * x + b plus some noise
    noise_std = 0.5
    noise = noise_std * np.random.randn(N)
    y = A.dot(true_x) + true_b + noise
    return A, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Definimos funciones que calculan la predicción del modelo y el error cuadrático medio sobre el conjunto de datos.
    """)
    return


@app.cell
def _(np):
    def predict(A, x, b):
        # A: shape (batch_size, 2)
        # x: shape (2,)
        # b: scalar
        return A.dot(x) + b # shape (batch_size,)

    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    return mean_squared_error, predict


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Definimos la función que calcula el gradiente aproximado sobre un lote de datos.

    El gradiente para los pesos $\mathbf{x}$ es igual a :

    \begin{equation}
    \dfrac{2}{m} \mathbf{A}^T(\mathbf{y}-\boldsymbol{A} \mathbf{x} - b)
    \end{equation}

    y para el intercepto $b$:

    \begin{equation}
    \dfrac{2}{m} \mathbf{1}^T (\mathbf{y}-\boldsymbol{A} \mathbf{x} - b)
    \end{equation}

    donde $\mathbf{1}$ es el vector de unos :

    \begin{equation}
    \mathbf{1}=\left[\begin{array}{l}
    1 \\
    1 \\
    1
    \end{array}\right]
    \end{equation}

    y $m$ es el tamaño de batch.
    """)
    return


@app.cell
def _(np, predict):
    def batch_gradients(A_batch, y_batch, x, b):
        # A_batch, x, b: same shape
        # y_batch: shape (batch_size,)
        batch_size = A_batch.shape[0]
        y_pred = predict(A_batch, x, b)
        residuals = (y_pred - y_batch)
        grad_x = (2.0 / batch_size) * A_batch.T.dot(residuals)
        grad_b = (2.0 / batch_size) * np.sum(residuals)
        return grad_x, grad_b

    return (batch_gradients,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ahora definimos los tres métodos a consideración :
    - Descenso de gradiente estándar (full-batch).
    - Descenso de gradiente estocástico con mini-batch.
    - Descenso de gradiente estocástico con un tamaño de batch de $n=1$. La tasa de aprendizaje es $\gamma$.
    """)
    return


@app.cell
def _(batch_gradients, mean_squared_error, np, predict):
    # Descenso por gradiente estándar
    def full_batch_gd(A, y, gamma=0.01, n_epochs=50):
        x = np.zeros(A.shape[1]) # unknown parameter vector
        b = 0.0
        mse_history = []
        for epoch in range(n_epochs):
            grad_x, grad_b = batch_gradients(A, y, x, b)
            x -= gamma * grad_x
            b -= gamma * grad_b

            y_pred = predict(A, x, b)
            mse = mean_squared_error(y, y_pred)
            mse_history.append(mse)

        results = {
            "x" : x, 
            "b" : b, 
            "mse_history" : mse_history
        }

        return results

    return (full_batch_gd,)


@app.cell
def _(batch_gradients, mean_squared_error, np, predict):
    # SGD con mini-batch (SGD con un tamaño de batch pequeño, n = 50)
    def mini_batch_sgd(A, y, batch_size=50, gamma=0.01, n_epochs=50):
        x = np.zeros(A.shape[1])
        b = 0.0
        N = A.shape[0]
        mse_history = []

        for epoch in range(n_epochs):
            indices = np.random.permutation(N) # Shuffle data
            for start in range(0, N, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                A_batch = A[batch_idx]
                y_batch = y[batch_idx]
                grad_x, grad_b = batch_gradients(A_batch, y_batch, x, b)
                x -= gamma * grad_x
                b -= gamma * grad_b
            # Check MSE after each epoch
            y_pred = predict(A, x, b)
            mse = mean_squared_error(y, y_pred)
            mse_history.append(mse)

        results = {
            "x" : x, 
            "b" : b, 
            "mse_history" : mse_history
        }

        return results

    return (mini_batch_sgd,)


@app.cell
def _(batch_gradients, mean_squared_error, np, predict):
    # SGD con tamaño de batch igual a 1
    def sgd_batch1(A, y, gamma=0.001, n_epochs=50):
        x = np.zeros(A.shape[1])
        b = 0.0
        N = A.shape[0]
        mse_history = []

        for epoch in range(n_epochs):
            indices = np.random.permutation(N)
            for i in indices:
                A_i = A[i:i+1] # shape (1,2)
                y_i = y[i:i+1] # shape (1,)
                grad_x, grad_b = batch_gradients(A_i, y_i, x, b)
                x -= gamma * grad_x
                b -= gamma * grad_b
            # MSE after epoch
            y_pred = predict(A, x, b)
            mse = mean_squared_error(y, y_pred)
            mse_history.append(mse)

        results = {
            "x" : x, 
            "b" : b, 
            "mse_history" : mse_history
        }

        return results

    return (sgd_batch1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Históricamente SGD "verdadero" solía usar un tamaño de batch de 1, aunque ahora en la práctica casi siempre se usa mini-batch. Para SGD con tamaño de batch de 1, se reduce la tasa de aprendizaje para disminuir la sensibilidad al ruido en los datos y el riesgo de divergencia.

    SGD y descenso por gradiente son los extremos de SGD por mini-batch.

    Ejecutemos los métodos:
    """)
    return


@app.cell
def _(A, full_batch_gd, mini_batch_sgd, sgd_batch1, y):
    # Guardemos los resultados de los métodos en un diccionario
    n_epochs = 50
    resultados_metodos = {
        "GD (full-batch)" : full_batch_gd(A, y, gamma=0.01, n_epochs=n_epochs), 
        "Mini-Batch GD" : mini_batch_sgd(A, y, batch_size=50, gamma=0.01, n_epochs=n_epochs), 
        "SGD (batch = 1)" : sgd_batch1(A, y, gamma=0.001, n_epochs=n_epochs)
    } 
    return n_epochs, resultados_metodos


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Comparemos los resultados de los tres métodos. Mini-batch y SGD (batch = 1) convergen en muchos menos épocas que el descenso de gradiente con batch completo.

    Sin embargo, una época denota una única pasada por todos los datos, de manera que una época con tamaño de batch igual a 1 involucra $n = 2000$ pasos individuales, cada uno utilizando un único punto de datos.
    """)
    return


@app.cell
def _(n_epochs, plt, resultados_metodos):
    ## Graficamos los resultados
    fig, ax = plt.subplots()

    for metodo, resultados in resultados_metodos.items():
        ax.semilogy(
            range(n_epochs), 
            resultados["mse_history"], 
            label = metodo, 
            marker='o', 
            linestyle='dotted'
        )
    ax.grid()
    ax.grid(which="minor", color="0.9")
    ax.legend()
    plt.ylabel("MSE (log scale)")
    plt.show()
    return


@app.cell
def _(resultados_metodos):
    resultados_metodos
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verificación de Resultados con [`statsmodels`](https://www.statsmodels.org/stable/index.html)
    """)
    return


@app.cell
def _(A, y):
    import statsmodels.api as sm

    X = sm.add_constant(A)
    model = sm.OLS(y, X)
    results = model.fit()

    print(results.summary())

    return X, sm


@app.cell
def _(np):
    def sigmoid(X):
        return 1.0 / (1.0 + np.exp(-X))

    def predict_log(X, w, b):
        return sigmoid(X.dot(w) + b)

    def loss_cross_entropy(y_true, y_pred):
        # Evitar log(0) con un pequeño épsilon numérico
        #eps = 1e-7
        #y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


    return loss_cross_entropy, predict_log


@app.cell
def _(np, predict_log):
    def batch_gradients_log(X_batch, y_batch, w, b):
        # X_batch, w, b: same shape
        # y_batch: shape (batch_size,)
        batch_size = X_batch.shape[0]
        y_pred = predict_log(X_batch, w, b)
        residuals = (y_pred - y_batch)
        grad_w = (2.0 / batch_size) * X_batch.T * residuals
        grad_b = (2.0 / batch_size) * np.sum(residuals)
        return grad_w, grad_b
    


    return


@app.cell
def _(X, loss_cross_entropy, np, predict_log, sm):
    X_log = sm.datasets.spector.load_pandas().exog.to_numpy()
    y_log = sm.datasets.spector.load_pandas().endog.to_numpy()
    batch_size = X_log.shape[0]

    w = np.zeros(X.shape[1]) # unknown parameter vector
    b = 0.0
    gamma = 0.00999

    mse_history = []
    for epoch in range(400_000):
        y_pred = predict_log(X_log, w, b)
        residuals = (y_pred - y_log)
    
        w -= gamma * (2.0 / batch_size) * X_log.T.dot(residuals)
        b -= gamma * (2.0 / batch_size) * np.sum(residuals)

        y_pred = predict_log(X_log, w, b)
        mse = loss_cross_entropy(y_log, y_pred)
        mse_history.append(mse)
    return mse_history, w


@app.cell
def _(mse_history):
    mse_history[-10:]
    return


@app.cell
def _(w):
    w
    return


@app.cell
def _(sm):
    spector_data = sm.datasets.spector.load_pandas()
    spector_data.exog = sm.add_constant(spector_data.exog)
    logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
    logit_res = logit_mod.fit()
    print(logit_res.summary())
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

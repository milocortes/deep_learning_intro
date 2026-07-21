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
    # Backpropagation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Backpropagation** es actualmente el algoritmo *core* detrás de *deep learning*. Sin este no podríamos entrenar redes neuronales en una cantidad razonable de tiempo.

    ## ¿Qué es Backpropagation?

    Entrenar una red neuronal involucra una *función de pérdida* (**loss function**), una función de los pesos y sesgos de la red que nos dice qué tan bien la red se desempeña en el conjunto de entrenamiento.

    Cuando utilizamos *descenso por gradiente*, usamos el gradiente para decidir como movernos de un punto a otro del landscape de la función de pérdida para encontrar donde la red se desempeña mejor.

    La méta del entrenamiento es minimizar la función de pérdida sobre el conjunto de entrenamiento.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Recordemos que los gradientes son funciones que aceptan un vector como insumo y regresan un valor escalar ($\mathbb{R}^n \rightarrow \mathbb{R}$).

    En una red neuronal, el vector de insumo corresponde a los pesos y a los sesgos, los parámetros que definen el desempeño de la red una vez que la arquitectura está definida.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Simbolicamente, podemos escribir la función de perdida como $L(\boldsymbol{\theta})$, donde $\boldsymbol{\theta}$ es un vector de todos los pesos y sesgos de la red.

    Nuestro objetivo es movernos a través del espacio de la función de pérdida para encontrar el mínimo, los valores específicos de $\boldsymbol{\theta}$ que generan la mínima pérdida $L$.

    Conseguiremos este objetivo al usar el gradiente de $L(\boldsymbol{\theta})$.

    Por lo tanto, para entrenar una red neuronal via descenso por gradiente, necesitamos conocer la contribución de cada peso y sesgo a la función de pérdida; esto es, necesitamos conocer  $\frac{\partial L}{\partial w}$ para cada peso (o sesgo) $w$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Backpropagation es el algoritmo que nos dice la contribución de cada peso y sesgo de la red en el valor de la función de pérdida ($\frac{\partial L}{\partial w}$).

    Con las derivadas parciales $\frac{\partial L}{\partial w}$, podemos usar descenso por gradiente para mejorar el desempeño de la red para los pasos siguientes del entrenamiento.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// attention | Atención!

    Before we go any further, a word on terminology. You'll often hear machine learning folks use backpropagation as a proxy for the entire process of training a neural network. Experienced practitioners understand what they mean, but people new to machine learning are sometimes a bit confused. To be explicit, backpropagation is the algorithm that finds the contribution of each weight and bias value to the network's error, the $\partial L / \partial w$ 's. Gradient descent is the algorithm that uses the $\partial L / \partial w$ 's to modify the weights and biases to improve the network's performance on the training set.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    En términos generales, Backpropagation es una aplicación de la regla de la cadena. El algoritmo fue propuesto por Rumelhart, Hinton, y Williams en su artículo de 1986 *Learning Representations by Back-propagating Errors*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Render an image from a URL
    mo.image(
        src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ98lgivOLO6ej4kGt1e38ontoTOwCScxYoWqirvnLccg&s=10",
        alt="Marimo logo",
        width=600,
        height=400,
        rounded=True,
        caption="Marimo logo",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Backpropagation comienza con la salida de la red, esto es, con la función de pérdida.

    Después se mueve *hacia atrás* (de ahí el nombre de "backpropagation") hacia capas cada vez más bajas de la red, propagando la señal de error para encontrar la contribución $\partial L / \partial w$ para cada peso y sesgo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Backpropagation a Mano
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Definamos una red neuronal simple, la cual acepta dos valores como input, tiene dos neuronal en su capa oculta, y tiene una sola neurona de salida, como muestra el siguiente digrama.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
    flowchart-elk LR
        subgraph "Input Layer"
            I1("$$x_0$$")
            I2("$$x_1$$")
        end

        subgraph "Hidden Layer"
            A1(("$$a_0$$"))
            A2(("$$a_1$$"))
        end

        subgraph "Output Layer"
            O1(("$$a_2$$"))
        end

        NodeB0["$$b_0$$"]  --> A1

        I1 -- "w0" -->A1
        I1 -- "w1" --> A2



        I2 -- "w2" --> A1
        I2 -- "w3" --> A2

        NodeB1["$$b_1$$"]  --> A2

        A1 -- "w4" --> O1
        A2 -- "w5" --> O1

        NodeB2["$$b_2$$"]  --> O1

        subgraph "Loss Function"
            O1 --> L["$$\dfrac{1}{2} (y - a_2)^2$$ "]
        end
    """
    ).style(width="700px").center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La figura presenta una red con seis pesos, $w_0, \dots, w_5$ y tres sesgos $b_0, b_1$ y $b_2$. Cada valor es un escalar.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Usaremos la función de activación sigmoid en la capa oculta,

    $$
    \sigma(x)=\frac{1}{1+e^{-x}}
    $$

    No usaremos función de activación en la capa de salida. Para entrenar la red, usaremos la función de pérdida de error cuadrado :

    $$
    L = \dfrac{1}{2}\big(y-a_2 \big)^2
    $$

    donde $y$ es la etiqueta, cero o uno, para un ejemplo del entrenamiento y $a_2$ es la salida de la red para las caracterísitcas asociadas con $y$, $x_0$ y $x_1$.

    Escribamos las ecuaciones para un recorrido hacia adelante con esta red, un recorrido que se desplaza de izquierda a derecha desde las entradas $x_0$ y $x_1$ a la salida $a_2$. Las ecuaciones son :

    $$
    \begin{align}
        &z_0=w_0 x_0+w_2 x_1+b_0 \nonumber \\
        &a_0=\sigma\left(z_0\right) \nonumber \\
        &z_1=w_1 x_0+w_3 x_1+b_1 \nonumber \\
        &a_1=\sigma\left(z_1\right) \nonumber \\
        &a_2=w_4 a_0+w_5 a_1+b_2 \tag{1}
    \end{align}
    $$

    Donde hemos introducido valores intermedios $z_0$ y $z_1$ para representar los argumentos de las funciones de activación. Note que $a_2$ no tiene función de activación.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    El argumento de la función de pérdida es $a_2$; $y$ es una constante fija. Sin embargo, $a_2$ depende directamente de $w_4$, $w_5$, $b_2$, y de los valores de $a_1$ y $a_0$, los cuales ellos mismos de $w_0$, $w_1$, $w_2$, $w_3$, $b_0$, $b_1$, $x_0$ y $x_1$.

    Por lo tanto, pensando en términos de pesos y sesgos, podemos escribir la función de pérdida como:

    $$
    L=L\left(w_0, w_1, w_2, w_3, w_4, w_5, b_0, b_1, b_2 ; x_0, x_1, y\right)=L(\boldsymbol{\theta} ; \mathbf{x}, y)
    $$

    donde $\boldsymbol{\theta}$ representa los pesos y los sesgos, las cuales son variables. Las partes después del punto y coma son constantes : el vector de entrada $\boldsymbol{x} = (x_0, x_1)$ y la etiqueta asociada $y$.

    Necesitamos el gradiente de la función de pérdida $\nabla L(\boldsymbol{\theta} ; \mathbf{x}, y)$. Explícitamente, necesitamos todas las derivadas parciales para todos los pesos y sesgos : nueve derivadas parciales en total.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cálculo de las Derivadas Parciales

    Necesitamos las expresiones de todas las derivadas parciales de la función de pérdida de la red. Además, necesitamos una expresión para la derivada de nuestra función de activación sigmoide.


    Comencemos con la función sigmoide, ya que un ingenioso truco permite expresar la derivada en términos de la propia sigmoide, un valor que se calcula durante el recorrido hacia adelante (**forward pass**).

    La derivada de la función sigmoide se muestra a continuación :

    $$
    \begin{align}
    & \sigma^{\prime}(x)=\frac{d}{d x}\left(\frac{1}{1+e^{-x}}\right)=\left(\frac{-1}{\left(1+e^{-x}\right)^2}\right)\left(-e^{-x}\right) \nonumber \\
    &=\frac{e^{-x}}{\left(1+e^{-x}\right)^2} \nonumber \\
    &=\left(\frac{1}{1+e^{-x}}\right)\left(\frac{e^{-x}}{1+e^{-x}}\right) \nonumber \\
    &=\sigma(x)\left(\frac{e^{-x}}{1+e^{-x}}\right) \nonumber \\
    &=\sigma(x)\left(\frac{1+e^{-x}-1}{1+e^{-x}}\right) \tag{2} \\
    &=\sigma(x)\left(\frac{1+e^{-x}}{1+e^{-x}}-\frac{1}{1+e^{-x}}\right) \nonumber \\
    &=\sigma(x)(1-\sigma(x)) \tag{3}
    \end{align}
    $$

    El truco consiste en sumar y restar uno en el numerador para transformar el factor en otra copia de la propia función sigmoide. Por tanto, la derivada de la sigmoide es el producto de la sigmoide por uno menos la sigmoide.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The trick of Equation 2 is to add and subtract one in the numerator to change the form of the factor to be another copy of the sigmoid itself. So, the derivative of the sigmoid is the product of the sigmoid and one minus the sigmoid. Looking back at Equation 1, we see that the forward pass computes the sigmoids, the activation functions, as a0 and a1. Therefore, during the derivation of the backpropagation partial derivatives, we’ll be able to substitute a0 and a1 via Equation 3 for the derivative of the sigmoid to avoid calculating it a second time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Comencemos con las derivadas. Como lo indica su nombre, con backpropagation trabajaremos hacia atrás desde la función de pérdida y aplicaremos la regla de la cadena para llegar a las expresiones que necesitamos. La derivada de la función de pérdida :

    $$
    L=\frac{1}{2}\left(y-a_2\right)^2
    $$

    es

    $$
    \frac{\partial L}{\partial a_2}=(2)\left(\frac{1}{2}\right)\left(y-a_2\right)(-1)=a_2-y \tag{4}
    $$

    Esto significa que, en todas partes de las expresiones que siguen, podemos reemplazar $\partial L / \partial a_2$ por $a_2 - y$. Recordemos que $y$ es la etiqueta para el ejemplo actual de entrenamiento y que calculamos $a_2$ durante **forward pass** como el producto de la red.

    Ahora encontremos las expresiones para $w_5$, $w_4$ y $b_2$, los parámetros usados para calcular $a_2$. La regla de la cadena nos dice que

    $$
    \frac{\partial L}{\partial w_5}=\left(\frac{\partial L}{\partial a_2}\right)\left(\frac{\partial a_2}{\partial w_5}\right)=\left(a_2-y\right) a_1  \tag{5}
    $$

    dado que

    $$
    \frac{\partial a_2}{\partial w_5}=\frac{\partial\left(w_4 a_0+w_5 a_1+b_2\right)}{\partial w_5}=a_1
    $$

    donde hemos sustituido el valor de $a$ de la ecuación 1.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Una lógica similar nos lleva a las expresiones para $w_4$ y $b_2$ :

    $$
    \begin{align}
    \frac{\partial L}{\partial w_4} & =\left(\frac{\partial L}{\partial a_2}\right)\left(\frac{\partial a_2}{\partial w_4}\right)=\left(a_2-y\right) a_0 \nonumber \\
    \frac{\partial L}{\partial b_2} & =\left(\frac{\partial L}{\partial a_2}\right)\left(\frac{\partial a_2}{\partial b_2}\right)=\left(a_2-y\right)(1)=\left(a_2-y\right) \tag{6}
    \end{align}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Tenemos tres de las nueve derivadas parciales que necesitamos. Escribamos ahora las expresiones para $b_1$, $w_1$ y $w_3$ :

    $$
    \begin{gathered}
    \frac{\partial L}{\partial b_1}=\left(\frac{\partial L}{\partial a_2}\right)\left(\frac{\partial a_2}{\partial a_1}\right)\left(\frac{\partial a_1}{\partial z_1}\right)\left(\frac{\partial z_1}{\partial b_1}\right)=\left(a_2-y\right) w_5 a_1\left(1-a_1\right) \\
    \frac{\partial L}{\partial w_1}=\left(\frac{\partial L}{\partial a_2}\right)\left(\frac{\partial a_2}{\partial a_1}\right)\left(\frac{\partial a_1}{\partial z_1}\right)\left(\frac{\partial z_1}{\partial w_1}\right)=\left(a_2-y\right) w_5 a_1\left(1-a_1\right) x_0 \\
    \frac{\partial L}{\partial w_3}=\left(\frac{\partial L}{\partial a_2}\right)\left(\frac{\partial a_2}{\partial a_1}\right)\left(\frac{\partial a_1}{\partial z_1}\right)\left(\frac{\partial z_1}{\partial w_3}\right)=\left(a_2-y\right) w_5 a_1\left(1-a_1\right) x_1
    \end{gathered}
    $$

    donde usamos :

    $$
    \frac{\partial a_1}{\partial z_1}=\sigma^{\prime}\left(z_1\right)=\sigma\left(z_1\right)\left(1-\sigma\left(z_1\right)\right)=a_1\left(1-a_1\right)
    $$

    sustituyendo $a_1$ por $\sigma(z_1)$ mientras calculamos $a_1$ durante el forward pass.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Un cálculo similar nos da las expresiones para las tres derivadas que nos faltan:

    $$
    \begin{gathered}
    \frac{\partial L}{\partial b_0}=\left(\frac{\partial L}{\partial a_2}\right)\left(\frac{\partial a_2}{\partial a_0}\right)\left(\frac{\partial a_0}{\partial z_0}\right)\left(\frac{\partial z_0}{\partial b_0}\right)=\left(a_2-y\right) w_4 a_0\left(1-a_0\right) \\
    \frac{\partial L}{\partial w_0}=\left(\frac{\partial L}{\partial a_2}\right)\left(\frac{\partial a_2}{\partial a_0}\right)\left(\frac{\partial a_0}{\partial z_0}\right)\left(\frac{\partial z_0}{\partial w_0}\right)=\left(a_2-y\right) w_4 a_0\left(1-a_0\right) x_0 \\
    \frac{\partial L}{\partial w_2}=\left(\frac{\partial L}{\partial a_2}\right)\left(\frac{\partial a_2}{\partial a_0}\right)\left(\frac{\partial a_0}{\partial z_0}\right)\left(\frac{\partial z_0}{\partial w_2}\right)=\left(a_2-y\right) w_4 a_0\left(1-a_0\right) x_1
    \end{gathered}
    $$

    Fue tedioso, pero ahora tenemos lo que necesitamos 🥳.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// attention | Atención!

    Cabe señalar, no obstante, que se trata de un proceso muy rígido: si modificamos la arquitectura de la red, la función de activación o la función de pérdida, debemos volver a derivar estas expresiones.
    ///

    Utilicemos estas expresiones para clasificar flores de iris.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ejecución en Python
    """)
    return


@app.cell
def _():
    import numpy as np
    from sklearn.datasets import load_iris


    def BuildDataset():
        """Create the dataset"""

        #  Get the dataset keeping the first two features
        iris = load_iris()
        x = iris["data"][:,:2]
        y = iris["target"]

        #  Standardize and keep only classes 0 and 1
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        i0 = np.where(y == 0)[0]
        i1 = np.where(y == 1)[0]
        x = np.vstack((x[i0],x[i1]))

        #  Train and test data
        xtrn = np.vstack((x[:35],x[50:85]))
        ytrn = np.array([0]*35 + [1]*35)
        xtst = np.vstack((x[35:50],x[85:]))
        ytst = np.array([0]*15+[1]*15)

        idx = np.argsort(np.random.random(70))
        xtrn = xtrn[idx]
        ytrn = ytrn[idx]
        idx = np.argsort(np.random.random(30))
        xtst = xtst[idx]
        ytst = ytst[idx]

        return xtrn, ytrn, xtst, ytst

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def Forward(net, x):
        """Pass the data through the network"""

        out = np.zeros(x.shape[0])

        for k in range(x.shape[0]):
            z0 = net["w0"]*x[k,0] + net["w2"]*x[k,1] + net["b0"]
            a0 = sigmoid(z0)
            z1 = net["w1"]*x[k,0] + net["w3"]*x[k,1] + net["b1"]
            a1 = sigmoid(z1)
            out[k] = net["w4"]*a0 + net["w5"]*a1 + net["b2"]

        return out

    def Evaluate(net, x, y):
        """Evaluate the network"""

        out = Forward(net, x)
        tn = fp = fn = tp = 0
        pred = []

        for i in range(len(y)):
            c = 0 if (out[i] < 0.5) else 1
            pred.append(c)
            if (c == 0) and (y[i] == 0):
                tn += 1
            elif (c == 0) and (y[i] == 1):
                fn += 1
            elif (c == 1) and (y[i] == 0):
                fp += 1
            else:
                tp += 1

        return tn,fp,fn,tp,pred


    def GradientDescent(net, x, y, epochs, eta):
        """Perform gradient descent"""

        for e in range(epochs):
            #  Pass over training set accumulating deltas
            dw0 = dw1 = dw2 = dw3 = dw4 = dw5 = db0 = db1 = db2 = 0.0

            for k in range(len(y)):
                #  Forward pass
                z0 = net["w0"]*x[k,0] + net["w2"]*x[k,1] + net["b0"]
                a0 = sigmoid(z0)
                z1 = net["w1"]*x[k,0] + net["w3"]*x[k,1] + net["b1"]
                a1 = sigmoid(z1)
                a2 = net["w4"]*a0 + net["w5"]*a1 + net["b2"]

                #  Backward pass
                """
                The next block of code implements the backward pass using the partial derivatives, Equations 4 through 8, to move                    the error (loss) backward through the network. We use the average loss over the training set to update the weights                   and biases. Therefore, we accumulate the contribution to the loss for each weight and bias value for each training                   example. This explains adding each new contribution to the total over the training set.
                """
                db2 += a2 - y[k]
                dw4 += (a2 - y[k]) * a0
                dw5 += (a2 - y[k]) * a1
                db1 += (a2 - y[k]) * net["w5"] * a1 * (1 - a1)
                dw1 += (a2 - y[k]) * net["w5"] * a1 * (1 - a1) * x[k,0]
                dw3 += (a2 - y[k]) * net["w5"] * a1 * (1 - a1) * x[k,1]
                db0 += (a2 - y[k]) * net["w4"] * a0 * (1 - a0)
                dw0 += (a2 - y[k]) * net["w4"] * a0 * (1 - a0) * x[k,0]
                dw2 += (a2 - y[k]) * net["w4"] * a0 * (1 - a0) * x[k,1]

            #  Use average deltas to update the network
            m = len(y)
            net["b2"] = net["b2"] - eta * db2 / m
            net["w4"] = net["w4"] - eta * dw4 / m
            net["w5"] = net["w5"] - eta * dw5 / m
            net["b1"] = net["b1"] - eta * db1 / m
            net["w1"] = net["w1"] - eta * dw1 / m
            net["w3"] = net["w3"] - eta * dw3 / m
            net["b0"] = net["b0"] - eta * db0 / m
            net["w0"] = net["w0"] - eta * dw0 / m
            net["w2"] = net["w2"] - eta * dw2 / m

        #  Training done, return the updated network
        return net


    return BuildDataset, Evaluate, GradientDescent, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// attention | Atención: Uso de los deltas promedio para actualizar la Red

    After passing each training example through the net and accumulating its contribution to the loss, we update the weights and         biases. The partial derivatives give us the gradient, the direction of maximal change; however, we want to minimize, so we           move in the direction opposite to the gradient, subtracting the average of the loss due to each weight and bias from its             current value.

    For example
    ```
    net["b2"] = net["b2"] - eta * db2 / m
    ```
    es

    $$
    b_2 \leftarrow b_2-\eta\left(\left.\frac{1}{m} \sum_{i=0}^{m-1} \frac{\partial L}{\partial b_2}\right|_{\boldsymbol{x}_i}\right)
    $$

    where $\eta=0.1$ is the learning rate and $m$ is the number of samples in the training set. The summation is over the partial for $b_2$ evaluated for each input sample, $\mathbf{x}_{\boldsymbol{i}}$, the average value of which, multiplied by the learning rate, is used to adjust $b_2$ for the next epoch. Another name we frequently use for the learning rate is step size. This parameter controls how quickly the weights and biases of the network step through the loss landscape toward a minimum value.


    ///
    """)
    return


@app.cell
def _(BuildDataset, Evaluate, GradientDescent, np):
    """Build and train a simple neural network"""

    epochs = 1000  # training epochs
    eta = 0.1      # learning rate

    #  Get the train/test data
    xtrn, ytrn, xtst, ytst = BuildDataset()

    #  Initialize the network
    net = {}
    net["b2"] = 0.0
    net["b1"] = 0.0
    net["b0"] = 0.0
    net["w5"] = 0.0001*(np.random.random() - 0.5)
    net["w4"] = 0.0001*(np.random.random() - 0.5)
    net["w3"] = 0.0001*(np.random.random() - 0.5)
    net["w2"] = 0.0001*(np.random.random() - 0.5)
    net["w1"] = 0.0001*(np.random.random() - 0.5)
    net["w0"] = 0.0001*(np.random.random() - 0.5)

    #  Do a forward pass to get initial performance
    tn0,fp0,fn0,tp0,pred0 = Evaluate(net, xtst, ytst)

    #  Gradient descent
    net = GradientDescent(net, xtrn, ytrn, epochs, eta)

    #  Final model performance
    tn,fp,fn,tp,pred = Evaluate(net, xtst, ytst)

    #  Summarize performance
    print()
    print("Training for %d epochs, learning rate %0.5f" % (epochs, eta))
    print()
    print("Before training:")
    print("    TN:%3d  FP:%3d" % (tn0, fp0))
    print("    FN:%3d  TP:%3d" % (fn0, tp0))
    print()
    print("After training:")
    print("    TN:%3d  FP:%3d" % (tn, fp))
    print("    FN:%3d  TP:%3d" % (fn, tp))
    print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Again, this exercise’s main point is to see how tedious and potentially error-prone it is to calculate derivatives by hand. The code above works with scalars; it doesn’t process vectors or matrices to take advantage of any symmetry possible by using a better representation of the backpropagation algorithm. Thankfully, we can do better. Let’s look again at the backpropagation algorithm for fully connected networks and see if we can use vectors and matrices to arrive at a more elegant approach.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backpropagation para una Red Completamente Conectada
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Red Completamente Conectada

    Una Red *Completamente Conectada* representa la arquitectura más popular y simple de redes neuronales. Cada neurona de la capa previa es conectada a cada neurona de la capa siguiente. De manera que si la capa previa tiene $m$ neuronas y la siguiente capa tiene $n$ neuronas, habrán $m \times n$ conexiones, cada una con su propio peso.

    El peso de la conexión de la $k$-ésima neurona en la capa $(l-1)$ a la neurona $j$-ésima en la capa $l$ se denota $w_{jk}^{(l)}$. Aquí, el orden de los subíndices es el destino (j) seguido del origen (k). Esto es ligeramente contraintuitivo, pero se sigue universalmente porque simplifica la notación matricial.


    La siguiente figura muestra una Red *Completamente Conectada*. En la Red la Capa lineal $l$ se genera a partir de la capa $(l − 1)$. Los pesos que pertenecen a la fila 1 de la matriz de pesos (provenientes de todas las neuronas de entrada, capa $(l − 1)$, que se suman para formar la neurona de salida 1) se muestran en negrita.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="images/fully_connected.png", width=600, caption="Red Completamente Conectada")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sea $a_0^{(l-1)}, a_1^{(l-1)}, \cdots, a_m^{(l-1)}$ valores que representan la salidas de las $m$ neuronas en la capa $(l-1)$. Sea $a_0^{(l)}, a_1^{(l)}, \cdots, a_n^{(l)}$ valores que representan las salidas de las $n$ neuronas en la capa $l$. Nota que tipicamente usamos el símbolo $a$, que representa la activación, para denotar la salida de neuronas individuales. Considera la $j$-ésima neurona de la capa $l$. Especialmente la neurona $z_1^{l}$. Observe los pesos que entran en él y las activaciones en su origen. Su salida es $a_j^{(l)}$, donde :

    $$
    \left.\begin{array}{l}
    z_j^{(l)}=\sum_{k=0}^m w_{j k}^{(l)} a_k^{(l-1)}+b_j^{(l)} \\
    a_j^{(l)}=\sigma\left(z_j^{(l)}\right)
    \end{array}\right\} \text { for } j=0 \cdots n
    $$

    Podemos reescribir la suma en esas ecuaciones como un producto punto entre los vectores de peso y activación :

    $$
    \left.\begin{array}{l}
    z_j^{(l)}=\left[\begin{array}{cccc}
    w_{j 0}^{(l)} & w_{j 1}^{(l)} & \cdots & w_{j m}^{(l)}
    \end{array}\right]\left[\begin{array}{c}
    a_0^{(l-1)} \\
    a_1^{(l-1)} \\
    \ldots \\
    a_m^{(l-1)}
    \end{array}\right]+b_j^{(l)} \\
    a_j^{(l)}=\sigma\left(z_j^{(l)}\right)
    \end{array}\right\} \text { for } j=0 \cdots n
    $$

    El conjunto completo de ecuaciones para todas las capas $j$ puede escribirse de forma compacta usando multiplicaciones de vectores y matrices,

    $$
    \begin{aligned}
    & \vec{z}^{(l)}=W^{(l)} \vec{a}^{(l-1)}+\vec{b}^{(l)} \\
    & \vec{a}^{(l)}=\sigma\left(\vec{z}^{(l)}\right)
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    donde
    - $W^{(l)}$ es una matriz $n \times m$ que representa los pesos de **todas las conexiones de la capa** $l-1$ **a la capa** $l$:

    $$
        W^{(l)}=\left[\begin{array}{cccc}
        w_{00}^{(l)} & w_{01}^{(l)} & \cdots & w_{0 m}^{(l)} \\
        w_{10}^{(l)} & w_{11}^{(l)} & \cdots & w_{1 m}^{(l)} \\
        \vdots & & & \\
        w_{j 0}^{(l)} & w_{j 1}^{(l)} & \cdots & w_{j m}^{(l)} \\
        \vdots & & & \\
        w_{n 0}^{(l)} & w_{n 1}^{(l)} & \cdots & w_{n m}^{(l)}
        \end{array}\right]
    $$

    - $\vec{a}^{(l)}$ representa las activaciones de la capa completa $l$. Aplicar la función sigmoide a un vector equivale a aplicarla individualmente a cada elemento del vector.

    $$
    \begin{array}{ll}
    \vec{a}^{(l)}=\left[\begin{array}{c}
    a_0^{(l)} \\
    a_1^{(l)} \\
    \ldots \\
    a_n^{(l)}
    \end{array}\right] & \vec{a}^{(l-1)}=\left[\begin{array}{c}
    a_0^{(l-1)} \\
    a_1^{(l-1)} \\
    \ldots \\
    \ldots \\
    a_m^{(l-1)}
    \end{array}\right] \\
    \vec{z}^{(l)}=\left[\begin{array}{c}
    z_0^{(l)} \\
    z_1^{(l)} \\
    \ldots \\
    z_n^{(l)}
    \end{array}\right] & \sigma\left(\vec{z}^{(l)}\right)=\left[\begin{array}{c}
    \sigma\left(z_0^{(l)}\right) \\
    \sigma\left(z_1^{(l)}\right) \\
    \ldots \\
    \sigma\left(z_n^{(l)}\right)
    \end{array}\right]
    \end{array}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Observación** : Las capa de una red completamente conectada pueden pensarse como vectores de funciones:

    $$
    \boldsymbol{y}=\boldsymbol{f(x)}
    $$

    donde la entrada de la capa es $\boldsymbol{x}$ y la salida es $\boldsymbol{y}$. La entrada $\boldsymbol{x}$ es la entrada a la red para una muestra de entrenamiento o, si se trabaja con una de las capas ocultas del modelo, la salida de la capa anterior. Ambos son vectores: Cada nodo de una capa produce una única salida escalar que, al agruparse, forma $\boldsymbol{y}$, un vector que representa la salida de la capa.

    El pase hacia adelante recorre las capas de la red en orden, realizando un mapeo de $\boldsymbol{x}_i$ a $\boldsymbol{y}_i$de manera que $\boldsymbol{y}_i$ se convierte en $\boldsymbol{x}_{i+1}$, la entrada a la capa $i+1$. Después que todas las capas son procesadas, usamos la salida de la capa final, llamémosla $\boldsymbol{h}$, para calcular la perdida, $L =(\boldsymbol{x},\boldsymbol{y_\text{true}})$. La pérdida es una medida de cuánto se equivoca la red respecto a la entrada $x$, la cual determinamos comparándola con la etiqueta verdadera $\boldsymbol{y_\text{true}}$.

    Necesitamos propagar el valor de pérdida, o el **error**, hacia atrás a través de la red; este es el **backpropagation step**. Para hacer esto para una red totalmente conectada que utiliza vectores por capa y matrices de pesos, primero debemos ver cómo ejecutar el **forward pass**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Para una capa completamente conectada el forward pass es


    $$
    \boldsymbol{y} = \boldsymbol{W}\boldsymbol{x} + \boldsymbol{b}
    $$

    donde $\boldsymbol{W}$ es la matriz de pesos, $\boldsymbol{x}$ es el vector de entrada, y $\boldsymbol{b}$ es el vector de sesgos.

    Para una capa de activación tenemos :

    $$
    \boldsymbol{y} = \boldsymbol{\sigma}(\boldsymbol{x})
    $$

    para cualquier función de activación $\boldsymbol{\sigma}$ (Nosotros continuaremos usando la función sigmoide). Aplicamos la función sigmoide escalar a cada elemento del vector de entrada para obtener el vector de salida:

    $$
    \boldsymbol{\sigma}(\boldsymbol{x})=\left[\sigma\left(x_0\right) \sigma\left(x_1\right) \ldots \sigma\left(x_{n-1}\right)\right]^{\top}
    $$

    La propagación hacia adelante conduce al resultado final y a la función de pérdida. La derivada de la función de pérdida con respecto a la salida de la red constituye el primer término de error. Para propagar dicho término de error hacia atrás a través del modelo, es necesario calcular cómo varía este en función de un cambio en la entrada de una capa, utilizando para ello la información sobre cómo cambia el error ante una modificación en la salida de la misma capa. En concreto, para cada capa, debemos saber cómo calcular

    $$
    \frac{\partial E}{\partial \boldsymbol{x}}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    necesitamos saber cómo varía el término de error ante un cambio en la entrada a la capa dado :

    $$
    \frac{\partial E}{\partial \boldsymbol{y}}
    $$

    es decir, cómo cambia el término de error ante una variación en la salida de la capa. La regla de la cadena nos indica cómo hacerlo:

    $$
    \frac{\partial E}{\partial \boldsymbol{x}}=\frac{\partial E}{\partial \boldsymbol{y}} \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}}
    $$

    donde $\partial E / \partial \mathbf{x}$ para la capa $l$ se convierte en $\partial E / \partial y$ para la capa $l-1$ en la medida que nos movemos hacia atrás en la red.

    Operacionalmente, el algoritmo Backpropagation sigue los siguientes pasos:

    - Ejecutar las propapagión hacia adelante para mapear $\boldsymbol{x} \rightarrow \boldsymbol{y}$, capa por capa, para obtener la capa de salida $\boldsymbol{h}$.
    - Calcular el valor de la derivada de la función de pérdida usando $\boldsymbol{h}$ y $\boldsymbol{y_{\text{true}}}$; $\partial E / \partial y$ para la capa de salida.
    - Repetir el procedimiento para todas las capas anteriores a fin de realizar el cálculo $\partial E / \partial \mathbf{x}$ a partir de $\partial E / \partial y$, generando que $\partial E / \partial \mathbf{x}$ para la capa $l$ se convierte en $\partial E / \partial y$ para la capa $l-1$

    Este algoritmo propaga el término de error hacia atrás a través de la red. Veamos cómo obtener las derivadas parciales necesarias según el tipo de capa, comenzando por la capa de activación.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Asumiremos que conocemos $\partial E / \partial y$ y que nuestro objetivo es $\partial E / \partial \mathbf{x}$. La regla de la cadena nos diría:

    $$
    \begin{aligned}
    \frac{\partial E}{\partial \boldsymbol{x}} & =\frac{\partial E}{\partial y} \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}} \\
    & =\left[\frac{\partial E}{\partial y_0} \frac{\partial y_0}{\partial x_0} \frac{\partial E}{\partial y_1} \frac{\partial y_1}{\partial x_1} \ldots\right]^{\top} \\
    & =\left[\frac{\partial E}{\partial y_0} \sigma^{\prime}\left(x_0\right) \frac{\partial E}{\partial y_1} \sigma^{\prime}\left(x_1\right) \ldots\right]^{\top} \\
    & =\frac{\partial E}{\partial y} \odot \sigma^{\prime}(\boldsymbol{x})
    \end{aligned}
    $$

    donde introducimos $\odot$ para representar el producto Hadamard.


    /// attention | Producto Hadamard
    El producto Hadamard es la multiplicación elemento por elemento de dos vectores o matrices
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ahora sabemos cómo propagar el término de error a través de una capa de activación. La única otra capa que estamos considerando es una capa totalmente conectada. Si desarrollamos la Ecuación 10.9, obtenemos

    $$
    \begin{aligned}
    \frac{\partial E}{\partial \boldsymbol{x}} & =\frac{\partial E}{\partial \boldsymbol{y}} \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}} \\
    & =\boldsymbol{W}^{\top} \frac{\partial E}{\partial \boldsymbol{y}}
    \end{aligned}
    $$

    dado que

    $$
    \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}}=\frac{\partial(\boldsymbol{W} \boldsymbol{x}+\boldsymbol{b})}{\partial \boldsymbol{x}}=\boldsymbol{W}^{\top}
    $$

    el resultado es $\boldsymbol{W}^{\top}$ y no $\boldsymbol{W}$ porque la derivada de una matriz por un vector, en la notación de denominador, es la transpuesta de la matriz en lugar de la matriz misma.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cálculo de las Derivadas Parciales de los Pesos y Sesgos

    Las ecuaciones 10.10 y 10.11 nos indican cómo propagar el término de error hacia atrás a través de la red. Sin embargo, el objetivo de la backpropagation es calcular cómo afectan al error los cambios en los pesos y los sesgos, de modo que podamos utilizar el descenso de gradiente. En concreto, para cada capa totalmente conectada, necesitamos expresiones para

    $$
    \frac{\partial E}{\partial \boldsymbol{W}} \text { y } \frac{\partial E}{\partial \boldsymbol{b}}
    $$

    dado

    $$
    \frac{\partial E}{\partial y} \text { and } \frac{\partial E}{\partial \boldsymbol{x}}
    $$

    Comencemos con $\partial E / \partial \mathbf{b}$. Aplicando la regla de la cadena una vez más se obtiene

    $$
    \begin{aligned}
    \frac{\partial E}{\partial \boldsymbol{b}} & =\frac{\partial E}{\partial \boldsymbol{y}} \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{b}} \\
    & =\frac{\partial E}{\partial \boldsymbol{y}} \frac{\partial(\boldsymbol{W} \boldsymbol{x}+\boldsymbol{b})}{\partial \boldsymbol{b}} \\
    & =\frac{\partial E}{\partial \boldsymbol{y}}(\mathbf{0}+\mathbf{1}) \\
    & =\frac{\partial E}{\partial \boldsymbol{y}}
    \end{aligned}
    $$

    lo que significa que el error debido al término de sesgo de una capa totalmente conectada es igual al error debido a la salida.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Es similar el cálculo para la matriz de pesos

    $$
    \begin{aligned}
    \frac{\partial E}{\partial \boldsymbol{W}} & =\frac{\partial E}{\partial \boldsymbol{y}} \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{W}} \\
    & =\frac{\partial E}{\partial \boldsymbol{y}} \frac{\partial(\boldsymbol{W} \boldsymbol{x}+\boldsymbol{b})}{\partial \boldsymbol{W}} \\
    & =\frac{\partial E}{\partial \boldsymbol{y}}\left(\boldsymbol{x}^{\top}+\mathbf{0}\right) \\
    & =\frac{\partial E}{\partial \boldsymbol{y}} \boldsymbol{x}^{\top}
    \end{aligned}
    $$

    La ecuación nos dice que el error debido a la matriz de pesos es el producto del error de la salida y de la entrada $\boldsymbol{x}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Las ecuaciones 10.10, 10.12, 10.12 y 10.13 se aplican a un único ejemplo de entrenamiento. Esto significa que, para una entrada específica a la red, estas ecuaciones —especialmente la 10.12 y la 10.13— nos indican la contribución de los sesgos y pesos de cualquier capa a la función de pérdida **para ese insumo muestra** 🤯.

    Para implementar Descenso por Gradiente, necesitamos acumular esos errores, los términos $\partial E / \partial \mathbf{W}$ y $\partial E / \partial \mathbf{b}$ sobre las muestras de entrenamiento.

    Para esto usamos el valor promedio de los errores para actualizar los pesos y sesgos al final de cada época o, como la implementaremos, minibatch.

    En general, sin embargo, para entrenar la red debemos realizar lo siguiente para cada muestra del minibatch:
    - Propagación hacia adelante a través de la red para crear la salida. En el camino, necesitamos almacenar la entrada en cada capa, ya que la necesitamos para implementar la retropropagación (esto es, necesitamos $\boldsymbol{x}^{\top}$ de la ecuación 10.13)
    - Calcule el valor de la derivada de la función de pérdida —que en nuestro caso es el error cuadrático medio— para utilizarlo como el primer término de error en la retropropagación.
    - Recorra las capas de la red en orden inverso, calculando $\partial E / \partial \mathbf{W}$ y $\partial E / \partial \mathbf{b}$ para cada capa completamente conectada. Esos valores son acumulados para cada muestra en el minibatch $(\Delta W, \Delta b)$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Una vez procesadas las muestras del minilote y acumulados los errores, llega el momento de implementar Descenso de Gradiente. Aquí es donde los pesos y sesgos de cada capa se actualizan mediante las reglas de actualización :

    $$
    \begin{aligned}
    \boldsymbol{W} & \leftarrow \boldsymbol{W}-\eta\left(\frac{1}{m} \Delta \boldsymbol{W}\right) \\
    \boldsymbol{b} & \leftarrow \boldsymbol{b}-\eta\left(\frac{1}{m} \Delta \boldsymbol{b}\right)
    \end{aligned}
    $$

    con $\Delta \boldsymbol{W}$ y $\Delta \boldsymbol{b}$ siendo los errores acumulados a lo largo del minilote y $m$ siendo el tamaño del minilote.

    La repetición de pasos de descenso de gradiente conduce a un conjunto final de pesos y sesgos: una red entrenada.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()

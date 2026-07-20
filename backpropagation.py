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
    """)
    return


if __name__ == "__main__":
    app.run()

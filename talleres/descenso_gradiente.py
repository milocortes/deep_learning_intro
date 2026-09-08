import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import numdifftools as nd

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método Steepest Descent

    Todos los métodos de iteración requieren especificar un punto de inicio $\boldsymbol{\theta}_{0}$. En cada iteración $t$ realizan una actualización siguiendo la siguiente regla:


    \begin{equation}
    	\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} + \rho_t \boldsymbol{d}_{t}
    \end{equation}


    donde $\rho_t$ se le conoce como **tamaño de paso** o **tasa de aprendizaje**, y $\boldsymbol{d}_t$ es una **dirección de descenso**.

    Cuando la dirección de descenso es igual al negativo del gradiente ($\textit{i.e}$ $\boldsymbol{d}_t = - \boldsymbol{g}_t $)(Recuerda que el gradiente apunta en la dirección de máximo incremento en $f$, por eso el negativo apunta en la dirección de máxima disminución), la dirección se le conoce como de **steepest descent**.


    \begin{equation}
    	\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} - \rho_t \boldsymbol{g}_{t}
    \end{equation}


    Utilizando una tasa de aprendizaje constante $\rho_t = \rho$, la regla de actualización es:


    \begin{equation}
    	\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} - \rho \boldsymbol{g}_{t}
    \end{equation}


    Para el caso univariado, la regla de actualización es:



    \begin{equation}
    	x_{t+1} = x_{t} - \rho f^\prime (x_{t})
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Steepest Descent en 1D

    Sea la función univariada:

    \begin{equation}
        f(x) = 6x^2 - 12x +3
    \end{equation}

    Graficamos la función
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Como puese verse, la función es una parábola, por lo cual tiene un mínimo local (global).

    ## Cálculo manual de la derivada
    Obtengamos el mínimo de forma analítica al obtener la derivada e igualar a cero.

    \begin{equation}
        \frac{d}{dx} (6x^2 - 12x +3) = 12x - 12
    \end{equation}

    Igualando $12x - 12=0$, tenemos que el mínimo es $x=1$.

    ## Cálculo de la derivada con diferenciación simbólica

    Obtengamos el minimo mediante el método steepest descent.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cálculo de la derivada con Diferenciación Automática

    La [diferenciación automática](https://en.wikipedia.org/wiki/Automatic_differentiation) es un método para evaluar derivadas de funciones representadas como programas usualamente conocido como gráfica de cómputo.[[Automatic Differentiation in Machine Learning: a Survey, Baydin et. al, 2018](https://arxiv.org/abs/1502.05767)]. El programa está compuesto por operaciones elementales como sumas, restas, multiplicaciones y divisiones.

    Una gráfica de cómputo representa una función donde los nodos son operaciones y las aristas son relaciones de entrada-salida. Los nodos hoja de una gráfica computacional son variables de entrada o constantes, y los nodos terminales son valores de salida de la función.

    Hay dos métodos de diferenciación automática usando una gráfica de cómputo.

    * **Forward accumulation** : el método usa **números duales** para recorrer el árbol desde las entradas hasta las salidas.
    * **Backward accumulation** : recorre el árbol de las salidas a las entradas.
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementación de Algoritmo de Optimización de Primer Orden Steepest Descent
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Steepest Descent en 2D

    Sea la función que recibe dos argumentos:


    \begin{equation}
            f(x,y) = 6x^2 + 9y^2 - 12x -14y +3
    \end{equation}

    Obtenemos el gradiente:

    \begin{equation}
    \nabla f(x,y)=  \begin{bmatrix}
    \frac{\partial f(x,y)}{\partial x} \\
    \frac{\partial f(x,y)}{\partial y}
    \end{bmatrix} =
     \begin{bmatrix}
    12x -12 \\
    18y -14
    \end{bmatrix}
    \end{equation}
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

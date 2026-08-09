# Introducción a Deep Learning

## Contenido
* [Introducción a optimización](slides/intro_optimizacion.pdf).
    - Maximización o minimización.
    - Formulación matemática.
    - Restricciones y regiones factibles.
    - Soluciones locales y globales.
    - Derivadas en múltiples dimensiones.
    - Gradientes
* [Métodos de Optimización](https://milocortes.github.io/bootcamp_matematicas/laboratorios/descenso_gradiente/index.html).
    - Métodos de Primer Orden.
        - Descenso de gradiente.
        - Momentum
        - Nesterov momentum
        - RMSprop
        - Adam
        - Descenso de gradiente estocástico.
    - Métodos de Segundo Orden.  

* [Aplicación Algoritmos de Optimización de Primer Orden](https://milocortes.github.io/bootcamp_matematicas/laboratorios/aplicaciones_primer_orden/index.html)
* [Deep Learning]()
    - Redes Neuronales
    - Entrenamiento de Redes Neuronales
    - Backpropagation
* [Procesamiento de Lenguaje Natural]()
    - Word Embeddings
        - Word2Vec
    - Semantic Search
    - Vector Databases

## Descarga del repositorio

Para descargar el repositorio utiliza la instrucción:

```
git clone https://github.com/milocortes/bootcamp_matematicas.git
```

## Sincronización del ambiente virtual

Sincronizamos las dependencias en nuestro ambiente virtual con la instrucción:

```bash 
uv sync
```
>Sincronizar (Syncing) es el proceso de instalar las versiones correctas de las dependencias de un lockfile en el ambiente del proyecto.

#### Inicio del servicio de Marimo

Para iniciar Marimo, ejecutamos :

```bash 
uv run marimo edit
```

#### Inicio del servicio de Jupyter Notebook

Para iniciar Jupyter Notebook, ejecutamos :

```bash 
uv run --with jupyter jupyter notebook
```

## Recursos adicionales

* [Algorithms - The Secret Rules of Modern Living - BBC documentary](https://www.youtube.com/watch?v=k2AqGongii0)
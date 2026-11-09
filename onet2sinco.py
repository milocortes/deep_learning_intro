import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Datos O*NET
    """)
    return


@app.cell
def _(pd):
    ### Cargamos datos de Skills de O*NET
    datos_filename = "Abilities"
    skills = pd.read_table(f"datos/ocupaciones/datos/{datos_filename}.txt")
    skills
    return datos_filename, skills


@app.cell
def _(skills):
    escala_medicion = "LV"
    ### Filtramos por Scale ID == "IM" y por el último levantamiento Date == "08/2023"
    skill_filtradas = skills[
        (skills["Scale ID"] == escala_medicion) 
    ][
        ["O*NET-SOC Code", "Element ID", "Element Name", "Data Value"]
    ].reset_index(
        drop = True
    ).groupby(
        ["O*NET-SOC Code", "Element ID", "Element Name"]
    ).agg(
        {
            "Data Value" : "mean"
        }
    ).reset_index()

    skill_filtradas
    return escala_medicion, skill_filtradas


@app.cell
def _(pd):
    ## Cargamos el archivo de correspondencias ONET-SINCO
    sinco = pd.read_excel(
        "datos/ocupaciones/datos/sinco-onet.xlsx"
    )
    sinco
    return (sinco,)


@app.cell
def _(sinco):
    ### Nos quedamos con las columnas que necesitamos
    sinco_min = sinco[
        ["onetsoccode", "title", "sinco", "nombre_sinco"]
    ]
    sinco_min
    return (sinco_min,)


@app.cell
def _(sinco_min, skill_filtradas):
    ### Reunimos los dataframes
    skill_sinco = skill_filtradas.merge(
        sinco_min, 
        left_on="O*NET-SOC Code", 
        right_on="onetsoccode", 
        how="left"
    )
    skill_sinco
    return (skill_sinco,)


@app.cell
def _(skill_sinco):
    ### Agrupamos por ocupacion SINCO y por las skills
    skill_sinco_agg = skill_sinco.groupby(
        ["sinco", "nombre_sinco", "Element ID", "Element Name"]
    ).agg(
        {
            "Data Value" : "mean"
        }
    ).reset_index()
    skill_sinco_agg
    return (skill_sinco_agg,)


@app.cell
def _(pd):
    ## Carga SINCO
    sinco_cat = pd.read_csv("datos/ocupaciones/datos/sinco_2019_2011.csv")[["cve_sinco_2019", "nombre_sinco_2019"]].dropna()

    ## Nos quedamos sólo con categorías a tres digitos
    sinco_cat = sinco_cat[sinco_cat["cve_sinco_2019"].apply(lambda x : len(x)==3)].reset_index(drop = True)
    sinco_cat
    return (sinco_cat,)


@app.cell
def _(sinco_cat, skill_sinco_agg):
    import polars as pl

    skill_sinco_3d = pl.from_pandas(
        skill_sinco_agg
    ).with_columns(
        sinco_3d = pl.col("sinco").cast(pl.String).map_elements(lambda x : x[:3])
    ).group_by(
        "sinco_3d", "Element ID", "Element Name"
    ).agg(
        pl.col("Data Value").mean()
    ).with_columns(
        year = pl.lit(2026)
    ).to_pandas()

    skill_sinco_3d = skill_sinco_3d.merge(
        sinco_cat, 
        left_on = "sinco_3d", 
        right_on= "cve_sinco_2019", 
        how = "left"
    )

    skill_sinco_3d
    return pl, skill_sinco_3d


@app.cell
def _(skill_sinco_3d):

    from ecomplexity import ecomplexity

    # Calculate complexity
    trade_cols = {'time':'year', 'loc':'sinco_3d', 'prod':'Element Name', 'val':'Data Value'}
    cdata = ecomplexity(skill_sinco_3d, trade_cols)
    cdata
    return (cdata,)


@app.cell
def _(cdata, pl):
    pl.from_pandas(cdata).select("sinco_3d", "nombre_sinco_2019", "eci").unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calculamos ingresos de la ENOE
    """)
    return


@app.cell
def _(pl):
    ## Cargamos datos
    sdemt226 = pl.read_parquet("datos/ocupaciones/datos/ENOE_SDEMT226.parquet")
    coe1t226 = pl.read_parquet("datos/ocupaciones/datos/ENOE_COE1T226.parquet")
    #coe2t226 = pl.read_parquet("datos/ocupaciones/datos/ENOE_COE2T226.parquet")

    ## Función que crea folio
    def crea_folio(df : pl.DataFrame) : 
        return df.with_columns(
            pl.concat_str(
                    [
                        pl.col("cd_a", "cve_ent", "con", "v_sel", "n_hog", "h_mud", "n_ren")
                    ],
                    separator="",
                ).alias("folio"),
        )

    sdemt226 = crea_folio(sdemt226)
    coe1t226 = crea_folio(coe1t226)
    #coe2t226 = crea_folio(coe2t226)
    return coe1t226, sdemt226


@app.cell
def _(coe1t226, pl, sdemt226):
    import numpy as np 

    ## Reune datos
    enoe = sdemt226.join(
        coe1t226, 
        on = "folio"
    )

    enoe = enoe.with_columns(
        sinco  = pl.col("p3").cast(pl.String).map_elements(lambda x : x[:3])
    ).drop_nulls(subset=["ingocup", "sinco"] )

    enoe
    return enoe, np


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(enoe, pl, sinco_cat):
    ## Incorporamos el esquema de muestreo
    import svy

    # Define the survey design
    design = svy.Design(stratum="est", psu="upm", wgt="fac_tri")

    # Create a sample object
    sample = svy.Sample(data=enoe, design=design)

    ingreso_sinco = sample.estimation.mean(y="ingocup", by = "sinco").to_polars()
    ingreso_sinco = ingreso_sinco.join(
        pl.from_pandas(sinco_cat), 
        left_on="sinco", 
        right_on="cve_sinco_2019"
    )

    ingreso_sinco
    return (ingreso_sinco,)


@app.cell
def _(pd):
    ## Carga SINCO
    sinco_1d = pd.read_csv("datos/ocupaciones/datos/sinco_2019_2011.csv")[["cve_sinco_2019", "nombre_sinco_2019"]].dropna()

    ## Nos quedamos sólo con categorías a tres digitos
    sinco_1d = sinco_1d[sinco_1d["cve_sinco_2019"].apply(lambda x : len(x)==1)].reset_index(drop = True)
    sinco_1d = sinco_1d.rename( columns=
        {
            "cve_sinco_2019" : "sinco_1d", 
            "nombre_sinco_2019" : "nombre_sinco_1d"
        }
    ).reset_index(drop = True)
    sinco_1d
    return (sinco_1d,)


@app.cell
def _(cdata, ingreso_sinco, pl, sinco_1d):
    sinco_eci_ingresos = pl.from_pandas(
        cdata
    ).select(
        "sinco_3d", "nombre_sinco_2019", "eci"
    ).unique().join(
        ingreso_sinco.filter(pl.col("cv") <= 0.2).select("sinco", "est"), 
        left_on="sinco_3d", 
        right_on="sinco"
    ).with_columns(
        log_ingresos = pl.col("est").log()
    )

    sinco_eci_ingresos = sinco_eci_ingresos.with_columns(
        sinco_1d = pl.col("sinco_3d").map_elements(lambda x : x[0] , return_dtype=pl.String)
    )

    sinco_eci_ingresos = sinco_eci_ingresos.join(
        pl.from_pandas(sinco_1d), 
        on = "sinco_1d", 
        how = "left"
    )
    sinco_eci_ingresos
    return (sinco_eci_ingresos,)


@app.cell
def _(sinco_eci_ingresos):
    import altair as alt

    ## Crea diagrama de dispersion
    scatter = alt.Chart(sinco_eci_ingresos).mark_circle(size=60).encode(
        x=alt.X('eci', scale=alt.Scale(zero=False)),
        y=alt.Y('est', scale=alt.Scale(zero=False)),
        color = "nombre_sinco_1d",
        tooltip=['nombre_sinco_2019', "est", "nombre_sinco_1d"]
    ).interactive()

    # Crear la línea de regresión
    line = alt.Chart(sinco_eci_ingresos).mark_line(color='red').transform_regression(
        'eci', 'est'
    ).encode(
        x='eci:Q',
        y='est:Q'
    )

    scatter + line
    return


@app.cell
def _():
    import seaborn as sns

    df = sns.load_dataset("titanic")
    df
    return df, sns


@app.cell
def _(df, sns):
    sns.violinplot(data=df, x="class", y="age", hue="alive", split=True, gap=.1, inner="quart")
    return


@app.cell
def _(pl, sinco_eci_ingresos):
    sinco_eci_ingresos_violin = sinco_eci_ingresos.select("eci", "est", "nombre_sinco_1d").with_columns(
        (pl.col("est") - pl.col("est").mean())/pl.col("est").std()
    ).unpivot(
        ["eci", "est"], index="nombre_sinco_1d"
    ).rename(
        {
            "nombre_sinco_1d" : "sinco_1d", 
            "variable" : "indicador"
        }
    )
    sinco_eci_ingresos_violin
    return (sinco_eci_ingresos_violin,)


@app.cell
def _(sinco_eci_ingresos_violin, sns):
    g = sns.violinplot(
        data=sinco_eci_ingresos_violin, x="sinco_1d", y="value", hue="indicador", split=True, gap=.1, inner="quart"
    )
    g#.set_xticklabels([item.get_text() for item in g.xaxis.get_ticklabels()], rotation = 45)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(sinco_eci_ingresos):
    sinco_eci_ingresos
    return


@app.cell
def _(sinco_eci_ingresos, sns):
    sns.distplot(sinco_eci_ingresos['eci'], hist = False, kde = True,
                     kde_kws = {'shade': True, 'linewidth': 3})
    return


@app.cell
def _(sinco_eci_ingresos, sns):
    # Density Plot and Histogram of all arrival delays
    sns.distplot(sinco_eci_ingresos['eci'], hist=True, kde=True, 
                 bins=int(180/5), color = 'darkblue', 
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 4})
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cálculo de Similaridad entre Ocupaciones
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Matriz de Especialización, $R$

    * Una Matriz de Especialización, $R$, se define al dividir cada entrada de la matriz $X_{c p}$ por la suma de sus respectivas filas o columnas.
    * Está medida se le conoce como **cociente de ubicación** o **Revealed Comparative Advantage, (RCA)**.


    \begin{equation}
        R_{c p}= \frac{x_{c p} / \Sigma_p x_{c p}}{\Sigma_c x_{c p} / \Sigma_c \Sigma_p x_{c p}}
    \end{equation}

    * Ubicaciones con $R_{c p} > 1$ se consideran **especializadas en la actividad $p$**.
    """)
    return


@app.cell
def _(pl, skill_sinco_3d):
    datos_rca = pl.from_pandas(skill_sinco_3d.dropna()).with_columns(
        rca = (
            pl.col("Data Value")/pl.col("Data Value").sum().over("nombre_sinco_2019")
        ) /
        (
            pl.col("Data Value").sum().over("Element Name")/pl.col("Data Value").sum()
        )

    )

    datos_rca
    return (datos_rca,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Matriz de Especialización Binaria, $M$

     Definimos la Matriz de Especialización Binaria, $M$, como

    \begin{equation}
    M_{c p}=\left\{\begin{array}{lll}
    1 & \text { if } & R_{c p} \geq R^{\star} \\
    0 & \text { if } & R_{c p}<R^{\star}
    \end{array}\right.
    \end{equation}

    donde $R^{\star}=1$ cuando usamos $R$ y $R^{\star}=0.25$ cuando usamos $R^{\text{pop}}$
    """)
    return


@app.cell
def _(datos_rca, pl):
    ## Umbral de RCA
    rca_umbral = 1.0

    datos_rca_m = datos_rca.with_columns(
        M = pl.when(
            pl.col("rca")>= rca_umbral   
        ).then(
            pl.lit(1)
        ).otherwise(
            pl.lit(0)
        )
    )
    datos_rca_m
    return (datos_rca_m,)


@app.cell
def _(datos_rca_m):
    M_df = datos_rca_m.pivot("Element Name", 
                      index = "nombre_sinco_2019", 
                      values = "M"
                ).fill_null(0).sort("nombre_sinco_2019")
    M_df
    return (M_df,)


@app.cell
def _(M_df, pl):
    M = M_df.select(
        pl.exclude("nombre_sinco_2019")
    ).to_numpy()

    M
    return (M,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Proximidad

    Hay múltiples formas de medir **Proximidad**. Algunas, como la **probabilidad condicional mínima**, miran a la colocalización o coaglomeración de actividades:

    \begin{equation}
      \phi_{p p^{\prime}}=\frac{\sum_c M_{c p} M_{c p^{\prime}}}{\max \left(M_p, M_{p,}\right)}
    \end{equation}
    """)
    return


@app.cell
def _(M, np):
    ## Calculamos diversidad
    diversidad = M.sum(axis = 1)
    D = np.diag(diversidad)

    ## Calculamos ubicuidad
    ubicuidad = M.sum(axis = 0)
    U = np.diag(ubicuidad)

    ## Calculamos proximidad
    proximity = M.T @ M / ubicuidad[np.newaxis, :]  
    proximity = np.minimum(proximity, proximity.T)
    proximity = np.nan_to_num(proximity)
    proximity
    return D, U, proximity


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Relatedness Density

    * Con la medida de proximidad, podemos calcular **Relatedness Density** como la fracción de actividades relacionadas presentes en una ubicación

    \begin{equation}
    \omega_{c p}=\frac{\sum_{p \prime} M_{c p \prime} \phi_{p p^{\prime}}}{\sum_{p,} \phi_{p p^{\prime}}} \quad \circ \quad \omega_{c p}=\frac{\sum_{c^{\prime}} M_{c^{\prime}} \phi_{c^{\prime} c}}{\sum_{c^{\prime}} \phi_{c^{\prime} c}}
    \end{equation}
    """)
    return


@app.cell
def _(M, np, proximity):
    density = (np.dot(M,proximity)/np.sum(proximity, axis=1))
    density = np.nan_to_num(density)
    density
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Distancia
    - La Proximidad mide la similaridad entre pares de actividades-productos.
    - Necesitamos una medida que cuantifique la **Distancia** entre las actividades especializadas en un país y las actividades donde no está especializada.


    \begin{equation}
    d_{c p}=\frac{\sum_{p'} (1 - M_{c p'}) \phi_{p p^{\prime}}}{\sum_{p,} \phi_{p p^{\prime}}}
    \end{equation}

    - La distancia nos da una idea de qué tan lejos está cada actividad dado el ecosistema productivo del país.
    """)
    return


@app.cell
def _(M, np, proximity):
    distancia = (np.dot((1 - M),proximity)/np.sum(proximity, axis=1))
    distancia = np.nan_to_num(distancia)
    distancia
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Economic Complexity

    - Las medidas de **Complejidad Económica** miden la capacidad económica con **métodos de reducción de dimensionalidad**.
    - Representan también funciones de producción generalizadas de dimensionalidad reducida.
    - Las medidas de complejidad económica pueden ser utilizadas para medir la presencia de múltiples factores económicos en una forma que es **agnóstica** sobre cuales podrían ser esos factores.
      - Formalmente, la complejidad $K_c$ de una ubicación $c$ y la complejidad $K_p$ de una actividad $p$ puede definirse como una función una de la otra:

    \begin{equation}
    K_c=f\left(M_{c p}, K_p\right)
    \end{equation}

    \begin{equation}
    K_p=g\left(M_{c p}, K_c\right)
    \end{equation}

    - Estas ecuaciones declaran que la complejidad de una ubicación es una función de la complejidad de las actividades que están presentes en esta, y viceversa.
    - Una economía es tan compleja como las actividades que puede realizar, y una actividad es tan compleja como los lugares que pueden realizarla.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ¿Cómo medimos la complejidad?

    - La idea de medir la complejidad usando estas ecuaciones acopladas fue introducida por Cesar Hidalgo y Ricardo Haussman. Los autores utilizan promedios simples para $f$ y $g$.
    - Las medidas resultantes se conocen como **Índice de Complejidad Económica, (ECI; $K_c$)** y el **Índice de Complejidad de Producto, (PCI; $K_p$)**.
    - Estas medidas están definidas por el siguiente sistema de ecuaciones:

    \begin{equation}
    K_c=\frac{1}{M_c} \sum_p M_{c p} K_p
    \end{equation}

    \begin{equation}
    K_p=\frac{1}{M_p} \sum_c M_{c p} K_c
    \end{equation}

    Reemplazando la segunda ecuación en la primera:

    \begin{equation}
        K_c=\widetilde{M}_{c c}, K_{c t}
    \end{equation}

    donde

    \begin{equation}
    \widetilde{M}_{c c \prime}=\sum_p \frac{M_{c p} M_{c \prime p}}{M_c M_p}
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Originalmente el cálculo de ECI y PCI se definió mediante un método iterativo llamado  **algoritmo de reflexión** que primero calcula la diversidad y ubicuidad para posteriormente y luego utiliza recursivamente la información de uno para corregir el otro.
    - Se puede demostrar que el método de reflección es equivalente a encontrar los eigenvalores de la matriz $\widetilde{M}$

    \begin{equation}
    \tilde{M}=D^{-1} M U^{-1} M^{\prime}
    \end{equation}

    donde $D$ es la matriz diagonal formada a partir del vector de diversidad y $U$ es la matriz diagonal formada a partir del vector de ubicuidad

    En el contexto de datos de comercio entre países, podemos pensar a $\widetilde{M}$ como una matriz de diversidad-ponderada (o normalizada) que refleja qué tan similares son las canastas exportadoras de los dos países, es decir, que tan similares son sus patrones de especialización.

    De la ecuación anterior podemos ver que:

    \begin{equation}
    \tilde{M}=D^{-1} S
    \end{equation}

    donde $S = M U^{-1} M^\prime$ es una matriz de similaridad simétrica en que cada elemento $S_{c c'}$ representa los productos que el pais $c$ tiene en común con el país $c'$, ponderado o normalizado por la inversa de la ubicuidad de cada producto.
    """)
    return


@app.cell
def _(D, M, U, np):
    # Calculamos M tilde cc
    M_tilde_cc = np.linalg.pinv(D) @ M @ np.linalg.pinv(U) @ M.T
    M_tilde_cc 
    return (M_tilde_cc,)


@app.cell
def _(M_df, M_tilde_cc, pd):
    ### Creamos la matriz de similaridad de ocupaciones
    matriz_similaridad_ocupaciones = pd.DataFrame(M_tilde_cc, columns=M_df["nombre_sinco_2019"])
    matriz_similaridad_ocupaciones["nombre_sinco_2019"] = M_df["nombre_sinco_2019"]
    matriz_similaridad_ocupaciones
    return (matriz_similaridad_ocupaciones,)


@app.cell
def _(cdata, matriz_similaridad_ocupaciones, pd, pl):
    similaridad_ocupaciones_eci = pl.from_pandas(
         pd.melt(matriz_similaridad_ocupaciones, id_vars="nombre_sinco_2019")
    )

    similaridad_ocupaciones_eci = similaridad_ocupaciones_eci.join(
        pl.from_pandas(cdata).select("nombre_sinco_2019", "eci").unique(),
        left_on="variable",
        right_on="nombre_sinco_2019",
        how="left"
    )

    similaridad_ocupaciones_eci = similaridad_ocupaciones_eci.with_columns(
        pl.col("eci")*pl.col("value")
    ).group_by("nombre_sinco_2019").agg(
        pl.col("eci").sum()
    ).rename({"eci" : "eci_esperado"})

    similaridad_ocupaciones_eci = similaridad_ocupaciones_eci.join(
        pl.from_pandas(cdata).select("nombre_sinco_2019", "eci", "sinco_3d").rename({"eci" : "eci_original"}).unique(), 
        on = "nombre_sinco_2019"
    )
    similaridad_ocupaciones_eci
    return (similaridad_ocupaciones_eci,)


@app.cell
def _(matriz_similaridad_ocupaciones, pd, pl, similaridad_ocupaciones_eci):
    similaridad_weighted_std = pl.from_pandas(
         pd.melt(matriz_similaridad_ocupaciones, id_vars="nombre_sinco_2019")
    ).join(
        similaridad_ocupaciones_eci.select("nombre_sinco_2019", "eci_original").unique(),
        left_on="variable",
        right_on="nombre_sinco_2019",
        how="left"
    ).join(
        similaridad_ocupaciones_eci.select("nombre_sinco_2019", "eci_esperado").unique(),
        left_on="nombre_sinco_2019",
        right_on="nombre_sinco_2019",
        how="left"
    ).with_columns(
        weighted_std = (
            pl.col("value")*((pl.col("eci_original")-pl.col("eci_esperado"))**2)
        )
    ).group_by("nombre_sinco_2019").agg(
        pl.col("weighted_std").sum()
    ).with_columns(
        pl.col("weighted_std")**(1/2)
    )
    similaridad_weighted_std
    return (similaridad_weighted_std,)


@app.cell
def _(similaridad_ocupaciones_eci, similaridad_weighted_std):
    similaridad_ocupaciones_eci_std = similaridad_ocupaciones_eci.join(
        similaridad_weighted_std, 
        on = "nombre_sinco_2019"
    ).select("sinco_3d", "nombre_sinco_2019", "eci_original", "eci_esperado", "weighted_std")

    similaridad_ocupaciones_eci_std
    return (similaridad_ocupaciones_eci_std,)


@app.cell
def _(datos_filename, escala_medicion, similaridad_ocupaciones_eci_std):
    resultados_eci_esperado = f"{datos_filename}_{escala_medicion}.csv"

    similaridad_ocupaciones_eci_std.write_csv(resultados_eci_esperado)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

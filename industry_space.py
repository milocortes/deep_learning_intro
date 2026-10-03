
from streamlit_echarts5 import st_echarts
import streamlit as st 
import json
import copy
import pandas as pd 

st.set_page_config(layout='wide')

zonas = pd.read_csv("datos/complejidad/nombres_zm.csv")
zonas_ramas_especializadas = pd.read_csv("datos/complejidad/especializacion_ramas_zm.csv")

st.sidebar.selectbox("Zona Metropolitana", zonas["zm"].to_list(), key="zm")

st.sidebar.selectbox("Año", [2003, 2008, 2013, 2018, 2023], key="anio")


with open("datos/complejidad/espacio_producto_format.json", "r") as f:
    graph = json.loads(f.read())

graph["links"] = [{ source : str(target) for source,target in node.items()} for node in graph["links"] ]

graph["categories"] = graph["categories"] + [{"name" : "No Especializado"}]



if st.session_state.zm != "Nacional":

    #print("actualizamos nodos")
    graph_to_plot = copy.deepcopy(graph)

    ramas_especializadas = zonas_ramas_especializadas.query(f"anio == {st.session_state.anio} and zm=='{st.session_state.zm }'")["rama_id"].to_list()

    print(ramas_especializadas)

    print(st.session_state.zm)

    
    for idx, node in enumerate(graph_to_plot["nodes"]):
        if node["id"] not in ramas_especializadas:
            graph_to_plot["nodes"][idx]["category"] = 11
            graph_to_plot["nodes"][idx]["label"] = {"show": False}
        else: 
            graph_to_plot["nodes"][idx]["label"] = {"show": True}
else:
    graph_to_plot = copy.deepcopy(graph)


option = {
    "title": {
        "text": "Espacio Producto",
        "subtext": "Default layout",
        "top": "bottom",
        "left": "right",
    },
    "tooltip": {},
    "legend": [{"data": [a["name"] for a in graph_to_plot["categories"]]}],
    "animationDuration": 1500,
    "animationEasingUpdate": "quinticInOut",
    "series": [
        {
            "name": "Rama",
            "type": "graph",
            "layout": "none",
            "data": graph_to_plot["nodes"],
            "links": graph_to_plot["links"],
            "categories": graph_to_plot["categories"],
            "roam": True,
            "label": { "position": "right", "formatter": "{b}"},
            "lineStyle": {"color": "source", "curveness": 0.3},
            "emphasis": {"focus": "adjacency", "lineStyle": {"width": 50}},
            "labelLayout" : {
                "hideOverlap": True
                },
                "scaleLimit": {
                "min": 0.4,
                "max": 2
                },
                
        }
    ],
}

themes = {
    # Global palette:
    "color":  [ "#91cc75" , "#d53e4f", "#f46d43", "#9e0142", "#fac858", "#5470c6", "#ffff00", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2", "#c4c4cc"],

}

st_echarts(option, height="650px", theme = themes)

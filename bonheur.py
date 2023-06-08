import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
import base64

st.set_page_config(layout="wide")

menu = st.container()

menu.title('Le bonheur national brut')
st.markdown(
    f"""
    <style>
    .css-10trblm.eqr7zpz0 {{ color: #263A29; }}
    p {{ margin-bottom: 0.5rem; }}
    h2 {{ text-align: center; }}
    </style>
    """,
    unsafe_allow_html=True
    )

with menu:
	choose = option_menu(None,["Introduction", "Datasets", "Visualisations", "Modélisations", "Conclusion"],
		icons=['globe-americas', 'database', 'bar-chart-line', 'calculator', 'balloon-heart'],
		default_index=0,
		orientation="horizontal",
		styles={
			"container": {"padding": "5!important", "background-color": "#41644A", "border-radius" : "4px"},
			"icon": {"font-size": "25px", "vertical-align" : "middle"}, 
			"nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color": "#F2E3DB", "--hover-color": "rgba(242, 227, 219, 0.3)"},
			"nav-link-selected": {"background-color": "rgba(242, 227, 219, 0.5)", "color" : "#41644A"},
		}
	)

if choose == "Introduction":
	intro1, intro2, intro3 = st.columns([1,8,1])
	#content = st.container()
	intro2.write('Ce projet a été fait dans le cadre de la formation Data Analyst au sein de l’organisme Data Scientest, promotion bootcamp avril 2023. Nous avons, à partir des connaissances acquise et de notre curiosité, tenté de répondre à la question suivante :') 
	intro2.header("QUELS FACTEURS ONT LE PLUS D'INFLUENCE SURLE BONHEUR DES INDIVIDUS ?")
	intro2.image('img/globe_beach.jpg')
	intro2.write('Ainsi, nous avons pu observer des facteurs politiques, économiques et sociaux.')
	intro2.write('L’objectif de ce projet est : ')
	intro2.subheader('Déterminer quels sont les facteurs pouvant expliquer le bonheur, mais aussi le poids de chaque facteur, et donc de comprendre les raisons pour lesquelles un pays est mieux classé qu’un autre.')
	intro2.write('Nous allons tenter de proposer un modèle parcimonieux mais ayant une bonne valeur explicative du bonheur national brut.')
	intro2.write('L’objectif parallèle à celui de l’élaboration du modèle est de présenter ces données de manière interactive, en utilisant des visualisations pertinentes, afin de mettre en évidence les combinaisons de facteurs qui répondent à notre questionnement principal')
	intro2.write('Composition de l’équipe de Data Analyst :') 
	intro2.caption('Francisco Comiran')
	intro2.caption('Zenaba Mogne')
	intro2.caption('Roxane Oubrerie')

elif choose == "Datasets":
	data1, data2, data3 = st.columns([1,8,1])
	data2.write("Datasets")

elif choose == "Visualisations":
	visu1, visu2, visu3 = st.columns([1,8,1])
	visu2.write("Visualisations")

elif choose == "Modélisations":
	model1, model2, model3 = st.columns([1,8,1])
	model2.write("Modélisations")

elif choose == "Conclusion":
	conclu1, conclu2, conclu3 = st.columns([1,8,1])
	conclu2.write("Conclusion")






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
    p a {{ text-decoration: none; color: #E86A33 !important}}
    p a:hover, p a:visited, p a:focus {{ text-decoration: none; font-weight: 600;}}
    h2 {{ text-align: center; }}
    [data-testid="stMarkdownContainer"] ul {{ list-style-position: inside; }}
    [data-testid="stCaptionContainer"], [data-testid="stExpander"], [data-testid="stMarkdownContainer"], h3, [data-testid="stImage"] {{ width: 80% !important; margin: 0 auto; }}
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
			"nav-link-selected": {"background-color": "#F2E3DB", "color" : "#41644A"},
		}
	)

if choose == "Introduction":
	intro = st.container()
	
	intro.write('Ce projet a été fait dans le cadre de la formation Data Analyst au sein de l’organisme Data Scientest, promotion bootcamp avril 2023. Nous avons, à partir des connaissances acquise et de notre curiosité, tenté de répondre à la question suivante :') 
	intro.header("QUELS FACTEURS ONT LE PLUS D'INFLUENCE SURLE BONHEUR DES INDIVIDUS ?")
	intro.image('img/globe_beach.jpg')
	intro.write('Ainsi, nous avons pu observer des facteurs politiques, économiques et sociaux.')
	intro.write('L’objectif de ce projet est : ')
	intro.subheader('Déterminer quels sont les facteurs pouvant expliquer le bonheur, mais aussi le poids de chaque facteur, et donc de comprendre les raisons pour lesquelles un pays est mieux classé qu’un autre.')
	intro.write('Nous allons tenter de proposer un modèle parcimonieux mais ayant une bonne valeur explicative du bonheur national brut.')
	intro.write('L’objectif parallèle à celui de l’élaboration du modèle est de présenter ces données de manière interactive, en utilisant des visualisations pertinentes, afin de mettre en évidence les combinaisons de facteurs qui répondent à notre questionnement principal')
	intro.write('Composition de l’équipe de Data Analyst :') 
	intro.caption('Francisco Comiran')
	intro.caption('Zenaba Mogne')
	intro.caption('Roxane Oubrerie')

elif choose == "Datasets":
	data = st.container()
	
	data.write('Afin d’obtenir un modèle et donc une réponse à notre problématique au plus proche de la réalité, il nous faut récolter des données de qualité.')

	data.subheader('1.Première étape : La récupération des données')
	data.write("La source principale des données est le [World Happiness Report](https://worldhappiness.report/), une enquête phare sur l'état du bonheur mondial.")
	data.write("Deux jeux de données de type csv ont pu être téléchargé sur [Kaggle](https://www.kaggle.com/ajaypalsinghlo/world-happiness-report-2021): ")
	data.markdown("- le World Happiness Report")
	data.markdown("- le World Happiness Report 2021")
	data.write("Ces jeux de données seront les sources initiales de nos données. Nous avons décidé de les agrémenter d’autres indicateurs pertinents.")

	data.subheader('2. Étoffer nos jeux de données')

	with st.expander("Avec un indicateur au sujet de la guerre"):
		st.write("""Création d’un dataset grâce aux données récupérée dans une [page wikipedia](https://en.wikipedia.org/wiki/List_of_armed_conflicts_in_2020.)  à l’aide du webscrapping via la librairie beautiful soup.""")
		st.write("""En téléchargement libre sur le [site de l'ucdp](https://ucdp.uu.se/encyclopedia)""")

	with st.expander("Avec un indicateur au sujet du chômage"):
		st.write("""Création d’un dataset grâce aux données du site Kaggle en téléchargement et libre de droit. Il provient initialement du site [data.worldbank](https://data.worldbank.org/).""")

	with st.expander("Avec d'autres facteurs sociaux et politiques"):
		st.write("""Création d’un dataset grâce aux bases de données [World Economics](https://www.worldeconomics.com/).""")

	data2 = st.container()

	data2.subheader('3. Mutualisation et préparation des dataframes finaux')
	data2.write("Nous choisissons de préparer les deux datasets, l’un sur 2021, l’autre sur plusieurs années.Les datasets ont été mergés par la colonne “Country”.")
	data2.image("img/df2021_final.jpg")
	data2.image("img/df_final.jpg")

	data2.subheader('Dataframes obtenus :')

	df_2021 = pd.read_csv('datasets/df2021_final.csv')
	with st.expander("DF2021_FINAL"):
		st.dataframe(df_2021)

	df = pd.read_csv('datasets/df_final.csv')
	with st.expander("DF_FINAL"):
		st.dataframe(df)

elif choose == "Visualisations":
	visu = st.container()
	visu.write('visualisations')





elif choose == "Modélisations":
	model1, model2, model3 = st.columns([1,8,1])
	model2.write("Modélisations")

elif choose == "Conclusion":
	conclu1, conclu2, conclu3 = st.columns([1,8,1])
	conclu2.write("Conclusion")






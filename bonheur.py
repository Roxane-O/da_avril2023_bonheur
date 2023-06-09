import streamlit as st
import folium
import json
from folium.plugins import Draw
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from streamlit_option_menu import option_menu
from  PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

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
    [data-testid="stCaptionContainer"], [data-testid="stExpander"], [data-testid="stMarkdownContainer"], h3, [data-testid="stImage"], [data-testid="stDataFrameResizable"], div.row-widget.stSelectbox {{ width: 80% !important; margin: 0 auto; }}
    [data-testid="stImage"] img {{ width: 100% !important; }}
    iframe {{ display: block; margin: 0 auto; width: 900px; }}
    [data-baseweb="tab-list"] [data-testid="stMarkdownContainer"] {{ width: 100% !important; }}
    .stTabs .stTabs [data-baseweb="tab-list"] {{ width: fit-content; margin: 0 auto; }}
    .stTabs .stTabs [data-baseweb="tab-border"] {{ background-color: transparent;}}
    [data-testid="stVerticalBlock"] > .stTabs {{ width: 80%; margin: 0 auto; }}
    [data-testid="stVerticalBlock"] > .stTabs [data-testid="stVerticalBlock"] > .stTabs {{ width: 100%; }}
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

####################
#   INTRODUCTION   #
####################
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

################
#   DATASETS   #
################
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
	data2.subheader('Dataframe obtenu :')

	df2021_final = pd.read_csv('datasets/df2021_final.csv')
	with st.expander("DF2021_FINAL"):
		st.dataframe(df2021_final)

	data3 = st.container()

	data3.image("img/df_final.jpg")
	data3.subheader('Dataframe obtenu :')
	df_final = pd.read_csv('datasets/df_final.csv')
	with st.expander("DF_FINAL"):
		st.dataframe(df_final)

######################
#   VISUALISATIONS   #
######################
elif choose == "Visualisations":
	visu = st.container()

	df1 = pd.read_csv('datasets/world-happiness-report-2021.csv', sep=',')

	#Trier le df par ordre décroissant
	df1_sorted = df1.sort_values(by = 'Ladder score', ascending = False)

	# Regrouper les pays par région en faisant la moyenne du Ladder Score
	df_region_ls = df1.groupby('Regional indicator').agg({'Ladder score' : 'mean'})

	# Trier le dataframe selon le Ladder Score
	df_region_ls = df_region_ls.sort_values(by = "Ladder score")

	visu.subheader("Analyse de la variable cible : L’échelle du bonheur")
	visu.markdown("- Top des pays les plus heureux et top 10 des pays les moins heureux en 2021")
	#Barplot sur les pays les plus et moins heureux
	fig = plt.figure(figsize=(10,10))

	plt.subplot(221)
	sns.barplot(x ='Country name', y = 'Ladder score', data=df1_sorted.head(10), palette=sns.color_palette("flare", 10))
	plt.xlabel('Noms des pays')
	plt.ylabel('Echelle du bonheur')
	plt.title('Top 10 des pays les plus heureux en 2021')
	plt.xticks(rotation=90)
	plt.ylim((0, 10));

	plt.subplot(222)
	sns.barplot(x ='Country name', y = 'Ladder score', data=df1_sorted.tail(10), palette=sns.color_palette("flare", 10))
	plt.xlabel('Noms des région')
	plt.ylabel('Echelle du bonheur')
	plt.title('Les 10 pays les moins heureux en 2021')
	plt.xticks(rotation=90)
	plt.ylim((0, 10));

	visu.pyplot(fig)

	with st.expander("Pourquoi ce graphique ?"):
		st.write("Ce graphique nous a permis de vérifier le côté déséquilibré de notre jeu de données (notes à 0). Puis cela nous a aidé à mieux visualiser la distribution du Ladder score afin de créer les classes de la variable le cas échéant.")

	
	visu2 = st.container()
	visu2.markdown("- Top des régions les plus heureuses en 2021")
	#Manipulation pour un barplot des ladder score par région

	# Regrouper les pays par région en faisant la moyenne du Ladder Score
	df_region_ls = df1.groupby('Regional indicator').agg({'Ladder score' : 'mean'})

	# Trier le dataframe selon le Ladder Score
	df_region_ls = df_region_ls.sort_values(by = "Ladder score")

	fig2 = plt.figure(figsize=(8,6))
	sns.barplot(y = df_region_ls.index, x = df_region_ls["Ladder score"], palette = sns.color_palette("flare", 10))
	plt.yticks(fontsize=10)
	plt.ylabel("")
	plt.xlabel("")
	plt.title("Classement des régions selon l'échelle du bonheur - 2021", fontsize = 12)

	visu2.pyplot(fig2, use_container_width = False)

	with st.expander("Pourquoi ce graphique ?"):
		st.write("Ce graphique nous permet de vérifier la distribution du Ladder score par région")

	visu3 = st.container()
	visu3.subheader("Matrices de corrélation des deux datasets")
	visu3.markdown("- Dataset 2021")

	df1_corr = df1.drop(['Country name', 'Regional indicator'], axis = 1)

	fig3 = plt.figure(figsize=(15,10))
	correlation_matrix = df1_corr.corr()
	sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
	plt.title("Matrice de corrélation")
	visu3.write(fig3)

	with st.expander("Pourquoi ce graphique ?"):
		st.write("Ce graphique nous apporte des informations concernant la corrélation entre les différentes variables du dataframe portant sur l'année 2021")

	visu4 = st.container()
	visu4.markdown("- Dataset longitudinal")

	df_final = pd.read_csv('datasets/df_final.csv')
	df_final_corr = df_final.drop(['Country name', 'year'], axis = 1)

	fig4 = plt.figure(figsize=(10,10))
	correlation_matrix = df_final_corr.corr()
	sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
	plt.title("Matrice de corrélation")
	visu4.write(fig4)

	with st.expander("Pourquoi ce graphique ?"):
		st.write("Ce graphique nous apporte des informations concernant la corrélation entre les différentes variables du dataframe longitudinal")


	visu5 = st.container()
	visu5.subheader("Carte du monde selon le score de bonheur")

	geolocator = Nominatim(user_agent='myapplication')

	latitudes = []
	longitudes = []

	#Boucle pour remplir les listes à partir des noms de pays du df1
	for country in df1['Country name']:
	    try:
	        location = geolocator.geocode(country, timeout=10)
	        if location:
	            latitudes.append(location.latitude)
	            longitudes.append(location.longitude)
	        else:
	            latitudes.append(None)
	            longitudes.append(None)
	    except GeocoderTimedOut as e:
	        print("Error: geocode failed on input %s with message %s" % (country, e))
	        latitudes.append(None)
	        longitudes.append(None)
	        
	#Coordonnées
	coordinates = list(zip(latitudes, longitudes))

	df1_geo = df1
	df1_geo['latitude']=latitudes
	df1_geo['longitude']=longitudes

	# Correction des coordonnés (à poursuivre)
	df1_geo.loc[df1_geo['Country name'] == 'Georgia', ['latitude', 'longitude']] = [42, 43.3]
	df1_geo.loc[df1_geo['Country name'] == 'Taiwan Province of China', ['latitude', 'longitude']] = [25.03, 121.3]
	df1_geo.loc[df1_geo['Country name'] == 'Hong Kong S.A.R. of China', ['latitude', 'longitude']] = [22.39, 114.109497]

	# Charger les données GeoJSON des pays
	with open('json/world-countries.json') as f:
	    geo_data = json.load(f)

	# Créer une carte centrée sur le monde
	m = folium.Map(location=[0, 0], zoom_start=2)

	# Créer une fonction pour définir la couleur en fonction du score Ladder
	def get_color(score):
	    if score > 7.5:
	        return 'darkgreen'
	    elif score > 7:
	        return 'green'
	    elif score > 6.5:
	        return 'lightgreen'
	    elif score > 6:
	        return 'yellow'
	    elif score > 5.5:
	        return 'orange'
	    else:
	        return 'red'

	# Ajouter une couche de remplissage à la carte pour chaque pays
	folium.Choropleth(
	    geo_data=geo_data,
	    name='choropleth',
	    data=df1_geo,
	    columns=['Country name', 'Ladder score'],
	    key_on='feature.properties.name',
	    fill_color='YlOrRd',
	    fill_opacity=0.7,
	    line_opacity=0.2,
	    legend_name='Ladder Score',
	    highlight=True,
	    overlay=True,
	    show=False,
	).add_to(m)

	# Parcourir le dataframe et ajouter un marqueur pour chaque pays
	for index, row in df1_geo.iterrows():
	    pays = row['Country name']
	    score = row['Ladder score']
	    # Récupérer les coordonnées géographiques du pays
	    coord = (row['latitude'], row['longitude'])
	    # Ajouter un marqueur avec la couleur correspondante
	    folium.Marker(location=coord, 
	                  icon=folium.Icon(color=get_color(score)), 
	                  tooltip=f"{pays}: {score}").add_to(m)

	output = st_folium(m, width=900, height=500)

####################
#   MODELISATION   #
####################
elif choose == "Modélisations":
	tab1, tab2 = st.tabs(["Modèles quantitatifs", "Modèles de classification"])

	with tab1:
		# Centrer le titre de la page
		st.markdown("<h2 style='text-align: center;'>Modèles Quantitatifs</h2>", unsafe_allow_html=True)

		tab2021, tabLongi = st.tabs(["Dataset 2021", "Dataset Longitudinal"])

		############
		#   2021   #
		############
		with tab2021:
			df2021_final = pd.read_csv('datasets/df2021_final.csv')


			# Afficher le sous-titre
			st.markdown("<h3>Dataset 2021</h3>", unsafe_allow_html=True)


			# Afficher le texte
			st.markdown("Aperçu des variables prédictives (avant standardisation):")

			# Affichage du df
			st.dataframe(df2021_final.head())


			# Afficher le sous-titre "Modèles"
			st.subheader("Modèles")


			####################
			#   Choix modèle   #
			####################
			# Définir les options de la liste déroulante
			options = ["Sélectionner modèle", "Régression Linéaire Multiple", "Régression Ridge", "Régression Lasso", "Régression Elastic Net"]

			# Afficher la liste déroulante
			selected_model = st.selectbox("Sélectionnez un modèle", options)

			# Afficher le contenu correspondant au modèle sélectionné
			if selected_model == "Régression Linéaire Multiple":
				######################
				#   Feats / Target   #
				######################
				# Variables prédictives
				predictors = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy',
				              'Freedom to make life choices', 'Perceptions of corruption', 'Law',
				              'Press_Freedom', 'Political_Rights', 'Inequality', 'Schooling',
				              'Unemployment rate', 'armed_conflicts', 'Generosity']

				# Variable dépendante
				dependent = 'Ladder score'


				####################
				#   Train / Test   #
				####################
				# Séparation des données en jeu d'entraînement et jeu de test en maintenant une moyenne similaire
				X_train, X_test, y_train, y_test = train_test_split(df2021_final[predictors], df2021_final[dependent], test_size=0.2, random_state=40)

				# Vérification des moyennes
				mean_train = np.mean(y_train)
				mean_test = np.mean(y_test)


				#######################
				#   Standardisation   #
				#######################
				# Colonnes à inclure dans le DataFrame
				columns = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy',
				           'Freedom to make life choices', 'Perceptions of corruption', 'Law',
				           'Press_Freedom', 'Political_Rights', 'Inequality', 'Schooling',
				           'Unemployment rate', 'armed_conflicts', 'Generosity']

				# Extraction des colonnes du jeu de données
				X_train = X_train[columns]
				X_test = X_test[columns]

				# Création d'un objet StandardScaler avec les paramètres souhaités
				scaler = StandardScaler(with_mean=True, with_std=True)

				# Mise à l'échelle des données d'entraînement
				X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=columns)

				# Mise à l'échelle des données de test
				X_test = pd.DataFrame(scaler.transform(X_test), columns=columns)


				#REGRESSION LINEAIRE
				# Création d'un objet modèle de régression linéaire
				model = LinearRegression()

				# Entraînement du modèle sur le jeu d'entraînement
				model.fit(X_train, y_train)

				st.markdown("- Standardisation des VI")
				st.markdown("- Entrainement 80% / Test 20%")
				st.markdown("- Approche par comparaison de modèles")
				st.markdown("- Exclusion des variables conflits armés, corruption perçue, droits politiques, inégalités et générosité")
			    
			    # Afficher l'équation du modèle de régression linéaire
				st.write("\n\n- **Equation du modèle:**")
				st.write("**Bonheur** = 0.46 × PIB + 0.21 × Soutien social + 0.32 × Espérance de vie en bonne santé + 0.19 × Liberté choix de vie - 0.26 × Droit + 0.38 × Liberté presse - 0.18 × Années de scolarité - 0.18 × Chômage")
			    
				#Affichage du graphique pour la prédiction du modèle
				pred_test = model.predict(X_test)

				fig = plt.figure(figsize = (6,6))
				plt.scatter(pred_test, y_test, c = 'orange')

				plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c = 'pink')
				plt.xlabel("prediction")
				plt.ylabel("vraie valeur")
				plt.title('Régression Linéaire pour la prédiction du score de bonheur (dataset 2021)')

				st.pyplot(fig)
			    
			    # Afficher les métriques sur le jeu de données d'entraînement
				st.write("\n\n- **Métriques sur le jeu de données d'entraînement :**")
				st.write("**MSE** : 0.18725914805070465\n\n"
			              "**MAE** : 0.3356944286948647\n\n"
			              "**R^2** : 0.8359853275283171")
			    
			    # Afficher les métriques sur le jeu de données de test
				st.write("\n\n- **Métriques sur le jeu de données de test :**")
				st.write("**MSE** : 0.20093223750843606\n\n"
			              "**MAE** : 0.3563574581451695\n\n"
			              "**R^2** : 0.8519184749009465")

			# Afficher le contenu correspondant au modèle sélectionné
			elif selected_model == "Régression Ridge":		
				#Constitution du df
				df = df2021_final[['Ladder score','Logged GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices','Perceptions of corruption','Law','Press_Freedom','Political_Rights','Inequality','Schooling','Unemployment rate','armed_conflicts', 'Generosity']]
				
				#Standardisation des variables
				df[df.columns] = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df))

				#Séparation du jeux de données
				data = df[['Logged GDP per capita', 'Social support', 'Healthy life expectancy','Freedom to make life choices', 'Perceptions of corruption', 'Law','Press_Freedom', 'Political_Rights', 'Inequality', 'Schooling','Unemployment rate', 'armed_conflicts', 'Generosity']]
				target = df['Ladder score']

				X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=789)

				#Regression ridge
				ridge_reg = RidgeCV(alphas= (0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100))
				ridge_reg.fit(X_train, y_train) 

				pred_test = ridge_reg.predict(X_test)

				st.markdown("- Standardisation de toutes les variables")
				st.markdown("- Entrainement 80% / Test 20%")
			    
			    # Afficher l'équation du modèle
				st.write("\n\n- **Equation du modèle:**")
				st.write("**Bonheur** = 0.43 × PIB + 0.32 × Soutien social + 0.22 × Espérance de vie en bonne santé + 0.15 × Liberté choix de vie - 0.08 × Corruption perçue - 0.21 × Droit + 0.27 × Liberté presse + 0.02 × Droits politiques - 0.00 × Inégalités - 0.16 × Années de scolarité - 0.14 × Chômage - 0.04 × Conflits armés + 0.05 × Générosité")
			    
				fig = plt.figure(figsize = (6,6))
				plt.scatter(pred_test, y_test, c = 'orange')

				plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c = 'green')
				plt.xlabel("prediction")
				plt.ylabel("vraie valeur")
				plt.title('Courbe de régression Ridge pour la prédiction du score de bonheur (dataset 2021)')
				st.pyplot(fig)

			    # Afficher les métriques sur le jeu d'entraînement
				st.write("\n\n- **Métriques sur le jeu d'entraînement :**")
				st.write("**MSE** : 0.15619765618683173\n\n"
					"**MAE** : 0.30295173204602516\n\n"
					"**R^2** : 0.8482302651797508")
			    
			    # Afficher les métriques sur le jeu de test
				st.write("\n\n- **Métriques sur le jeu de test :**")
				st.write("**MSE** : 0.1870771776297484\n\n"
			              "**MAE** : 0.36310379499170137\n\n"
			              "**R^2** : 0.7876422672542858")
			    
			# Afficher le contenu correspondant au modèle sélectionné
			elif selected_model == "Régression Lasso":
				#Constitution du df
				df = df2021_final[['Ladder score','Logged GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices','Perceptions of corruption','Law','Press_Freedom','Political_Rights','Inequality','Schooling','Unemployment rate','armed_conflicts', 'Generosity']]
				
				#Standardisation des variables
				df[df.columns] = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df))

				#Séparation du jeux de données
				data = df[['Logged GDP per capita', 'Social support', 'Healthy life expectancy','Freedom to make life choices', 'Perceptions of corruption', 'Law','Press_Freedom', 'Political_Rights', 'Inequality', 'Schooling','Unemployment rate', 'armed_conflicts', 'Generosity']]
				target = df['Ladder score']

				X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=789)

				lasso_reg = Lasso()
				lasso_reg.fit(X_train, y_train)

				alpha = 0  # Valeur d'alpha à ajuster selon vos besoins
				lasso_model = Lasso(alpha=alpha)
				lasso_model.fit(X_train, y_train)

				# Prédiction du modèle Lasso sur le jeu de test
				pred_test = lasso_model.predict(X_test)
				

				st.markdown("- Standardisation de toutes les variables")
				st.markdown("- Entrainement 80% / Test 20%")
			    
			    # Afficher l'équation du modèle
				st.write("\n\n- **Equation du modèle:**")
				st.write("**Bonheur** = 0.47 × PIB + 0.31 × Soutien social + 0.22 × Espérance de vie en bonne santé + 0.15 × Liberté choix de vie - 0.09 × Corruption perçue - 0.24 × Droit + 0.29 × Liberté presse + 0.01 × Droits politiques + 0.01 × Inégalités - 0.17 × Années de scolarité - 0.15 × Chômage - 0.04 × Conflits armés + 0.05 × Générosité")
			        
			    # Affichage d'un nuage de points avec les prédictions du modèle Lasso (axe x) et les vraies valeurs (axe y) du jeu de test
				fig = plt.figure(figsize=(6, 6))
				plt.scatter(pred_test, y_test, c='orange')

				# Affichage d'une ligne diagonale représentant l'identité (x = y) pour visualiser l'ajustement du modèle
				plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c='red')

				plt.xlabel("prediction")
				plt.ylabel("vraie valeur")
				plt.title('Courbe de régression Lasso pour la prédiction du score de bonheur (dataset 2021)')
				st.pyplot(fig)
			    
			    # Afficher les métriques sur le jeu d'entraînement
				st.write("\n\n- **Métriques sur le jeu d'entraînement :**")
				st.write("**MSE** : 0.15656140168987856\n\n"
			              "**MAE** : 0.3037389672150526\n\n"
			              "**R^2** : 0.8478768312045735")
			    
			    # Afficher les métriques sur le jeu de test
				st.write("\n\n- **Métriques sur le jeu de test :**")
				st.write("**MSE** : 0.18627690494010946\n\n"
			              "**MAE** : 0.3630074756055885\n\n"
			              "**R^2** : 0.788550684283574")
			    
			# Afficher le contenu correspondant au modèle sélectionné
			elif selected_model == "Régression Elastic Net":
				#Constitution du df
				df = df2021_final[['Ladder score','Logged GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices','Perceptions of corruption','Law','Press_Freedom','Political_Rights','Inequality','Schooling','Unemployment rate','armed_conflicts', 'Generosity']]
				
				#Standardisation des variables
				df[df.columns] = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df))

				#Séparation du jeux de données
				data = df[['Logged GDP per capita', 'Social support', 'Healthy life expectancy','Freedom to make life choices', 'Perceptions of corruption', 'Law','Press_Freedom', 'Political_Rights', 'Inequality', 'Schooling','Unemployment rate', 'armed_conflicts', 'Generosity']]
				target = df['Ladder score']

				X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=789)

				# Création d'un objet modèle Elastic Net avec une validation croisée à 8 plis et une liste de rapports L1/L2 spécifiés
				model_en = ElasticNetCV(cv=8, l1_ratio=(0.1, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99), 
				                        alphas=(0.001, 0.01, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0))


				# Entraînement du modèle Elastic Net sur les données d'entraînement
				model_en.fit(X_train, y_train)

				st.markdown("- Standardisation de toutes les variables")
				st.markdown("- Entrainement 80% / Test 20%")
			    
			    # Afficher l'équation du modèle
				st.write("\n\n- **Equation du modèle:**")
				st.write("**Bonheur** = 0.420728 × PIB + 0.319837 × Support social + 0.217963 × Espérance de vie en bonne santé + 0.153546 × Liberté de choix de vie + 0.077608 × Corruption perçue - 0.191576 × Droit + 0.269812 × Liberté de la presse + 0.013726 × Droits politiques - 0.003178 × Inégalités - 0.156633 × Années de scolarité - 0.140324 × Taux de chômage - 0.036933 × Conflits armés + 0.045715 × Générosité")
			        
			    # Afficher le graph correspondant
				pred_test = model_en.predict(X_test)

				fig = plt.figure(figsize = (6,6))
				plt.scatter(pred_test, y_test, c = 'orange')

				plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c = 'blue')
				plt.xlabel("prediction")
				plt.ylabel("vraie valeur")
				plt.title('Courbe de régression Elastic Net pour la prédiction du score de bonheur (dataset 2021)')
				
				st.pyplot(fig)
			    
			    # Afficher les métriques sur le jeu d'entraînement
				st.write("\n\n- **Métriques sur le jeu d'entraînement :**")
				st.write("**MSE** : 0.3954930955674146\n\n"
			              "**R^2** : 0.8480192880379052")
			    
			    # Afficher les métriques sur le jeu de test
				st.write("\n\n- **Métriques sur le jeu de test :**")
				st.write("**MSE** : 0.4334633441149267\n\n"
			              "**R^2** : 0.7867190703534337")

			else:
				st.write("Aucun modèle sélectionné")

		with tabLongi:
			####################
			#   Longitudinal   #
			####################
			df_final = pd.read_csv('datasets/df_final.csv')

			# Afficher le sous-titre
			st.markdown("<h3>Dataset 2011 - 2020</h3>", unsafe_allow_html=True)

			# Afficher le texte
			st.markdown("Aperçu des variables prédictives (avant standardisation):")

			# Affichage du df
			st.dataframe(df_final.head())

			# Afficher le sous-titre "Modèles"
			st.subheader("Modèles")

			# Définir les options de la deuxième liste déroulante
			options_2 = ["Sélectionner modèle", "Régression Linéaire Multiple", "Régression Ridge", "Régression Lasso", "Régression Elastic Net"]

			# Afficher la deuxième liste déroulante
			selected_model_2 = st.selectbox("Sélectionnez un modèle (Dataset 2011 - 2020)", options_2)

			# Afficher le contenu correspondant au modèle sélectionné
			if selected_model_2 == "Régression Linéaire Multiple":
				# Variables prédictives
				predictors = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth', 'Freedom to make life choices', 'Unemployment rate', 'War country', 'Generosity']
				# Variable dépendante
				dependent = 'Life Ladder'

				# Filtrage des données pour l'entraînement (2011 à 2018)
				train_data = df_final[(df_final['year'] >= 2011) & (df_final['year'] <= 2018)]

				# Filtrage des données pour le test (2019 à 2020)
				test_data = df_final[(df_final['year'] >= 2019) & (df_final['year'] <= 2020)]

				# Variables prédictives pour l'entraînement
				X_train = train_data[predictors]

				# Variable dépendante pour l'entraînement
				y_train = train_data[dependent]

				# Variables prédictives pour le test
				X_test = test_data[predictors]

				# Variable dépendante pour le test
				y_test = test_data[dependent]

				#STANDARDISATION DES VARIABLES
				# Extraction des variables du jeu de données
				X_train = X_train[predictors]
				X_test = X_test[predictors]

				# Création d'un objet StandardScaler avec les paramètres souhaités
				scaler = StandardScaler(with_mean=True, with_std=True)

				# Mise à l'échelle des données d'entraînement
				X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=predictors)

				# Mise à l'échelle des données de test
				X_test = pd.DataFrame(scaler.transform(X_test), columns=predictors)

				model = LinearRegression()
				model.fit(X_train, y_train)

				st.markdown("- Standardisation des VI")
				st.markdown("- Entrainement 80% (2011-2018) / Test 20% (2019-2020)")
				st.markdown("- Approche par comparaison de modèles")
				st.markdown("- Exclusion de la variable conflits armés")

			    # Afficher l'équation du modèle de régression linéaire
				st.write("\n\n- **Equation du modèle :**")
				st.write("Bonheur = 0.52 × PIB + 2.27 × Soutien social + 0.32 × Espérance de vie en bonne santé + 1.46 × Liberté de choix - 0.02 × Taux de chômage + 0.62 × Générosité")

			    # Afficher le graph correspondant
				pred_test = model.predict(X_test)

				fig = plt.figure(figsize = (6,6))
				plt.scatter(pred_test, y_test, c = 'orange')

				plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c = 'pink')
				plt.xlabel("prediction")
				plt.ylabel("vraie valeur")
				plt.title('Régression Linéaire pour la prédiction du score de bonheur (dataset longitudinal)')
				st.pyplot(fig)

			    # Afficher les métriques sur le jeu de données d'entraînement
				st.write("\n\n- **Métriques sur le jeu de données d'entraînement :**")
				st.write("MSE (Mean Squared Error) : 0.26469127594001707\n\n"
			              "MAE (Mean Absolute Error) : 0.4063954536271683\n\n"
			              "R^2 (Coefficient de détermination) : 0.7594485002408284")

			    # Afficher les métriques sur le jeu de données de test
				st.write("\n\n- **Métriques sur le jeu de données de test :**")
				st.write("MSE (Mean Squared Error) : 0.22702134110641828\n\n"
			              "MAE (Mean Absolute Error) : 0.3721927072625658\n\n"
			              "R^2 (Coefficient de détermination) : 0.772605418008979")


			# Afficher le contenu correspondant au modèle sélectionné
			elif selected_model_2 == "Régression Ridge":
				#Suppression de colonnes inutiles
				df = df_final.drop(['Positive affect', 'Negative affect'], axis=1)

				# Variables prédictives
				predictors = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth', 'Freedom to make life choices', 'Unemployment rate', 'War country', 'Generosity']

				# Variable dépendante
				dependent = 'Life Ladder'

				# Filtrage des données pour l'entraînement (2011 à 2018)
				train_data = df[(df['year'] >= 2011) & (df['year'] <= 2018)]

				# Filtrage des données pour le test (2019 à 2020)
				test_data = df[(df['year'] >= 2019) & (df['year'] <= 2020)]

				# Variables prédictives pour l'entraînement
				X_train = train_data[predictors]

				# Variable dépendante pour l'entraînement
				y_train = train_data[dependent]

				# Variables prédictives pour le test
				X_test = test_data[predictors]

				# Variable dépendante pour le test
				y_test = test_data[dependent]

				# Copie du DataFrame d'entraînement
				df_train_scaled = train_data.copy()

				# Standardisation des variables prédictives pour l'entraînement
				scaler_predictors_train = preprocessing.StandardScaler()
				df_train_scaled[predictors] = scaler_predictors_train.fit_transform(train_data[predictors])

				# Standardisation de la variable dépendante pour l'entraînement
				scaler_dependent_train = preprocessing.StandardScaler()
				df_train_scaled[dependent] = scaler_dependent_train.fit_transform(train_data[dependent].values.reshape(-1, 1))

				# Copie du DataFrame de test
				df_test_scaled = test_data.copy()

				# Standardisation des variables prédictives pour le test
				scaler_predictors_test = preprocessing.StandardScaler()
				df_test_scaled[predictors] = scaler_predictors_test.fit_transform(test_data[predictors])

				# Standardisation de la variable dépendante pour le test
				scaler_dependent_test = preprocessing.StandardScaler()
				df_test_scaled[dependent] = scaler_dependent_test.fit_transform(test_data[dependent].values.reshape(-1, 1))

				# Assigner les DataFrames standardisés à X_train, y_train, X_test et y_test
				X_train = df_train_scaled[predictors]
				y_train = df_train_scaled[dependent]
				X_test = df_test_scaled[predictors]
				y_test = df_test_scaled[dependent]

				ridge_reg = RidgeCV(alphas= (0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100))
				ridge_reg.fit(X_train, y_train) 

				# Prédictions sur le jeu d'entraînement
				y_train_pred = ridge_reg.predict(X_train)

				# Prédictions sur le jeu de test
				y_test_pred = ridge_reg.predict(X_test)

				st.markdown("- Standardisation de toutes les variables")
				st.markdown("- Entrainement 80% (2011-2018) / Test 20% (2019-2020)")
			    
			    # Afficher l'équation du modèle de régression ridge
				st.write("\n\n- **Equation du modèle :**")
				st.write("Bonheur = 0.461 × PIB + 0.210 × Soutien social + 0.184 × Espérance de vie en bonne santé + 0.172 × Liberté choix de vie - 0.123 × Taux de chômage + 0.032 × Conflits armés + 0.087 × Générosité")
			    
			    #Prédiction du modèle Ridge sur le jeu de test
				pred_test = ridge_reg.predict(X_test)

				# Affichage d'un nuage de points avec les prédictions sur l'axe des abscisses et les vraies valeurs sur l'axe des ordonnées
				fig = plt.figure(figsize = (6,6))
				plt.scatter(pred_test, y_test, c = 'orange')

				# Traçage d'une ligne reliant le point le plus bas au point le plus haut des vraies valeurs pour créer une ligne de référence diagonale
				plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c = 'green')
				plt.xlabel("prediction")
				plt.ylabel("vraie valeur")
				plt.title('Courbe de régression Ridge pour la prédiction du score de bonheur')
				st.pyplot(fig)
			    
			    # Afficher les métriques sur le jeu d'entraînement
				st.write("\n\n- **Métriques sur le jeu d'entraînement :**")
				st.write("MSE (Mean Squared Error) : 0.23960708229878258\n\n"
			              "MAE (Mean Absolute Error) : 0.3876269167933225\n\n"
			              "R^2 (Coefficient de détermination) : 0.7603929177012174")
			    
			    # Afficher les métriques sur le jeu de test
				st.write("\n\n- **Métriques sur le jeu de test :**")
				st.write("MSE (Mean Squared Error) : 0.21973698995404617\n\n"
			              "MAE (Mean Absolute Error) : 0.36613331632479373\n\n"
			              "R^2 (Coefficient de détermination) : 0.7802630100459538")


			# Afficher le contenu correspondant au modèle sélectionné
			elif selected_model_2 == "Régression Lasso":
				#Suppression de colonnes inutiles
				df = df_final.drop(['Positive affect', 'Negative affect'], axis=1)

				# Variables prédictives
				predictors = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth', 'Freedom to make life choices', 'Unemployment rate', 'War country', 'Generosity']

				# Variable dépendante
				dependent = 'Life Ladder'

				# Filtrage des données pour l'entraînement (2011 à 2018)
				train_data = df[(df['year'] >= 2011) & (df['year'] <= 2018)]

				# Filtrage des données pour le test (2019 à 2020)
				test_data = df[(df['year'] >= 2019) & (df['year'] <= 2020)]

				# Variables prédictives pour l'entraînement
				X_train = train_data[predictors]

				# Variable dépendante pour l'entraînement
				y_train = train_data[dependent]

				# Variables prédictives pour le test
				X_test = test_data[predictors]

				# Variable dépendante pour le test
				y_test = test_data[dependent]

				# Copie du DataFrame d'entraînement
				df_train_scaled = train_data.copy()

				# Standardisation des variables prédictives pour l'entraînement
				scaler_predictors_train = preprocessing.StandardScaler()
				df_train_scaled[predictors] = scaler_predictors_train.fit_transform(train_data[predictors])

				# Standardisation de la variable dépendante pour l'entraînement
				scaler_dependent_train = preprocessing.StandardScaler()
				df_train_scaled[dependent] = scaler_dependent_train.fit_transform(train_data[dependent].values.reshape(-1, 1))

				# Copie du DataFrame de test
				df_test_scaled = test_data.copy()

				# Standardisation des variables prédictives pour le test
				scaler_predictors_test = preprocessing.StandardScaler()
				df_test_scaled[predictors] = scaler_predictors_test.fit_transform(test_data[predictors])

				# Standardisation de la variable dépendante pour le test
				scaler_dependent_test = preprocessing.StandardScaler()
				df_test_scaled[dependent] = scaler_dependent_test.fit_transform(test_data[dependent].values.reshape(-1, 1))

				# Assigner les DataFrames standardisés à X_train, y_train, X_test et y_test
				X_train = df_train_scaled[predictors]
				y_train = df_train_scaled[dependent]
				X_test = df_test_scaled[predictors]
				y_test = df_test_scaled[dependent]

				# Création d'un objet modèle Lasso avec un paramètre d'alpha (régularisation)
				alpha = 0  # Valeur d'alpha à ajuster selon vos besoins
				lasso_model = Lasso(alpha=alpha)

				# Entraînement du modèle sur les données d'entraînement
				lasso_model.fit(X_train, y_train)

				st.markdown("- Standardisation de toutes les variables")
				st.markdown("- Entrainement 80% (2011-2018) / Test 20% (2019-2020)")
			    
			    # Afficher l'équation du modèle de régression lasso
				st.write("\n\n- **Equation du modèle :**")
				st.write("Bonheur = 0.48 × PIB + 0.21 × Soutien social + 0.17 × Espérance de vie en bonne santé + 0.17 × Liberté choix de vie - 0.13 × Taux de chômage + 0.03 × Conflits armés + 0.09 × Générosité")
			    
			    # Prédiction du modèle Lasso sur le jeu de test
				pred_test = lasso_model.predict(X_test)

				fig = plt.figure(figsize = (6,6))

				# Tracé du nuage de points avec les prédictions sur l'axe des abscisses et les vraies valeurs sur l'axe des ordonnées
				plt.scatter(pred_test, y_test, c = 'orange')

				# Tracé d'une ligne diagonale allant du minimum au maximum des vraies valeurs, en couleur rouge
				plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c = 'red')

				plt.xlabel("prediction")
				plt.ylabel("vraie valeur")
				plt.title('Courbe de régression Lasso pour la prédiction du score de bonheur (dataset longitudinal)')
				st.pyplot(fig)

			    # Afficher les métriques sur le jeu d'entraînement
				st.write("\n\n- **Métriques sur le jeu d'entraînement :**")
				st.write("MSE (Mean Squared Error) : 0.23959984207565396\n\n"
			              "MAE (Mean Absolute Error) : 0.38792293718141463\n\n"
			              "R^2 (Coefficient de détermination) : 0.760400157924346")
			    
			    # Afficher les métriques sur le jeu de test
				st.write("\n\n- **Métriques sur le jeu de test :**")
				st.write("MSE (Mean Squared Error) : 0.22003783375138894\n\n"
			              "MAE (Mean Absolute Error) : 0.36690209101886684\n\n"
			              "R^2 (Coefficient de détermination) : 0.779962166248611")


			# Afficher le contenu correspondant au modèle sélectionné
			elif selected_model_2 == "Régression Elastic Net":
				#Suppression de colonnes inutiles
				df = df_final.drop(['Positive affect', 'Negative affect'], axis=1)

				# Variables prédictives
				predictors = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth', 'Freedom to make life choices', 'Unemployment rate', 'War country', 'Generosity']

				# Variable dépendante
				dependent = 'Life Ladder'

				# Filtrage des données pour l'entraînement (2011 à 2018)
				train_data = df[(df['year'] >= 2011) & (df['year'] <= 2018)]

				# Filtrage des données pour le test (2019 à 2020)
				test_data = df[(df['year'] >= 2019) & (df['year'] <= 2020)]

				# Variables prédictives pour l'entraînement
				X_train = train_data[predictors]

				# Variable dépendante pour l'entraînement
				y_train = train_data[dependent]

				# Variables prédictives pour le test
				X_test = test_data[predictors]

				# Variable dépendante pour le test
				y_test = test_data[dependent]

				# Copie du DataFrame d'entraînement
				df_train_scaled = train_data.copy()

				# Standardisation des variables prédictives pour l'entraînement
				scaler_predictors_train = preprocessing.StandardScaler()
				df_train_scaled[predictors] = scaler_predictors_train.fit_transform(train_data[predictors])

				# Standardisation de la variable dépendante pour l'entraînement
				scaler_dependent_train = preprocessing.StandardScaler()
				df_train_scaled[dependent] = scaler_dependent_train.fit_transform(train_data[dependent].values.reshape(-1, 1))

				# Copie du DataFrame de test
				df_test_scaled = test_data.copy()

				# Standardisation des variables prédictives pour le test
				scaler_predictors_test = preprocessing.StandardScaler()
				df_test_scaled[predictors] = scaler_predictors_test.fit_transform(test_data[predictors])

				# Standardisation de la variable dépendante pour le test
				scaler_dependent_test = preprocessing.StandardScaler()
				df_test_scaled[dependent] = scaler_dependent_test.fit_transform(test_data[dependent].values.reshape(-1, 1))

				# Assigner les DataFrames standardisés à X_train, y_train, X_test et y_test
				X_train = df_train_scaled[predictors]
				y_train = df_train_scaled[dependent]
				X_test = df_test_scaled[predictors]
				y_test = df_test_scaled[dependent]

				ridge_reg = RidgeCV(alphas= (0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100))
				ridge_reg.fit(X_train, y_train) 

				# Prédictions sur le jeu d'entraînement
				y_train_pred = ridge_reg.predict(X_train)

				# Prédictions sur le jeu de test
				y_test_pred = ridge_reg.predict(X_test)

				# Création d'un objet modèle Elastic Net avec une validation croisée à 8 plis et une liste de rapports L1/L2 spécifiés
				model_en = ElasticNetCV(cv=8, l1_ratio=(0.1, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99), 
				                        alphas=(0.001, 0.01, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0))

				# Entraînement du modèle Elastic Net sur les données d'entraînement
				model_en.fit(X_train, y_train)

				st.markdown("- Standardisation de toutes les variables")
				st.markdown("- Entrainement 80% (2011-2018) / Test 20% (2019-2020)")
		    
				# Afficher l'équation du modèle de régression Elastic Net
				st.write("\n\n- **Equation du modèle :**")
				st.write("Bonheur = 0.432 × PIB + 0.208 × Soutien social + 0.190 × Espérance de vie en bonne santé + 0.169 × Liberté choix de vie - 0.112 × Taux de chômage + 0.016 × Conflits armés + 0.080 × Générosité")

				# Prédiction du modèle Elastic net sur le jeu de test
				pred_test = model_en.predict(X_test)

				fig = plt.figure(figsize = (6,6))
				plt.scatter(pred_test, y_test, c = 'orange')

				plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c = 'blue')
				plt.xlabel("prediction")
				plt.ylabel("vraie valeur")
				plt.title('Courbe de régression Elastic Net pour la prédiction du score de bonheur (dataset longitudinal)')
				st.pyplot(fig)

				# Afficher les métriques sur le jeu d'entraînement
				st.write("\n\n- **Métriques sur le jeu d'entraînement :**")
				st.write("MSE (Mean Squared Error) : 0.23959984207565396\n\n"
				          "R^2 (Coefficient de détermination) : 0.7673401365035228")

				# Afficher les métriques sur le jeu de test
				st.write("\n\n- **Métriques sur le jeu de test :**")
				st.write("MSE (Mean Squared Error) : 0.22003783375138894\n\n"
	              "R^2 (Coefficient de détermination) : 0.7376272826647554")

			else:
				st.write("Aucun modèle sélectionné")

	with tab2:
		# Centrer le titre de la page
		st.markdown("<h2 style='text-align: center;'>Modèles de Classification</h2>", unsafe_allow_html=True)




##################
#   CONCLUSION   #
##################
elif choose == "Conclusion":
	conclu1, conclu2, conclu3 = st.columns([1,8,1])
	conclu2.write("Conclusion")






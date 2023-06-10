# base
import streamlit as st
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
from  PIL import Image

# dataviz
import matplotlib.pyplot as plt
import seaborn as sns

# map
import folium
import json
from folium.plugins import Draw
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# preprocessing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# modeles
from sklearn import tree
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV, Lasso, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier

#resultats
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import plot_tree 

# stats
import statsmodels.api as sm

st.set_page_config(
	layout = "wide", 
	page_title = "Le bonheur national brut",  # String or None. Strings get appended with "• Streamlit". 
	page_icon = ":smile:"  # String, anything supported by st.image, or None.
)

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
	    .stTabs .stTabs [data-baseweb="tab-list"] {{ width: fit-content; margin: 0 auto; }}
	    .stTabs .stTabs [data-baseweb="tab-border"] {{ background-color: transparent;}}
	    span[aria-disabled="true"] {{ background-color: #263A29 !important; color: #F2E3DB; }}
	    [data-testid="stExpander"] {{ width: 80%; margin: 0 auto; }}
    </style>
    """,
    unsafe_allow_html=True
    )

with menu:
	choose = option_menu(None,["Introduction", "Datasets", "Visualisations", "Modélisations", "Conclusion", "Quiz"],
		icons=['globe-americas', 'database', 'bar-chart-line', 'calculator', 'emoji-smile', 'patch-question'],
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

	colLeft, colMid, colRight = st.columns([1, 8, 1])

	colMid.header("COMMENT PRÉDIRE LE NIVEAU DE BONHEUR D'UN PAYS ?")
	colMid.write("\n\n")
	colMid.write("\n\n")
	colMid.image('img/globe_beach.jpg')
	colMid.write("\n\n")
	colMid.write("\n\n")
	colMid.write("Ce projet a été réalisé dans le cadre de la formation Data Analyst au sein de l’organisme Data Scientest. Pour répondre à cette question, nous observé des facteurs politiques, économiques et sociaux.")
	colMid.write("\n\n")
	colMid.markdown("<h4>OBJECTIFS</h4>", unsafe_allow_html=True)
	colMid.markdown("- Déterminer quels sont les facteurs pouvant expliquer le bonheur, mais aussi le poids de chaque facteur, et donc de comprendre les raisons pour lesquelles un pays est mieux classé qu’un autre.")
	colMid.markdown("- Proposer un modèle parcimonieux mais ayant une bonne valeur explicative du bonheur national brut.")
	colMid.markdown("- Présenter ces données de manière interactive, en utilisant des visualisations pertinentes, afin de mettre en évidence les combinaisons de facteurs qui répondent à notre questionnement principal")	
	colMid.write("\n\n")
	colMid.write("\n\n")
	colMid.write('Composition de l’équipe de Data Analyst :') 
	colMid.caption('Francisco Comiran')
	colMid.caption('Zenaba Mogne')
	colMid.caption('Roxane Oubrerie')

################
#   DATASETS   #
################
elif choose == "Datasets":

	colLeft, colMid, colRight = st.columns([1, 8, 1])

	colMid.write('Afin d’obtenir un modèle et donc une réponse à notre problématique au plus proche de la réalité, il nous faut récolter des données de qualité.')

	colMid.subheader('1.Première étape : La récupération des données')
	colMid.write("La source principale des données est le [World Happiness Report](https://worldhappiness.report/), une enquête phare sur l'état du bonheur mondial.")
	colMid.write("Deux jeux de données de type csv ont pu être téléchargé sur [Kaggle](https://www.kaggle.com/ajaypalsinghlo/world-happiness-report-2021): ")
	
	with st.expander("le World Happiness Report"):
		whr = pd.read_csv('datasets/world-happiness-report.csv')
		st.dataframe(whr)

	with st.expander("le World Happiness Report 2021"):
		whr2021 = pd.read_csv('datasets/world-happiness-report-2021.csv')
		st.dataframe(whr2021)

	colLeft, colMid, colRight = st.columns([1, 8, 1])
	colMid.write("\n\n")
	colMid.write("Ces jeux de données seront les sources initiales de nos données. Nous avons décidé de les agrémenter d’autres indicateurs pertinents.")
	colMid.write("\n\n")
	colMid.write("\n\n")
	colMid.subheader('2. Étoffer nos jeux de données')

	with st.expander("Avec un indicateur au sujet de la guerre"):
		st.write("""Création d’un dataset grâce aux données récupérées dans une [page wikipedia](https://en.wikipedia.org/wiki/List_of_armed_conflicts_in_2020.)  à l’aide du webscrapping via la librairie beautiful soup.""")
		st.write("""En téléchargement libre sur le [site de l'ucdp](https://ucdp.uu.se/encyclopedia)""")
		df_war = pd.read_csv("datasets/war_casualties.csv")
		st.dataframe(df_war.head())

	with st.expander("Avec un indicateur au sujet du chômage"):
		st.write("""Création d’un dataset grâce aux données du site Kaggle en téléchargement et libre de droit. Il provient initialement du site [data.worldbank](https://data.worldbank.org/).""")
		df_unemployment = pd.read_csv("datasets/unemployment_analysis.csv")
		st.dataframe(df_unemployment.head())

	with st.expander("Avec d'autres facteurs sociaux et politiques"):
		st.write("""Création d’un dataset grâce aux bases de données [World Economics](https://www.worldeconomics.com/).""")
		
		st.write("ESG Governance")
		df_gov = pd.read_excel("datasets/ESG-Governance.xlsx", sheet_name = "AllData-ByCountryName", skiprows = 6)
		st.dataframe(df_gov.head())

		st.write("ESG Social Index")
		df_social = pd.read_excel("datasets/ESG-Social-Index.xlsx", sheet_name = "AllData-ByCountryName", skiprows = 6)
		st.dataframe(df_social.head())
		
		st.write("Inequality Index")
		df_inequality = pd.read_excel("datasets/Inequality-Index.xlsx", sheet_name = "AllData-ByCountryName", skiprows = 6)
		st.dataframe(df_inequality.head())

	colLeft, colMid, colRight = st.columns([1, 8, 1])

	colMid.write("\n\n")
	colMid.write("\n\n")
	colMid.subheader('3. Mutualisation et préparation des dataframes finaux')
	colMid.write("Nous choisissons de préparer les deux datasets, l’un sur 2021, l’autre sur plusieurs années. Les datasets ont été mergés par la colonne “Country”.")

	colImgLeft, colImgMid, colImgRight = st.columns([3, 6, 3])
	colImgMid.image("img/df2021_final.jpg")

	colLeft, colMid, colRight = st.columns([1, 8, 1])
	colMid.write("\n\n")
	colMid.subheader('Dataframe obtenu :')

	df2021_final = pd.read_csv('datasets/df2021_final.csv')
	with st.expander("DF2021_FINAL"):
		st.dataframe(df2021_final)

	colImgLeft, colImgMid, colImgRight = st.columns([3, 6, 3])
	colImgMid.image("img/df_final.jpg")

	colLeft, colMid, colRight = st.columns([1, 8, 1])
	colMid.write("\n\n")
	colMid.subheader('Dataframe obtenu :')
	df_final = pd.read_csv('datasets/df_final.csv')
	with st.expander("DF_FINAL"):
		st.dataframe(df_final)

######################
#   VISUALISATIONS   #
######################
elif choose == "Visualisations":

	colLeft, colMid, colRight = st.columns([1, 8, 1])

	df1 = pd.read_csv('datasets/world-happiness-report-2021.csv', sep=',')

	#Trier le df par ordre décroissant
	df1_sorted = df1.sort_values(by = 'Ladder score', ascending = False)

	# Regrouper les pays par région en faisant la moyenne du Ladder Score
	df_region_ls = df1.groupby('Regional indicator').agg({'Ladder score' : 'mean'})

	# Trier le dataframe selon le Ladder Score
	df_region_ls = df_region_ls.sort_values(by = "Ladder score")

	colMid.subheader("Analyse de la variable cible : L’échelle du bonheur")
	colMid.markdown("- Top des pays les plus heureux et top 10 des pays les moins heureux en 2021")
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

	colImgLeft, colImgMid, colImgRight = st.columns([2, 8, 2])
	colImgMid.pyplot(fig)

	with st.expander("Pourquoi ce graphique ?"):
		st.write("Ce graphique nous a permis de vérifier le côté déséquilibré de notre jeu de données (notes à 0). Puis cela nous a aidé à mieux visualiser la distribution du Ladder score afin de créer les classes de la variable le cas échéant.")

	colLeft, colMid, colRight = st.columns([1, 8, 1])
	colMid.markdown("- Top des régions les plus heureuses en 2021")
	#Manipulation pour un barplot des ladder score par région

	# Regrouper les pays par région en faisant la moyenne du Ladder Score
	df_region_ls = df1.groupby('Regional indicator').agg({'Ladder score' : 'mean'})

	# Trier le dataframe selon le Ladder Score
	df_region_ls = df_region_ls.sort_values(by = "Ladder score")

	fig2 = plt.figure(figsize=(6,6))
	sns.barplot(y = df_region_ls.index, x = df_region_ls["Ladder score"], palette = sns.color_palette("flare", 10))
	plt.yticks(fontsize=10)
	plt.ylabel("")
	plt.xlabel("")
	plt.title("Classement des régions selon l'échelle du bonheur - 2021", fontsize = 12)

	colImgLeft, colImgMid, colImgRight = st.columns([3, 6, 3])
	colImgMid.pyplot(fig2)

	with st.expander("Pourquoi ce graphique ?"):
		st.write("Ce graphique nous permet de vérifier la distribution du Ladder score par région")

	# Carte du monde

	colLeft, colMid, colRight = st.columns([1, 8, 1])
	colMid.subheader("Matrices de corrélation des deux datasets")
	colMid.markdown("- Dataset 2021")

	df1_corr = df1.drop(['Country name', 'Regional indicator'], axis = 1)

	fig3 = plt.figure(figsize=(15,10))
	correlation_matrix = df1_corr.corr()
	sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
	plt.title("Matrice de corrélation")

	colImgLeft, colImgMid, colImgRight = st.columns([2, 8, 2])
	colImgMid.write(fig3)

	with st.expander("Pourquoi ce graphique ?"):
		st.write("Ce graphique nous apporte des informations concernant la corrélation entre les différentes variables du dataframe portant sur l'année 2021")

	colLeft, colMid, colRight = st.columns([1, 8, 1])
	colMid.markdown("- Dataset longitudinal")

	df_final = pd.read_csv('datasets/df_final.csv')
	df_final_corr = df_final.drop(['Country name', 'year'], axis = 1)

	fig4 = plt.figure(figsize=(10,10))
	correlation_matrix = df_final_corr.corr()
	sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
	plt.title("Matrice de corrélation")
	colImgLeft, colImgMid, colImgRight = st.columns([3, 6, 3])
	colImgMid.write(fig4)

	with st.expander("Pourquoi ce graphique ?"):
		st.write("Ce graphique nous apporte des informations concernant la corrélation entre les différentes variables du dataframe longitudinal")

####################
#   MODELISATION   #
####################
elif choose == "Modélisations":
	colLeft, colMid, colRight = st.columns([1, 8, 1])

	tab1, tab2, tab3 = colMid.tabs(["Modèles quantitatifs", "Modèles de classification", "Simulation"])

	with tab1:
		# Centrer le titre de la page
		st.markdown("<h2 style='text-align: center;'>Modèles Quantitatifs</h2>", unsafe_allow_html=True)

		tab2021, tabLongi = st.tabs(["Dataset 2021", "Dataset Longitudinal"])

		############
		#   2021   #
		############
		with tab2021:
			df2021_final = pd.read_csv('datasets/df2021_final.csv')

			colLeft, colMid, colRight = st.columns([1, 10, 1])

			# Afficher le sous-titre
			colMid.markdown("<h3>Dataset 2021</h3>", unsafe_allow_html=True)


			# Afficher le texte
			colMid.markdown("Aperçu des variables prédictives (avant standardisation):")

			# Affichage du df
			colMid.dataframe(df2021_final.head())


			# Afficher le sous-titre "Modèles"
			colMid.subheader("Modèles")


			####################
			#   Choix modèle   #
			####################
			# Définir les options de la liste déroulante
			options = ["Sélectionner un modèle", "Régression Linéaire Multiple", "Régression Ridge", "Régression Lasso", "Régression Elastic Net"]

			# Afficher la liste déroulante
			selected_model = colMid.selectbox("Sélectionnez un modèle", options)

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

				colInLeft, colInMid, colInRight = st.columns([1, 8, 1])

				colInMid.markdown("- Standardisation des VI")
				colInMid.markdown("- Entrainement 80% / Test 20%")
				colInMid.markdown("- Approche par comparaison de modèles")
				colInMid.markdown("- Exclusion des variables conflits armés, droits politiques, inégalités et générosité")
			    
			    # Afficher l'équation du modèle de régression linéaire
				colInMid.write("\n\n- **Equation du modèle:**")
				colInMid.write("**Bonheur** = 0.46 × PIB + 0.21 × Soutien social + 0.32 × Espérance de vie en bonne santé + 0.19 × Liberté choix de vie - 0.26 × Droit + 0.38 × Liberté presse - 0.18 × Années de scolarité - 0.18 × Chômage - 0.14 × Corruption perçue")
				colInMid.write("\n\n")
				colInMid.write("\n\n")

				#Affichage du graphique pour la prédiction du modèle
				pred_test = model.predict(X_test)

				fig = plt.figure(figsize = (6,6))
				plt.scatter(pred_test, y_test, c = 'orange')

				plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c = 'pink')
				plt.xlabel("prediction")
				plt.ylabel("vraie valeur")
				plt.title('Courbe de régression Linéaire pour la prédiction du score de bonheur (dataset 2021)')

				colImgLeft, colImgMid, colImgRight = st.columns([2, 8, 2])
				colImgMid.pyplot(fig)

				colExtLeft, colMetLeft, colMetRight, colExtRight = st.columns([1, 4, 4, 1])
			    
			    # Afficher les métriques sur le jeu de données d'entraînement
				colMetLeft.write("**Métriques sur le jeu de données d'entraînement :**")
				colMetLeft.write("**MSE** : 0.18725914805070465\n\n"
			              "**MAE** : 0.3356944286948647\n\n"
			              "**R^2** : 0.8359853275283171")
			    
			    # Afficher les métriques sur le jeu de données de test
				colMetRight.write("**Métriques sur le jeu de données de test :**")
				colMetRight.write("**MSE** : 0.20093223750843606\n\n"
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

				colInLeft, colInMid, colInRight = st.columns([1, 8, 1])

				colInMid.markdown("- Standardisation de toutes les variables")
				colInMid.markdown("- Entrainement 80% / Test 20%")
			    
			    # Afficher l'équation du modèle
				colInMid.write("\n\n- **Equation du modèle:**")
				colInMid.write("**Bonheur** = 0.43 × PIB + 0.32 × Soutien social + 0.22 × Espérance de vie en bonne santé + 0.15 × Liberté choix de vie - 0.08 × Corruption perçue - 0.21 × Droit + 0.27 × Liberté presse + 0.02 × Droits politiques - 0.00 × Inégalités - 0.16 × Années de scolarité - 0.14 × Chômage - 0.04 × Conflits armés + 0.05 × Générosité")
				colInMid.write('\n\n')
				colInMid.write('\n\n')

				fig = plt.figure(figsize = (6,6))
				plt.scatter(pred_test, y_test, c = 'orange')

				plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c = 'green')
				plt.xlabel("prediction")
				plt.ylabel("vraie valeur")
				plt.title('Courbe de régression Ridge pour la prédiction du score de bonheur (dataset 2021)')
				
				colImgLeft, colImgMid, colImgRight = st.columns([2, 8, 2])
				colImgMid.pyplot(fig)

				colExtLeft, colMetLeft, colMetRight, colExtRight = st.columns([1, 4, 4, 1])

			    # Afficher les métriques sur le jeu d'entraînement
				colMetLeft.write("**Métriques sur le jeu d'entraînement :**")
				colMetLeft.write("**MSE** : 0.15619765618683173\n\n"
					"**MAE** : 0.30295173204602516\n\n"
					"**R^2** : 0.8482302651797508")
			    
			    # Afficher les métriques sur le jeu de test
				colMetRight.write("**Métriques sur le jeu de test :**")
				colMetRight.write("**MSE** : 0.1870771776297484\n\n"
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
				
				colInLeft, colInMid, colInRight = st.columns([1, 8, 1])

				colInMid.markdown("- Standardisation de toutes les variables")
				colInMid.markdown("- Entrainement 80% / Test 20%")
			    
			    # Afficher l'équation du modèle
				colInMid.write("\n\n- **Equation du modèle:**")
				colInMid.write("**Bonheur** = 0.47 × PIB + 0.31 × Soutien social + 0.22 × Espérance de vie en bonne santé + 0.15 × Liberté choix de vie - 0.09 × Corruption perçue - 0.24 × Droit + 0.29 × Liberté presse + 0.01 × Droits politiques + 0.01 × Inégalités - 0.17 × Années de scolarité - 0.15 × Chômage - 0.04 × Conflits armés + 0.05 × Générosité")
				
				colInMid.write("\n\n")
				colInMid.write("\n\n")

			    # Affichage d'un nuage de points avec les prédictions du modèle Lasso (axe x) et les vraies valeurs (axe y) du jeu de test
				fig = plt.figure(figsize=(6, 6))
				plt.scatter(pred_test, y_test, c='orange')

				# Affichage d'une ligne diagonale représentant l'identité (x = y) pour visualiser l'ajustement du modèle
				plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c='red')

				plt.xlabel("prediction")
				plt.ylabel("vraie valeur")
				plt.title('Courbe de régression Lasso pour la prédiction du score de bonheur (dataset 2021)')
				
				colImgLeft, colImgMid, colImgRight = st.columns([2, 8, 2])
				colImgMid.pyplot(fig)

				colExtLeft, colMetLeft, colMetRight, colExtRight = st.columns([1, 4, 4, 1])
			    
			    # Afficher les métriques sur le jeu d'entraînement
				colMetLeft.write("**Métriques sur le jeu d'entraînement :**")
				colMetLeft.write("**MSE** : 0.15656140168987856\n\n"
			              "**MAE** : 0.3037389672150526\n\n"
			              "**R^2** : 0.8478768312045735")
			    
			    # Afficher les métriques sur le jeu de test
				colMetRight.write("**Métriques sur le jeu de test :**")
				colMetRight.write("**MSE** : 0.18627690494010946\n\n"
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

				colInLeft, colInMid, colInRight = st.columns([1, 8, 1])

				colInMid.markdown("- Standardisation de toutes les variables")
				colInMid.markdown("- Entrainement 80% / Test 20%")
			    
			    # Afficher l'équation du modèle
				colInMid.write("\n\n- **Equation du modèle:**")
				colInMid.write("**Bonheur** = 0.420728 × PIB + 0.319837 × Support social + 0.217963 × Espérance de vie en bonne santé + 0.153546 × Liberté de choix de vie + 0.077608 × Corruption perçue - 0.191576 × Droit + 0.269812 × Liberté de la presse + 0.013726 × Droits politiques - 0.003178 × Inégalités - 0.156633 × Années de scolarité - 0.140324 × Taux de chômage - 0.036933 × Conflits armés + 0.045715 × Générosité")

				colInMid.write("\n\n")
				colInMid.write("\n\n")

			    # Afficher le graph correspondant
				pred_test = model_en.predict(X_test)

				fig = plt.figure(figsize = (6,6))
				plt.scatter(pred_test, y_test, c = 'orange')

				plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c = 'blue')
				plt.xlabel("prediction")
				plt.ylabel("vraie valeur")
				plt.title('Courbe de régression Elastic Net pour la prédiction du score de bonheur (dataset 2021)')
				
				colImgLeft, colImgMid, colImgRight = st.columns([2, 8, 2])
				colImgMid.pyplot(fig)

				colExtLeft, colMetLeft, colMetRight, colExtRight = st.columns([1, 4, 4, 1])
			    
			    # Afficher les métriques sur le jeu d'entraînement
				colMetLeft.write("**Métriques sur le jeu d'entraînement :**")
				colMetLeft.write("**MSE** : 0.3954930955674146\n\n"
			              "**R^2** : 0.8480192880379052")
			    
			    # Afficher les métriques sur le jeu de test
				colMetRight.write("**Métriques sur le jeu de test :**")
				colMetRight.write("**MSE** : 0.4334633441149267\n\n"
			              "**R^2** : 0.7867190703534337")

			else:
				colMid.write("Aucun modèle sélectionné")

		with tabLongi:
			####################
			#   Longitudinal   #
			####################
			df_final = pd.read_csv('datasets/df_final.csv')

			colLeft, colMid, colRight = st.columns([1, 10, 1])

			# Afficher le sous-titre
			colMid.markdown("<h3>Dataset 2011 - 2020</h3>", unsafe_allow_html=True)

			# Afficher le texte
			colMid.markdown("Aperçu des variables prédictives (avant standardisation):")

			# Affichage du df
			colMid.dataframe(df_final.head())

			# Afficher le sous-titre "Modèles"
			colMid.subheader("Modèles")

			# Définir les options de la deuxième liste déroulante
			options_2 = ["Sélectionner un modèle", "Régression Linéaire Multiple", "Régression Ridge", "Régression Lasso", "Régression Elastic Net"]

			# Afficher la deuxième liste déroulante
			selected_model_2 = colMid.selectbox("Sélectionnez un modèle (Dataset 2011 - 2020)", options_2)

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

				colInLeft, colInMid, colInRight = st.columns([1, 8, 1])

				colInMid.markdown("- Standardisation des VI")
				colInMid.markdown("- Entrainement 80% (2011-2018) / Test 20% (2019-2020)")
				colInMid.markdown("- Approche par comparaison de modèles")
				colInMid.markdown("- Exclusion de la variable conflits armés")

			    # Afficher l'équation du modèle de régression linéaire
				colInMid.write("\n\n- **Equation du modèle :**")
				colInMid.write("Bonheur = 0.52 × PIB + 2.27 × Soutien social + 0.32 × Espérance de vie en bonne santé + 1.46 × Liberté de choix - 0.02 × Taux de chômage + 0.62 × Générosité")

				colInMid.write("\n\n")
				colInMid.write("\n\n")

			    # Afficher le graph correspondant
				pred_test = model.predict(X_test)

				fig = plt.figure(figsize = (6,6))
				plt.scatter(pred_test, y_test, c = 'orange')

				plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c = 'pink')
				plt.xlabel("prediction")
				plt.ylabel("vraie valeur")
				plt.title('Régression Linéaire pour la prédiction du score de bonheur (dataset longitudinal)')
				
				colImgLeft, colImgMid, colImgRight = st.columns([2, 8, 2])
				colImgMid.pyplot(fig)

				colExtLeft, colMetLeft, colMetRight, colExtRight = st.columns([1, 4, 4, 1])

			    # Afficher les métriques sur le jeu de données d'entraînement
				colMetLeft.write("**Métriques sur le jeu de données d'entraînement :**")
				colMetLeft.write("MSE (Mean Squared Error) : 0.26469127594001707\n\n"
			              "MAE (Mean Absolute Error) : 0.4063954536271683\n\n"
			              "R^2 (Coefficient de détermination) : 0.7594485002408284")

			    # Afficher les métriques sur le jeu de données de test
				colMetRight.write("**Métriques sur le jeu de données de test :**")
				colMetRight.write("MSE (Mean Squared Error) : 0.22702134110641828\n\n"
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

				colInLeft, colInMid, colInRight = st.columns([1, 8, 1])

				colInMid.markdown("- Standardisation de toutes les variables")
				colInMid.markdown("- Entrainement 80% (2011-2018) / Test 20% (2019-2020)")
			    
			    # Afficher l'équation du modèle de régression ridge
				colInMid.write("\n\n- **Equation du modèle :**")
				colInMid.write("Bonheur = 0.461 × PIB + 0.210 × Soutien social + 0.184 × Espérance de vie en bonne santé + 0.172 × Liberté choix de vie - 0.123 × Taux de chômage + 0.032 × Conflits armés + 0.087 × Générosité")
			    
				colInMid.write("\n\n")
				colInMid.write("\n\n")

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
				
				colImgLeft, colImgMid, colImgRight = st.columns([2, 8, 2])
				colImgMid.pyplot(fig)

				colExtLeft, colMetLeft, colMetRight, colExtRight = st.columns([1, 4, 4, 1])
			    
			    # Afficher les métriques sur le jeu d'entraînement
				colMetLeft.write("**Métriques sur le jeu d'entraînement :**")
				colMetLeft.write("MSE (Mean Squared Error) : 0.23960708229878258\n\n"
			              "MAE (Mean Absolute Error) : 0.3876269167933225\n\n"
			              "R^2 (Coefficient de détermination) : 0.7603929177012174")
			    
			    # Afficher les métriques sur le jeu de test
				colMetRight.write("**Métriques sur le jeu de test :**")
				colMetRight.write("MSE (Mean Squared Error) : 0.21973698995404617\n\n"
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

				colInLeft, colInMid, colInRight = st.columns([1, 8, 1])

				colInMid.markdown("- Standardisation de toutes les variables")
				colInMid.markdown("- Entrainement 80% (2011-2018) / Test 20% (2019-2020)")
			    
			    # Afficher l'équation du modèle de régression lasso
				colInMid.write("\n\n- **Equation du modèle :**")
				colInMid.write("Bonheur = 0.48 × PIB + 0.21 × Soutien social + 0.17 × Espérance de vie en bonne santé + 0.17 × Liberté choix de vie - 0.13 × Taux de chômage + 0.03 × Conflits armés + 0.09 × Générosité")

				colInMid.write("\n\n")
				colInMid.write("\n\n")

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
				
				colImgLeft, colImgMid, colImgRight = st.columns([2, 8, 2])
				colImgMid.pyplot(fig)

				colExtLeft, colMetLeft, colMetRight, colExtRight = st.columns([1, 4, 4, 1])

			    # Afficher les métriques sur le jeu d'entraînement
				colMetLeft.write("\n\n- **Métriques sur le jeu d'entraînement :**")
				colMetLeft.write("MSE (Mean Squared Error) : 0.23959984207565396\n\n"
			              "MAE (Mean Absolute Error) : 0.38792293718141463\n\n"
			              "R^2 (Coefficient de détermination) : 0.760400157924346")
			    
			    # Afficher les métriques sur le jeu de test
				colMetRight.write("\n\n- **Métriques sur le jeu de test :**")
				colMetRight.write("MSE (Mean Squared Error) : 0.22003783375138894\n\n"
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

				colInLeft, colInMid, colInRight = st.columns([1, 8, 1])

				colInMid.markdown("- Standardisation de toutes les variables")
				colInMid.markdown("- Entrainement 80% (2011-2018) / Test 20% (2019-2020)")
		    
				# Afficher l'équation du modèle de régression Elastic Net
				colInMid.write("\n\n- **Equation du modèle :**")
				colInMid.write("Bonheur = 0.432 × PIB + 0.208 × Soutien social + 0.190 × Espérance de vie en bonne santé + 0.169 × Liberté choix de vie - 0.112 × Taux de chômage + 0.016 × Conflits armés + 0.080 × Générosité")

				colInMid.write("\n\n")
				colInMid.write("\n\n")

				# Prédiction du modèle Elastic net sur le jeu de test
				pred_test = model_en.predict(X_test)

				fig = plt.figure(figsize = (6,6))
				plt.scatter(pred_test, y_test, c = 'orange')

				plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), c = 'blue')
				plt.xlabel("prediction")
				plt.ylabel("vraie valeur")
				plt.title('Courbe de régression Elastic Net pour la prédiction du score de bonheur (dataset longitudinal)')
				
				colImgLeft, colImgMid, colImgRight = st.columns([2, 8, 2])
				colImgMid.pyplot(fig)

				colExtLeft, colMetLeft, colMetRight, colExtRight = st.columns([1, 4, 4, 1])

				# Afficher les métriques sur le jeu d'entraînement
				colMetLeft.write("**Métriques sur le jeu d'entraînement :**")
				colMetLeft.write("MSE (Mean Squared Error) : 0.23959984207565396\n\n"
				          "R^2 (Coefficient de détermination) : 0.7673401365035228")

				# Afficher les métriques sur le jeu de test
				colMetRight.write("**Métriques sur le jeu de test :**")
				colMetRight.write("MSE (Mean Squared Error) : 0.22003783375138894\n\n"
	              "R^2 (Coefficient de détermination) : 0.7376272826647554")

			else:
				colMid.write("Aucun modèle sélectionné")

	with tab2:
		# Centrer le titre de la page
		st.markdown("<h2 style='text-align: center;'>Modèles de Classification</h2>", unsafe_allow_html=True)

		tab2021, tabLongi = st.tabs(["Dataset 2021", "Dataset Longitudinal"])

		############
		#   2021   #
		############
		with tab2021:
			df= pd.read_csv('datasets/df2021_final.csv')

			colLeft, colMid, colRight = st.columns([1, 10, 1])

			# Afficher le sous-titre
			colMid.markdown("<h3>Dataset 2021</h3>", unsafe_allow_html=True)


			# Afficher le texte
			colMid.markdown("Aperçu des variables prédictives (avant standardisation):")

			# Affichage du df
			colMid.dataframe(df2021_final.head())

			# Création de la colonne catégorielle correspondant au Ladder score divisé en 3 classes (tercile)
			df['hapiness_categ'] = pd.qcut(df['Ladder score'], q=[0, 0.33, 0.66, 1], labels=['Low', 'Medium', 'High'])

			# Supression des colonnes inutiles
			df = df.drop(['Country name', 'Regional indicator', 'Ladder score'], axis=1)

			#Separer variable cible des variables explicatives
			X = df.drop('hapiness_categ', axis = 1)
			y = df['hapiness_categ']

			# Divisez les données en ensembles d'entraînement et de test :
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 42)

			#Standardiser les valeurs
			sc = StandardScaler()
			X_train[X_train.columns] = sc.fit_transform(X_train[X_train.columns])
			X_test[X_test.columns] = sc.transform(X_test[X_test.columns])

			# Encodage de la variable cible
			le = LabelEncoder()
			y_train = le.fit_transform(y_train)
			y_test = le.transform(y_test)

			# Afficher le sous-titre "Modèles"
			colMid.subheader("Modèles")

			# Définir les options de la liste déroulante
			options3 = ["Sélectionner un modèle", "Régression Logistique", "Arbre de Décision", "Random Forest Classifier"]

			# Afficher la liste déroulante
			selected_model3 = colMid.selectbox("Choisissez un modèle pour explorer ses résultats et évaluations :", options3)


			# Afficher le contenu correspondant au modèle sélectionné
			if selected_model3 == "Régression Logistique":

				colInLeft, colInMid, colInRight = st.columns([1, 8, 1])

				colInMid.markdown("- Standardisation des VI")
				colInMid.markdown("- Entrainement 80% / Test 20%")
				colInMid.markdown("- Transformation de la variable 'bonheur' en la divisant en trois catégories équilibrées, basées sur des terciles")
				colInMid.markdown("- Accuracy train : 0.8660714285714286")
				colInMid.markdown("- Accuracy test : 0.6071428571428571")

				reglog = LogisticRegression(random_state = 42)
				reglog.fit(X_train, y_train)

				#Predire les classes
				y_pred = reglog.predict(X_test)

				col1, col2, col3 = st.columns([2,4,2])

				# Affichage de la matrice de confusion et du rapport de classification
				col2.caption('Matrice de confusion')
				col2.dataframe(pd.crosstab(y_test,y_pred, rownames=['Realité'], colnames=['Prédiction']))

				report = classification_report(y_test, y_pred, output_dict = True)
				df_report = pd.DataFrame(report).transpose()
				col2.caption('Rapport de classification')
				col2.dataframe(df_report)

			# Afficher le contenu correspondant au modèle sélectionné
			elif selected_model3 == "Arbre de Décision":

				colInLeft, colInMid, colInRight = st.columns([1, 8, 1])

				colInMid.markdown("- Standardisation des VI")
				colInMid.markdown("- Entrainement 80% / Test 20%")
				colInMid.markdown("- Transformation de la variable 'bonheur' en la divisant en trois catégories équilibrées, basées sur des terciles")
				colInMid.markdown("- Sélection des 3 variables les + influentes et ré-entraînement")
				colInMid.markdown("- Accuracy train : 1.0")
				colInMid.markdown("- Accuracy test : 0.5357142857142857")

				# entrainement de l'arbre de décision
				clf = tree.DecisionTreeClassifier(random_state = 42)
				clf.fit(X_train, y_train)

				y_pred = clf.predict(X_test)

				# Création d'un DataFrame pour stocker les importances des features
				feat_importances = pd.DataFrame(clf.feature_importances_, index=X.columns, columns=["Importance"])

				# Tri des features par ordre décroissant d'importance
				feat_importances.sort_values(by='Importance', ascending=False, inplace=True)

				fig = plt.figure(figsize = (8,8))
				plt.bar(x = feat_importances.index, height = feat_importances.Importance)
				plt.xticks(rotation=90)

				# Tracé d'un diagramme en barres pour visualiser les importances des features
				stcolImgLeft, colImgMid, colImgRight = st.columns([2, 8, 2])
				colImgMid.pyplot(fig)

				#Nouvel entraînement avec 3 variables
				X_train_new = X_train[['Law','Social support','Press_Freedom']]
				X_test_new = X_test[['Law','Social support','Press_Freedom']]

				clf = tree.DecisionTreeClassifier(random_state=42) 
				  
				clf.fit(X_train_new, y_train)

				col1, col2, col3 = st.columns([2,4,2])

				y_pred = clf.predict(X_test_new)

				# Affichage de la matrice de confusion et du rapport de classification
				col2.caption('Matrice de confusion')
				col2.dataframe(pd.crosstab(y_test,y_pred, rownames=['Realité'], colnames=['Prédiction']))

				report = classification_report(y_test, y_pred, output_dict = True)
				df_report = pd.DataFrame(report).transpose()
				col2.caption('Rapport de classification')
				col2.dataframe(df_report)

				clf = tree.DecisionTreeClassifier(random_state=42,max_depth = 3) 

				clf.fit(X_train_new, y_train)

				arbre, ax = plt.subplots(figsize=(10, 10)) 

				plot_tree(clf, 
				          feature_names = ['Law','Social support','Press_Freedom'],
				          class_names = ['Low','Medium','High'],
				          filled = True, 
				          rounded = True)

				colImgLeft, colImgMid, colImgRight = st.columns([2, 8, 2])
				colImgMid.pyplot(arbre)

			# Afficher le contenu correspondant au modèle sélectionné
			elif selected_model3 == "Random Forest Classifier":

				colInLeft, colInMid, colInRight = st.columns([1, 8, 1])
				
				colInMid.markdown("- Standardisation des VI")
				colInMid.markdown("- Entrainement 80% / Test 20%")
				colInMid.markdown("- Transformation de la variable 'bonheur' en la divisant en trois catégories équilibrées, basées sur des terciles")
				colInMid.markdown("- Réechantillonnage")

				rf = RandomForestClassifier(random_state = 42)
				rf.fit(X_train, y_train)

				col1, col2, col3 = st.columns([2,4,2])

				y_pred = rf.predict(X_test)

				# Affichage de la matrice de confusion et du rapport de classification
				col2.caption('Matrice de confusion')
				col2.dataframe(pd.crosstab(y_test,y_pred, rownames=['Realité'], colnames=['Prédiction']))

				report = classification_report(y_test, y_pred, output_dict = True)
				df_report = pd.DataFrame(report).transpose()
				col2.caption('Rapport de classification')
				col2.dataframe(df_report)

			else:
				colMid.write("Aucun modèle sélectionné")

		with tabLongi:
			####################
			#   Longitudinal   #
			####################
			df_final = pd.read_csv('datasets/df_final.csv')

			# création de la colonne catégorielle correspondant au score de bonheur, basé sur les tertiles
			df_final['hapiness_categ'] = pd.qcut(df_final['Life Ladder'], q=[0, 0.33, 0.66, 1], labels=['Low', 'Medium', 'High'])
			df_final['hapiness_categ'].value_counts()

			# suppression des colonnes inutiles
			df_final = df_final.drop(['Country name', 'year', 'Life Ladder', 'Positive affect', 'Negative affect'], axis = 1)

			# séparation des variables explicatives et de la variable cible
			feats = df_final.drop('hapiness_categ', axis = 1)
			target = df_final.hapiness_categ

			# Création des jeux de données de test et d'entrainement
			X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.20, random_state = 42)

			# Standardisation des variables explicatives
			sc = StandardScaler()

			X_train[X_train.columns] = sc.fit_transform(X_train[X_train.columns])
			X_test[X_test.columns] = sc.transform(X_test[X_test.columns])

			# Encodage de la variable cible
			le = LabelEncoder()

			y_train = le.fit_transform(y_train)
			y_test = le.transform(y_test)

			colLeft, colMid, colRight = st.columns([1, 10, 1])

			# Afficher le sous-titre
			colMid.markdown("<h3>Dataset 2011 - 2020</h3>", unsafe_allow_html=True)

			# Afficher le texte
			colMid.markdown("Aperçu des variables prédictives (avant standardisation):")

			# Affichage du df
			colMid.dataframe(df_final.head())

			# Afficher le sous-titre "Modèles"
			colMid.subheader("Modèles")

			# Définir les options de la liste déroulante
			options4 = ["Sélectionner un modèle", "Régression Logistique", "Arbre de Décision", "Random Forest Classifier"]

			# Afficher la liste déroulante
			#selected_model4 = st.selectbox("Choisissez un modèle pour explorer ses résultats et évaluations :", options4)
			selected_model4 = colMid.selectbox("Choisissez un modèle pour explorer ses résultats et évaluations :", options4, key="model4_selection")

			# Afficher le contenu correspondant au modèle sélectionné
			if selected_model4 == "Régression Logistique":

				colInLeft, colInMid, colInRight = st.columns([1, 8, 1])

				colInMid.markdown("- Standardisation des VI")
				colInMid.markdown("- Entrainement 80% / Test 20%")
				colInMid.markdown("- Transformation de la variable 'bonheur' en la divisant en trois catégories équilibrées, basées sur des terciles")

				# Application d'un régression logistique
				reglog = LogisticRegression(random_state = 42)
				reglog.fit(X_train, y_train)

				colExtLeft, colMetLeft, colMetRight, colExtRight = st.columns([1, 5, 5, 1])

				colMetLeft.markdown("<h4>Sans hyperparamètres</h4>", unsafe_allow_html=True)

				y_pred = reglog.predict(X_test)

				# Affichage de la matrice de confusion et du rapport de classification
				colMetLeft.caption('Matrice de confusion')
				colMetLeft.dataframe(pd.crosstab(y_test,y_pred, rownames=['Realité'], colnames=['Prédiction']))

				report = classification_report(y_test, y_pred, output_dict = True)
				df_report = pd.DataFrame(report).transpose()
				colMetLeft.caption('Rapport de classification')
				colMetLeft.dataframe(df_report)

				colMetRight.markdown("<h4>Avec hyperparamètres</h4>", unsafe_allow_html=True)

				#Nouvelle application de la regression logistique avec modification des hyperparamètre
				reglog2 = LogisticRegression(C = 0.05963623316594643, penalty = 'l2', solver = 'lbfgs', random_state = 42)
				reglog2.fit(X_train, y_train)

				# classification_report et matrice de confusion
				y_pred_2 = reglog2.predict(X_test)

				y_pred = reglog.predict(X_test)

				# Affichage de la matrice de confusion et du rapport de classification
				colMetRight.caption('Matrice de confusion')
				colMetRight.dataframe(pd.crosstab(y_test,y_pred, rownames=['Realité'], colnames=['Prédiction']))

				report = classification_report(y_test, y_pred, output_dict = True)
				df_report = pd.DataFrame(report).transpose()
				colMetRight.caption('Rapport de classification')
				colMetRight.dataframe(df_report)
			    
			# Afficher le contenu correspondant au modèle sélectionné
			elif selected_model4 == "Arbre de Décision":

				colInLeft, colInMid, colInRight = st.columns([1, 8, 1])

				colInMid.markdown("- Standardisation des VI")
				colInMid.markdown("- Entrainement 80% / Test 20%")
				colInMid.markdown("- Transformation de la variable 'bonheur' en la divisant en trois catégories équilibrées, basées sur des terciles")
				colInMid.markdown("- Sélection des 3 variables les + influentes et ré-entraînement")

				clf = tree.DecisionTreeClassifier(random_state = 42)
				clf.fit(X_train, y_train)

				col1, col2, col3 = st.columns([2,4,2])

				y_pred = clf.predict(X_test)

				# Affichage de la matrice de confusion et du rapport de classification
				col2.caption('Matrice de confusion')
				col2.dataframe(pd.crosstab(y_test,y_pred, rownames=['Realité'], colnames=['Prédiction']))

				report = classification_report(y_test, y_pred, output_dict = True)
				df_report = pd.DataFrame(report).transpose()
				col2.caption('Rapport de classification')
				col2.dataframe(df_report)

				#Rentrainement du modèle avec la modification des hyperparamètres
				clf2 = tree.DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 2)
				clf2.fit(X_train, y_train)

				# Création d'un DataFrame pour stocker les importances des features
				feat_importances = pd.DataFrame(clf2.feature_importances_, index=feats.columns, columns=["Importance"])

				# Tri des features par ordre décroissant d'importance
				feat_importances.sort_values(by='Importance', ascending=False, inplace=True)

				fig = plt.figure(figsize = (8,8))
				plt.bar(x = feat_importances.index, height = feat_importances.Importance)
				plt.xticks(rotation=90)

				# Tracé d'un diagramme en barres pour visualiser les importances des features
				colImgLeft, colImgMid, colImgRight = st.columns([2, 8, 2])
				colImgMid.pyplot(fig)

				X_train_new = X_train[['Log GDP per capita','Unemployment rate','Healthy life expectancy at birth']]
				X_test_new = X_test[['Log GDP per capita','Unemployment rate','Healthy life expectancy at birth']]

				clf2 = tree.DecisionTreeClassifier(random_state=42,max_depth = 3) 

				clf2.fit(X_train_new, y_train)

				arbre, ax = plt.subplots(figsize=(10, 10))  

				plot_tree(clf2, 
				          feature_names = ['Log GDP per capita','Unemployment rate','Healthy life expectancy at birth'],
				          class_names = ['Low','Medium','High'],
				          filled = True, 
				          rounded = True)

				colImgLeft, colImgMid, colImgRight = st.columns([2, 8, 2])
				colImgMid.pyplot(arbre)

			# Afficher le contenu correspondant au modèle sélectionné
			elif selected_model4 == "Random Forest Classifier":

				colInLeft, colInMid, colInRight = st.columns([1, 8, 1])
				
				colInMid.markdown("- Standardisation des VI")
				colInMid.markdown("- Entrainement 80% / Test 20%")
				colInMid.markdown("- Transformation de la variable 'bonheur' en la divisant en trois catégories équilibrées, basées sur des terciles")
				colInMid.markdown("- Réechantillonnage")

				colExtLeft, colMetLeft, colMetRight, colExtRight = st.columns([1, 5, 5, 1])

				colMetLeft.markdown("<h4>Sans hyperparamètres</h4>", unsafe_allow_html=True)

				# Random Forest Classifier
				rf = RandomForestClassifier(random_state=42)
				rf.fit(X_train, y_train)

				y_pred = rf.predict(X_test)

				# Affichage de la matrice de confusion et du rapport de classification
				colMetLeft.caption('Matrice de confusion')
				colMetLeft.dataframe(pd.crosstab(y_test,y_pred, rownames=['Realité'], colnames=['Prédiction']))

				report = classification_report(y_test, y_pred, output_dict = True)
				df_report = pd.DataFrame(report).transpose()
				colMetLeft.caption('Rapport de classification')
				colMetLeft.dataframe(df_report)

				colMetRight.markdown("<h4>Avec hyperparamètres</h4>", unsafe_allow_html=True)
				#Réentrainement du modèle avec modification des hyperparamtres
				rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 500, max_depth=8, criterion='gini')
				rfc1.fit(X_train, y_train)

				y_pred = rfc1.predict(X_test)

				# Affichage de la matrice de confusion et du rapport de classification
				colMetRight.caption('Matrice de confusion')
				colMetRight.dataframe(pd.crosstab(y_test,y_pred, rownames=['Realité'], colnames=['Prédiction']))

				report = classification_report(y_test, y_pred, output_dict = True)
				df_report = pd.DataFrame(report).transpose()
				colMetRight.caption('Rapport de classification')
				colMetRight.dataframe(df_report)

			else:
				colMid.write("Aucun modèle sélectionné")

	with tab3:
		# Charger le jeu de données
		data = pd.read_csv("datasets/df2021_final.csv")

		# Liste des variables dans le jeu de données
		variables = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy',
		             'Freedom to make life choices', 'Perceptions of corruption', 'Law',
		             'Press_Freedom', 'Political_Rights', 'Inequality', 'Schooling',
		             'Unemployment rate', 'armed_conflicts', 'Generosity']

		# Interface utilisateur pour choisir les variables
		selected_variables = []

		st.subheader('Modèle du Bonheur Intéractif')

		st.write("Composez votre propre modèle du bonheur en sélectionnant les variables qui vous semblent pertinentes. Ensuite, vous pourrez observer le poids de chaque variable dans la prédiction du bonheur d'un pays en fonction de votre modèle. Enfin, vous pourrez constater quel pourcentage de la variance du bonheur est expliqué par votre modèle")
		
		col1, col2, col3, col4 = st.columns(4)

		if col1.checkbox(variables[0]):
			selected_variables.append(variables[0])
		if col1.checkbox(variables[4]):
			selected_variables.append(variables[4])
		if col1.checkbox(variables[8]):
			selected_variables.append(variables[8])
		if col1.checkbox(variables[12]):
			selected_variables.append(variables[12])

		if col2.checkbox(variables[3]):
			selected_variables.append(variables[3])
		if col2.checkbox(variables[5]):
			selected_variables.append(variables[5])
		if col2.checkbox(variables[9]):
			selected_variables.append(variables[9])

		if col3.checkbox(variables[2]):
			selected_variables.append(variables[2])
		if col3.checkbox(variables[6]):
			selected_variables.append(variables[6])
		if col3.checkbox(variables[10]):
			selected_variables.append(variables[10])

		if col4.checkbox(variables[1]):
			selected_variables.append(variables[1])
		if col4.checkbox(variables[7]):
			selected_variables.append(variables[7])
		if col4.checkbox(variables[11]):
			selected_variables.append(variables[11])

		if len(selected_variables) == 0:
		    st.warning("Veuillez sélectionner au moins une variable.")
		else:
		    # Variable dépendante
		    dependent = 'Ladder score'

		    # Séparation des données en jeu d'entraînement et jeu de test
		    X_train, X_test, y_train, y_test = train_test_split(data[selected_variables], data[dependent],
		                                                        test_size=0.2, random_state=40)

		    # Création d'un objet StandardScaler
		    scaler = StandardScaler()

		    # Standardisation des variables prédictives pour le jeu d'entraînement
		    X_train_scaled = scaler.fit_transform(X_train)
		    # Remplace les valeurs de X_train par les valeurs standardisées
		    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)

		    # Réinitialiser les indices des données
		    X_train.reset_index(drop=True, inplace=True)
		    y_train.reset_index(drop=True, inplace=True)

		    # Création d'un objet modèle de régression linéaire
		    X_train_with_constant = sm.add_constant(X_train)
		    model_sm = sm.OLS(y_train, X_train_with_constant)

		    # Entraînement du modèle sur le jeu d'entraînement
		    model = LinearRegression()
		    model.fit(X_train, y_train)

		    # Prédictions sur le jeu d'entraînement
		    y_train_pred = model.predict(X_train)

		    # Prédictions sur le jeu de test
		    y_test_pred = model.predict(X_test)

		    # Métriques sur le jeu d'entraînement
		    mse_train = mean_squared_error(y_train, y_train_pred)
		    mae_train = mean_absolute_error(y_train, y_train_pred)
		    r2_train = r2_score(y_train, y_train_pred)

		    # Obtenir les p-values des coefficients
		    results = model_sm.fit()
		    p_values = results.pvalues[1:]  # Exclure le coefficient de constante

		    # Récupération des coefficients des variables sélectionnées
		    coefficients = pd.DataFrame({'Variable': X_train.columns, 'Coefficient': model.coef_})

		    # Affichage des VI sélectionnées avec les coefficients et les p-values
		    selected_variables_interactive = st.multiselect("Variables indépendantes sélectionnées", variables,
		                                                    default=selected_variables,
		                                                    disabled = True)

		    # Récupération des coefficients des variables sélectionnées
		    coefficients_selected = coefficients[coefficients['Variable'].isin(selected_variables_interactive)]

		    # Affichage du barplot des coefficients
		    fig, ax = plt.subplots(figsize=(10, 6))
		    sns.barplot(x='Variable', y='Coefficient', data=coefficients_selected, ax=ax)
		    ax.set_xticklabels(selected_variables_interactive, rotation=90)
		    ax.set_ylabel('Coefficient')
		    ax.set_title('Coefficients des variables indépendantes')

		    colImgLeft, colImgMid, colImgRight = st.columns([2, 8, 2])
		    colImgMid.pyplot(fig)

		    # Affichage des métriques
		    st.write("Part de variance expliquée (R2) :", r2_train)

##################
#   CONCLUSION   #
##################
elif choose == "Conclusion":

	colLeft, colMid, colRight = st.columns([1, 8, 1])

	colMid.subheader("Conclusion")

	colMid.write("Les deux meilleurs modèles pour prédire l'indice de bonheur brut national sont la régression linéaire multiple utilisant une approche comparative de modèles et le random forest classifier.")
	colMid.write("Nous souhaitions connaître les variables ayant le plus de poids dans le calcul du score de bonheur afin d'en tirer une équation.")
	colMid.write("\n\n")
	colMid.write("\n\n")

	colMid.markdown("<h5>Bonheur = 0.46 × PIB + 0.21 × Soutien social + 0.32 × Espérance de vie en bonne santé + 0.19 × Liberté choix de vie - 0.26 × Droit + 0.38 × Liberté presse - 0.18 × Années de scolarité - 0.18 × Chômage - 0.14 × Corruption perçue</h5>", unsafe_allow_html=True)
	colMid.write("\n\n")
	colMid.write("\n\n")

	variables = ['PIB', 'Liberté Presse', 'Santé',
             'Droit', 'Soutien Social', 'Scolarité', 'Liberté Vie', 'Chômage',
             'Corruption Perçue']
	coefficients = [0.4570331728929467, 0.3824984257759499, 0.31686315162364254, -0.2627542194623681,
	                0.21160954293664438, -0.18416223137371948, 0.19481694441003133, -0.17599495021472106,
	                -0.14400512347906697]
	lower_bounds = [0.2225637288951422, 0.2576001210282243, 0.13668452174330462, -0.468403336569577,
	                0.06829144981340154, -0.3321163539671077, 0.08496750030310063, -0.27409283166793974,
	                -0.24685320316512274]
	upper_bounds = [0.6915026168907512, 0.5073967305236755, 0.4970417815039805, -0.05710510235515917,
	                0.3549276360598872, -0.03620810878033123, 0.30466638851696204, -0.07789706876150237,
	                -0.04115704379301119]
	fig, ax = plt.subplots()
	colors = ['#E86A33' if c < 0 else '#41644A' for c in coefficients]
	ax.bar(variables, coefficients, yerr=[np.subtract(coefficients, lower_bounds), np.subtract(upper_bounds, coefficients)], capsize=5, color=colors)
	ax.set_ylabel('Coefficients')
	ax.set_xlabel('Variables')
	ax.set_title('Poids des variables dans la régression linéaire finale')
	plt.xticks(rotation=90)

	colImgLeft, colImgMid, colImgRight = st.columns([2, 8, 2])
	colImgMid.pyplot(fig)

############
#   QUIZ   #
############
elif choose == "Quiz":

	colLeft, colMid, colRight = st.columns([1, 8, 1])

	colMid.markdown("<h3 style = 'text-align: center;'>Quizz Bonheur: êtes-vous heureux?</h3>", unsafe_allow_html=True)
	colMid.caption("Répondez à quelques question et découvrez votre score de bonheur.")

	df2021_final = pd.read_csv("datasets/df2021_final.csv")
	cols = ['Logged GDP per capita', 'Press_Freedom', 'Healthy life expectancy', 'Law', 'Social support', 'Freedom to make life choices', 'Schooling', 'Unemployment rate', 'Perceptions of corruption']

	options = []
	options.append('Pays')
	for country in df2021_final["Country name"]:
		options.append(country)

	option = colMid.selectbox(
		label = 'Choisissez votre pays',
		options = options)

	if option == 'Pays':
		colMid.write('Veuillez sélectionner un pays')

	else:
		colInLeft, colInMid, colInRight = st.columns([2, 6, 2])

		ss = colInMid.slider("**Lorsque vous avez des soucis, avez-vous des proches sur qui compter ?**\n\n0 = Pas d'accord, 10 = D'accord",
			0, 10, 5)

		social_support = ss / 10

		
		lc = colInMid.slider("**Êtes-vous satisfait de votre liberté de faire des choix de vie ?**\n\n0 = Pas satisfait, 10 = Satisfait",
			0, 10, 5)

		life_choices = lc / 10

		pc1 = colInMid.slider("**La corruption est-elle répandue au sein du gouvernement ?**\n\n0 = Pas d'accord, 10 = D'accord",
			0, 10, 5)

		pc2 = colInMid.slider("**La corruption est-elle répandue au sein des entreprises ?**\n\n0 = Pas d'accord, 10 = D'accord",
			0, 10, 5)
		
		perception_corruption = (pc1 + pc2) / 20

		selected_country = option

		df_row = df2021_final[df2021_final["Country name"] == selected_country]

		X_train = df2021_final[cols]
		y_train = df2021_final['Ladder score']

		df_row['Social support'] = social_support
		df_row['Freedom to make life choices'] = life_choices
		df_row['Perceptions of corruption'] = perception_corruption

		X_test = df_row[cols]

		# Création d'un objet modèle de régression linéaire
		model = LinearRegression()

		# Entraînement du modèle sur le jeu d'entraînement
		model.fit(X_train, y_train)

		pred_test = model.predict(X_test)


		col1, col2, col3, col4, col5 = st.columns([1,3,3,3,1])

		col2.write("**Score de bonheur personnel :**")
		col2.write(np.round(pred_test.item(), 2))

		col3.write("**Pays sélectionné :**")
		col3.write(df_row["Country name"].item())

		col4.write("**Score général du pays :**")
		col4.write(np.round(df_row["Ladder score"].item(), 2))








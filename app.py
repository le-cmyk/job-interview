# to run the app : streamlit run app.py
# to have the correct version  : pipreqs --encoding=utf8 --force

#All libraries importation
from libraries import *
from functions import *
from data_import import data_load




# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Job interview", page_icon=":bar_chart:", layout="wide")

#region Configuration page 

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)




#endregion


# ---- SIDEBAR ----
st.sidebar.header("Please import your dataset:")

df= data_load()

# ---- MAINPAGE ----

# Utiliser la classe "centered-elements" pour les éléments que vous voulez centrer
st.markdown('<div class="centered-elements">', unsafe_allow_html=True)
st.title(":bar_chart: Job interview")
st.markdown("##")
st.subheader("Let's dive in", divider='rainbow')
st.markdown('</div>', unsafe_allow_html=True)
# Appliquer le style CSS pour centrer ces éléments spécifiques
st.markdown(
    """
    <style>
    .centered-elements {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#region Visualisation data
visu = st.expander("Visualisation data")
visu.dataframe(df)
#endregion


#region question 1

def question1(data):
    # Initialize an empty dictionary to store the results
    result_dict = {}

    # Utilisez df.info() pour obtenir les informations du DataFrame
    buffer = StringIO()
    data.info(buf=buffer)

    # Utilisez la méthode getvalue() pour obtenir les informations sous forme de chaîne de caractères
    result_dict["data_info"] ={"data_info": buffer.getvalue()}

    # Count the number of observations (rows) in the dataset
    num_observations = data.shape[0]
    result_dict["num_observations"] = num_observations

    # Check for missing values in the dataset
    missing_values = data.isnull().sum()
    result_dict["missing_values"] = missing_values

    return result_dict

reponse_question (df, "**1)** Combien y a t-il d'observations dans ce dataset? Y a t-il des valeurs manquantes ?", question1)
#endregion

#region question 2 

statement_question2 = """**2)** Réaliser l'imputation des valeurs manquantes pour la variable "Experience" avec :

a. la valeur médiane pour les data scientists 

b. la valeur moyenne pour les data engineers """


def question2(data):
    results = {}  # Dictionnaire pour stocker les résultats

    # Vérification du nom des différents métiers
    metiers_uniques = data['Metier'].unique()
    results['Metiers_uniques'] = metiers_uniques

    # Vérification des changements
    null_counts_before = data.isnull().sum()
    results['Null_counts_before'] = null_counts_before

    # Calcul de la médiane pour les data scientists
    median_ds_exp = data[data['Metier'] == 'Data scientist']['Experience'].median()
    results['Median_ds_exp'] = median_ds_exp

    # Calcul de la moyenne pour les data engineers
    mean_de_exp = data[data['Metier'] == 'Data engineer']['Experience'].mean()
    results['Mean_de_exp'] = mean_de_exp

    # Imputation pour les data scientists avec la médiane
    data.loc[(data['Metier'] == 'Data scientist') & (data['Experience'].isnull()), 'Experience'] = median_ds_exp

    # Imputation pour les data engineers avec la moyenne
    data.loc[(data['Metier'] == 'Data engineer') & (data['Experience'].isnull()), 'Experience'] = mean_de_exp

    # Vérification des changements après imputation
    null_counts_after = data.isnull().sum()
    results['Null_counts_after'] = null_counts_after

    return results

reponse_question (df, statement_question2, question2)

#endregion 

#region question 3 

def question3(data):
    results = {}  # Dictionnaire pour stocker les résultats

    # Calcul de l'expérience moyenne pour Data Scientist, Lead Data Scientist et Data Engineer
    avg_exp_data_scientist = data[data['Metier'] == 'Data scientist']['Experience'].mean()
    avg_exp_lead_data_scientist = data[data['Metier'] == 'Lead data scientist']['Experience'].mean()
    avg_exp_data_engineer = data[data['Metier'] == 'Data engineer']['Experience'].mean()

    results['Avg_exp_data_scientist'] = avg_exp_data_scientist
    results['Avg_exp_lead_data_scientist'] = avg_exp_lead_data_scientist
    results['Avg_exp_data_engineer'] = avg_exp_data_engineer

    # Calcul de l'expérience moyenne par métier
    avg_experience_by_job = data.groupby('Metier')['Experience'].mean().reset_index()
    results['Avg_experience_by_job'] = avg_experience_by_job

    return results

statement_question3 ="""**3)** Combien d'années d'expériences ont, en moyenne, chacun des profils : le 
data scientist, le lead data scientist et le data engineer en moyenne ?"""

reponse_question (df, statement_question3, question3)


#endregion

#region question 4

statement_question4="""**4)** Faire la représentation graphique de votre choix afin de comparer le 
nombre moyen d'années d'expériences pour chaque métier"""

def question4(data):
    results = {}  # Dictionnaire pour stocker les résultats

    # Calcul de l'expérience moyenne par métier
    avg_experience_by_job = data.groupby('Metier')['Experience'].mean().reset_index()
    results['Avg_experience_by_job'] = avg_experience_by_job

    # Création du graphique
    fig=plt.figure(figsize=(10, 6))
    sns.barplot(x='Metier', y='Experience', data=avg_experience_by_job)
    plt.xticks(rotation=45)
    plt.title('Average Years of Experience by Job Title')
    plt.xlabel('Job Title')
    plt.ylabel('Average Years of Experience')

    # Enregistrement du graphique sous forme d'image dans le dictionnaire
    results['Bar_chart_image'] = fig

    return results


reponse_question (df, statement_question4, question4)


#endregion

#region question 5

def question5(data):
    results = {}  # Dictionnaire pour stocker les résultats

    # Définir une fonction pour catégoriser l'expérience
    def categorize_experience(x):
        if x <= 2:
            return 'débutant'
        elif x <= 5:
            return 'confirmé'
        elif x <= 10:
            return 'avancé'
        else:
            return 'expert'

    # Appliquer la fonction à la colonne 'Experience'
    data['Exp_label'] = data['Experience'].apply(categorize_experience)

    # Ajouter les données catégorisées au dictionnaire des résultats
    results['Categorized_data'] = data[['Experience', 'Exp_label']].head()

    return results

statement_question5 = """**5)** Transformer la variable continue 'Experience' en une nouvelle variable
catégorielle 'Exp_label' à 4 modalités : débutant, confirmé, avancé et expert. 
Veuillez expliquer votre choix de la règle de transformation. """


reponse_question(df , statement_question5,question5)

#endregion 

#region question 6

statement_question6="""**6)** Quelles sont les 5 technologies les plus utilisées ? Faites un graphique"""

def question6(data):
    results = {}  # Dictionnaire pour stocker les résultats

    # Séparation des technologies et comptage des occurrences
    all_technologies = data['Technologies'].str.split('/').explode()
    tech_counts_alt = all_technologies.value_counts().head(5)

    # Conversion en DataFrame pour le graphique
    tech_df_alt = tech_counts_alt.reset_index()
    tech_df_alt.columns = ['Technology', 'Count']

    #with counter 
    # # Séparer les technologies et compter les occurrences
    # tech_list = data['Technologies'].str.cat(sep='/').split('/')
    # tech_counts = Counter(tech_list)

    # # Obtenir les 5 technologies les plus courantes
    # most_common_techs = tech_counts.most_common(5)

    # # Convertir le résultat en DataFrame pour le graphique
    # tech_df = pd.DataFrame(most_common_techs, columns=['Technology', 'Count'])


    # Création du graphique avec seaborn
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x='Technology', y='Count', data=tech_df_alt)
    plt.title('Top 5 Technologies Used (Without Counter)')
    plt.xlabel('Technology')
    plt.ylabel('Frequency')

    results['Top_Technologies_Bar_Chart'] = fig

    return results


reponse_question(df, statement_question6,question6)
#endregion

#region question 7 

statement_question7 = """**7)** Réaliser une méthode de clustering non supervisée de votre choix pour faire 
apparaître le nombre de clusters que vous jugerez pertinents. Donnez les 
caractéristiques de chacun des clusters. 

a. Justifier le nombre de clusters 

b. Justifier la performance de votre algorithme grâce à une métrique. 

c. Interpréter votre résultat."""

def question7(data):
    results = {}  # Dictionnaire pour stocker les résultats

    # Preprocessing the data
    data.dropna(inplace=True)

    # Séparer les colonnes numériques des colonnes catégorielles
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=['number']).columns.tolist()

    # Créer un transformateur de colonnes
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),  # Pour les colonnes numériques, normalisation
            ('cat', Pipeline([
                #('binary_encoding', ce.BinaryEncoder(cols=categorical_cols)),  # Encodage binaire pour les catégories
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]), categorical_cols)
        ])

    # Appliquer le prétraitement sur les données
    df_clustering = preprocessor.fit_transform(data)

    # Determining the optimal number of clusters using the Elbow Method
    inertia = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(df_clustering)
        inertia.append(kmeans.inertia_)

    # Plotting the Elbow Curve
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(1, 10), inertia, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.xticks(range(1, 10))

    # Adding vertical dashed lines for better readability
    for k in range(1, 10):
        plt.axvline(x=k, linestyle='--', color='gray', alpha=0.7)

    # Enregistrez le graphique dans un fichier temporaire pour Streamlit
    results['Elbow_Curve'] = fig

    num_clusters = 3

    # Exécution de K-Means avec le nombre choisi de clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_clustering)

    # Calcul du score de silhouette pour évaluer la performance de l'algorithme
    silhouette_avg = silhouette_score(df_clustering, clusters)
    results['Silhouette_Score'] = silhouette_avg

    # Conversion de la matrice sparse en un tableau numpy dense
    data_dense = df_clustering.toarray()

    # Calcul de l'indice Davies-Bouldin pour évaluer la performance de l'algorithme
    db_score = davies_bouldin_score(data_dense, clusters)
    results['Davies_Bouldin_Index'] = db_score

    # Calcul des statistiques des caractéristiques pour chaque cluster
    cluster_stats = pd.DataFrame(data, columns=data.columns)
    cluster_stats['Cluster'] = clusters

    # Statistiques globales par cluster
    cluster_means = cluster_stats.groupby('Cluster').describe()
    results['Cluster_Statistics'] = cluster_means

    # Statistiques spécifiques aux colonnes catégorielles
    categorical_cols = data.select_dtypes(exclude=['number']).columns.tolist()
    cluster_means_categorical = cluster_stats.groupby('Cluster')[categorical_cols].describe()
    results['Categorical_Cluster_Statistics'] = cluster_means_categorical

    # Statistiques de l'expérience par cluster
    exp_label_counts = cluster_stats.groupby('Cluster').Exp_label.value_counts()
    results['Experience_Label_Counts_By_Cluster'] = exp_label_counts

    return results

commentaire="""
Prétraitement des données : Nous devons transformer les variables catégorielles en numériques et normaliser les données pour une meilleure performance de l'algorithme de clustering.

Choix du nombre de clusters : Nous utiliserons la méthode du coude (Elbow Method) pour déterminer le nombre approprié de clusters.

Exécution de K-Means : Nous appliquerons l'algorithme K-Means avec le nombre de clusters choisi.

Évaluation de la performance : Nous utiliserons le score de la silhouette pour évaluer la performance de l'algorithme.

Interprétation des résultats : Nous analyserons les caractéristiques de chaque cluster.


Score de silhouette

- Un score proche de 1 indique que les points de données sont bien séparés par rapport aux autres clusters, ce qui suggère une bonne qualité de clustering.
- Un score proche de 0 indique une certaine superposition ou ambiguïté dans la séparation des clusters.
- Un score proche de -1 indique que les points de données sont mal regroupés et qu'ils sont plus similaires aux points des autres clusters qu'à leur propre cluster.

Index Davies-Bouldin

- Plus le score est bas, meilleure est la qualité du clustering. Un score de zéro indiquerait un clustering parfait, mais cela est rarement atteint dans la pratique.
- Un score proche de zéro signifie que les clusters sont bien séparés et compacts. C'est un signe de clustering de haute qualité.
- Un score plus élevé indique une plus grande similitude entre les clusters ou une mauvaise séparation, ce qui suggère une moins bonne qualité de clustering. Cela pourrait signifier qu'il y a de l'ambiguïté dans la séparation des clusters.
- Il n'y a pas de limite stricte pour ce score, il doit être interprété en comparant différents résultats de clustering avec d'autres méthodes ou en ajustant le nombre de clusters pour obtenir un score satisfaisant.    



**Remarque:** Les clusters ont été répartis suivant l'expérience ce qui parait cohérent avec notre dataset"""

reponse_question(df, statement_question7,question7,commentaire)

#endregion

#region question 8

statement_question8 = """**8)** Réaliser la prédiction des métiers manquants dans la base de données par 
l'algorithme de votre choix 

a. Justifier la performance de votre algorithme grâce à une métrique. 

b. Interpréter votre résultat"""

from data.result_classifier import *
def question8(data):

    # results = {}  # Dictionnaire pour stocker les résultats

    # # Drop missing values
    # data.dropna(inplace=True)

    # # Separate numerical and categorical columns
    # numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
    # categorical_cols = data.select_dtypes(exclude=['number']).columns.tolist()
    # categorical_cols.remove("Metier")  # Assuming 'Metier' is your target variable
    # categorical_cols.remove("Technologies")

    # df_classifier = data[categorical_cols + numerical_cols].copy()

    # # Normalizing numerical columns
    # for col in numerical_cols:
    #     df_classifier[col] = (df_classifier[col] - df_classifier[col].mean()) / df_classifier[col].std()

    # # One-hot encoding categorical columns
    # df_classifier = pd.get_dummies(df_classifier, columns=categorical_cols, drop_first=True)

    # all_technologies = data['Technologies'].str.get_dummies(sep='/')
    # df_classifier = pd.concat([df_classifier, all_technologies], axis=1)


    # # Split the data into train and test sets
    # X=df_classifier
    # y=data.Metier

    # X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
    # clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    # models,predictions = clf.fit(X_train, X_test, y_train, y_test)

    # results["model"]=models

    # return results

    return {"Lazzy result":result_classifier}

commentaire8 = """
C'est trop long de faire tourner le model en entier mais voila le résultat des différents models"""


reponse_question(df, statement_question8,question8,commentaire8)

#endregion 






# - LIENS

LIEN = {
    "Léo Dujourd'hui": "https://leo-dujourd-hui-digital-cv.streamlit.app",
}
SOURCES ={
    "Github": "https://github.com/le-cmyk/job-interview"
}



# - Téléchargement des données 


c_1, c_2,c_3 = st.columns(3)
with c_1:
    for clé, link in LIEN.items():
        st.write(f"Made by : [{clé}]({link})")
with c_2:
    for clé, link in SOURCES.items():
        st.write(f"[{clé}]({link})")
with c_3:
    st.download_button(
        label="Télécharger le fichier CSV",
        data=df.to_csv(index=False),
        file_name="data.csv",
        key='download_button')




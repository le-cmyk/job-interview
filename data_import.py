#All libraries importation
from libraries import *

def clean_data(df):
    """Fonction pour nettoyer et convertir les données du DataFrame."""
    for col in df.columns:
        # Tenter de convertir en float en remplaçant les virgules par des points
        try:
            df[col] = df[col].str.replace(',', '.').astype(float)
        except:
            pass
    return df

def load_data(uploaded_file):
    """Fonction pour charger les données depuis un fichier CSV."""
    if uploaded_file is not None:
        # Charger le fichier fourni par l'utilisateur
        df = pd.read_csv(uploaded_file)
    else:
        # Charger le fichier par défaut
        df = pd.read_csv('data/data.csv')
    return clean_data(df)

#@st.cache_data
def data_load():

    # Téléchargement de fichier par l'utilisateur
    uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV", type="csv")

    # Charger les données
    df = load_data(uploaded_file)

    return df




import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder

# 🔧 Configuration
#model_path = r"C:\Users\mohamed.mammeri\Documents\Machine learning\dpe_model.joblib"
model_path = r"dpe_model.joblib"
#ordinal_categories_path = r"C:\Users\mohamed.mammeri\Documents\Machine learning\ordinal_categories.csv"
ordinal_categories_path = "ordinal_categories.csv"

# Charger le modèle
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Modèle de prédiction non trouvé à {model_path}.")
    st.stop()

# Charger les catégories ordinales
try:
    ordinal_df = pd.read_csv(ordinal_categories_path, encoding='utf-8', quoting=1)  # QUOTE_ALL
except FileNotFoundError:
    st.error(f"Fichier {ordinal_categories_path} non trouvé.")
    st.stop()

# Variables utiles (sans etiquette_dpe)
variables_utiles = [
    'annee_construction',
    'surface_habitable_logement',
    'classe_altitude',
    'code_departement_ban',
    'classe_inertie_batiment',
    'qualite_isolation_enveloppe',
    'qualite_isolation_murs',
    'qualite_isolation_menuiseries',
    'type_generateur_chauffage_principal',
    'type_generateur_chauffage_principal_ecs',
    'type_ventilation'
]

categorical_columns = [
    'classe_altitude', 'code_departement_ban', 'classe_inertie_batiment',
    'qualite_isolation_enveloppe', 'qualite_isolation_murs', 'qualite_isolation_menuiseries',
    'type_generateur_chauffage_principal', 'type_generateur_chauffage_principal_ecs',
    'type_ventilation'
]

# Créer le dictionnaire des catégories ordinales
ordinal_categories = {}
for col in categorical_columns:
    col_categories = ordinal_df[ordinal_df['variable'] == col][['category', 'ordinal_rank']].sort_values('ordinal_rank')
    if col_categories.empty:
        st.error(f"Aucune catégorie trouvée pour {col} dans {ordinal_categories_path}")
        st.stop()
    ordinal_categories[col] = col_categories['category'].tolist()
    
# Injecter du CSS pour personnaliser le bouton
customized_button = st.markdown("""
    <style >
     div[data-testid="stFormSubmitButton"] {
        display: flex !important;
        justify-content: flex-end !important;
        margin-top: 10px !important;
    }
    div.stButton > button:first-child {
        background-color: #28a745;
        color:#ffffff;
    }
    div.stButton > button:hover {
        background-color: #218838;
        color:#ffffff;
        }
    </style>""", unsafe_allow_html=True)

# Titre de l'application
st.title("Prédiction du DPE d'un appartement")

# Formulaire
st.subheader("Remplissez les informations de l'appartement")
with st.form("dpe_form"):
    # Variables numériques
    annee_construction = st.slider("Année de construction", min_value=1850, max_value=2024, value=1960)
    surface_habitable_logement = st.slider("Surface habitable (m²)", min_value=10.0, max_value=500.0, value=100.0, step=0.1, format="%.1f")

    # Variables catégoriques
    inputs = {}
    for col in categorical_columns:
        label = col.replace('_', ' ').title()
        inputs[col] = st.selectbox(f"{label}", options=ordinal_categories[col], index=0)

    # Conteneur pour le bouton
    with st.container():
        submitted = st.form_submit_button("Prédire le DPE ✅")

if submitted:
    # Créer un DataFrame avec les entrées
    data = {
        'annee_construction': [annee_construction],
        'surface_habitable_logement': [surface_habitable_logement],
        **{col: [inputs[col]] for col in categorical_columns}
    }
    df_input = pd.DataFrame(data, columns=variables_utiles)

    # Encoder les variables catégoriques
    for col in categorical_columns:
        encoder = OrdinalEncoder(categories=[ordinal_categories[col]], handle_unknown='use_encoded_value', unknown_value=-1)
        df_input[col] = encoder.fit_transform(df_input[[col]])

    # Vérifier les valeurs -1
    if (df_input[categorical_columns] < 0).any().any():
        st.error("Une ou plusieurs entrées catégoriques sont invalides.")
        st.stop()

    # Prédiction
    proba = model.predict_proba(df_input)
    # Les classes sont dans l'ordre de model.classes_ ([0, 1, 2, 3, 4, 5, 6] → G, F, E, D, C, B, A)
    class_labels = ['G', 'F', 'E', 'D', 'C', 'B', 'A']

    # Extraire les 3 classes majoritaires
    top3_indices = np.argsort(-proba, axis=1)[0, :2]
    top3_probs = proba[0, top3_indices]
    top3_classes = [class_labels[idx] for idx in top3_indices]

    # Afficher les résultats
    st.subheader("Résultat de la prédiction")
    st.write("Les 2 classes DPE les plus probables :")
    result_df = pd.DataFrame({
        'Classe DPE probable': top3_classes,
        'Probabilité ': [str(round(prob * 100))+' %' for prob in top3_probs]
    })
    st.dataframe(result_df.set_index(result_df.columns[0]))
    #st.table(result_df)

    # # Vérifier que top1_class correspond à la prédiction
    # pred = model.predict(df_input)[0]
    # pred_class = class_labels[int(pred)]
    # if pred_class != top3_classes[0]:
        # st.error(f"Incohérence : Prédiction ({pred_class}) ne correspond pas à top1_class ({top3_classes[0]}).")

    # Afficher les données entrées pour vérification
    st.subheader("Données entrées")
    st.write(df_input)
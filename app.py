# app.py - Application Complète TB Diagnostic Pro
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import hashlib
import datetime
from datetime import date
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import sqlite3
import os
warnings.filterwarnings('ignore')

# =============================================================================
# 🎨 CONFIGURATION AVANCÉE DU DESIGN
# =============================================================================
st.set_page_config(
    page_title="TB Diagnostic Pro", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🩺"
)

# =============================================================================
# 🎨 STYLE CSS PERSONNALISÉ
# =============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
    /* Style général */
    .main {
        background-color: #f8f9fa;
    }
    
    /* En-têtes avec dégradé */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Cartes avec ombres */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    /* Boutons stylisés */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar stylisée */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    
    /* Métriques améliorées */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Onglets personnalisés */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f3f4;
        border-radius: 5px 5px 0px 0px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Inputs stylisés */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        border: 2px solid #e0e0e0;
        border-radius: 5px;
        padding: 0.5rem;
    }
    
    /* Barre de progression */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Alertes personnalisées */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
    }
    
    /* Séparateurs */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        margin: 2rem 0;
        border-radius: 2px;
    }
    
    /* Badges */
    .risk-badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }
    
    .risk-low { background-color: #d4edda; color: #155724; }
    .risk-medium { background-color: #fff3cd; color: #856404; }
    .risk-high { background-color: #f8d7da; color: #721c24; }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# 🔹 SYSTÈME D'AUTHENTIFICATION AMÉLIORÉ
# =============================================================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_auth():
    if 'users' not in st.session_state:
        st.session_state.users = {
            "admin": {"password": hash_password("admin123"), "role": "admin", "name": "Administrateur"},
            "medecin": {"password": hash_password("medecin123"), "role": "medecin", "name": "Dr Dupont"}
        }
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None

def login_register_page():
    # Application du CSS
    inject_custom_css()
    
    # Container principal centré
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Carte de connexion
        st.markdown("""
        <div style='background: white; padding: 3rem; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);'>
            <div class='main-header' style='text-align: center; margin: -3rem -3rem 2rem -3rem; border-radius: 10px 10px 0 0;'>
                <h1 style='color: white; margin: 0;'>🫁 TB Diagnostic Pro</h1>
                <p style='color: white; opacity: 0.9; margin: 0.5rem 0 0 0;'>Système Expert de Diagnostic de la Tuberculose</p>
            </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🚀 **Connexion**", "👤 **Créer un Compte**"])
        
        with tab1:
            with st.form("login_form"):
                st.subheader("Connectez-vous à votre compte")
                
                username = st.text_input(
                    "**Nom d'utilisateur**",
                    placeholder="Entrez votre nom d'utilisateur...",
                    key="login_username"
                )
                
                password = st.text_input(
                    "**Mot de passe**", 
                    type="password",
                    placeholder="Entrez votre mot de passe...",
                    key="login_password"
                )
                
                col_btn1, col_btn2 = st.columns([2, 1])
                with col_btn1:
                    login_btn = st.form_submit_button(
                        "**Se connecter** 🚀", 
                        use_container_width=True,
                        type="primary"
                    )
                
                if login_btn:
                    if username in st.session_state.users and st.session_state.users[username]["password"] == hash_password(password):
                        st.session_state.logged_in = True
                        st.session_state.current_user = username
                        st.success(f"✅ Connexion réussie! Bienvenue {st.session_state.users[username]['name']}")
                        st.rerun()
                    else:
                        st.error("❌ Identifiants incorrects")
        
        with tab2:
            with st.form("register_form"):
                st.subheader("Créez votre compte")
                
                new_username = st.text_input(
                    "**Nom d'utilisateur**",
                    placeholder="Choisissez un nom d'utilisateur...",
                    key="reg_username"
                )
                
                col_pass1, col_pass2 = st.columns(2)
                with col_pass1:
                    new_password = st.text_input(
                        "**Mot de passe**", 
                        type="password",
                        placeholder="Créez un mot de passe...",
                        key="reg_password"
                    )
                with col_pass2:
                    confirm_password = st.text_input(
                        "**Confirmer le mot de passe**", 
                        type="password",
                        placeholder="Confirmez le mot de passe...",
                        key="reg_confirm"
                    )
                
                full_name = st.text_input(
                    "**Nom complet**",
                    placeholder="Entrez votre nom complet...",
                    key="reg_fullname"
                )
                
                role = st.selectbox(
                    "**Rôle**", 
                    ["medecin", "infirmier"],
                    key="reg_role"
                )
                
                register_btn = st.form_submit_button(
                    "**Créer le compte** 👤", 
                    use_container_width=True,
                    type="primary"
                )
                
                if register_btn:
                    if new_username in st.session_state.users:
                        st.error("❌ Ce nom d'utilisateur existe déjà")
                    elif new_password != confirm_password:
                        st.error("❌ Les mots de passe ne correspondent pas")
                    elif len(new_password) < 4:
                        st.error("❌ Le mot de passe doit avoir au moins 4 caractères")
                    else:
                        st.session_state.users[new_username] = {
                            "password": hash_password(new_password),
                            "role": role,
                            "name": full_name
                        }
                        st.success("✅ Compte créé avec succès! Vous pouvez maintenant vous connecter.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 🔹 BASE DE DONNÉES - OPTIONS MULTIPLES
# =============================================================================
@st.cache_resource
def get_db_connection():
    """
    Essaye plusieurs méthodes de connexion à la base de données
    Priorité: MySQL -> SQLite -> Données simulées
    """
    
    # Option 1: MySQL (original)
    try:
        engine = create_engine("mysql+mysqlconnector://root:123456789@localhost/tb_database")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        st.sidebar.success("✅ Connecté à MySQL")
        return engine
    except Exception as e:
        st.sidebar.warning(f"❌ MySQL non disponible: {e}")
    
    # Option 2: SQLite (fallback)
    try:
        # Créer un dossier data s'il n'existe pas
        if not os.path.exists('data'):
            os.makedirs('data')
        
        sqlite_path = "data/tb_database.db"
        engine = create_engine(f"sqlite:///{sqlite_path}")
        
        # Créer la table si elle n'existe pas
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cin VARCHAR(20),
                    nom VARCHAR(100),
                    prenom VARCHAR(100),
                    age INT,
                    genre VARCHAR(10),
                    poids FLOAT,
                    taille FLOAT,
                    imc FLOAT,
                    douleur_thoracique VARCHAR(20),
                    intensite_toux INT,
                    essoufflement INT,
                    production_crachats VARCHAR(20),
                    sang_crachats VARCHAR(20),
                    fievre VARCHAR(20),
                    fatigue INT,
                    sueurs_nocturnes VARCHAR(20),
                    perte_poids FLOAT,
                    tabagisme VARCHAR(30),
                    antecedents_tb VARCHAR(30),
                    prediction INT,
                    probabilite FLOAT,
                    niveau_risque VARCHAR(20),
                    medecin_traitant VARCHAR(100),
                    date_consultation DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
        
        st.sidebar.success("✅ Connecté à SQLite")
        return engine
        
    except Exception as e:
        st.sidebar.error(f"❌ SQLite échoué: {e}")
        return None

def create_sample_data():
    """Crée des données d'exemple réalistes pour la démonstration"""
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'cin': ['AB123456', 'CD789012', 'EF345678', 'GH901234', 'IJ567890',
                'KL123456', 'MN789012', 'OP345678', 'QR901234', 'ST567890'],
        'nom': ['DUPONT', 'MARTIN', 'BERNARD', 'PETIT', 'ROBERT', 
                'RICHARD', 'DURAND', 'DUBOIS', 'MOREAU', 'LAURENT'],
        'prenom': ['Jean', 'Marie', 'Pierre', 'Sophie', 'Paul', 
                  'Nathalie', 'Michel', 'Catherine', 'Philippe', 'Isabelle'],
        'age': [35, 42, 28, 55, 67, 31, 45, 38, 52, 29],
        'genre': ['Homme', 'Femme', 'Homme', 'Femme', 'Homme', 
                 'Femme', 'Homme', 'Femme', 'Homme', 'Femme'],
        'poids': [70.0, 65.0, 80.0, 60.0, 75.0, 58.0, 85.0, 62.0, 78.0, 55.0],
        'taille': [170.0, 165.0, 175.0, 160.0, 172.0, 163.0, 178.0, 166.0, 174.0, 162.0],
        'imc': [24.2, 23.9, 26.1, 23.4, 25.4, 21.8, 26.8, 22.5, 25.8, 21.0],
        'douleur_thoracique': ['Légère', 'Aucune', 'Modérée', 'Sévère', 'Modérée', 
                              'Légère', 'Aucune', 'Modérée', 'Sévère', 'Légère'],
        'intensite_toux': [3, 7, 5, 9, 6, 2, 8, 4, 7, 3],
        'essoufflement': [2, 6, 4, 8, 5, 1, 7, 3, 6, 2],
        'production_crachats': ['Faible', 'Moyenne', 'Faible', 'Importante', 'Moyenne', 
                               'Aucune', 'Importante', 'Faible', 'Moyenne', 'Aucune'],
        'sang_crachats': ['Non', 'Oui', 'Non', 'Abondant', 'Oui', 
                         'Non', 'Abondant', 'Non', 'Oui', 'Non'],
        'fievre': ['<38°C', '38-39°C', 'Absente', '>39°C', '38-39°C', 
                  'Absente', '>39°C', '<38°C', '38-39°C', 'Absente'],
        'fatigue': [4, 8, 3, 9, 7, 2, 8, 5, 6, 3],
        'sueurs_nocturnes': ['Occasionnelles', 'Fréquentes', 'Non', 'Très fréquentes', 'Fréquentes', 
                            'Non', 'Très fréquentes', 'Occasionnelles', 'Fréquentes', 'Non'],
        'perte_poids': [1.5, 3.2, 0.5, 4.5, 2.8, 0.2, 5.1, 1.2, 3.5, 0.8],
        'tabagisme': ['<10/jour', 'Ancien fumeur', 'Jamais fumé', '>10/jour', 'Ancien fumeur',
                     'Jamais fumé', '>10/jour', '<10/jour', 'Ancien fumeur', 'Jamais fumé'],
        'antecedents_tb': ['Non', 'Oui, traité', 'Non', 'Oui, récurrent', 'Oui, traité',
                          'Non', 'Oui, récurrent', 'Non', 'Oui, traité', 'Non'],
        'prediction': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        'probabilite': [0.25, 0.78, 0.32, 0.85, 0.72, 0.18, 0.91, 0.28, 0.68, 0.22],
        'niveau_risque': ['Faible', 'Élevé', 'Faible', 'Élevé', 'Élevé', 
                         'Faible', 'Élevé', 'Faible', 'Modéré', 'Faible'],
        'medecin_traitant': ['Dr Dupont', 'Dr Martin', 'Dr Bernard', 'Dr Petit', 'Dr Robert',
                            'Dr Richard', 'Dr Durand', 'Dr Dubois', 'Dr Moreau', 'Dr Laurent'],
        'date_consultation': [date.today()] * 10
    }
    return pd.DataFrame(sample_data)

# =============================================================================
# 🔹 MODÈLE ML ET PRÉDICTION
# =============================================================================
def predict_tuberculosis(patient_data):
    """Simulation de prédiction ML"""
    try:
        risk_score = 0
        
        # Symptômes respiratoires
        if patient_data['intensite_toux'] > 5:
            risk_score += 0.3
        if patient_data['sang_crachats'] > 0:
            risk_score += 0.4
        if patient_data['douleur_thoracique'] > 1:
            risk_score += 0.2
            
        # Symptômes généraux
        if patient_data['fievre'] > 1:
            risk_score += 0.2
        if patient_data['sueurs_nocturnes'] > 1:
            risk_score += 0.1
        if patient_data['perte_poids'] > 2:
            risk_score += 0.2
            
        # Facteurs de risque
        if patient_data['antecedents_tb'] > 0:
            risk_score += 0.3
        if patient_data['tabagisme'] > 1:
            risk_score += 0.1
            
        # Ajustement par âge
        if patient_data['age'] < 10 or patient_data['age'] > 60:
            risk_score += 0.1
            
        probability = min(0.95, risk_score)
        prediction = 1 if probability > 0.5 else 0
        
        return prediction, probability
        
    except Exception as e:
        st.error(f"Erreur prédiction: {e}")
        return 0, 0.0

def calculate_risk_level(probability):
    if probability < 0.3:
        return "Faible", "green", "🟢"
    elif probability < 0.7:
        return "Modéré", "orange", "🟡"
    else:
        return "Élevé", "red", "🔴"

# =============================================================================
# 🔹 FONCTIONS DE GESTION DE LA BASE DE DONNÉES
# =============================================================================
def save_patient_data(engine, patient_data):
    """Sauvegarde les données du patient dans la base"""
    try:
        if engine:
            save_df = pd.DataFrame([patient_data])
            save_df.to_sql("patients", con=engine, if_exists="append", index=False)
            return True
        else:
            # Si pas de base de données, sauvegarde en session
            if 'patients' not in st.session_state:
                st.session_state.patients = []
            st.session_state.patients.append(patient_data)
            return True
    except Exception as e:
        st.error(f"❌ Erreur sauvegarde: {e}")
        return False

def load_patient_data(engine):
    """Charge les données des patients"""
    try:
        if engine:
            # Vérifier si la table existe
            try:
                df = pd.read_sql("SELECT * FROM patients", con=engine)
                if not df.empty:
                    return df
                else:
                    # Table vide, créer des données d'exemple
                    sample_df = create_sample_data()
                    sample_df.to_sql("patients", con=engine, if_exists="replace", index=False)
                    return sample_df
            except Exception as e:
                # Table n'existe pas, créer avec données d'exemple
                st.sidebar.warning("Table patients non trouvée, création...")
                sample_df = create_sample_data()
                sample_df.to_sql("patients", con=engine, if_exists="replace", index=False)
                return sample_df
        
        # Si pas de base de données, utiliser les données de session
        if 'patients' in st.session_state and st.session_state.patients:
            return pd.DataFrame(st.session_state.patients)
        else:
            # Créer et sauvegarder des données d'exemple
            sample_df = create_sample_data()
            if 'patients' not in st.session_state:
                st.session_state.patients = []
            st.session_state.patients.extend(sample_df.to_dict('records'))
            return sample_df
            
    except Exception as e:
        st.error(f"❌ Erreur chargement données: {e}")
        # Retourner des données d'exemple en cas d'erreur
        return create_sample_data()

# =============================================================================
# 🔹 PAGES DE L'APPLICATION AMÉLIORÉES
# =============================================================================
def diagnostic_page(engine):
    inject_custom_css()
    
    # En-tête amélioré
    st.markdown("""
    <div class='main-header'>
        <h1 style='color: white; margin: 0;'>🩺 Diagnostic de la Tuberculose</h1>
        <p style='color: white; opacity: 0.9; margin: 0;'>Évaluation complète des patients et analyse des risques</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Formulaire en deux colonnes avec cartes
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Informations patient dans une carte
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.header("👤 Informations du Patient")
        
        subcol1, subcol2, subcol3 = st.columns(3)
        with subcol1:
            cin = st.text_input("**CIN**", placeholder="AB123456", key="cin")
            nom = st.text_input("**Nom**", placeholder="DUPONT", key="nom")
        with subcol2:
            prenom = st.text_input("**Prénom**", placeholder="Jean", key="prenom")
            age = st.slider("**Âge**", 1, 100, 35, key="age")
        with subcol3:
            genre = st.selectbox("**Genre**", ["Homme", "Femme"], key="genre")
            medecin = st.text_input("**Médecin traitant**", 
                                  value=st.session_state.users[st.session_state.current_user]['name'],
                                  key="medecin")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Données anthropométriques dans une carte
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("📊 Données Anthropométriques")
        anthro_col1, anthro_col2, anthro_col3 = st.columns(3)
        with anthro_col1:
            poids = st.number_input("**Poids (kg)**", 30.0, 200.0, 70.0, step=0.5, key="poids")
        with anthro_col2:
            taille = st.number_input("**Taille (cm)**", 100.0, 220.0, 170.0, step=1.0, key="taille")
        with anthro_col3:
            if taille > 0:
                imc = poids / ((taille/100) ** 2)
                st.metric("**IMC**", f"{imc:.1f}", 
                         delta="Normal" if 18.5 <= imc <= 24.9 else "Attention")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Symptômes cliniques dans des cartes
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.header("🩺 Symptômes Cliniques")
        
        st.subheader("🫁 Symptômes Respiratoires")
        resp_col1, resp_col2, resp_col3 = st.columns(3)
        with resp_col1:
            intensite_toux = st.slider("**Intensité de la toux**", 0, 10, 0, key="toux")
            essoufflement = st.slider("**Essoufflement**", 0, 10, 0, key="essoufflement")
        with resp_col2:
            douleur_thoracique = st.selectbox("**Douleur thoracique**", 
                                            ["Aucune", "Légère", "Modérée", "Sévère"], key="douleur")
            production_crachats = st.selectbox("**Production de crachats**", 
                                             ["Aucune", "Faible", "Moyenne", "Importante"], key="crachats")
        with resp_col3:
            sang_crachats = st.selectbox("**Sang dans les crachats**", 
                                       ["Non", "Oui", "Abondant"], key="sang")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("🌡️ Symptômes Généraux")
        gen_col1, gen_col2, gen_col3 = st.columns(3)
        with gen_col1:
            fievre = st.selectbox("**Fièvre**", 
                                ["Absente", "<38°C", "38-39°C", ">39°C"], key="fievre")
            fatigue = st.slider("**Fatigue**", 0, 10, 0, key="fatigue")
        with gen_col2:
            sueurs_nocturnes = st.selectbox("**Sueurs nocturnes**", 
                                          ["Non", "Occasionnelles", "Fréquentes", "Très fréquentes"], key="sueurs")
            perte_poids = st.slider("**Perte de poids (kg)**", 0.0, 20.0, 0.0, step=0.5, key="perte_poids")
        with gen_col3:
            tabagisme = st.selectbox("**Tabagisme**", 
                                   ["Jamais fumé", "Ancien fumeur", "<10/jour", ">10/jour"], key="tabagisme")
            antecedents_tb = st.selectbox("**Antécédents TB**", 
                                        ["Non", "Oui, traité", "Oui, récurrent"], key="antecedents")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Panneau de contrôle latéral
        st.markdown("<div class='custom-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>", 
                   unsafe_allow_html=True)
        st.subheader("🎯 Contrôles")
        st.markdown("""
        <div style='color: white;'>
        <p>Remplissez tous les champs obligatoires et cliquez sur le bouton pour lancer le diagnostic.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Indicateur de complétion
        champs_obligatoires = [cin, nom, prenom]
        completion = sum(1 for champ in champs_obligatoires if champ) / len(champs_obligatoires) * 100
        st.metric("**Complétion du formulaire**", f"{completion:.0f}%")
        st.progress(int(completion))
        
        # Bouton de diagnostic stylisé
        if st.button("**🎯 Lancer le Diagnostic**", use_container_width=True, type="primary"):
            if completion < 100:
                st.error("Veuillez remplir tous les champs obligatoires (*)")
            else:
                # Mapping des valeurs
                mapping = {
                    'Genre': {'Homme': 1, 'Femme': 0},
                    'Douleur_Thoracique': {'Aucune': 0, 'Légère': 1, 'Modérée': 2, 'Sévère': 3},
                    'Fievre': {'Absente': 0, '<38°C': 1, '38-39°C': 2, '>39°C': 3},
                    'Sueurs_Nocturnes': {'Non': 0, 'Occasionnelles': 1, 'Fréquentes': 2, 'Très fréquentes': 3},
                    'Production_Crachats': {'Aucune': 0, 'Faible': 1, 'Moyenne': 2, 'Importante': 3},
                    'Sang_Crachats': {'Non': 0, 'Oui': 1, 'Abondant': 2},
                    'Tabagisme': {'Jamais fumé': 0, 'Ancien fumeur': 1, '<10/jour': 2, '>10/jour': 3},
                    'Antecedents_TB': {'Non': 0, 'Oui, traité': 1, 'Oui, récurrent': 2}
                }
                
                # Préparation des données
                input_data = {
                    'age': age,
                    'genre': mapping['Genre'][genre],
                    'douleur_thoracique': mapping['Douleur_Thoracique'][douleur_thoracique],
                    'intensite_toux': intensite_toux,
                    'essoufflement': essoufflement,
                    'fatigue': fatigue,
                    'perte_poids': perte_poids,
                    'fievre': mapping['Fievre'][fievre],
                    'sueurs_nocturnes': mapping['Sueurs_Nocturnes'][sueurs_nocturnes],
                    'production_crachats': mapping['Production_Crachats'][production_crachats],
                    'sang_crachats': mapping['Sang_Crachats'][sang_crachats],
                    'tabagisme': mapping['Tabagisme'][tabagisme],
                    'antecedents_tb': mapping['Antecedents_TB'][antecedents_tb]
                }
                
                # Prédiction
                with st.spinner("🔍 Analyse des symptômes en cours..."):
                    prediction, probability = predict_tuberculosis(input_data)
                    niveau_risque, couleur_risque, emoji_risque = calculate_risk_level(probability)
                
                # Affichage résultats dans la sidebar
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.subheader("📊 Résultats")
                
                # Badge de risque
                risk_class = f"risk-{niveau_risque.lower()}"
                st.markdown(f"""
                <div class='risk-badge {risk_class}'>
                    {emoji_risque} Niveau de Risque: {niveau_risque}
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("**Probabilité**", f"{probability:.1%}")
                st.metric("**Recommandation**", 
                         "🔴 Consultation urgente" if prediction == 1 else "🟢 Surveillance")
                
                # Jauge de risque
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={'text': f"Niveau de Risque - {niveau_risque}"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': couleur_risque},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}],
                    }
                ))
                fig_gauge.update_layout(height=250)
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Graphique radar
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.subheader("📈 Profil des Symptômes")
                symptoms_data = {
                    'Symptôme': ['Toux', 'Essoufflement', 'Fatigue', 'Douleur thoracique', 'Fièvre', 'Perte poids'],
                    'Intensité': [
                        intensite_toux, essoufflement, fatigue,
                        mapping['Douleur_Thoracique'][douleur_thoracique],
                        mapping['Fievre'][fievre], perte_poids * 2
                    ]
                }
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=symptoms_data['Intensité'],
                    theta=symptoms_data['Symptôme'],
                    fill='toself',
                    name='Symptômes',
                    fillcolor='rgba(102, 126, 234, 0.3)',
                    line=dict(color='rgb(102, 126, 234)')
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 10])), 
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Recommandations
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.subheader("💡 Plan de Prise en Charge")
                if prediction == 1:
                    st.error("""
                    **🔴 Actions Immédiates:**
                    - Consultation médicale urgente
                    - Radiographie thoracique
                    - Examen des crachats
                    - Isolement préventif
                    - Déclaration aux autorités
                    """)
                else:
                    st.success("""
                    **🟢 Surveillance Standard:**
                    - Contrôle dans 1 mois
                    - Surveillance des symptômes
                    - Mesures d'hygiène
                    - Consultation si aggravation
                    """)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Sauvegarde des données
                patient_data = {
                    "cin": cin, "nom": nom, "prenom": prenom, "age": age,
                    "genre": genre, "poids": poids, "taille": taille, "imc": imc,
                    "douleur_thoracique": douleur_thoracique, "intensite_toux": intensite_toux,
                    "essoufflement": essoufflement, "production_crachats": production_crachats,
                    "sang_crachats": sang_crachats, "fievre": fievre, "fatigue": fatigue,
                    "sueurs_nocturnes": sueurs_nocturnes, "perte_poids": perte_poids,
                    "tabagisme": tabagisme, "antecedents_tb": antecedents_tb,
                    "prediction": prediction, "probabilite": probability,
                    "niveau_risque": niveau_risque, "medecin_traitant": medecin,
                    "date_consultation": date.today()
                }
                
                if save_patient_data(engine, patient_data):
                    st.success("✅ Données sauvegardées avec succès!")
                else:
                    st.warning("⚠️ Données sauvegardées en session (base de données non disponible)")

def dashboard_page(engine):
    inject_custom_css()
    
    # En-tête amélioré
    st.markdown("""
    <div class='main-header'>
        <h1 style='color: white; margin: 0;'>📊 Tableau de Bord Analytique</h1>
        <p style='color: white; opacity: 0.9; margin: 0;'>Surveillance et analyse des données patients</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Charger les données
        df = load_patient_data(engine)
        
        if df.empty:
            st.info("📝 Aucune donnée patient disponible")
            return
        
        # Debug: Afficher les colonnes disponibles
        st.sidebar.write("🔍 Colonnes disponibles:", df.columns.tolist())
        
        # KPI dans des cartes - CORRIGÉ
        st.subheader("📈 Indicateurs Clés de Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            total_patients = len(df)
            st.metric("**Total Patients**", total_patients, "Patients")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            # Compter les cas à risque (prediction = 1)
            if 'prediction' in df.columns:
                cas_risque = df['prediction'].sum()
            else:
                # Si pas de colonne prediction, estimer basé sur niveau_risque
                if 'niveau_risque' in df.columns:
                    cas_risque = len(df[df['niveau_risque'].isin(['Élevé', 'Modéré'])])
                else:
                    cas_risque = 0
            st.metric("**Cas à Risque**", cas_risque, f"{cas_risque} cas")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            # Calculer le taux de risque
            taux_risque = (cas_risque / total_patients * 100) if total_patients > 0 else 0
            st.metric("**Taux de Risque**", f"{taux_risque:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            # Calculer l'âge moyen
            if 'age' in df.columns:
                age_moyen = df['age'].mean()
            else:
                age_moyen = 0
            st.metric("**Âge Moyen**", f"{age_moyen:.1f} ans")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Graphiques
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("📊 Visualisations des Données")
        col1, col2 = st.columns(2)
        
        with col1:
            # Répartition par genre
            if 'genre' in df.columns:
                fig_genre = px.pie(df, names='genre', title="🔄 Répartition par Genre",
                                 color_discrete_sequence=px.colors.sequential.Blues_r)
                st.plotly_chart(fig_genre, use_container_width=True)
            else:
                st.info("📊 Données de genre non disponibles")
            
            # Distribution par âge
            if 'age' in df.columns:
                fig_age = px.histogram(df, x='age', title="📅 Distribution par Âge", nbins=20,
                                     color_discrete_sequence=['#667eea'])
                st.plotly_chart(fig_age, use_container_width=True)
            else:
                st.info("📊 Données d'âge non disponibles")
        
        with col2:
            # Répartition du risque
            if 'niveau_risque' in df.columns:
                risque_counts = df['niveau_risque'].value_counts()
                fig_risque = px.bar(x=risque_counts.index, 
                                  y=risque_counts.values,
                                  title="⚠️ Répartition du Niveau de Risque",
                                  labels={'x': 'Niveau de Risque', 'y': 'Nombre de Patients'},
                                  color=risque_counts.index,
                                  color_discrete_map={'Faible': 'green', 'Modéré': 'orange', 'Élevé': 'red'})
                st.plotly_chart(fig_risque, use_container_width=True)
            else:
                st.info("📊 Données de risque non disponibles")
            
            # Évolution temporelle
            if 'date_consultation' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date_consultation']).dt.date
                    daily_cases = df.groupby('date').size().reset_index(name='count')
                    if len(daily_cases) > 1:
                        fig_trend = px.line(daily_cases, x='date', y='count', 
                                          title="📈 Évolution des Consultations",
                                          color_discrete_sequence=['#764ba2'])
                        st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.info("📈 Données temporelles insuffisantes")
                except Exception as e:
                    st.info(f"📈 Erreur traitement dates: {e}")
            else:
                st.info("📈 Données temporelles non disponibles")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Données brutes
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("📋 Données des Patients")
        display_columns = ['cin', 'nom', 'prenom', 'age', 'genre', 'niveau_risque']
        available_columns = [col for col in display_columns if col in df.columns]
        
        # Ajouter une recherche
        search_term = st.text_input("🔍 Rechercher un patient...")
        if search_term:
            filtered_df = df[df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]
        else:
            filtered_df = df
        
        if not available_columns:
            st.warning("Aucune colonne de données disponible")
        else:
            st.dataframe(filtered_df[available_columns].head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Erreur chargement données: {e}")

def advanced_analysis_page(engine):
    inject_custom_css()
    
    # En-tête amélioré
    st.markdown("""
    <div class='main-header'>
        <h1 style='color: white; margin: 0;'>🔬 Analyse Avancée des Données</h1>
        <p style='color: white; opacity: 0.9; margin: 0;'>Machine Learning et Analytics avancés</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Charger les données
        df = load_patient_data(engine)
        
        if df.empty:
            st.info("📝 Aucune donnée patient disponible")
            return
        
        # Afficher les informations de base dans une carte
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.info(f"📊 Dataset chargé: {df.shape[0]} patients, {df.shape[1]} variables")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Initialiser l'analyseur
        analyzer = AdvancedDataAnalyzer(df=df)
        
        # Onglets d'analyse stylisés
        tab1, tab2, tab3 = st.tabs(["📈 **Analyse Exploratoire**", "🎯 **Clustering**", "📊 **Analyse des Features**"])
        
        with tab1:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("Analyse Exploratoire des Données")
            if st.button("🚀 Lancer l'Analyse Exploratoire", use_container_width=True, key="eda"):
                with st.spinner("Analyse en cours..."):
                    results = analyzer.comprehensive_eda()
                st.success("✅ Analyse exploratoire terminée!")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("Analyse de Clustering")
            n_clusters = st.slider("**Nombre de clusters**", 2, 6, 3, key="clusters")
            
            if st.button("🎯 Effectuer le Clustering", use_container_width=True, key="cluster_btn"):
                with st.spinner("Clustering en cours..."):
                    analyzer.preprocess_data(target_column='prediction')
                    clusters = analyzer.perform_clustering(n_clusters=n_clusters)
                    
                if clusters is not None:
                    st.success(f"✅ Clustering terminé avec {n_clusters} clusters")
                else:
                    st.error("❌ Échec du clustering")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("Analyse des Caractéristiques")
            st.info("Cette analyse identifie les variables les plus importantes pour prédire le risque TB")
            
            if st.button("📊 Analyser l'Importance des Features", use_container_width=True, key="features"):
                with st.spinner("Analyse des caractéristiques..."):
                    analyzer.preprocess_data(target_column='prediction')
                    feature_importance = analyzer.advanced_feature_analysis()
                
                if feature_importance is not None:
                    st.subheader("Top 10 des Caractéristiques Importantes")
                    st.dataframe(feature_importance.head(10), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Erreur d'analyse: {e}")

# =============================================================================
# 🔹 APPLICATION PRINCIPALE AMÉLIORÉE
# =============================================================================
def main():
    # Initialisation
    init_auth()
    
    # Page de connexion si non connecté
    if not st.session_state.logged_in:
        login_register_page()
        return
    
    # Application du CSS pour les pages principales
    inject_custom_css()
    
    # Connexion base de données
    engine = get_db_connection()
    
    # Sidebar navigation améliorée
    with st.sidebar:
        # En-tête sidebar
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 2rem;'>
            <h3 style='color: white; margin: 0;'>🫁 TB Diagnostic Pro</h3>
            <p style='color: white; opacity: 0.9; margin: 0; font-size: 0.9rem;'>Système Expert</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Informations utilisateur
        st.markdown(f"""
        <div class='custom-card' style='margin-bottom: 1rem;'>
            <h4 style='margin: 0;'>👋 Bonjour, {st.session_state.users[st.session_state.current_user]['name']}</h4>
            <p style='margin: 0; color: #666;'>Rôle: <strong>{st.session_state.users[st.session_state.current_user]['role']}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        
        # Navigation selon le rôle
        if st.session_state.users[st.session_state.current_user]['role'] in ['admin', 'medecin']:
            pages = ["🩺 Diagnostic", "📊 Dashboard", "🔬 Analyse Avancée", "🚪 Déconnexion"]
        else:
            pages = ["🩺 Diagnostic", "📊 Dashboard", "🚪 Déconnexion"]
            
        st.subheader("📋 Navigation")
        for page in pages:
            if st.button(page, use_container_width=True, key=f"nav_{page}"):
                st.session_state.current_page = page
                st.rerun()
        
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        
        # Statut base de données
        if engine:
            st.success("✅ **Base de données connectée**")
            try:
                df = load_patient_data(engine)
                st.info(f"📁 **{len(df)}** patients enregistrés")
            except:
                st.info("📁 **Données de démonstration**")
        else:
            st.warning("⚠️ **Mode démonstration**")
            st.info("📁 **Données en session**")
        
        # Pied de page
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
            <p>TB Diagnostic Pro v2.0</p>
            <p>© 2024 - Système Médical Expert</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gestion des pages
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🩺 Diagnostic"
    
    if st.session_state.current_page == "🩺 Diagnostic":
        diagnostic_page(engine)
    elif st.session_state.current_page == "📊 Dashboard":
        dashboard_page(engine)
    elif st.session_state.current_page == "🔬 Analyse Avancée":
        advanced_analysis_page(engine)
    elif st.session_state.current_page == "🚪 Déconnexion":
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.current_page = "🩺 Diagnostic"
        st.success("✅ Déconnexion réussie!")
        st.rerun()

# =============================================================================
# 🔹 CLASSE ANALYSE AVANCÉE
# =============================================================================
class AdvancedDataAnalyzer:
    def __init__(self, data_path=None, df=None):
        if df is not None:
            self.df = self._clean_dataframe(df)
        elif data_path:
            self.df = self._clean_dataframe(pd.read_csv(data_path))
        else:
            raise ValueError("Fournir soit un DataFrame soit un chemin de fichier")
        
        self.original_df = self.df.copy()
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.kmeans = None
        self.analysis_results = {}
    
    def _clean_dataframe(self, df):
        """Nettoie le DataFrame et convertit les types de données"""
        df_clean = df.copy()
        
        # Supprimer les colonnes non numériques problématiques pour l'analyse
        columns_to_drop = ['cin', 'nom', 'prenom', 'medecin_traitant', 'date_consultation', 'created_at']
        for col in columns_to_drop:
            if col in df_clean.columns:
                df_clean = df_clean.drop(columns=[col])
        
        # Convertir les colonnes catégorielles en numériques
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df_clean[col].nunique() <= 10:
                dummies = pd.get_dummies(df_clean[col], prefix=col)
                df_clean = pd.concat([df_clean, dummies], axis=1)
                df_clean = df_clean.drop(columns=[col])
            else:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        
        for col in df_clean.columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except:
                    df_clean = df_clean.drop(columns=[col])
        
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
        
        return df_clean
    
    def comprehensive_eda(self):
        """Analyse exploratoire complète des données"""
        st.subheader("📊 Analyse Exploratoire des Données")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Dimensions", f"{self.df.shape[0]} lignes × {self.df.shape[1]} colonnes")
        with col2:
            st.metric("Valeurs Manquantes", self.df.isnull().sum().sum())
        with col3:
            numeric_cols = len(self.df.select_dtypes(include=[np.number]).columns)
            st.metric("Colonnes Numériques", numeric_cols)
        with col4:
            st.metric("Colonnes Total", len(self.df.columns))
        
        st.subheader("Aperçu des Données")
        st.dataframe(self.df.head())
        
        st.subheader("Statistiques Descriptives")
        st.dataframe(self.df.describe())
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.subheader("Distribution des Variables Numériques")
            cols_to_plot = numeric_cols[:min(4, len(numeric_cols))]
            for col in cols_to_plot:
                fig = px.histogram(self.df, x=col, title=f"Distribution de {col}")
                st.plotly_chart(fig, use_container_width=True)
        
        return self.analysis_results
    
    def preprocess_data(self, target_column=None, normalize=True):
        """Prétraitement avancé des données"""
        for col in self.df.columns:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.df = self.df.fillna(self.df.median(numeric_only=True))
        
        if target_column and target_column in self.df.columns:
            self.X = self.df.drop(columns=[target_column])
            self.y = self.df[target_column]
        else:
            self.X = self.df.copy()
            self.y = None
        
        if normalize and len(self.X.columns) > 0:
            try:
                self.X_scaled = self.scaler.fit_transform(self.X)
                st.success(f"✅ Données prétraitées: {self.X.shape}")
            except Exception as e:
                st.error(f"❌ Erreur lors de la normalisation: {e}")
                self.X_scaled = self.X.values
        else:
            self.X_scaled = self.X.values
        
    def perform_clustering(self, n_clusters=3):
        """Effectue un clustering K-means avancé"""
        if not hasattr(self, 'X_scaled'):
            self.preprocess_data()
        
        if len(self.X.columns) == 0:
            st.error("❌ Aucune caractéristique disponible pour le clustering")
            return None
        
        wcss = []
        k_range = range(1, min(8, len(self.X) // 2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            wcss.append(kmeans.inertia_)
        
        fig = px.line(x=list(k_range), y=wcss, title='Méthode du Coude pour le Nombre Optimal de Clusters')
        fig.update_layout(xaxis_title='Nombre de Clusters', yaxis_title='WCSS')
        st.plotly_chart(fig, use_container_width=True)
        
        try:
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = self.kmeans.fit_predict(self.X_scaled)
            
            self.df['cluster'] = clusters
            self.analysis_results['clusters'] = clusters
            
            if self.X_scaled.shape[1] >= 2:
                pca_2d = PCA(n_components=2)
                X_pca = pca_2d.fit_transform(self.X_scaled)
                
                viz_df = pd.DataFrame({
                    'PC1': X_pca[:, 0],
                    'PC2': X_pca[:, 1],
                    'Cluster': clusters
                })
                
                fig = px.scatter(viz_df, x='PC1', y='PC2', color='Cluster', 
                                title=f'Visualisation des Clusters (PCA) - {n_clusters} clusters',
                                color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
            
            cluster_dist = pd.Series(clusters).value_counts().sort_index()
            fig = px.pie(values=cluster_dist.values, names=[f'Cluster {i}' for i in cluster_dist.index], 
                        title="Distribution des Clusters")
            st.plotly_chart(fig, use_container_width=True)
            
            return clusters
            
        except Exception as e:
            st.error(f"❌ Erreur lors du clustering: {e}")
            return None
    
    def advanced_feature_analysis(self):
        """Analyse avancée des caractéristiques avec importance"""
        if not hasattr(self, 'X') or not hasattr(self, 'y'):
            st.warning("⚠️ Aucune variable cible définie pour l'analyse des features")
            return None
        
        if self.y is None:
            st.warning("⚠️ Veuillez spécifier une variable cible")
            return None
        
        try:
            if len(np.unique(self.y)) > 10:
                model = RandomForestRegressor(random_state=42, n_estimators=50)
            else:
                model = RandomForestClassifier(random_state=42, n_estimators=50)
            
            model.fit(self.X, self.y)
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            top_features = feature_importance.head(10)
            fig = px.bar(top_features, 
                        x='importance', 
                        y='feature',
                        orientation='h',
                        title='Top 10 des Caractéristiques les Plus Importantes')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            self.analysis_results['feature_importance'] = feature_importance
            return feature_importance
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse des features: {e}")
            return None

if __name__ == "__main__":
    main()


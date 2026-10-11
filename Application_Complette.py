import streamlit as st
import pandas as pd
import io
import os
import re
import urllib.parse

st.set_page_config(page_title="Gestionnaire d'Inventaire Films", page_icon="🎬", layout="wide")

st.title("🎬 Application de Gestion et Consultation - Inventaire Films")

# --- FONCTION DE CHARGEMENT ROBUSTE DE FICHIER CSV ---
def load_csv_file(filepath_or_buffer):
    """Charge un fichier CSV en testant plusieurs encodages et séparateurs."""
    if isinstance(filepath_or_buffer, str):
        with open(filepath_or_buffer, 'rb') as f:
            bytes_data = f.read()
    else:
        bytes_data = filepath_or_buffer.read()

    df_loaded = None
    for sep in [';', ',', '\t', '|']:
        for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                test_df = pd.read_csv(
                    io.BytesIO(bytes_data),
                    encoding=enc,
                    sep=sep,
                    low_memory=False,
                    on_bad_lines='skip'
                )
                if test_df.shape[1] > 1 and len(test_df) > 0:
                    df_loaded = test_df
                    break
            except Exception:
                continue
        if df_loaded is not None:
            break

    if df_loaded is None:
        # Dernier recours
        df_loaded = pd.read_csv(io.BytesIO(bytes_data), encoding='latin-1', low_memory=False, on_bad_lines='skip')

    return df_loaded


# --- DÉTECTION ET CHARGEMENT AUTOMATIQUE AU DÉMARRAGE ---
if 'df_films' not in st.session_state:
    loaded_df = None
    source_name = "Aucune source trouvée"

    # Liste des répertoires et fichiers à vérifier en priorité
    candidate_locations = [
        r"C:\_Test_Code_Python\uploads\outputs",
        r"C:\_Test_Code_Python\uploads",
        r"C:\_Test_Code_Python",
        os.path.join(".", "uploads", "outputs"),
        os.path.join(".", "outputs"),
        os.path.join(".", "uploads"),
        os.path.join("/outputs"),
        os.path.join("/uploads"),
        os.getcwd()
    ]

    found_file = None

    for loc in candidate_locations:
        if not loc or not os.path.exists(loc):
            continue

        # Si c'est directement un fichier CSV / TXT
        if os.path.isfile(loc) and loc.lower().endswith(('.csv', '.txt')):
            found_file = loc
            break

        # Si c'est un dossier, on recherche le CSV le plus approprié
        if os.path.isdir(loc):
            try:
                entries = [os.path.join(loc, f) for f in os.listdir(loc) if f.lower().endswith(('.csv', '.txt'))]
                if entries:
                    # Trier par date de modification (le plus récent d'abord)
                    entries.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    found_file = entries[0]
                    break
            except Exception:
                continue

    if found_file and os.path.exists(found_file):
        try:
            loaded_df = load_csv_file(found_file)
            source_name = found_file
        except Exception as e:
            st.sidebar.warning(f"Erreur lors du chargement auto de {found_file} : {e}")

    # Données par défaut si aucun fichier trouvé
    if loaded_df is None or loaded_df.empty:
        data_default = {
            'Code_Barres': [f"3700000{i:03d}" for i in range(1, 40)],
            'Titre': [f"Film Exemple {i}" for i in range(1, 40)],
            'IMDB_ID': [f"tt{1000000+i}" if i % 2 == 0 else "N/A" for i in range(1, 40)],
            'Annee': [2000 + (i % 24) for i in range(1, 40)],
            'Genres': ['Action' if i % 2 == 0 else 'Drame' for i in range(1, 40)],
            'Support': ['Blu-ray' if i % 3 == 0 else 'DVD' for i in range(1, 40)]
        }
        loaded_df = pd.DataFrame(data_default)
        source_name = "Jeu de données de démonstration"

    st.session_state['df_films'] = loaded_df
    st.session_state['source_name'] = source_name


# --- SIDEBAR : GESTION DU FICHIER ET RECHARGEMENT ---
st.sidebar.header("📂 Fichier & Source de données")

# Saisie manuelle de chemin personnalisé au besoin
custom_path = st.sidebar.text_input(
    "Chemin du dossier / fichier local :",
    value=r"C:\_Test_Code_Python\uploads\outputs"
)

if st.sidebar.button("🔄 Charger depuis ce chemin"):
    if os.path.exists(custom_path):
        target_path = custom_path
        if os.path.isdir(custom_path):
            files = [os.path.join(custom_path, f) for f in os.listdir(custom_path) if f.lower().endswith(('.csv', '.txt'))]
            if files:
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                target_path = files[0]
            else:
                target_path = None
        
        if target_path and os.path.isfile(target_path):
            try:
                st.session_state['df_films'] = load_csv_file(target_path)
                st.session_state['source_name'] = target_path
                st.sidebar.success(f"Fichier chargé : {os.path.basename(target_path)}")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Erreur de lecture : {e}")
        else:
            st.sidebar.error("Aucun fichier .csv trouvé dans ce répertoire.")
    else:
        st.sidebar.error("Chemin introuvable.")

# Import par téléversement direct
uploaded = st.sidebar.file_uploader("Ou importer un fichier CSV manuellement :", type=["csv", "txt"])
if uploaded is not None:
    try:
        st.session_state['df_films'] = load_csv_file(uploaded)
        st.session_state['source_name'] = uploaded.name
        st.sidebar.success(f"Import réussi : {uploaded.name}")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Erreur import : {e}")

# Récupération et nettoyage du DataFrame actif
raw_df = st.session_state['df_films'].copy()
raw_df.columns = [str(c).strip().replace('\ufeff', '') for c in raw_df.columns]

# 1. Suppression définitive de la colonne date_steelbook si présente
cols_drop = [c for c in raw_df.columns if 'date_steelbook' in c.lower() or 'date steelbook' in c.lower()]
if cols_drop:
    raw_df = raw_df.drop(columns=cols_drop)

# 2. Nettoyage des colonnes code-barres (suppression des .0 résiduels des floats)
for c in raw_df.columns:
    if any(k in c.lower() for k in ['code', 'barre', 'ean']):
        raw_df[c] = raw_df[c].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        raw_df.loc[raw_df[c] == 'nan', c] = ''

# 3. DÉTECTION ET CRÉATION DE LA COLONNE 'Lien IMDb' GARANTIE FONCTIONNELLE
# Recherche de colonnes IMDb ou ID
col_imdb_existing = next((c for c in raw_df.columns if 'imdb' in c.lower() and ('lien' in c.lower() or 'url' in c.lower() or 'link' in c.lower())), None)
col_imdb_id = next((c for c in raw_df.columns if 'imdb' in c.lower() and 'id' in c.lower()), None)
col_title = next((c for c in raw_df.columns if 'titre' in c.lower() or 'title' in c.lower() or 'film' in c.lower()), raw_df.columns[0])

def build_valid_imdb_url(row):
    """Garantit une URL valide http(s):// pour chaque ligne."""
    # 1. Si une URL complète existe déjà
    if col_imdb_existing and pd.notna(row.get(col_imdb_existing)):
        val = str(row[col_imdb_existing]).strip()
        if val.startswith(('http://', 'https://')):
            return val
        # Si c'est un code tt dans la colonne lien
        tt_match = re.search(r'(tt\d+)', val)
        if tt_match:
            return f"https://www.imdb.com/title/{tt_match.group(1)}/"

    # 2. Si un ID tt1234567 existe dans une colonne ID
    if col_imdb_id and pd.notna(row.get(col_imdb_id)):
        val_id = str(row[col_imdb_id]).strip()
        tt_match = re.search(r'(tt\d+)', val_id)
        if tt_match:
            return f"https://www.imdb.com/title/{tt_match.group(1)}/"

    # 3. Fallback : lien de recherche IMDb direct avec le titre
    titre_val = str(row.get(col_title, '')).strip()
    if titre_val and titre_val.lower() != 'nan':
        encoded_title = urllib.parse.quote_plus(titre_val)
        return f"https://www.imdb.com/find/?q={encoded_title}"

    return "https://www.imdb.com/"

# Création d'une colonne uniforme 'Lien IMDb'
raw_df['Lien IMDb'] = raw_df.apply(build_valid_imdb_url, axis=1)

# Informations sur la source active
st.sidebar.info(f"📁 Source active :\n**{os.path.basename(str(st.session_state.get('source_name', '')))}**\n\n({len(raw_df)} enregistrements)")

# Export CSV
csv_data = raw_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Exporter l'inventaire en CSV",
    data=csv_data,
    file_name="inventaire_films_export.csv",
    mime="text/csv"
)

# --- NAVIGATION DES 3 FONCTIONS ---
st.sidebar.markdown("---")
st.sidebar.header("🧭 Menu des Fonctions")
menu_choice = st.sidebar.radio("Sélectionnez la fonction :", [
    "1° Consultation globale avec filtres",
    "2° Saisie Code-Barres (5 avant / 5 après - Tri Alpha)",
    "3° Recherche par titre (Français / Global) & IMDb"
])

# Rendu universel du tableau pour toutes les vues
def display_inventory_table(df_to_show, key_id):
    if df_to_show.empty:
        st.warning("Aucun enregistrement à afficher.")
        return

    # Configuration des colonnes
    column_config = {
        "Lien IMDb": st.column_config.LinkColumn(
            "Lien IMDb",
            help="Ouvrir la fiche officielle du film sur IMDb",
            display_text="🎬 Voir sur IMDb ↗",
            width="medium"
        )
    }

    # Élargir les colonnes de titres
    for col in df_to_show.columns:
        if any(w in col.lower() for w in ['titre', 'title', 'nom']):
            column_config[col] = st.column_config.TextColumn(col, width="large")
        elif any(w in col.lower() for w in ['code', 'barre', 'ean']):
            column_config[col] = st.column_config.TextColumn(col, width="medium")

    st.dataframe(
        df_to_show,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        key=f"df_table_{key_id}"
    )


# ==============================================================================
# FONCTION 1 : CONSULTATION GLOBALE AVEC FILTRES
# ==============================================================================
if menu_choice == "1° Consultation globale avec filtres":
    st.header("🗂️ Consultation Globale & Filtres Multi-critères")
    st.markdown("Filtrer l'inventaire complet en combinant les critères de sélection.")

    cols_for_filters = [
        c for c in raw_df.columns 
        if c != 'Lien IMDb' and not any(k in c.lower() for k in ['code', 'barre', 'ean']) and raw_df[c].nunique() < 250
    ]

    filtered_df = raw_df.copy()

    if cols_for_filters:
        f_cols = st.columns(3)
        for idx, col in enumerate(cols_for_filters[:6]):
            col_target = f_cols[idx % 3]
            vals = sorted([str(x) for x in raw_df[col].dropna().unique() if str(x).strip() not in ['', 'nan', 'N/A']])
            if vals:
                with col_target:
                    selected = st.multiselect(f"Filtrer par {col} :", vals, key=f"f1_sel_{col}")
                    if selected:
                        filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]

    st.markdown(f"**Nombre de films affichés :** `{len(filtered_df)}`")
    display_inventory_table(filtered_df, key_id="consultation")


# ==============================================================================
# FONCTION 2 : SAISIE CODE-BARRES (5 AVANT / 5 APRÈS - TRI ALPHABÉTIQUE)
# ==============================================================================
elif menu_choice == "2° Saisie Code-Barres (5 avant / 5 après - Tri Alpha)":
    st.header("📦 Saisie Code-Barres & Voisinage Alphabétique")
    st.markdown("Recherche d'un film par son code-barres et affichage de ses **5 voisins avant et 5 voisins après** par ordre alphabétique.")

    # Identification de la colonne de tri alphabétique et code-barres
    sort_col = next((c for c in raw_df.columns if 'titre' in c.lower() and ('francais' in c.lower() or 'vf' in c.lower())), col_title)
    col_code = next((c for c in raw_df.columns if any(k in c.lower() for k in ['code', 'barre', 'ean'])), raw_df.columns[0])

    # Tri par ordre alphabétique insensible à la casse
    sorted_df = raw_df.sort_values(
        by=sort_col,
        key=lambda x: x.astype(str).str.lower()
    ).reset_index(drop=True)

    barcode_input = st.text_input("Saisissez ou scannez le code-barres :", "").strip()

    if barcode_input:
        match_idx = sorted_df.index[sorted_df[col_code].astype(str).str.strip() == barcode_input].tolist()
        if not match_idx:
            match_idx = sorted_df.index[sorted_df[col_code].astype(str).str.contains(barcode_input, na=False)].tolist()

        if match_idx:
            idx = match_idx[0]
            st.success(f"🎯 Film identifié à la position **{idx + 1}** sur {len(sorted_df)} (Classement alphabétique sur `{sort_col}`).")

            start_idx = max(0, idx - 5)
            end_idx = min(len(sorted_df), idx + 6)

            subset_df = sorted_df.iloc[start_idx:end_idx].copy()
            subset_df.insert(0, "Position", [i + 1 for i in range(start_idx, end_idx)])
            subset_df.insert(1, "Repère", ["⭐ FILM CIBLÉ" if i == idx else "" for i in range(start_idx, end_idx)])

            display_inventory_table(subset_df, key_id="barcode_match")
        else:
            st.error("❌ Aucun film ne correspond à ce code-barres.")
            st.markdown("### Inventaire complet (Ordre alphabétique) :")
            display_inventory_table(sorted_df, key_id="barcode_fallback")
    else:
        st.markdown("### Inventaire complet (Ordre alphabétique) :")
        display_inventory_table(sorted_df, key_id="barcode_default")


# ==============================================================================
# FONCTION 3 : RECHERCHE PAR TITRE (FRANÇAIS / GLOBAL) & IMDB
# ==============================================================================
else:
    st.header("🔍 Recherche par Titre (Français / Global) & Lien IMDb")
    st.markdown("Recherche rapide et insensible à la casse dans toutes les colonnes de titre.")

    query_title = st.text_input("Entrez tout ou partie du titre recherché :", "").strip()

    # Détection des colonnes de titres
    title_cols = [c for c in raw_df.columns if any(k in c.lower() for k in ['titre', 'title', 'nom'])]
    if not title_cols:
        title_cols = [col_title]

    if query_title:
        q_norm = query_title.lower()
        mask = False
        for tc in title_cols:
            mask = mask | raw_df[tc].astype(str).str.lower().str.contains(q_norm, na=False)

        results_df = raw_df[mask].copy()

        if not results_df.empty:
            st.success(f"✨ **{len(results_df)}** film(s) correspondant(s) trouvé(s) :")
            display_inventory_table(results_df, key_id="title_search_results")
        else:
            st.warning(f"Aucun film trouvé contenant « {query_title} ».")
            st.markdown("### Inventaire complet :")
            display_inventory_table(raw_df, key_id="title_fallback")
    else:
        st.markdown("### Inventaire complet :")
        display_inventory_table(raw_df, key_id="title_default")

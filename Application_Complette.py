import streamlit as st
import pandas as pd
import io
import os
import re
import urllib.parse
from collections import Counter
try:
    import altair as alt
except ImportError:
    alt = None

st.set_page_config(
    page_title="Gestionnaire d'Inventaire Films & IMDb V4.4.0",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FONCTION UNIVERSELLE DE CHARGEMENT & PRÉPARATION ---
def load_and_prepare_csv(filepath_or_buffer):
    """Charge le fichier CSV consolidé (avec Année, Durée et Studio), convertit les identifiants en vrai type numérique pour un tri exact."""
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
                    dtype=str,
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
        df_loaded = pd.read_csv(io.BytesIO(bytes_data), encoding='latin-1', dtype=str, low_memory=False, on_bad_lines='skip')

    # Nettoyage des en-têtes
    df_loaded.columns = [str(c).strip().replace('\ufeff', '') for c in df_loaded.columns]

    # Suppression date_steelbook résiduelle si présente
    cols_drop = [c for c in df_loaded.columns if 'date_steelbook' in c.lower() or 'date steelbook' in c.lower()]
    if cols_drop:
        df_loaded = df_loaded.drop(columns=cols_drop)

    # 1. CONVERSION NUMÉRIQUE STRICTE DES IDENTIFIANTS (Int64 pour éviter le tri alphabétique '1', '10', '2')
    # ID MyMovie = Position physique du boîtier dans My Movies (1 à 4301)
    if 'pos_source' in df_loaded.columns:
        df_loaded['ID MyMovie'] = pd.to_numeric(df_loaded['pos_source'], errors='coerce').astype('Int64')
    elif 'position' in df_loaded.columns:
        df_loaded['ID MyMovie'] = pd.to_numeric(df_loaded['position'], errors='coerce').astype('Int64')
    elif 'id' in df_loaded.columns:
        df_loaded['ID MyMovie'] = pd.to_numeric(df_loaded['id'], errors='coerce').astype('Int64')

    # ID Unique = Numéro séquentiel strict du film (1 à 4698 / 4771)
    if 'id_film' in df_loaded.columns:
        df_loaded['ID Unique'] = pd.to_numeric(df_loaded['id_film'], errors='coerce').astype('Int64')
    elif 'id' in df_loaded.columns:
        df_loaded['ID Unique'] = pd.to_numeric(df_loaded['id'], errors='coerce').astype('Int64')

    # Réf Entrée = Code d'extraction détaillé (ex: 1, 1_001, 1_002...)
    if 'id' in df_loaded.columns and 'id_film' in df_loaded.columns:
        df_loaded['Réf Entrée'] = df_loaded['id'].fillna('').astype(str).str.strip()

    # 2. Nettoyage strict des codes-barres (texte pur, pas de .0 ni notation scientifique)
    for c in df_loaded.columns:
        if any(k in c.lower() for k in ['code', 'barre', 'barcode', 'ean']):
            df_loaded[c] = df_loaded[c].fillna('').astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            df_loaded.loc[df_loaded[c].isin(['nan', 'none', 'null']), c] = ''

    # 3. Conversion chronologique stricte des dates d'ajout
    for date_col in ['date_ajout', 'date_coffret', 'date_sortie']:
        if date_col in df_loaded.columns:
            df_loaded[date_col] = pd.to_datetime(df_loaded[date_col], format='%d/%m/%Y', errors='coerce')

    # 4. Conversion numérique de l'ANNÉE DE SORTIE (annee)
    col_annee = next((c for c in df_loaded.columns if c.lower() in ['annee', 'année', 'year', 'release_year']), None)
    if col_annee:
        df_loaded['annee_num'] = pd.to_numeric(df_loaded[col_annee].astype(str).str.extract(r'(\d{4})')[0], errors='coerce').astype('Int64')

    # 5. Conversion numérique de la DURÉE EN MINUTES (duree_minutes)
    col_duree = next((c for c in df_loaded.columns if 'duree' in c.lower() or 'durée' in c.lower() or 'runtime' in c.lower() or 'minutes' in c.lower()), None)
    if col_duree:
        df_loaded['duree_num'] = pd.to_numeric(df_loaded[col_duree].astype(str).str.extract(r'(\d+)')[0], errors='coerce').astype('Int64')

    # 6. Extraction propre des Genres si type_film est présent
    if 'type_film' in df_loaded.columns and 'genres' not in df_loaded.columns:
        df_loaded['genres'] = df_loaded['type_film'].str.extract(r'\((.*?)\)')[0].fillna('')
        df_loaded['genres'] = df_loaded['genres'].str.replace(';', ', ')

    # 7. Conversion numérique de la note IMDb
    col_rating = next((c for c in df_loaded.columns if 'rating' in c.lower() or 'note' in c.lower()), None)
    if col_rating:
        df_loaded['rating_num'] = pd.to_numeric(df_loaded[col_rating].astype(str).str.replace(',', '.'), errors='coerce')

    # 8. Détection des titres français et VO
    col_t_fr = next((c for c in df_loaded.columns if c.lower() in ['titre', 'titre_fr', 'titre_francais', 'titre_vf']), None)
    if not col_t_fr:
        col_t_fr = next((c for c in df_loaded.columns if 'titre' in c.lower() or 'title' in c.lower()), df_loaded.columns[0])

    col_t_vo = next((c for c in df_loaded.columns if c.lower() in ['titre_anglais', 'titre_original', 'titre_en_imdb', 'titre_global', 'titre_vo'] and c != col_t_fr), None)

    # 9. Synthèse claire des supports physiques & packagings
    def extract_supports_str(row):
        badges = []
        if str(row.get('4K', '')).strip().upper() == 'OUI': badges.append('4K UHD')
        if str(row.get('BLURAY', '')).strip().upper() == 'OUI': badges.append('Blu-ray')
        if str(row.get('3D', '')).strip().upper() == 'OUI': badges.append('3D')
        if str(row.get('DVD', '')).strip().upper() == 'OUI': badges.append('DVD')
        if str(row.get('STEELBOOK', '')).strip().upper() == 'OUI': badges.append('Steelbook')
        if str(row.get('COFFRET', '')).strip().upper() == 'OUI': badges.append('Coffret')

        if not badges:
            for s_name, col in [('DVD', 'nb_DVD'), ('Blu-ray', 'nb_BLURAY'), ('4K UHD', 'nb_4K'), ('Coffret', 'nb_Coffret')]:
                if col in row and pd.to_numeric(row[col], errors='coerce') > 0:
                    badges.append(s_name)
            if 'jaquette' in row and str(row['jaquette']).strip():
                badges.append(str(row['jaquette']).strip())
            if '3D' in row and str(row['3D']).strip().lower() == 'oui':
                badges.append('3D')

        return ' | '.join(badges) if badges else '-'

    df_loaded['supports'] = df_loaded.apply(extract_supports_str, axis=1)

    # 10. Colonne Lien IMDb cliquable garantie
    col_raw_lien = next((c for c in df_loaded.columns if 'imdb' in c.lower() and ('lien' in c.lower() or 'url' in c.lower() or 'link' in c.lower())), None)
    col_raw_id = next((c for c in df_loaded.columns if 'imdb' in c.lower() and ('no_' in c.lower() or 'id' in c.lower()) and c != col_raw_lien), None)

    def generate_imdb_link(row):
        if col_raw_lien and pd.notna(row.get(col_raw_lien)):
            l_val = str(row[col_raw_lien]).strip()
            if l_val.startswith(('http://', 'https://')):
                return l_val
            tt_m = re.search(r'(tt\d+)', l_val)
            if tt_m:
                return f"https://www.imdb.com/title/{tt_m.group(1)}/"

        if col_raw_id and pd.notna(row.get(col_raw_id)):
            id_val = str(row[col_raw_id]).strip()
            tt_m = re.search(r'(tt\d+)', id_val)
            if tt_m:
                return f"https://www.imdb.com/title/{tt_m.group(1)}/"

        t_val = str(row.get(col_t_fr, '')).strip()
        if not t_val and col_t_vo:
            t_val = str(row.get(col_t_vo, '')).strip()

        if t_val and t_val.lower() not in ['nan', '', 'none']:
            return f"https://www.imdb.com/find/?q={urllib.parse.quote_plus(t_val)}"

        return "https://www.imdb.com/"

    df_loaded['Lien IMDb'] = df_loaded.apply(generate_imdb_link, axis=1)
    if col_raw_lien and col_raw_lien in df_loaded.columns and col_raw_lien != 'Lien IMDb':
        df_loaded = df_loaded.drop(columns=[col_raw_lien])

    return df_loaded


# --- DÉTECTION ET CHARGEMENT DU FICHIER ---
st.sidebar.header("📂 Données de l'Inventaire")

candidate_paths = [
    r"C:\_Test_Code_Python\uploads\Liste_Complette_Film_Consolidee_Final.csv",
    r"C:\_Test_Code_Python\uploads\outputs\Liste_Complette_Film_Consolidee_Final.csv",
    r"C:\_Test_Code_Python\Liste_Complette_Film_Consolidee_Final.csv",
    r"/workspace/files/personal_files/uploads/Liste_Complette_Film_Consolidee_Final.csv",
    r"/uploads/Liste_Complette_Film_Consolidee_Final.csv",
    r"Liste_Complette_Film_Consolidee_Final.csv",
    r"C:\_Test_Code_Python\uploads\Liste_Complette_Film_Consolidee_Final_NOUVEAU_2.csv",
    r"C:\_Test_Code_Python\uploads\outputs\Liste_Complette_Film_Consolidee_Final_NOUVEAU_2.csv",
    r"C:\_Test_Code_Python\Liste_Complette_Film_Consolidee_Final_NOUVEAU_2.csv",
    r"/workspace/files/personal_files/uploads/Liste_Complette_Film_Consolidee_Final_NOUVEAU_2.csv",
    r"/uploads/Liste_Complette_Film_Consolidee_Final_NOUVEAU_2.csv",
    r"Liste_Complette_Film_Consolidee_Final_NOUVEAU_2.csv",
    r"C:\_Test_Code_Python\uploads",
    r"C:\_Test_Code_Python",
    r"./uploads",
    r"."
]

if 'df_films' not in st.session_state:
    loaded_df = None
    loaded_name = None

    for p in candidate_paths:
        if not p or not os.path.exists(p):
            continue
        if os.path.isfile(p) and p.lower().endswith(('.csv', '.txt')):
            try:
                loaded_df = load_and_prepare_csv(p)
                loaded_name = os.path.basename(p)
                break
            except Exception:
                continue
        elif os.path.isdir(p):
            try:
                files = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(('.csv', '.txt'))]
                files.sort(key=lambda x: ('final' in x.lower(), '44' in x.lower(), 'nouveau_2' in x.lower(), 'nouveau' in x.lower(), os.path.getmtime(x)), reverse=True)
                if files:
                    loaded_df = load_and_prepare_csv(files[0])
                    loaded_name = os.path.basename(files[0])
                    break
            except Exception:
                continue

    if loaded_df is None or loaded_df.empty:
        # Données de démonstration numériques
        data_demo = {
            'ID MyMovie': pd.Series(range(1, 31), dtype='Int64'),
            'ID Unique': pd.Series(range(1, 31), dtype='Int64'),
            'barcode': [f"3700000{i:03d}" for i in range(1, 31)],
            'titre': [f"Film Exemple {i}" for i in range(1, 31)],
            'annee': [f"{2000 + (i % 25)}" for i in range(1, 31)],
            'duree_minutes': [f"{85 + (i * 3)}" for i in range(1, 31)],
            'genres': ['Action, Sci-Fi' if i % 2 == 0 else 'Comedy, Drama' for i in range(1, 31)],
            'directeur': ['Steven Spielberg' if i % 3 == 0 else 'Christopher Nolan' for i in range(1, 31)],
            'acteurs_principaux': ['Tom Hanks, Leonardo DiCaprio' for _ in range(30)],
            'Production_Corp': ['WARNER BROS.' if i % 2 == 0 else 'UNIVERSAL PICTURES' for i in range(1, 31)],
            'rating_imdb': ['7.8' if i % 2 == 0 else '8.3' for i in range(1, 31)],
            'date_ajout': pd.date_range(start='2021-01-01', periods=30, freq='M'),
            'pays': ['France' if i % 2 == 0 else 'USA' for i in range(1, 31)],
            'supports': ['Blu-ray | Steelbook' if i % 2 == 0 else 'DVD' for i in range(1, 31)],
            'DVD': ['OUI' if i % 2 != 0 else '' for i in range(1, 31)],
            'BLURAY': ['OUI' if i % 2 == 0 else '' for i in range(1, 31)],
            '4K': ['OUI' if i % 4 == 0 else '' for i in range(1, 31)],
            'STEELBOOK': ['OUI' if i % 3 == 0 else '' for i in range(1, 31)],
            'COFFRET': ['OUI' if i % 5 == 0 else '' for i in range(1, 31)],
            '3D': ['OUI' if i % 6 == 0 else '' for i in range(1, 31)]
        }
        loaded_df = pd.DataFrame(data_demo)
        loaded_df['annee_num'] = pd.to_numeric(loaded_df['annee'], errors='coerce').astype('Int64')
        loaded_df['duree_num'] = pd.to_numeric(loaded_df['duree_minutes'], errors='coerce').astype('Int64')
        loaded_df['rating_num'] = pd.to_numeric(loaded_df['rating_imdb'], errors='coerce')
        loaded_df['Lien IMDb'] = [f"https://www.imdb.com/title/tt{1000000+i}/" for i in range(1, 31)]
        loaded_name = "Mode Démo"

    st.session_state['df_films'] = loaded_df
    st.session_state['source_name'] = loaded_name

# Contrôle manuel du chemin
input_custom_path = st.sidebar.text_input(
    "Chemin du fichier / dossier CSV :",
    value=st.session_state.get('source_name', r"C:\_Test_Code_Python\uploads\Liste_Complette_Film_Consolidee_Final.csv")
)

if st.sidebar.button("🔄 Charger / Recharger ce fichier"):
    if os.path.exists(input_custom_path):
        target = input_custom_path
        if os.path.isdir(input_custom_path):
            candidates = [os.path.join(input_custom_path, f) for f in os.listdir(input_custom_path) if f.lower().endswith(('.csv', '.txt'))]
            candidates.sort(key=lambda x: ('final' in x.lower(), '44' in x.lower(), 'nouveau_2' in x.lower(), 'nouveau' in x.lower(), os.path.getmtime(x)), reverse=True)
            target = candidates[0] if candidates else None

        if target and os.path.isfile(target):
            try:
                st.session_state['df_films'] = load_and_prepare_csv(target)
                st.session_state['source_name'] = os.path.basename(target)
                st.sidebar.success(f"Fichier chargé : {os.path.basename(target)}")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Erreur : {e}")
        else:
            st.sidebar.error("Aucun fichier CSV valide trouvé.")
    else:
        st.sidebar.error("Chemin introuvable.")

# Import par téléversement
up_file = st.sidebar.file_uploader("Ou téléverser un fichier CSV :", type=["csv", "txt"])
if up_file is not None:
    try:
        st.session_state['df_films'] = load_and_prepare_csv(up_file)
        st.session_state['source_name'] = up_file.name
        st.sidebar.success(f"Import réussi : {up_file.name}")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Erreur d'import : {e}")

df = st.session_state['df_films']
st.sidebar.info(f"📊 Source active : **{st.session_state.get('source_name', 'Inconnu')}**\n\n📌 **{len(df):,}** enregistrements".replace(',', ' '))

# Export CSV propre
df_export = df.copy()
for dc in ['date_ajout', 'date_coffret', 'date_sortie']:
    if dc in df_export.columns and pd.api.types.is_datetime64_any_dtype(df_export[dc]):
        df_export[dc] = df_export[dc].dt.strftime('%d/%m/%Y').fillna('')

csv_bytes = df_export.to_csv(index=False, sep=';').encode('utf-8')
st.sidebar.download_button(
    label="💾 Exporter en CSV (séparateur ;)",
    data=csv_bytes,
    file_name="inventaire_films_consolide.csv",
    mime="text/csv"
)

# --- NAVIGATION ---
st.sidebar.markdown("---")
st.sidebar.header("🧭 Menu des Fonctions")
menu = st.sidebar.radio("Sélectionnez la fonction :", [
    "1° Consultation globale avec filtres enrichis",
    "2° Saisie Code-Barres (5 avant / 5 après - Tri Alpha)",
    "3° Recherche par titre / réalisateur / acteur & IMDb",
    "4° Statistiques complètes de la collection"
])

# Détection unifiée des colonnes
col_titre_fr = next((c for c in df.columns if c.lower() in ['titre', 'titre_fr', 'titre_francais', 'titre_vf']), None)
if not col_titre_fr:
    col_titre_fr = next((c for c in df.columns if 'titre' in c.lower() or 'title' in c.lower()), df.columns[0])

col_titre_vo = next((c for c in df.columns if c.lower() in ['titre_anglais', 'titre_original', 'titre_en_imdb', 'titre_global', 'titre_vo'] and c != col_titre_fr), None)
col_barcode = next((c for c in df.columns if any(k in c.lower() for k in ['code', 'barre', 'barcode', 'ean'])), df.columns[0])
col_rating = next((c for c in df.columns if 'rating' in c.lower() or 'note' in c.lower()), None)
col_director = next((c for c in df.columns if 'directeur' in c.lower() or 'director' in c.lower() or 'realisateur' in c.lower()), None)
col_actors = next((c for c in df.columns if 'acteur' in c.lower() or 'actor' in c.lower() or 'casting' in c.lower()), None)
col_genres = 'genres' if 'genres' in df.columns else next((c for c in df.columns if 'genre' in c.lower() or 'type' in c.lower()), None)

# Colonne Studio / Production
col_production = next((c for c in df.columns if 'production' in c.lower() or 'corp' in c.lower() or 'studio' in c.lower()), None)

# Colonnes Année et Durée
has_annee = 'annee_num' in df.columns or 'annee' in df.columns
has_duree = 'duree_num' in df.columns or 'duree_minutes' in df.columns

# Fonction d'affichage standardisée avec configuration Streamlit propre
def render_custom_table(dataframe, key_id):
    if dataframe.empty:
        st.warning("Aucun résultat ne correspond aux critères sélectionnés.")
        return

    col_cfg = {
        "Lien IMDb": st.column_config.LinkColumn(
            "Lien IMDb",
            help="Ouvrir la fiche officielle du film sur IMDb",
            display_text="🎬 Voir sur IMDb ↗",
            width="medium"
        )
    }

    # Configuration des identifiants physiques et uniques en NUMBERCOLUMN (Tri numérique natif garanti)
    if 'ID MyMovie' in dataframe.columns:
        col_cfg['ID MyMovie'] = st.column_config.NumberColumn(
            "🏷️ ID MyMovie",
            help="Emplacement physique du boîtier dans My Movies (# position du support)",
            format="%d",
            width="small"
        )
    if 'ID Unique' in dataframe.columns:
        col_cfg['ID Unique'] = st.column_config.NumberColumn(
            "🔑 ID Unique",
            help="Identifiant séquentiel unique du film",
            format="%d",
            width="small"
        )
    if 'Réf Entrée' in dataframe.columns:
        col_cfg['Réf Entrée'] = st.column_config.TextColumn(
            "📑 Réf",
            help="Référence détaillée (ex: 1_001 pour un sous-film dans un coffret)",
            width="small"
        )

    # Configuration Année & Durée
    if 'annee_num' in dataframe.columns:
        col_cfg['annee_num'] = st.column_config.NumberColumn(
            "📅 Année",
            format="%d",
            help="Année de sortie en salles",
            width="small"
        )
    elif 'annee' in dataframe.columns:
        col_cfg['annee'] = st.column_config.TextColumn("📅 Année", help="Année de sortie", width="small")

    if 'duree_num' in dataframe.columns:
        col_cfg['duree_num'] = st.column_config.NumberColumn(
            "⏱️ Durée",
            format="%d min",
            help="Durée totale du film en minutes",
            width="small"
        )
    elif 'duree_minutes' in dataframe.columns:
        col_cfg['duree_minutes'] = st.column_config.TextColumn("⏱️ Durée", help="Durée en minutes", width="small")

    # Configuration des dates d'ajout
    if 'date_ajout' in dataframe.columns and pd.api.types.is_datetime64_any_dtype(dataframe['date_ajout']):
        col_cfg['date_ajout'] = st.column_config.DateColumn(
            "Date d'ajout",
            format="DD/MM/YYYY",
            help="Date d'ajout dans la vidéothèque",
            width="medium"
        )

    # Note IMDb
    if 'rating_imdb' in dataframe.columns:
        col_cfg['rating_imdb'] = st.column_config.TextColumn("⭐ Note IMDb", width="small")
    elif 'rating_num' in dataframe.columns:
        col_cfg['rating_num'] = st.column_config.NumberColumn("⭐ Note IMDb", format="%.1f", width="small")

    # Supports / Formats
    if 'supports' in dataframe.columns:
        col_cfg['supports'] = st.column_config.TextColumn("💿 Formats", width="medium")

    # Largeurs de colonnes textuelles
    if col_titre_fr in dataframe.columns:
        col_cfg[col_titre_fr] = st.column_config.TextColumn(col_titre_fr, width="large")
    if col_titre_vo and col_titre_vo in dataframe.columns:
        col_cfg[col_titre_vo] = st.column_config.TextColumn(col_titre_vo, width="medium")
    if col_director and col_director in dataframe.columns:
        col_cfg[col_director] = st.column_config.TextColumn(col_director, width="medium")
    if col_actors and col_actors in dataframe.columns:
        col_cfg[col_actors] = st.column_config.TextColumn(col_actors, width="large")
    if col_production and col_production in dataframe.columns:
        col_cfg[col_production] = st.column_config.TextColumn("🏢 Production / Studio", width="medium", help="Société de production ou studio du film")
    if col_barcode in dataframe.columns:
        col_cfg[col_barcode] = st.column_config.TextColumn(col_barcode, width="medium")

    display_df = dataframe.copy()
    if 'rating_num' in display_df.columns and 'rating_imdb' in display_df.columns:
        display_df = display_df.drop(columns=['rating_num'])

    # Si on utilise annee_num et duree_num, on masque les chaînes brutes redondantes
    cols_to_exclude = ['source', 'pos_source', 'page', 'id', 'id_film']
    if 'annee_num' in display_df.columns and 'annee' in display_df.columns:
        cols_to_exclude.append('annee')
    if 'duree_num' in display_df.columns and 'duree_minutes' in display_df.columns:
        cols_to_exclude.append('duree_minutes')

    # Ordre de présentation clair avec Année & Durée au cœur de la fiche
    col_annee_disp = 'annee_num' if 'annee_num' in display_df.columns else ('annee' if 'annee' in display_df.columns else None)
    col_duree_disp = 'duree_num' if 'duree_num' in display_df.columns else ('duree_minutes' if 'duree_minutes' in display_df.columns else None)

    preferred_order = [c for c in [
        'Position', 'Repère', 'ID MyMovie', 'ID Unique', 'Réf Entrée', col_titre_fr,
        col_annee_disp, col_duree_disp, 'rating_imdb', 'supports', col_genres, col_director,
        col_actors, col_production, 'date_ajout', 'pays', col_barcode, 'Lien IMDb'
    ] if c and c in display_df.columns]

    other_cols = [c for c in display_df.columns if c not in preferred_order and c not in cols_to_exclude]
    display_df = display_df[preferred_order + other_cols]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config=col_cfg,
        key=f"tab_view_{key_id}"
    )


# --- FONCTION GRAPHIQUE BARRES TRI DÉCROISSANT STRICT (ALTAIR / NATIVE) ---
def render_descending_bar_chart(df_data, cat_col, val_col, height=480, orientation="Horizontale", color_scheme="reds"):
    if df_data.empty:
        st.warning("Aucune donnée disponible.")
        return

    # Tri explicite du DataFrame par valeur strictement décroissante (du plus grand au plus petit)
    sorted_df = df_data.sort_values(by=val_col, ascending=False).reset_index(drop=True)

    if alt is not None:
        if orientation == "Horizontale":
            bars = alt.Chart(sorted_df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                y=alt.Y(
                    f"{cat_col}:N",
                    sort=alt.EncodingSortField(field=val_col, order="descending"),
                    title=cat_col,
                    axis=alt.Axis(labelOverlap=False, labelLimit=400, labelFontSize=12)
                ),
                x=alt.X(f"{val_col}:Q", title=val_col),
                tooltip=[
                    alt.Tooltip(f"{cat_col}:N", title=cat_col),
                    alt.Tooltip(f"{val_col}:Q", title="Nombre de films")
                ],
                color=alt.Color(f"{val_col}:Q", scale=alt.Scale(scheme=color_scheme), legend=None)
            )
            text_labels = alt.Chart(sorted_df).mark_text(
                align='left',
                baseline='middle',
                dx=6,
                fontSize=11,
                fontWeight='bold'
            ).encode(
                y=alt.Y(f"{cat_col}:N", sort=alt.EncodingSortField(field=val_col, order="descending")),
                x=alt.X(f"{val_col}:Q"),
                text=alt.Text(f"{val_col}:Q")
            )
            chart = (bars + text_labels).properties(
                height=max(height, len(sorted_df) * 26)
            )
        else:
            bars = alt.Chart(sorted_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X(
                    f"{cat_col}:N",
                    sort=alt.EncodingSortField(field=val_col, order="descending"),
                    title=cat_col,
                    axis=alt.Axis(labelAngle=-65, labelOverlap=False, labelLimit=350, labelFontSize=10)
                ),
                y=alt.Y(f"{val_col}:Q", title=val_col),
                tooltip=[
                    alt.Tooltip(f"{cat_col}:N", title=cat_col),
                    alt.Tooltip(f"{val_col}:Q", title="Nombre de films")
                ],
                color=alt.Color(f"{val_col}:Q", scale=alt.Scale(scheme=color_scheme), legend=None)
            )
            text_labels = alt.Chart(sorted_df).mark_text(
                align='center',
                baseline='bottom',
                dy=-4,
                fontSize=10,
                fontWeight='bold'
            ).encode(
                x=alt.X(f"{cat_col}:N", sort=alt.EncodingSortField(field=val_col, order="descending")),
                y=alt.Y(f"{val_col}:Q"),
                text=alt.Text(f"{val_col}:Q")
            )
            chart = (bars + text_labels).properties(height=height)

        st.altair_chart(chart, use_container_width=True)
    else:
        st.bar_chart(sorted_df.set_index(cat_col)[val_col])


# ==============================================================================
# FONCTION 1 : CONSULTATION GLOBALE AVEC FILTRES ENRICHIS
# ==============================================================================
if menu == "1° Consultation globale avec filtres enrichis":
    st.title("🗂️ Consultation Globale & Filtres Multi-critères")
    st.markdown("Retrouvez vos films physiquement via l'**ID MyMovie** (boîtier) ou l'**ID Unique**, avec un **tri numérique strict** (1, 2, ..., 10, ..., 4301).")

    filtered_df = df.copy()

    # Barre de recherche textuelle globale (avec Année, ID MyMovie, ID Unique et Studio inclus)
    search_text = st.text_input("🔍 Recherche rapide (Titre VF/VO, Année, ID MyMovie, ID Unique, Réalisateur, Acteur, Studio, Code-barres...) :", "").strip().lower()
    if search_text:
        mask = False
        cols_to_search = ['ID MyMovie', 'ID Unique', 'Réf Entrée', col_titre_fr, col_titre_vo, 'annee', 'annee_num', 'duree_minutes', col_barcode, col_director, col_actors, col_genres, col_production, 'supports']
        for c in cols_to_search:
            if c and c in filtered_df.columns:
                mask = mask | filtered_df[c].astype(str).str.lower().str.contains(search_text, na=False)
        filtered_df = filtered_df[mask]

    # Filtres interactifs enrichis (5 colonnes)
    f_c1, f_c2, f_c3, f_c4, f_c5 = st.columns(5)

    # 1. Filtre Formats Physiques & Packagings
    with f_c1:
        support_options = []
        if 'DVD' in df.columns: support_options.append('DVD')
        if 'BLURAY' in df.columns: support_options.append('Blu-ray')
        if '4K' in df.columns: support_options.append('4K UHD')
        if 'STEELBOOK' in df.columns: support_options.append('Steelbook')
        if 'COFFRET' in df.columns: support_options.append('Coffret')
        if '3D' in df.columns: support_options.append('3D')

        if not support_options and 'jaquette' in df.columns:
            support_options = sorted([x for x in df['jaquette'].dropna().unique() if str(x).strip()])

        selected_supports = st.multiselect("💿 Formats / Supports :", support_options, key="f_supports")
        if selected_supports:
            mask_sup = False
            for sup in selected_supports:
                if sup == 'DVD' and 'DVD' in filtered_df.columns:
                    mask_sup = mask_sup | (filtered_df['DVD'].astype(str).str.upper() == 'OUI')
                elif sup == 'Blu-ray' and 'BLURAY' in filtered_df.columns:
                    mask_sup = mask_sup | (filtered_df['BLURAY'].astype(str).str.upper() == 'OUI')
                elif sup == '4K UHD' and '4K' in filtered_df.columns:
                    mask_sup = mask_sup | (filtered_df['4K'].astype(str).str.upper() == 'OUI')
                elif sup == 'Steelbook' and 'STEELBOOK' in filtered_df.columns:
                    mask_sup = mask_sup | (filtered_df['STEELBOOK'].astype(str).str.upper() == 'OUI')
                elif sup == 'Coffret' and 'COFFRET' in filtered_df.columns:
                    mask_sup = mask_sup | (filtered_df['COFFRET'].astype(str).str.upper() == 'OUI')
                elif sup == '3D' and '3D' in filtered_df.columns:
                    mask_sup = mask_sup | (filtered_df['3D'].astype(str).str.upper() == 'OUI')
                elif 'supports' in filtered_df.columns:
                    mask_sup = mask_sup | filtered_df['supports'].astype(str).str.contains(sup, case=False, na=False)
            filtered_df = filtered_df[mask_sup]

    # 2. Filtre Genres Cinématographiques
    with f_c2:
        if col_genres and col_genres in df.columns:
            genre_set = set()
            for g_val in df[col_genres].dropna():
                for sub_g in str(g_val).split(','):
                    clean_g = sub_g.strip()
                    if clean_g: genre_set.add(clean_g)
            selected_genres = st.multiselect("🎭 Genres du film :", sorted(list(genre_set)), key="f_genres")
            if selected_genres:
                mask_g = False
                for g in selected_genres:
                    mask_g = mask_g | filtered_df[col_genres].astype(str).str.contains(r'\b' + re.escape(g) + r'\b', na=False)
                filtered_df = filtered_df[mask_g]

    # 3. Filtre Note IMDb minimale
    with f_c3:
        if 'rating_num' in df.columns and df['rating_num'].notna().sum() > 0:
            min_r = st.slider("⭐ Note IMDb minimale :", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
            if min_r > 0:
                filtered_df = filtered_df[filtered_df['rating_num'] >= min_r]
        else:
            st.info("Notes IMDb non disponibles.")

    # 4. Filtre Réalisateurs (Top 100)
    with f_c4:
        if col_director and col_director in df.columns:
            top_directors = [d for d in df[col_director].value_counts().head(100).index if str(d).strip()]
            sel_director = st.multiselect("🎬 Réalisateur (Top 100) :", sorted(top_directors), key="f_director")
            if sel_director:
                filtered_df = filtered_df[filtered_df[col_director].isin(sel_director)]

    # 5. Filtre Studios & Sociétés de Production
    with f_c5:
        if col_production and col_production in df.columns:
            all_studios = sorted([str(s).strip() for s in df[col_production].dropna().unique() if str(s).strip()])
            sel_studios = st.multiselect("🏢 Studio / Production :", all_studios, key="f_studios")
            if sel_studios:
                filtered_df = filtered_df[filtered_df[col_production].isin(sel_studios)]

    # Volet de filtres chronologiques & de durée
    with st.expander("⏳ Filtres chronologiques & de durée (Année de sortie & Durée en minutes)", expanded=False):
        flt_y1, flt_y2 = st.columns(2)
        with flt_y1:
            if 'annee_num' in df.columns and df['annee_num'].notna().sum() > 0:
                min_y = int(df['annee_num'].dropna().min())
                max_y = int(df['annee_num'].dropna().max())
                sel_years = st.slider("📅 Période de sortie (Année min - max) :", min_value=min_y, max_value=max_y, value=(min_y, max_y))
                filtered_df = filtered_df[filtered_df['annee_num'].isna() | ((filtered_df['annee_num'] >= sel_years[0]) & (filtered_df['annee_num'] <= sel_years[1]))]
            else:
                st.info("Données d'année non disponibles.")
        with flt_y2:
            if 'duree_num' in df.columns and df['duree_num'].notna().sum() > 0:
                min_d = int(df['duree_num'].dropna().min())
                max_d = int(df['duree_num'].dropna().max())
                sel_duree = st.slider("⏱️ Durée du film (en minutes) :", min_value=min_d, max_value=max_d, value=(min_d, max_d))
                filtered_df = filtered_df[filtered_df['duree_num'].isna() | ((filtered_df['duree_num'] >= sel_duree[0]) & (filtered_df['duree_num'] <= sel_duree[1]))]
            else:
                st.info("Données de durée non disponibles.")

    # --- SÉLECTEUR DE TRI DÉDIÉ (NUMÉRIQUE, CHRONOLOGIQUE, ALPHABÉTIQUE, DURÉE) ---
    st.markdown("---")
    s_col1, s_col2 = st.columns([2, 2])

    sort_options = [
        "ID MyMovie (Numérique 1 ➔ N)",
        "ID MyMovie (Numérique N ➔ 1)",
        "ID Unique (Numérique 1 ➔ N)",
        "ID Unique (Numérique N ➔ 1)",
        "Année de sortie (Du plus récent au plus ancien)",
        "Année de sortie (Du plus ancien au plus récent)",
        "Durée (Du plus long au plus court)",
        "Durée (Du plus court au plus long)",
        "Date d'ajout (Du plus récent au plus ancien)",
        "Date d'ajout (Du plus ancien au plus récent)",
        "Note IMDb (Meilleure note d'abord)",
        "Titre alphabétique (A ➔ Z)",
        "Titre alphabétique (Z ➔ A)",
        "Ordre d'origine du fichier"
    ]
    if 'rating_num' not in df.columns:
        sort_options.remove("Note IMDb (Meilleure note d'abord)")
    if 'annee_num' not in df.columns:
        sort_options.remove("Année de sortie (Du plus récent au plus ancien)")
        sort_options.remove("Année de sortie (Du plus ancien au plus récent)")
    if 'duree_num' not in df.columns:
        sort_options.remove("Durée (Du plus long au plus court)")
        sort_options.remove("Durée (Du plus court au plus long)")

    with s_col1:
        sort_mode = st.selectbox("🔽 Trier l'affichage par :", sort_options, index=0)

    # Exécution du tri sélectionné (tri purement numérique pour les ID, Année et Durée)
    if sort_mode == "ID MyMovie (Numérique 1 ➔ N)":
        if 'ID MyMovie' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by='ID MyMovie', ascending=True, na_position='last')
    elif sort_mode == "ID MyMovie (Numérique N ➔ 1)":
        if 'ID MyMovie' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by='ID MyMovie', ascending=False, na_position='last')
    elif sort_mode == "ID Unique (Numérique 1 ➔ N)":
        if 'ID Unique' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by='ID Unique', ascending=True, na_position='last')
    elif sort_mode == "ID Unique (Numérique N ➔ 1)":
        if 'ID Unique' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by='ID Unique', ascending=False, na_position='last')
    elif sort_mode == "Année de sortie (Du plus récent au plus ancien)":
        if 'annee_num' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by='annee_num', ascending=False, na_position='last')
    elif sort_mode == "Année de sortie (Du plus ancien au plus récent)":
        if 'annee_num' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by='annee_num', ascending=True, na_position='last')
    elif sort_mode == "Durée (Du plus long au plus court)":
        if 'duree_num' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by='duree_num', ascending=False, na_position='last')
    elif sort_mode == "Durée (Du plus court au plus long)":
        if 'duree_num' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by='duree_num', ascending=True, na_position='last')
    elif sort_mode == "Date d'ajout (Du plus récent au plus ancien)":
        if 'date_ajout' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by='date_ajout', ascending=False, na_position='last')
    elif sort_mode == "Date d'ajout (Du plus ancien au plus récent)":
        if 'date_ajout' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by='date_ajout', ascending=True, na_position='last')
    elif sort_mode == "Note IMDb (Meilleure note d'abord)":
        if 'rating_num' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by='rating_num', ascending=False, na_position='last')
    elif sort_mode == "Titre alphabétique (A ➔ Z)":
        filtered_df = filtered_df.sort_values(by=col_titre_fr, key=lambda x: x.astype(str).str.lower(), ascending=True)
    elif sort_mode == "Titre alphabétique (Z ➔ A)":
        filtered_df = filtered_df.sort_values(by=col_titre_fr, key=lambda x: x.astype(str).str.lower(), ascending=False)

    with s_col2:
        st.markdown(f"<div style='padding-top: 28px; font-weight: bold;'>Films affichés : {len(filtered_df):,} / {len(df):,}</div>".replace(',', ' '), unsafe_allow_html=True)

    render_custom_table(filtered_df, key_id="f1_consultation")


# ==============================================================================
# FONCTION 2 : SAISIE CODE-BARRES (5 AVANT / 5 APRÈS - TRI ALPHA)
# ==============================================================================
elif menu == "2° Saisie Code-Barres (5 avant / 5 après - Tri Alpha)":
    st.title("📦 Saisie Code-Barres & Voisinage Alphabétique")
    st.markdown(f"Entrez ou scannez un code-barres : l'inventaire est trié alphabétiquement sur **`{col_titre_fr}`** et affiche le film identifié ainsi que **ses 5 voisins précédents et ses 5 voisins suivants** avec son **ID MyMovie** pour le retrouver physiquement.")

    sorted_df = df.sort_values(
        by=col_titre_fr,
        key=lambda x: x.astype(str).str.lower()
    ).reset_index(drop=True)

    cb_in = st.text_input("Saisissez ou scannez le code-barres :", "").strip()

    if cb_in:
        mask_match = (sorted_df[col_barcode].astype(str).str.strip() == cb_in)
        if not mask_match.any():
            mask_match = sorted_df[col_barcode].astype(str).str.contains(cb_in, na=False)

        match_indices = sorted_df.index[mask_match].tolist()

        if match_indices:
            idx = match_indices[0]
            matched_row = sorted_df.iloc[idx]
            film_name = matched_row[col_titre_fr]

            st.success(f"🎯 Film identifié : **{film_name}** (Position alphabétique **{idx + 1}** sur {len(sorted_df)}).")

            # Fiche détaillée avec localisation physique MyMovie, Année, Durée, Studio & ID Unique
            ci_c1, ci_c2, ci_c3 = st.columns(3)
            with ci_c1:
                if 'ID MyMovie' in matched_row and pd.notna(matched_row['ID MyMovie']):
                    st.markdown(f"**🏷️ ID MyMovie (Boîtier Physique) :** `#{matched_row['ID MyMovie']}`")
                if 'ID Unique' in matched_row and pd.notna(matched_row['ID Unique']):
                    st.markdown(f"**🔑 ID Unique :** `{matched_row['ID Unique']}`")
                if 'Réf Entrée' in matched_row and pd.notna(matched_row['Réf Entrée']) and str(matched_row['Réf Entrée']) != str(matched_row.get('ID Unique')):
                    st.markdown(f"**📑 Réf Entrée :** `{matched_row['Réf Entrée']}`")
                if col_director and pd.notna(matched_row.get(col_director)):
                    st.markdown(f"**🎬 Réalisateur :** {matched_row[col_director]}")
            with ci_c2:
                # Année & Durée
                annee_val = matched_row.get('annee_num') if 'annee_num' in matched_row and pd.notna(matched_row.get('annee_num')) else matched_row.get('annee')
                if pd.notna(annee_val) and str(annee_val).strip():
                    st.markdown(f"**📅 Année de sortie :** `{annee_val}`")

                duree_val = matched_row.get('duree_num') if 'duree_num' in matched_row and pd.notna(matched_row.get('duree_num')) else matched_row.get('duree_minutes')
                if pd.notna(duree_val) and str(duree_val).strip():
                    try:
                        d_int = int(float(str(duree_val)))
                        h = d_int // 60
                        m = d_int % 60
                        st.markdown(f"**⏱️ Durée :** `{d_int} min` ({h}h {m:02d})")
                    except Exception:
                        st.markdown(f"**⏱️ Durée :** `{duree_val} min`")

                if 'supports' in matched_row and pd.notna(matched_row['supports']):
                    st.markdown(f"**💿 Formats :** `{matched_row['supports']}`")
                if col_actors and pd.notna(matched_row.get(col_actors)):
                    st.markdown(f"**👥 Acteurs :** {matched_row[col_actors]}")
                if col_genres and pd.notna(matched_row.get(col_genres)):
                    st.markdown(f"**🎭 Genres :** {matched_row[col_genres]}")
            with ci_c3:
                if col_production and pd.notna(matched_row.get(col_production)) and str(matched_row[col_production]).strip():
                    st.markdown(f"**🏢 Production / Studio :** `{matched_row[col_production]}`")
                if 'date_ajout' in matched_row and pd.notna(matched_row['date_ajout']):
                    d_str = matched_row['date_ajout'].strftime('%d/%m/%Y') if hasattr(matched_row['date_ajout'], 'strftime') else str(matched_row['date_ajout'])
                    st.markdown(f"**📅 Date d'ajout :** {d_str}")
                if 'rating_imdb' in matched_row and pd.notna(matched_row['rating_imdb']):
                    st.markdown(f"**⭐ Note IMDb :** `{matched_row['rating_imdb']} / 10`")
                if 'Lien IMDb' in matched_row and matched_row['Lien IMDb']:
                    st.markdown(f"[🎬 Ouvrir sur IMDb ↗]({matched_row['Lien IMDb']})")

            st.markdown("---")
            start_i = max(0, idx - 5)
            end_i = min(len(sorted_df), idx + 6)

            subset = sorted_df.iloc[start_i:end_i].copy()
            subset.insert(0, "Position", [i + 1 for i in range(start_i, end_i)])
            subset.insert(1, "Repère", ["⭐ FILM CIBLÉ" if i == idx else "" for i in range(start_i, end_i)])

            render_custom_table(subset, key_id="f2_barcode_match")
        else:
            st.error(f"❌ Aucun film trouvé avec le code-barres « {cb_in} ».")
            st.markdown("### Inventaire complet (Ordre Alphabétique) :")
            render_custom_table(sorted_df, key_id="f2_barcode_fallback")
    else:
        st.markdown("### Inventaire complet (Ordre Alphabétique) :")
        render_custom_table(sorted_df, key_id="f2_barcode_default")


# ==============================================================================
# FONCTION 3 : RECHERCHE PAR TITRE / RÉALISATEUR / ACTEUR / STUDIO / ANNÉE / ID & IMDB
# ==============================================================================
elif menu == "3° Recherche par titre / réalisateur / acteur & IMDb":
    st.title("🔍 Recherche par Titre, Réalisateur, Acteur, Année, Studio, ou ID")
    st.markdown("Recherche multi-critères instantanée : titres, **Année**, **ID MyMovie**, **ID Unique**, réalisateurs, casting et studios.")

    query = st.text_input("Saisissez un mot-clé (Titre, Année, ID MyMovie, ID Unique, Réalisateur, Acteur ou Studio) :", "").strip()

    title_cols_to_check = [c for c in ['ID MyMovie', 'ID Unique', 'Réf Entrée', col_titre_fr, col_titre_vo, 'annee', 'annee_num', col_director, col_actors, col_production] if c and c in df.columns]

    if query:
        q_clean = query.lower()
        mask_q = False
        for c in title_cols_to_check:
            mask_q = mask_q | df[c].astype(str).str.lower().str.contains(q_clean, na=False)

        res_df = df[mask_q].copy()

        if not res_df.empty:
            st.success(f"✨ **{len(res_df)}** film(s) correspondant(s) trouvé(s) pour « {query} » :")
            render_custom_table(res_df, key_id="f3_title_match")
        else:
            st.warning(f"Aucun film trouvé contenant « {query} ».")
            st.markdown("### Inventaire complet :")
            render_custom_table(df, key_id="f3_title_fallback")
    else:
        st.markdown("### Inventaire complet :")
        render_custom_table(df, key_id="f3_title_default")


# ==============================================================================
# FONCTION 4 : STATISTIQUES COMPLÈTES DE LA COLLECTION
# ==============================================================================
else:
    st.title("📊 Statistiques Complètes de la Collection")
    st.markdown("Tableau de bord exhaustif de votre vidéothèque : Supports, Boîtiers My Movies, Décennies, Durées, Genres, Réalisateurs, Acteurs, Studios et Notes IMDb.")

    tot_entries = len(df)
    tot_uniques = df[col_titre_fr].nunique() if col_titre_fr in df.columns else tot_entries
    tot_mymovies_boxes = df['ID MyMovie'].nunique() if 'ID MyMovie' in df.columns else 0

    has_ratings = 'rating_num' in df.columns and df['rating_num'].notna().sum() > 0
    mean_rating = round(df['rating_num'].mean(), 2) if has_ratings else None
    count_top_rated = int((df['rating_num'] >= 8.0).sum()) if has_ratings else 0
    count_directors = df[col_director].nunique() if col_director and col_director in df.columns else 0
    count_studios = df[col_production].dropna().str.strip().nunique() if col_production and col_production in df.columns else 0

    # Statistiques Année & Durée
    has_annee_stats = 'annee_num' in df.columns and df['annee_num'].notna().sum() > 0
    min_year = int(df['annee_num'].dropna().min()) if has_annee_stats else None
    max_year = int(df['annee_num'].dropna().max()) if has_annee_stats else None
    median_year = int(df['annee_num'].dropna().median()) if has_annee_stats else None

    has_duree_stats = 'duree_num' in df.columns and df['duree_num'].notna().sum() > 0
    mean_duree = int(df['duree_num'].dropna().mean()) if has_duree_stats else None

    # --- KPI CARDS ENRICHIES (Ligne 1 & Ligne 2) ---
    st.subheader("📌 Indicateurs Clés")
    kpi_c1 = st.columns(6)
    kpi_c1[0].metric("🎬 Total Titres / Films", f"{tot_entries:,}".replace(',', ' '))
    kpi_c1[1].metric("📦 Boîtiers Physiques My Movies", f"{tot_mymovies_boxes:,}".replace(',', ' '))
    kpi_c1[2].metric("🏢 Studios & Productions", f"{count_studios:,}".replace(',', ' '))
    if has_ratings:
        kpi_c1[3].metric("⭐ Note IMDb Moyenne", f"{mean_rating} / 10")
        kpi_c1[4].metric("🏆 Chefs-d'œuvre (≥ 8/10)", f"{count_top_rated:,}".replace(',', ' '))
    else:
        kpi_c1[3].metric("⭐ Note IMDb", "N/A")
        kpi_c1[4].metric("🏆 Top Notés", "N/A")
    kpi_c1[5].metric("🎥 Réalisateurs répertoriés", f"{count_directors:,}".replace(',', ' '))

    # Seconde rangée d'indicateurs : Année et Durée
    if has_annee_stats or has_duree_stats:
        kpi_c2 = st.columns(4)
        if has_annee_stats:
            kpi_c2[0].metric("📅 Période Cinématographique", f"{min_year} ➔ {max_year}")
            kpi_c2[1].metric("🎯 Année Médiane", f"{median_year}")
        if has_duree_stats:
            h_m = mean_duree // 60
            m_m = mean_duree % 60
            kpi_c2[2].metric("⏱️ Durée Moyenne d'un Film", f"{mean_duree} min ({h_m}h {m_m:02d})")
            fresque_count = int((df['duree_num'] >= 150).sum())
            kpi_c2[3].metric("⏳ Fresques Longues (≥ 2h30)", f"{fresque_count:,}".replace(',', ' '))

    st.markdown("---")

    # --- SECTION SUPPORTS PHYSIQUES & BOÎTIERS ---
    st.subheader("💿 Répartition des Formats Physiques & Packagings")

    count_dvd = int((df['DVD'].astype(str).str.upper() == 'OUI').sum()) if 'DVD' in df.columns else int(pd.to_numeric(df.get('nb_DVD', 0), errors='coerce').fillna(0).sum())
    count_br = int((df['BLURAY'].astype(str).str.upper() == 'OUI').sum()) if 'BLURAY' in df.columns else int(pd.to_numeric(df.get('nb_BLURAY', 0), errors='coerce').fillna(0).sum())
    count_4k = int((df['4K'].astype(str).str.upper() == 'OUI').sum()) if '4K' in df.columns else int(pd.to_numeric(df.get('nb_4K', 0), errors='coerce').fillna(0).sum())
    count_sb = int((df['STEELBOOK'].astype(str).str.upper() == 'OUI').sum()) if 'STEELBOOK' in df.columns else int((df.get('jaquette', '').astype(str).str.lower() == 'steelbook').sum())
    count_cf = int((df['COFFRET'].astype(str).str.upper() == 'OUI').sum()) if 'COFFRET' in df.columns else int((df.get('jaquette', '').astype(str).str.lower() == 'coffret').sum())
    count_3d = int((df['3D'].astype(str).str.upper() == 'OUI').sum()) if '3D' in df.columns else int((df.get('3D', '').astype(str).str.lower() == 'oui').sum())

    sup_kpis = st.columns(6)
    sup_kpis[0].metric("📀 DVD", f"{count_dvd:,}".replace(',', ' '))
    sup_kpis[1].metric("🔵 Blu-ray", f"{count_br:,}".replace(',', ' '))
    sup_kpis[2].metric("⚫ 4K UHD", f"{count_4k:,}".replace(',', ' '))
    sup_kpis[3].metric("🧲 Steelbooks", f"{count_sb:,}".replace(',', ' '))
    sup_kpis[4].metric("📦 Coffrets", f"{count_cf:,}".replace(',', ' '))
    sup_kpis[5].metric("🕶️ Versions 3D", f"{count_3d:,}".replace(',', ' '))

    df_sup_chart = pd.DataFrame({
        "Format": ["DVD", "Blu-ray", "4K UHD", "Steelbook", "Coffret", "3D"],
        "Nombre": [count_dvd, count_br, count_4k, count_sb, count_cf, count_3d]
    }).set_index("Format")
    st.bar_chart(df_sup_chart)

    st.markdown("---")

    # ==========================================================================
    # --- ANALYSES CHRONOLOGIQUES : DÉCENNIES & TRANCHES DE DURÉE ---
    # ==========================================================================
    st.subheader("⏳ Analyses Temporelles : Décennies de Sortie & Tranches de Durée")
    an_c1, an_c2 = st.columns(2)

    with an_c1:
        st.markdown("### 📅 Répartition par Décennie de Sortie")
        if has_annee_stats:
            decades = (df['annee_num'] // 10 * 10).dropna().astype(int).astype(str) + 's'
            decades_vc = decades.value_counts().sort_index()
            st.bar_chart(decades_vc)
        else:
            st.info("Données d'année non disponibles.")

    with an_c2:
        st.markdown("### ⏱️ Répartition par Tranches de Durée")
        if has_duree_stats:
            bins = [0, 89, 109, 139, 999]
            labels = ['< 90 min (Court / Animé)', '90 - 110 min (Standard)', '110 - 140 min (Long)', '≥ 140 min (Fresque / Épique)']
            duree_cuts = pd.cut(df['duree_num'], bins=bins, labels=labels)
            st.bar_chart(duree_cuts.value_counts().sort_index())
        else:
            st.info("Données de durée non disponibles.")

    st.markdown("---")

    # ==========================================================================
    # --- LES GRANDS TOPS : GENRES (15), RÉALISATEURS (60), ACTEURS (35), STUDIOS (20) ---
    # ==========================================================================
    st.subheader("🏆 Les Grands Tops de la Vidéothèque (Tri Décroissant : Tous les noms visibles)")

    col_opt1, col_opt2 = st.columns([3, 1])
    with col_opt1:
        chart_orient_choice = st.radio(
            "📐 Disposition des graphiques :",
            [
                "Barres Horizontales (De haut en bas : recommandé, 100% des noms lisibles à l'horizontale)",
                "Barres Verticales (De gauche à droite : du plus grand au plus petit)"
            ],
            index=0,
            horizontal=True
        )
    opt_orient = "Horizontale" if "Horizontales" in chart_orient_choice else "Verticale"

    st.markdown("---")

    # 1. TOP 15 GENRES
    st.markdown("### 🎭 Top 15 des Genres Cinématographiques")
    if col_genres and col_genres in df.columns:
        genres_list = [g.strip() for val in df[col_genres].dropna() for g in re.split(r'[,;]', str(val)) if g.strip()]
        if genres_list:
            c_genres = Counter(genres_list).most_common(15)
            df_genres = pd.DataFrame(c_genres, columns=['Genre', 'Nombre de Films'])
            render_descending_bar_chart(df_genres, 'Genre', 'Nombre de Films', height=400, orientation=opt_orient, color_scheme="reds")

            with st.expander("📋 Voir le tableau détaillé du Top 15 des Genres (Rang 1 à 15)"):
                df_genres_tbl = df_genres.copy()
                df_genres_tbl.insert(0, 'Rang', range(1, len(df_genres_tbl) + 1))
                st.dataframe(df_genres_tbl, hide_index=True, use_container_width=True)
    else:
        st.info("Information sur les genres non disponible dans ce fichier.")

    st.markdown("---")

    # 2. TOP 60 RÉALISATEURS
    st.markdown("### 🎬 Top 60 des Réalisateurs les plus présents")
    if col_director and col_director in df.columns:
        dirs_list = [d.strip() for val in df[col_director].dropna() for d in re.split(r'[,;]', str(val)) if d.strip() and d.strip().lower() not in ['nan', 'none', 'null', '-']]
        if dirs_list:
            c_dirs = Counter(dirs_list).most_common(60)
            df_dirs = pd.DataFrame(c_dirs, columns=['Réalisateur', 'Films'])
            render_descending_bar_chart(df_dirs, 'Réalisateur', 'Films', height=550, orientation=opt_orient, color_scheme="blues")

            with st.expander("📋 Voir le tableau détaillé du Top 60 des Réalisateurs (Rang 1 à 60)"):
                df_dirs_tbl = df_dirs.copy()
                df_dirs_tbl.insert(0, 'Rang', range(1, len(df_dirs_tbl) + 1))
                st.dataframe(df_dirs_tbl, hide_index=True, use_container_width=True)
    else:
        st.info("Information sur les réalisateurs non disponible dans ce fichier.")

    st.markdown("---")

    # 3. TOP 35 ACTEURS
    st.markdown("### 👥 Top 35 des Acteurs les plus fréquents")
    if col_actors and col_actors in df.columns:
        actors_list = [a.strip() for val in df[col_actors].dropna() for a in re.split(r'[,;]', str(val)) if a.strip() and a.strip().lower() not in ['nan', 'none', 'null', '-']]
        if actors_list:
            c_actors = Counter(actors_list).most_common(35)
            df_actors = pd.DataFrame(c_actors, columns=['Acteur', 'Films'])
            render_descending_bar_chart(df_actors, 'Acteur', 'Films', height=500, orientation=opt_orient, color_scheme="tealblues")

            with st.expander("📋 Voir le tableau détaillé du Top 35 des Acteurs (Rang 1 à 35)"):
                df_actors_tbl = df_actors.copy()
                df_actors_tbl.insert(0, 'Rang', range(1, len(df_actors_tbl) + 1))
                st.dataframe(df_actors_tbl, hide_index=True, use_container_width=True)
    else:
        st.info("Information sur les acteurs non disponible dans ce fichier.")

    st.markdown("---")

    # 4. TOP 20 STUDIOS & SOCIÉTÉS DE PRODUCTION
    if col_production and col_production in df.columns:
        st.markdown("### 🏢 Top 20 des Sociétés de Production & Studios")
        prod_counts = df[col_production].dropna().astype(str).str.strip().value_counts().head(20)
        if not prod_counts.empty:
            df_studios = pd.DataFrame({'Studio': prod_counts.index, 'Films': prod_counts.values})
            render_descending_bar_chart(df_studios, 'Studio', 'Films', height=450, orientation=opt_orient, color_scheme="purples")

            with st.expander("📋 Voir le tableau détaillé de tous les Studios répertoriés (Décroissant)"):
                all_prod = df[col_production].dropna().astype(str).str.strip().value_counts()
                df_all_studios = pd.DataFrame({'Studio': all_prod.index, 'Films': all_prod.values})
                df_all_studios.insert(0, 'Rang', range(1, len(df_all_studios) + 1))
                st.dataframe(df_all_studios, hide_index=True, use_container_width=True)

        st.markdown("---")

    # --- AUTRES ANALYSES : NOTES IMDB & CHRONOLOGIE DES AJOUTS ---
    g_c1, g_c2 = st.columns(2)

    with g_c1:
        st.subheader("⭐ Distribution des Notes IMDb")
        if has_ratings:
            bins = [0, 4.9, 5.9, 6.9, 7.9, 10.0]
            labels = ['< 5.0 (Médiocre)', '5.0 - 5.9 (Moyen)', '6.0 - 6.9 (Bon)', '7.0 - 7.9 (Très Bon)', '≥ 8.0 (Chef-d’œuvre)']
            rating_cuts = pd.cut(df['rating_num'], bins=bins, labels=labels)
            st.bar_chart(rating_cuts.value_counts().sort_index())
        else:
            st.info("Notes IMDb non disponibles dans ce fichier.")

    with g_c2:
        st.subheader("📅 Évolution des Ajouts par Année")
        if 'date_ajout' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['date_ajout']):
                annee_counts = df['date_ajout'].dt.year.value_counts().sort_index()
            else:
                annees = df['date_ajout'].astype(str).str.extract(r'(\d{4})$')[0].dropna()
                annee_counts = annees.value_counts().sort_index()
            if not annee_counts.empty:
                st.bar_chart(annee_counts)
        else:
            st.info("Dates d'ajout non disponibles dans ce fichier.")

    st.markdown("---")
    st.subheader("🌍 Répartition Géographique (Top 10 Pays)")
    if 'pays' in df.columns:
        pays_counts = df['pays'].value_counts().head(10)
        df_pays = pd.DataFrame({
            "Pays": pays_counts.index,
            "Titres": pays_counts.values,
            "Part (%)": [round(v / max(tot_entries, 1) * 100, 1) for v in pays_counts.values]
        })
        st.dataframe(df_pays, hide_index=True, use_container_width=True)

    # --- ONGLETS DE ZOOM INTERACTIF ---
    st.markdown("---")
    st.subheader("🎯 Zooms & Listes Spécifiques")

    tabs = st.tabs([
        "🏆 Chefs-d'œuvre (≥ 8.0)",
        "⏳ Fresques & Films Longs (≥ 150 min)",
        "🎞️ Classiques (≤ 1980)",
        "🧲 Steelbooks",
        "📦 Coffrets",
        "⚫ 4K UHD",
        "🕶️ Versions 3D",
        "🎬 Grands Réalisateurs",
        "🏢 Studios & Productions"
    ])

    with tabs[0]:
        if has_ratings:
            top_movies = df[df['rating_num'] >= 8.0].sort_values(by='rating_num', ascending=False)
            st.info(f"**{len(top_movies)}** films d'exception notés 8.0 ou plus sur IMDb :")
            render_custom_table(top_movies, key_id="zoom_top_rated")
        else:
            st.warning("Notes IMDb indisponibles.")

    with tabs[1]:
        if has_duree_stats:
            long_movies = df[df['duree_num'] >= 150].sort_values(by='duree_num', ascending=False)
            st.info(f"**{len(long_movies)}** films épiques ou fresques de 2h30 ou plus (≥ 150 min) :")
            render_custom_table(long_movies, key_id="zoom_long_movies")
        else:
            st.warning("Durées indisponibles.")

    with tabs[2]:
        if has_annee_stats:
            classic_movies = df[df['annee_num'] <= 1980].sort_values(by='annee_num', ascending=True)
            st.info(f"**{len(classic_movies)}** grands classiques du cinéma sortis jusqu'en 1980 :")
            render_custom_table(classic_movies, key_id="zoom_classic_movies")
        else:
            st.warning("Années de sortie indisponibles.")

    with tabs[3]:
        sb_rows = df[df['STEELBOOK'].astype(str).str.upper() == 'OUI'] if 'STEELBOOK' in df.columns else df[df.get('jaquette', '').astype(str).str.lower() == 'steelbook']
        st.info(f"**{len(sb_rows)}** éditions Steelbook répertoriées :")
        render_custom_table(sb_rows, key_id="zoom_steelbooks")

    with tabs[4]:
        cf_rows = df[df['COFFRET'].astype(str).str.upper() == 'OUI'] if 'COFFRET' in df.columns else df[(df.get('jaquette', '').astype(str).str.lower() == 'coffret') | (pd.to_numeric(df.get('nb_Coffret', 0), errors='coerce') > 0)]
        st.info(f"**{len(cf_rows)}** films issus de Coffrets :")
        render_custom_table(cf_rows, key_id="zoom_coffrets")

    with tabs[5]:
        k4_rows = df[df['4K'].astype(str).str.upper() == 'OUI'] if '4K' in df.columns else df[pd.to_numeric(df.get('nb_4K', 0), errors='coerce') > 0]
        st.info(f"**{len(k4_rows)}** éditions 4K Ultra HD :")
        render_custom_table(k4_rows, key_id="zoom_4k")

    with tabs[6]:
        d3_rows = df[df['3D'].astype(str).str.upper() == 'OUI'] if '3D' in df.columns else df[df.get('3D', '').astype(str).str.lower() == 'oui']
        st.info(f"**{len(d3_rows)}** éditions 3D :")
        render_custom_table(d3_rows, key_id="zoom_3d")

    with tabs[7]:
        if col_director and col_director in df.columns:
            top_5_directors = df[col_director].dropna().value_counts().head(5).index.tolist()
            cult_movies = df[df[col_director].isin(top_5_directors)].sort_values(by=col_director)
            st.info(f"Films des 5 réalisateurs les plus représentés ({', '.join(top_5_directors)}) :")
            render_custom_table(cult_movies, key_id="zoom_cult_directors")
        else:
            st.warning("Réalisateurs indisponibles.")

    with tabs[8]:
        if col_production and col_production in df.columns:
            st.info("Explorez les films par studio ou maison de production :")
            available_studios = sorted([s for s in df[col_production].dropna().unique() if str(s).strip()])
            selected_zoom_studio = st.selectbox("Sélectionnez un studio :", available_studios, key="zoom_sel_studio")
            if selected_zoom_studio:
                studio_films = df[df[col_production].astype(str).str.strip() == selected_zoom_studio]
                st.success(f"**{len(studio_films)}** films produits/édités par **{selected_zoom_studio}** :")
                render_custom_table(studio_films, key_id="zoom_studio_films")
        else:
            st.warning("Information studio non disponible.")

import streamlit as st
import pandas as pd
import io
import os
import re
import urllib.parse

st.set_page_config(
    page_title="Gestionnaire d'Inventaire Films & Coffrets",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FONCTION ROBUSTE DE LECTURE DU CSV (DTYPE=STR POUR PRÉSERVER LES CODES-BARRES) ---
def load_and_prepare_csv(filepath_or_buffer):
    """Charge le fichier CSV sans altérer les codes-barres (dtype=str) et prépare les colonnes."""
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

    # Nettoyage des noms de colonnes
    df_loaded.columns = [str(c).strip().replace('\ufeff', '') for c in df_loaded.columns]

    # Suppression de la colonne date_steelbook si présente
    cols_drop = [c for c in df_loaded.columns if 'date_steelbook' in c.lower() or 'date steelbook' in c.lower()]
    if cols_drop:
        df_loaded = df_loaded.drop(columns=cols_drop)

    # Nettoyage systématique des codes-barres (évite .0 ou notations scientifiques)
    for c in df_loaded.columns:
        if any(k in c.lower() for k in ['code', 'barre', 'ean']):
            df_loaded[c] = df_loaded[c].fillna('').astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            df_loaded.loc[df_loaded[c].isin(['nan', 'none', 'null']), c] = ''

    # Normalisation du titre français et du titre VO
    col_t_fr = next((c for c in df_loaded.columns if 'titre' in c.lower() and ('_fr' in c.lower() or 'francais' in c.lower() or 'vf' in c.lower())), None)
    if not col_t_fr:
        col_t_fr = next((c for c in df_loaded.columns if 'titre' in c.lower() or 'title' in c.lower()), df_loaded.columns[0])

    col_t_vo = next((c for c in df_loaded.columns if 'titre' in c.lower() and ('_en' in c.lower() or 'imdb' in c.lower() or 'global' in c.lower() or 'vo' in c.lower()) and c != col_t_fr), None)

    # Recherche des colonnes IMDb existantes
    col_raw_lien = next((c for c in df_loaded.columns if 'imdb' in c.lower() and ('lien' in c.lower() or 'url' in c.lower() or 'link' in c.lower())), None)
    col_raw_id = next((c for c in df_loaded.columns if 'imdb' in c.lower() and 'id' in c.lower() and c != col_raw_lien), None)

    def generate_imdb_link(row):
        # 1. Lien direct existant
        if col_raw_lien and pd.notna(row.get(col_raw_lien)):
            l_val = str(row[col_raw_lien]).strip()
            if l_val.startswith(('http://', 'https://')):
                return l_val
            tt_m = re.search(r'(tt\d+)', l_val)
            if tt_m:
                return f"https://www.imdb.com/title/{tt_m.group(1)}/"

        # 2. Code TT existant
        if col_raw_id and pd.notna(row.get(col_raw_id)):
            id_val = str(row[col_raw_id]).strip()
            tt_m = re.search(r'(tt\d+)', id_val)
            if tt_m:
                return f"https://www.imdb.com/title/{tt_m.group(1)}/"

        # 3. Fallback : lien de recherche IMDb direct avec le titre
        t_val = str(row.get(col_t_fr, '')).strip()
        if not t_val and col_t_vo:
            t_val = str(row.get(col_t_vo, '')).strip()

        if t_val and t_val.lower() not in ['nan', '', 'none']:
            return f"https://www.imdb.com/find/?q={urllib.parse.quote_plus(t_val)}"

        return "https://www.imdb.com/"

    # Remplacement de l'ancienne colonne lien_imdb ou création de Lien IMDb
    df_loaded['Lien IMDb'] = df_loaded.apply(generate_imdb_link, axis=1)
    if col_raw_lien and col_raw_lien in df_loaded.columns and col_raw_lien != 'Lien IMDb':
        df_loaded = df_loaded.drop(columns=[col_raw_lien])

    return df_loaded


# --- DÉTECTION ET CHARGEMENT DU FICHIER ---
st.sidebar.header("📂 Données de l'Inventaire")

candidate_paths = [
    r"/uploads/inventaire_complet_coffrets.csv",
    r"C:\_Test_Code_Python\uploads\inventaire_complet_coffrets.csv",
    r"C:\_Test_Code_Python\uploads\outputs\inventaire_complet_coffrets.csv",
    r"C:\_Test_Code_Python\inventaire_complet_coffrets.csv",
    r"inventaire_complet_coffrets.csv",
    r"C:\_Test_Code_Python\uploads\outputs",
    r"C:\_Test_Code_Python\uploads",
    r"C:\_Test_Code_Python",
    r"./uploads",
    r"./outputs",
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
                # Privilégier un fichier avec "coffret" ou "inventaire"
                files.sort(key=lambda x: ('coffret' in x.lower(), 'inventaire' in x.lower(), os.path.getmtime(x)), reverse=True)
                if files:
                    loaded_df = load_and_prepare_csv(files[0])
                    loaded_name = os.path.basename(files[0])
                    break
            except Exception:
                continue

    if loaded_df is None or loaded_df.empty:
        # Données de démonstration
        data_demo = {
            'ordre': [str(i) for i in range(1, 41)],
            'code_barre': [f"3700000{i:03d}" for i in range(1, 41)],
            'titre_fr': [f"Film Exemple {i}" for i in range(1, 41)],
            'pays': ['France' if i % 2 == 0 else 'Belgium' for i in range(1, 41)],
            'nb_DVD': ['1' if i % 2 == 0 else '0' for i in range(1, 41)],
            'nb_BLURAY': ['1' if i % 3 == 0 else '0' for i in range(1, 41)],
            'nb_4K': ['1' if i % 5 == 0 else '0' for i in range(1, 41)],
            'nb_Coffret': ['1' if i % 10 == 0 else '0' for i in range(1, 41)],
            '3D': ['oui' if i % 8 == 0 else 'non' for i in range(1, 41)],
            'jaquette': ['Coffret' if i % 10 == 0 else ('Steelbook' if i % 4 == 0 else 'Plastic case') for i in range(1, 41)]
        }
        loaded_df = pd.DataFrame(data_demo)
        loaded_df['Lien IMDb'] = [f"https://www.imdb.com/title/tt{1000000+i}/" for i in range(1, 41)]
        loaded_name = "Jeu de test démo"

    st.session_state['df_films'] = loaded_df
    st.session_state['source_name'] = loaded_name

# Option pour recharger ou spécifier un autre fichier
input_custom_path = st.sidebar.text_input(
    "Chemin du dossier / fichier CSV :",
    value=st.session_state.get('source_name', r"C:\_Test_Code_Python\uploads\outputs")
)

if st.sidebar.button("🔄 Charger / Recharger ce fichier"):
    if os.path.exists(input_custom_path):
        target = input_custom_path
        if os.path.isdir(input_custom_path):
            candidates = [os.path.join(input_custom_path, f) for f in os.listdir(input_custom_path) if f.lower().endswith(('.csv', '.txt'))]
            candidates.sort(key=lambda x: ('coffret' in x.lower(), os.path.getmtime(x)), reverse=True)
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

# Import par upload
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
st.sidebar.info(f"📊 Fichier actif : **{st.session_state.get('source_name', 'Inconnu')}**\n\n📌 **{len(df):,}** enregistrements".replace(',', ' '))

# Export CSV
csv_export = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="💾 Exporter en CSV",
    data=csv_export,
    file_name="inventaire_films_export.csv",
    mime="text/csv"
)

# --- NAVIGATION ---
st.sidebar.markdown("---")
st.sidebar.header("🧭 Menu des Fonctions")
menu = st.sidebar.radio("Sélectionnez la fonction :", [
    "1° Consultation globale avec filtres",
    "2° Saisie Code-Barres (5 avant / 5 après - Tri Alpha)",
    "3° Recherche par titre (Français / Global) & IMDb",
    "4° Statistiques complètes de la collection"
])

# Identifications des colonnes clés
col_titre_fr = next((c for c in df.columns if 'titre' in c.lower() and ('_fr' in c.lower() or 'francais' in c.lower() or 'vf' in c.lower())), None)
if not col_titre_fr:
    col_titre_fr = next((c for c in df.columns if 'titre' in c.lower() or 'title' in c.lower()), df.columns[0])

col_titre_vo = next((c for c in df.columns if 'titre' in c.lower() and ('_en' in c.lower() or 'imdb' in c.lower() or 'global' in c.lower() or 'vo' in c.lower()) and c != col_titre_fr), None)

col_barcode = next((c for c in df.columns if any(k in c.lower() for k in ['code', 'barre', 'ean'])), df.columns[0])

# Fonction standardisée pour l'affichage de tableaux interactifs
def render_custom_table(dataframe, key_id):
    if dataframe.empty:
        st.warning("Aucun résultat ne correspond aux critères sélectionnés.")
        return

    # Configuration Streamlit avec liens IMDb cliquables
    col_cfg = {
        "Lien IMDb": st.column_config.LinkColumn(
            "Lien IMDb",
            help="Ouvrir la fiche officielle du film sur IMDb",
            display_text="🎬 Voir sur IMDb ↗",
            width="medium"
        )
    }

    for col in dataframe.columns:
        if any(w in col.lower() for w in ['titre', 'title', 'nom']):
            col_cfg[col] = st.column_config.TextColumn(col, width="large")
        elif any(w in col.lower() for w in ['code', 'barre', 'ean']):
            col_cfg[col] = st.column_config.TextColumn(col, width="medium")

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
        column_config=col_cfg,
        key=f"tab_view_{key_id}"
    )


# ==============================================================================
# FONCTION 1 : CONSULTATION GLOBALE AVEC FILTRES
# ==============================================================================
if menu == "1° Consultation globale avec filtres":
    st.title("🗂️ Consultation Globale & Filtres Multi-critères")
    st.markdown("Recherchez et filtrez l'inventaire en direct selon les types de jaquette, supports, pays, etc.")

    filtered_df = df.copy()

    # Barre de recherche textuelle rapide
    search_text = st.text_input("🔍 Recherche rapide par mot-clé (Titre, Réalisateur, etc.) :", "").strip().lower()
    if search_text:
        mask = False
        for c in [col_titre_fr, col_titre_vo, col_barcode]:
            if c and c in filtered_df.columns:
                mask = mask | filtered_df[c].astype(str).str.lower().str.contains(search_text, na=False)
        filtered_df = filtered_df[mask]

    # Filtres interactifs par colonnes
    f_cols = st.columns(4)

    # 1. Filtre Jaquette
    if 'jaquette' in df.columns:
        with f_cols[0]:
            jaq_vals = sorted([x for x in df['jaquette'].dropna().unique() if str(x).strip()])
            sel_jaq = st.multiselect("Type de Jaquette :", jaq_vals, key="f_jaq")
            if sel_jaq:
                filtered_df = filtered_df[filtered_df['jaquette'].isin(sel_jaq)]

    # 2. Filtre Support Présent (DVD, Blu-ray, 4K, Coffret)
    with f_cols[1]:
        sup_choices = []
        if 'nb_DVD' in df.columns: sup_choices.append("DVD")
        if 'nb_BLURAY' in df.columns: sup_choices.append("Blu-ray")
        if 'nb_4K' in df.columns: sup_choices.append("4K UHD")
        if 'nb_Coffret' in df.columns: sup_choices.append("Coffret")

        sel_sup = st.multiselect("Format présent :", sup_choices, key="f_sup")
        if sel_sup:
            mask_s = pd.Series(False, index=filtered_df.index)
            if "DVD" in sel_sup and 'nb_DVD' in filtered_df.columns:
                mask_s = mask_s | (pd.to_numeric(filtered_df['nb_DVD'], errors='coerce').fillna(0) > 0)
            if "Blu-ray" in sel_sup and 'nb_BLURAY' in filtered_df.columns:
                mask_s = mask_s | (pd.to_numeric(filtered_df['nb_BLURAY'], errors='coerce').fillna(0) > 0)
            if "4K UHD" in sel_sup and 'nb_4K' in filtered_df.columns:
                mask_s = mask_s | (pd.to_numeric(filtered_df['nb_4K'], errors='coerce').fillna(0) > 0)
            if "Coffret" in sel_sup and 'nb_Coffret' in filtered_df.columns:
                mask_s = mask_s | (pd.to_numeric(filtered_df['nb_Coffret'], errors='coerce').fillna(0) > 0)
            filtered_df = filtered_df[mask_s]

    # 3. Filtre 3D
    if '3D' in df.columns:
        with f_cols[2]:
            vals_3d = sorted([x for x in df['3D'].dropna().unique() if str(x).strip()])
            sel_3d = st.multiselect("Version 3D :", vals_3d, key="f_3d")
            if sel_3d:
                filtered_df = filtered_df[filtered_df['3D'].isin(sel_3d)]

    # 4. Filtre Pays
    if 'pays' in df.columns:
        with f_cols[3]:
            pays_vals = [p for p in df['pays'].value_counts().index if str(p).strip()][:15]
            sel_pays = st.multiselect("Pays d'édition :", pays_vals, key="f_pays")
            if sel_pays:
                filtered_df = filtered_df[filtered_df['pays'].isin(sel_pays)]

    st.markdown(f"**Films correspondants affichés :** `{len(filtered_df):,}` / `{len(df):,}`".replace(',', ' '))
    render_custom_table(filtered_df, key_id="f1_consultation")


# ==============================================================================
# FONCTION 2 : SAISIE CODE-BARRES (5 AVANT / 5 APRÈS - TRI ALPHA)
# ==============================================================================
elif menu == "2° Saisie Code-Barres (5 avant / 5 après - Tri Alpha)":
    st.title("📦 Saisie Code-Barres & Voisinage Alphabétique")
    st.markdown(f"Entrez ou scannez un code-barres : l'inventaire est trié alphabétiquement sur **`{col_titre_fr}`** et affiche le film identifié ainsi que **ses 5 voisins précédents et ses 5 voisins suivants**.")

    # Tri par ordre alphabétique insensible à la casse
    sorted_df = df.sort_values(
        by=col_titre_fr,
        key=lambda x: x.astype(str).str.lower()
    ).reset_index(drop=True)

    cb_in = st.text_input("Saisissez ou scannez le code-barres :", "").strip()

    if cb_in:
        # Recherche exacte ou partielle sur le code-barre nettoyé
        mask_match = (sorted_df[col_barcode].astype(str).str.strip() == cb_in)
        if not mask_match.any():
            mask_match = sorted_df[col_barcode].astype(str).str.contains(cb_in, na=False)

        match_indices = sorted_df.index[mask_match].tolist()

        if match_indices:
            idx = match_indices[0]
            film_name = sorted_df.iloc[idx][col_titre_fr]
            st.success(f"🎯 Film identifié : **{film_name}** (Position **{idx + 1}** sur {len(sorted_df)}).")

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
# FONCTION 3 : RECHERCHE PAR TITRE & IMDB
# ==============================================================================
elif menu == "3° Recherche par titre (Français / Global) & IMDb":
    st.title("🔍 Recherche par Titre & Accès Direct IMDb")
    st.markdown("Recherche rapide et insensible à la casse dans les titres en français et originaux.")

    query = st.text_input("Saisissez tout ou partie du titre à rechercher :", "").strip()

    title_cols_to_check = [c for c in [col_titre_fr, col_titre_vo] if c and c in df.columns]

    if query:
        q_clean = query.lower()
        mask_title = False
        for c in title_cols_to_check:
            mask_title = mask_title | df[c].astype(str).str.lower().str.contains(q_clean, na=False)

        res_df = df[mask_title].copy()

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
    st.markdown("Tableau de bord exhaustif des supports physiques, packagings, éditions 3D et coffrets.")

    # Conversions numériques pour les calculs
    tot_entries = len(df)
    tot_uniques = df[col_titre_fr].nunique() if col_titre_fr in df.columns else tot_entries

    sum_dvd = int(pd.to_numeric(df['nb_DVD'], errors='coerce').fillna(0).sum()) if 'nb_DVD' in df.columns else 0
    sum_br = int(pd.to_numeric(df['nb_BLURAY'], errors='coerce').fillna(0).sum()) if 'nb_BLURAY' in df.columns else 0
    sum_4k = int(pd.to_numeric(df['nb_4K'], errors='coerce').fillna(0).sum()) if 'nb_4K' in df.columns else 0
    sum_coffret_num = int(pd.to_numeric(df['nb_Coffret'], errors='coerce').fillna(0).sum()) if 'nb_Coffret' in df.columns else 0

    tot_discs = sum_dvd + sum_br + sum_4k

    # Packagings / Jaquettes
    count_steelbook = int((df['jaquette'].astype(str).str.lower() == 'steelbook').sum()) if 'jaquette' in df.columns else 0
    count_coffret_jaq = int((df['jaquette'].astype(str).str.lower() == 'coffret').sum()) if 'jaquette' in df.columns else 0
    count_plastic = int((df['jaquette'].astype(str).str.lower() == 'plastic case').sum()) if 'jaquette' in df.columns else 0

    # 3D
    count_3d = int((df['3D'].astype(str).str.lower() == 'oui').sum()) if '3D' in df.columns else 0

    # --- KPI CARDS : CHIFFRES CLÉS ---
    st.subheader("📌 Indicateurs Clés de la Vidéothèque")
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("🎬 Total Titres / Entrées", f"{tot_entries:,}".replace(',', ' '), help="Nombre total de lignes dans la base")
    kpi_cols[1].metric("💿 Total Disques Physiques", f"{tot_discs:,}".replace(',', ' '), help="Cumul des DVD + Blu-ray + 4K")
    kpi_cols[2].metric("📀 Nombre de DVD", f"{sum_dvd:,}".replace(',', ' '))
    kpi_cols[3].metric("🔵 Nombre de Blu-ray", f"{sum_br:,}".replace(',', ' '))
    kpi_cols[4].metric("⚫ Nombre de 4K UHD", f"{sum_4k:,}".replace(',', ' '))

    st.markdown("---")

    kpi_cols2 = st.columns(4)
    kpi_cols2[0].metric("📦 Films issus de Coffrets", f"{count_coffret_jaq:,}".replace(',', ' '), help="Titres intégrés dans un format Coffret")
    kpi_cols2[1].metric("🧲 Boîtiers Steelbooks", f"{count_steelbook:,}".replace(',', ' '), help="Éditions boîtier métal")
    kpi_cols2[2].metric("👓 Éditions 3D", f"{count_3d:,}".replace(',', ' '), help="Films compatibles 3D")
    kpi_cols2[3].metric("🎞️ Titres Uniques (VF)", f"{tot_uniques:,}".replace(',', ' '), help="Titres distincts hors doublons d'éditions")

    st.markdown("---")

    # --- SECTION 1 & 2 : RÉPARTITION SUPPORTS ET PACKAGINGS ---
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("💿 Répartition par Support Physique")
        sup_data = {
            "Support": ["DVD", "Blu-ray", "4K Ultra HD"],
            "Nombre de Disques": [sum_dvd, sum_br, sum_4k],
            "Part (%)": [
                round(sum_dvd / max(tot_discs, 1) * 100, 1),
                round(sum_br / max(tot_discs, 1) * 100, 1),
                round(sum_4k / max(tot_discs, 1) * 100, 1)
            ]
        }
        df_sup_stats = pd.DataFrame(sup_data)
        st.dataframe(df_sup_stats, hide_index=True, use_container_width=True)

        chart_sup = pd.DataFrame({
            "Disques": [sum_dvd, sum_br, sum_4k]
        }, index=["DVD", "Blu-ray", "4K Ultra HD"])
        st.bar_chart(chart_sup)

    with col_g2:
        st.subheader("📦 Répartition des Types de Packagings")
        jaq_counts = df['jaquette'].value_counts() if 'jaquette' in df.columns else pd.Series()
        df_jaq_stats = pd.DataFrame({
            "Type de Jaquette": jaq_counts.index,
            "Nombre d'éditions": jaq_counts.values,
            "Part (%)": [round(v / max(tot_entries, 1) * 100, 1) for v in jaq_counts.values]
        })
        st.dataframe(df_jaq_stats, hide_index=True, use_container_width=True)

        chart_jaq = pd.DataFrame({
            "Éditions": jaq_counts.values
        }, index=jaq_counts.index)
        st.bar_chart(chart_jaq)

    st.markdown("---")

    # --- SECTION 3 : HISTORIQUE DES AJOUTS & TOP PAYS ---
    col_g3, col_g4 = st.columns(2)

    with col_g3:
        st.subheader("🌍 Répartition Géographique (Top 10 Pays)")
        if 'pays' in df.columns:
            pays_counts = df['pays'].value_counts().head(10)
            df_pays = pd.DataFrame({
                "Pays": pays_counts.index,
                "Titres": pays_counts.values,
                "Part (%)": [round(v / max(tot_entries, 1) * 100, 1) for v in pays_counts.values]
            })
            st.dataframe(df_pays, hide_index=True, use_container_width=True)

    with col_g4:
        st.subheader("📅 Évolution des Ajouts par Année")
        if 'date_ajout' in df.columns:
            annees = df['date_ajout'].astype(str).str.extract(r'(\d{4})$')[0].dropna()
            annee_counts = annees.value_counts().sort_index()
            if not annee_counts.empty:
                st.bar_chart(annee_counts)

    # --- SECTION 4 : FOCUS COFFRETS & STEELBOOKS ---
    st.markdown("---")
    st.subheader("🎯 Zoom Détaillé sur les Éditions Spéciales")

    tab_coffrets, tab_steelbooks, tab_3d = st.tabs(["📦 Liste des Coffrets", "🧲 Liste des Steelbooks", "👓 Liste des Films 3D"])

    with tab_coffrets:
        coffret_rows = df[
            (df['jaquette'].astype(str).str.lower() == 'coffret') |
            (pd.to_numeric(df.get('nb_Coffret', 0), errors='coerce').fillna(0) > 0)
        ]
        st.info(f"**{len(coffret_rows)}** titres répertoriés dans un coffret :")
        render_custom_table(coffret_rows, key_id="stats_coffrets")

    with tab_steelbooks:
        steelbook_rows = df[df['jaquette'].astype(str).str.lower() == 'steelbook']
        st.info(f"**{len(steelbook_rows)}** éditions Steelbook dans la collection :")
        render_custom_table(steelbook_rows, key_id="stats_steelbooks")

    with tab_3d:
        rows_3d = df[df['3D'].astype(str).str.lower() == 'oui']
        st.info(f"**{len(rows_3d)}** films disponibles en 3D :")
        render_custom_table(rows_3d, key_id="stats_3d")

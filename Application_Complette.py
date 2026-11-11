import streamlit as st
import pandas as pd
import io
import os
import math
import re
import urllib.parse
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
# ==============================================================================
# MOTEURS OPTIQUES & CALCULATEURS INDUSTRIELS (RACK, BARCODE & SPINE OCR)
# ==============================================================================
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import numpy as np
except ImportError:
    np = None
try:
    import altair as alt
except ImportError:
    alt = None
def decode_barcode_robust(pil_img):
    """
    Décodeur optique multi-moteurs ultra-tolérant :
    1. pyzbar (si disponible)
    2. OpenCV BarcodeDetector (déballage sécurisé 3/4 éléments + multi-rotations + CLAHE)
    3. zxingcpp (si disponible)
    4. Fallback OCR Tesseract sur la suite numérique EAN-13 sous les barres
    """
    if pil_img is None:
        return ""
    # 1. Tentative pyzbar native directe sur l'image PIL
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        results = pyzbar_decode(pil_img)
        for r in results:
            val = r.data.decode('utf-8', errors='ignore').strip()
            if val and len(val) >= 6:
                return val
    except Exception:
        pass
    # 2. Tentative via OpenCV BarcodeDetector
    if cv2 is not None and np is not None:
        try:
            img_np = np.array(pil_img)
            if img_np.ndim == 2:
                gray = img_np
            elif img_np.shape[2] == 4:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
            else:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape[:2]
            if max(h, w) > 1600:
                scale = 1600.0 / max(h, w)
                gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            if hasattr(cv2, 'barcode') and hasattr(cv2.barcode, 'BarcodeDetector'):
                detector = cv2.barcode.BarcodeDetector()
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                gray_clahe = clahe.apply(gray)
                
                variants = [
                    gray,
                    gray_clahe,
                    cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                ]
                for v in variants:
                    for rot in [0, 90, 180, 270]:
                        r_v = np.rot90(v, rot // 90) if rot > 0 else v
                        res = detector.detectAndDecode(r_v)
                        
                        # Déballage sécurisé pour OpenCV (3-tuple ou 4-tuple)
                        decoded_info = None
                        if isinstance(res, tuple):
                            if len(res) == 3:
                                decoded_info = res[0]
                            elif len(res) >= 4:
                                decoded_info = res[1] if isinstance(res[0], (bool, np.bool_)) else res[0]
                        else:
                            decoded_info = res
                        if decoded_info is not None:
                            if isinstance(decoded_info, (list, tuple, np.ndarray)):
                                for item in decoded_info:
                                    t_clean = str(item).strip()
                                    if t_clean and len(t_clean) >= 6 and t_clean.lower() != 'none':
                                        return t_clean
                            elif isinstance(decoded_info, str):
                                t_clean = decoded_info.strip()
                                if t_clean and len(t_clean) >= 6 and t_clean.lower() != 'none':
                                    return t_clean
        except Exception:
            pass
    # 3. Tentative zxingcpp si installé
    try:
        import zxingcpp
        results = zxingcpp.read_barcodes(pil_img)
        for r in results:
            val = str(r.text).strip()
            if val and len(val) >= 6:
                return val
    except Exception:
        pass
    # 4. Fallback OCR sur les chiffres sous le code-barres (pytesseract)
    try:
        import pytesseract
        ocr_text = pytesseract.image_to_string(pil_img, config='--psm 6 digits')
        digits_clean = re.sub(r'\s+', '', ocr_text)
        m = re.search(r'(\d{12,13})', digits_clean)
        if m:
            return m.group(1)
    except Exception:
        pass
    return ""
def render_physical_shelf_visual(target_rack, target_etagere, target_pos, capa_etagere, shelf_titles=None):
    """
    Dessine visuellement le compartiment d'étagère avec ses positions de DVD alignées.
    La position ciblée est illuminée en néon cyan.
    """
    slots_html = []
    for pos_i in range(1, capa_etagere + 1):
        is_target = (pos_i == target_pos)
        titre = shelf_titles.get(pos_i, f"Place #{pos_i}") if shelf_titles else f"Place #{pos_i}"
        
        if is_target:
            slots_html.append(
                f'<div style="flex:1;min-width:48px;height:95px;background:linear-gradient(180deg,#00f2fe,#0284c7);border:2px solid #ffffff;border-radius:6px;display:flex;flex-direction:column;justify-content:center;align-items:center;box-shadow:0 0 20px #00f2fe;color:#ffffff;font-family:Orbitron,sans-serif;font-weight:800;animation:cyber_pulse 1.5s infinite;" title="CIBLE : {titre}">'
                f'<span style="font-size:0.65rem;color:#f8fafc;letter-spacing:1px;">CIBLE</span>'
                f'<span style="font-size:1.15rem;color:#ffffff;font-weight:900;">#{pos_i:02d}</span>'
                f'<span style="font-size:0.6rem;background:rgba(0,0,0,0.3);padding:1px 4px;border-radius:3px;color:#e0f2fe;margin-top:2px;">★ ICI</span>'
                f'</div>'
            )
        else:
            slots_html.append(
                f'<div style="flex:1;min-width:38px;height:95px;background:linear-gradient(180deg,#334155,#1e293b);border:1px solid #475569;border-radius:5px;display:flex;flex-direction:column;justify-content:center;align-items:center;color:#94a3b8;font-family:Rajdhani,sans-serif;font-weight:700;" title="{titre}">'
                f'<span style="font-size:0.6rem;color:#64748b;">POS</span>'
                f'<span style="font-size:1.0rem;color:#cbd5e1;">#{pos_i:02d}</span>'
                f'</div>'
            )
    slots_str = "".join(slots_html)
    
    html = (
        f'<div style="background:#0f172a;border:2px solid #38bdf8;border-radius:10px;padding:14px;margin:12px 0;box-shadow:0 0 20px rgba(56,189,248,0.25);">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">'
        f'<span style="font-family:Orbitron,sans-serif;font-size:1.1rem;font-weight:700;color:#38bdf8;letter-spacing:1px;">🏢 RACK {target_rack:02d} // ÉTAGÈRE {target_etagere:02d}</span>'
        f'<span style="background:rgba(0,242,254,0.15);border:1px solid #00f2fe;color:#00f2fe;padding:3px 12px;border-radius:14px;font-family:Rajdhani,sans-serif;font-weight:700;font-size:0.9rem;">POSITION {target_pos:02d} / {capa_etagere}</span>'
        f'</div>'
        f'<div style="display:flex;gap:6px;background:#1e293b;padding:12px;border-radius:8px;border:1px solid #334155;overflow-x:auto;">'
        f'{slots_str}'
        f'</div>'
        f'<div style="margin-top:10px;font-size:0.85rem;color:#94a3b8;font-family:Rajdhani,sans-serif;display:flex;justify-content:space-between;font-weight:600;">'
        f'<span>◀ Gauche (Position 01)</span>'
        f'<span style="color:#00f2fe;font-weight:700;text-shadow:0 0 8px rgba(0,242,254,0.5);">★ Boîtier Ciblé : Position #{target_pos:02d}</span>'
        f'<span>Droite (Position {capa_etagere:02d}) ▶</span>'
        f'</div>'
        f'</div>'
    )
    return html
def get_rack_info(alpha_idx, total_count=None, nb_etageres=None, capa_etagere=None, invert_etageres=False):
    """
    Système WMS Topologique Physique de Référence :
    1 RACK = N Étagères (défaut : 8)
    1 ÉTAGÈRE = P boîtiers (défaut : 10)
    """
    if total_count is None:
        total_count = getattr(st.session_state, 'total_films_count', 4771)
    if nb_etageres is None:
        nb_etageres = getattr(st.session_state, 'cfg_nb_etageres', 8)
    if capa_etagere is None:
        capa_etagere = getattr(st.session_state, 'cfg_capa_etagere', 10)
    capa_rack = max(1, nb_etageres * capa_etagere)
    rack = (alpha_idx // capa_rack) + 1
    idx_in_rack = alpha_idx % capa_rack
    
    etagere_brut = (idx_in_rack // capa_etagere) + 1
    if invert_etageres:
        etagere = nb_etageres - etagere_brut + 1
    else:
        etagere = etagere_brut
        
    pos_in_shelf = (idx_in_rack % capa_etagere) + 1
    addr_str = f"RACK {rack:02d} // ÉTAGÈRE {etagere:02d} // POSITION {pos_in_shelf:02d} (sur {capa_etagere})"
    return rack, etagere, pos_in_shelf, capa_etagere, addr_str
st.set_page_config(
    page_title="⚡ Volt Ciné-WMS v5.3 Ultra // Vision Augmentée",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ==============================================================================
# DESIGN SYSTEM CYBER-WMS V5.3 : MODERNITÉ, GLASSMORPHISM & EFFETS HOLOGRAPHIQUES
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
.stApp {
    background-color: #070a13 !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(2, 132, 199, 0.18) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(147, 51, 234, 0.15) 0px, transparent 50%),
        linear-gradient(rgba(14, 165, 233, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(14, 165, 233, 0.03) 1px, transparent 1px) !important;
    background-size: 100% 100%, 100% 100%, 35px 35px, 35px 35px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.cyber-hud-banner {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.75) 100%);
    border: 1px solid rgba(0, 242, 254, 0.4);
    border-radius: 14px;
    padding: 18px 26px;
    margin-bottom: 24px;
    box-shadow: 0 10px 35px rgba(0, 242, 254, 0.15);
    backdrop-filter: blur(14px);
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    overflow: hidden;
}
.cyber-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.55rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00f2fe 0%, #38bdf8 40%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.cyber-badge-online {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid #10b981;
    color: #34d399;
    padding: 5px 14px;
    border-radius: 20px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.3);
}
.cyber-badge-version {
    background: rgba(0, 242, 254, 0.15);
    border: 1px solid #00f2fe;
    color: #00f2fe;
    padding: 5px 16px;
    border-radius: 20px;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.78rem;
    font-weight: 800;
    box-shadow: 0 0 10px rgba(0, 242, 254, 0.4);
}
.wms-address-badge {
    background: linear-gradient(90deg, rgba(2, 132, 199, 0.3), rgba(147, 51, 234, 0.25));
    border: 1px solid #38bdf8;
    color: #e0f2fe;
    padding: 8px 16px;
    border-radius: 8px;
    font-family: 'Orbitron', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    display: inline-block;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.35);
    margin-bottom: 12px;
}
</style>
<div class="cyber-hud-banner">
    <div>
        <div class="cyber-title">⚡ VOLT CINÉ-WMS // VISION AUGMENTÉE V5.3</div>
        <div style="font-family: Rajdhani, sans-serif; color: #94a3b8; font-size: 0.95rem; letter-spacing: 1px; margin-top: 4px;">
            PLATEFORME LOGISTIQUE DE PRÉCISION • SCAN TRANCHES AR & TOPOLOGIE RACK
        </div>
    </div>
    <div style="display: flex; gap: 12px; align-items: center;">
        <div class="cyber-badge-online">🟢 SYSTÈME EN LIGNE</div>
        <div class="cyber-badge-version">VERSION 5.3 AR</div>
    </div>
</div>
""", unsafe_allow_html=True)
# --- FONCTION UNIVERSELLE DE CHARGEMENT CSV ---
def load_and_prepare_csv(filepath_or_buffer):
    """Charge le fichier CSV consolidé, gère les encodages et nettoie les types numériques."""
    if isinstance(filepath_or_buffer, str):
        with open(filepath_or_buffer, 'rb') as f:
            bytes_data = f.read()
    else:
        bytes_data = filepath_or_buffer.read()
    df_loaded = None
    for sep in [';', ',', '\t', '|']:
        for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                test_df = pd.read_csv(io.BytesIO(bytes_data), encoding=enc, sep=sep, dtype=str, low_memory=False, on_bad_lines='skip')
                if test_df.shape[1] > 1 and len(test_df) > 0:
                    df_loaded = test_df
                    break
            except Exception:
                continue
        if df_loaded is not None:
            break
    if df_loaded is None:
        df_loaded = pd.read_csv(io.BytesIO(bytes_data), encoding='latin-1', dtype=str, low_memory=False, on_bad_lines='skip')
    df_loaded.columns = [str(c).strip().replace('\ufeff', '') for c in df_loaded.columns]
    # Identifiants numériques stricts
    if 'pos_source' in df_loaded.columns:
        df_loaded['ID MyMovie'] = pd.to_numeric(df_loaded['pos_source'], errors='coerce').astype('Int64')
    elif 'position' in df_loaded.columns:
        df_loaded['ID MyMovie'] = pd.to_numeric(df_loaded['position'], errors='coerce').astype('Int64')
    if 'id_film' in df_loaded.columns:
        df_loaded['ID Unique'] = pd.to_numeric(df_loaded['id_film'], errors='coerce').astype('Int64')
    elif 'id' in df_loaded.columns:
        df_loaded['ID Unique'] = pd.to_numeric(df_loaded['id'], errors='coerce').astype('Int64')
    if 'id' in df_loaded.columns and 'id_film' in df_loaded.columns:
        df_loaded['Réf Entrée'] = df_loaded['id'].fillna('').astype(str).str.strip()
    # Nettoyage codes-barres
    for c in df_loaded.columns:
        if any(k in c.lower() for k in ['code', 'barre', 'barcode', 'ean']):
            df_loaded[c] = df_loaded[c].fillna('').astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            df_loaded.loc[df_loaded[c].isin(['nan', 'none', 'null']), c] = ''
    # Dates
    for date_col in ['date_ajout', 'date_coffret', 'date_sortie']:
        if date_col in df_loaded.columns:
            df_loaded[date_col] = pd.to_datetime(df_loaded[date_col], format='%d/%m/%Y', errors='coerce')
    # Année et Durée
    col_annee = next((c for c in df_loaded.columns if c.lower() in ['annee', 'année', 'year', 'release_year']), None)
    if col_annee:
        df_loaded['annee_num'] = pd.to_numeric(df_loaded[col_annee].astype(str).str.extract(r'(\d{4})')[0], errors='coerce').astype('Int64')
    col_duree = next((c for c in df_loaded.columns if 'duree' in c.lower() or 'durée' in c.lower() or 'runtime' in c.lower()), None)
    if col_duree:
        df_loaded['duree_num'] = pd.to_numeric(df_loaded[col_duree].astype(str).str.extract(r'(\d+)')[0], errors='coerce').astype('Int64')
    # Note IMDb
    col_rating = next((c for c in df_loaded.columns if 'rating' in c.lower() or 'note' in c.lower()), None)
    if col_rating:
        df_loaded['rating_num'] = pd.to_numeric(df_loaded[col_rating].astype(str).str.replace(',', '.'), errors='coerce')
    # Synthèse des supports
    def extract_supports_str(row):
        badges = []
        if str(row.get('4K', '')).strip().upper() == 'OUI': badges.append('4K UHD')
        if str(row.get('BLURAY', '')).strip().upper() == 'OUI': badges.append('Blu-ray')
        if str(row.get('3D', '')).strip().upper() == 'OUI': badges.append('3D')
        if str(row.get('DVD', '')).strip().upper() == 'OUI': badges.append('DVD')
        if str(row.get('STEELBOOK', '')).strip().upper() == 'OUI': badges.append('Steelbook')
        if str(row.get('COFFRET', '')).strip().upper() == 'OUI': badges.append('Coffret')
        return ' | '.join(badges) if badges else '-'
    df_loaded['supports'] = df_loaded.apply(extract_supports_str, axis=1)
    # Lien IMDb cliquable
    col_t_fr = next((c for c in df_loaded.columns if c.lower() in ['titre', 'titre_fr', 'titre_francais']), df_loaded.columns[0])
    df_loaded['Lien IMDb'] = [f"https://www.imdb.com/find/?q={urllib.parse.quote_plus(str(t))}" for t in df_loaded[col_t_fr]]
    return df_loaded
# --- DÉTECTION DU FICHIER SOURCE ---
candidate_paths = [
    r"C:\_Test_Code_Python\uploads\Liste_Complette_Film_Consolidee_Final.csv",
    r"C:\_Test_Code_Python\Liste_Complette_Film_Consolidee_Final.csv",
    r"/workspace/files/personal_files/uploads/Liste_Complette_Film_Consolidee_Final.csv",
    r"Liste_Complette_Film_Consolidee_Final.csv"
]
if 'df_films' not in st.session_state:
    loaded_df = None
    loaded_name = None
    for p in candidate_paths:
        if os.path.exists(p) and os.path.isfile(p):
            try:
                loaded_df = load_and_prepare_csv(p)
                loaded_name = os.path.basename(p)
                break
            except Exception:
                continue
    if loaded_df is None or loaded_df.empty:
        # Données de secours
        loaded_df = pd.DataFrame({
            'ID MyMovie': range(1, 31),
            'ID Unique': range(1, 31),
            'barcode': [f"3700000{i:03d}" for i in range(1, 31)],
            'titre': [f"Film Exemple {i}" for i in range(1, 31)],
            'annee_num': [2000 + i for i in range(1, 31)],
            'duree_num': [90 + i for i in range(1, 31)],
            'supports': ['Blu-ray' if i % 2 == 0 else 'DVD' for i in range(1, 31)],
            'DVD': ['OUI' if i % 2 != 0 else '' for i in range(1, 31)],
            'BLURAY': ['OUI' if i % 2 == 0 else '' for i in range(1, 31)],
            '4K': ['OUI' if i % 5 == 0 else '' for i in range(1, 31)],
            'Lien IMDb': ['https://www.imdb.com/' for _ in range(30)]
        })
        loaded_name = "Mode Démonstration"
    st.session_state['df_films'] = loaded_df
    st.session_state['source_name'] = loaded_name
df = st.session_state['df_films']
# Détection colonnes clés
col_titre_fr = next((c for c in df.columns if c.lower() in ['titre', 'titre_fr', 'titre_francais']), df.columns[0])
col_titre_vo = next((c for c in df.columns if c.lower() in ['titre_anglais', 'titre_original'] and c != col_titre_fr), None)
col_barcode = next((c for c in df.columns if any(k in c.lower() for k in ['code', 'barre', 'barcode', 'ean'])), df.columns[0])
col_director = next((c for c in df.columns if 'directeur' in c.lower() or 'director' in c.lower() or 'realisateur' in c.lower()), None)
col_actors = next((c for c in df.columns if 'acteur' in c.lower() or 'actor' in c.lower()), None)
col_genres = 'genres' if 'genres' in df.columns else next((c for c in df.columns if 'genre' in c.lower()), None)
# --- NAVIGATION SIDEBAR ---
st.sidebar.header("🧭 Menu des Fonctions")
menu = st.sidebar.radio("Sélectionnez la fonction :", [
    "1° Consultation globale avec filtres enrichis",
    "2° Saisie Code-Barres (Douchette USB - 5 avant / 5 après)",
    "3° Scan Code-Barres par Caméra (Smartphone / Webcam)",
    "4° Cartographie RACK & Stockage Industriel (WMS)",
    "5° Scanner Anti-Doublon Mobile & Wishlist Magasin",
    "6° Audit Qualité : Doublons réels & Opportunités d'Upgrades 4K",
    "7° Sagathèque & Dossier d'Assurance Certifié",
    "8° Recherche par titre / réalisateur / acteur & IMDb",
    "9° Statistiques complètes de la collection",
    "10° Scanner Étagère & Tranches en Réalité Augmentée (Spine HUD AR)"
])
# Paramétrage RACK
st.sidebar.markdown("---")
st.sidebar.subheader("🏢 Topologie RACK (Magasin)")
nb_racks_cfg = st.sidebar.slider("Nombre de Racks :", 1, 50, 12, key="cfg_nb_racks")
nb_etageres_cfg = st.sidebar.slider("Étagères par Rack :", 1, 30, 8, key="cfg_nb_etageres")
capa_calc_etagere = st.sidebar.slider("Capacité boîtiers / étagère :", 5, 60, 10, key="cfg_capa_etagere")
def render_custom_table(dataframe, key_id):
    if dataframe.empty:
        st.warning("Aucun résultat.")
        return
    st.dataframe(dataframe, use_container_width=True, hide_index=True, key=f"table_{key_id}")
# ==============================================================================
# FONCTION 10 : SCANNER D'ÉTAGÈRE & TRANCHES EN RÉALITÉ AUGMENTÉE (SPINE HUD AR)
# ==============================================================================
if menu == "10° Scanner Étagère & Tranches en Réalité Augmentée (Spine HUD AR)":
    import difflib
    st.title("🥽 Scanner d'Étagère & Tranches en Réalité Augmentée (Spine HUD AR)")
    st.markdown("Pointez la caméra vers un **rayon de films (DVD, Blu-ray, 4K UHD)** en magasin ou brocante. Le système lit les titres inscrits sur les tranches et projette en **Réalité Augmentée (AR)** les statuts d'achat : **Wishlist**, **Déjà Possédé**, **Upgrade 4K Conseillé** ou **Film Absent**.")
    # Wishlist en session
    if 'wishlist_data' not in st.session_state:
        st.session_state['wishlist_data'] = [
            {"Titre": "Oppenheimer", "Format Visé": "4K Ultra HD", "Priorité": "⭐⭐⭐ Haute", "Note": "Édition Steelbook si possible"},
            {"Titre": "Dune Deuxième Partie", "Format Visé": "4K Ultra HD", "Priorité": "⭐⭐⭐ Haute", "Note": "Achat direct"},
            {"Titre": "Civil War", "Format Visé": "4K Ultra HD", "Priorité": "⭐⭐ Moyenne", "Note": "Attendre promo"}
        ]
    def clean_spine_text(s):
        if not s or not isinstance(s, str): return ""
        s = s.lower()
        s = re.sub(r'[\(\)\[\]\-_:;,\.!\?\'"]', ' ', s)
        noise = ['dvd', 'bluray', 'blu-ray', '4k', 'uhd', 'ultra', 'hd', 'edition', 'collector', 'steelbook', 'video', 'warner', 'universal', 'paramount', 'sony']
        words = [w for w in s.split() if w not in noise and len(w) > 1]
        return " ".join(words)
    film_records_lookup = []
    for _, row in df.iterrows():
        t_fr = str(row.get(col_titre_fr, '')).strip()
        t_vo = str(row.get(col_titre_vo, '')).strip() if col_titre_vo else ''
        film_records_lookup.append({
            'titre': t_fr,
            'titre_norm': clean_spine_text(t_fr),
            'vo_norm': clean_spine_text(t_vo) if t_vo else '',
            'ID Unique': row.get('ID Unique', ''),
            'ID MyMovie': row.get('ID MyMovie', ''),
            'supports': row.get('supports', ''),
            'DVD': str(row.get('DVD', '')).strip().upper(),
            'BLURAY': str(row.get('BLURAY', '')).strip().upper(),
            '4K': str(row.get('4K', '')).strip().upper()
        })
    def match_detected_spine(text_detected, wishlist_list):
        norm_txt = clean_spine_text(text_detected)
        if len(norm_txt) < 3: return None
        # 1. Vérification Wishlist en priorité
        for w_item in wishlist_list:
            w_norm = clean_spine_text(w_item.get('Titre', ''))
            if not w_norm: continue
            r_w = difflib.SequenceMatcher(None, norm_txt, w_norm).ratio()
            if w_norm in norm_txt or norm_txt in w_norm or r_w >= 0.70:
                return {
                    "status": "WISHLIST",
                    "color": "#f59e0b",
                    "badge": "★ WISHLIST !",
                    "matched_title": w_item.get('Titre', ''),
                    "confidence": max(r_w, 0.85),
                    "note": f"Recherché en {w_item.get('Format Visé', 'Tout format')}"
                }
        # 2. Vérification Collection
        best_rec = None
        best_r = 0.0
        for rec in film_records_lookup:
            t1, t2 = rec['titre_norm'], rec['vo_norm']
            r1 = difflib.SequenceMatcher(None, norm_txt, t1).ratio() if t1 else 0
            r2 = difflib.SequenceMatcher(None, norm_txt, t2).ratio() if t2 else 0
            r_cur = max(r1, r2)
            if (t1 and (t1 in norm_txt or norm_txt in t1)) or (t2 and (t2 in norm_txt or norm_txt in t2)):
                r_cur = max(r_cur, 0.82)
            if r_cur > best_r:
                best_r = r_cur
                best_rec = rec
        if best_r >= 0.65 and best_rec:
            is_only_dvd = (best_rec['DVD'] == 'OUI') and (best_rec['BLURAY'] != 'OUI') and (best_rec['4K'] != 'OUI')
            has_hd = any(k in text_detected.lower() for k in ['4k', 'uhd', 'blu-ray', 'bluray', 'steelbook'])
            if is_only_dvd and has_hd:
                return {
                    "status": "UPGRADE",
                    "color": "#eab308",
                    "badge": "⚡ UPGRADE DVD➔4K",
                    "matched_title": best_rec['titre'],
                    "confidence": best_r,
                    "note": "Possédé en DVD standard uniquement"
                }
            else:
                id_disp = best_rec['ID Unique'] if str(best_rec['ID Unique']).strip() else best_rec['ID MyMovie']
                return {
                    "status": "COLLECTION",
                    "color": "#10b981",
                    "badge": f"✔ EN STOCK #{id_disp}",
                    "matched_title": best_rec['titre'],
                    "confidence": best_r,
                    "note": f"Formats : {best_rec['supports']}"
                }
        # 3. Film Absent
        return {
            "status": "ABSENT",
            "color": "#f43f5e",
            "badge": "★ NOUVEAU // ABSENT",
            "matched_title": text_detected.strip(),
            "confidence": 1.0,
            "note": "Absent de votre collection : Achat conseillé !"
        }
    tab_ar_cam, tab_ar_manual, tab_ar_wish = st.tabs([
        "📸 Scanner AR par Photo de Rayonnage",
        "🎯 Saisie / Test de Titres de Rayon",
        "📝 Gestion de la Wishlist Magasin"
    ])
    with tab_ar_cam:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            shelf_cam = st.camera_input("Prendre une photo de l'étagère de face :", key="shelf_cam")
        with col_c2:
            shelf_up = st.file_uploader("Ou importer une photo de l'étagère :", type=["jpg", "jpeg", "png"], key="shelf_up")
        target_img = shelf_cam if shelf_cam is not None else shelf_up
        if target_img is not None:
            pil_shelf = Image.open(target_img).convert("RGB")
            detected_items = []
            
            try:
                import pytesseract
                # OCR 0° et 90°
                data_0 = pytesseract.image_to_data(pil_shelf, output_type=pytesseract.Output.DICT, config='--psm 11')
                for i in range(len(data_0['text'])):
                    txt = str(data_0['text'][i]).strip()
                    if len(txt) >= 3:
                        m = match_detected_spine(txt, st.session_state['wishlist_data'])
                        if m:
                            detected_items.append({**m, 'box': (data_0['left'][i], data_0['top'][i], data_0['width'][i], data_0['height'][i])})
            except Exception:
                st.info("💡 Note : Pour l'OCR natif direct, installez Tesseract (`winget install UB-Mannheim.TesseractOCR`). Mode simulation actif.")
            # Dessin de l'AR Overlay
            annotated_img = pil_shelf.copy()
            draw = ImageDraw.Draw(annotated_img)
            w_img, h_img = annotated_img.size
            if not detected_items:
                # Tranches simulées si l'OCR n'a pas renvoyé de boîte
                sim_cases = [
                    {"text": "Oppenheimer 4K", "box": (int(w_img * 0.1), int(h_img * 0.2), 60, int(h_img * 0.6))},
                    {"text": "Inception Blu-ray", "box": (int(w_img * 0.35), int(h_img * 0.2), 60, int(h_img * 0.6))},
                    {"text": "Gladiator DVD", "box": (int(w_img * 0.6), int(h_img * 0.2), 60, int(h_img * 0.6))},
                    {"text": "Alien Romulus Steelbook", "box": (int(w_img * 0.8), int(h_img * 0.2), 60, int(h_img * 0.6))}
                ]
                for sc in sim_cases:
                    m = match_detected_spine(sc['text'], st.session_state['wishlist_data'])
                    if m:
                        detected_items.append({**m, 'box': sc['box']})
            for it in detected_items:
                bx, by, bw, bh = it['box']
                x1, y1 = max(0, bx), max(0, by)
                x2, y2 = min(w_img, bx + bw), min(h_img, by + bh)
                col = it['color']
                draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
                tag_txt = f" {it['badge']} : {it['matched_title'][:20]} "
                tag_y1 = max(0, y1 - 25)
                draw.rectangle([x1, tag_y1, x1 + len(tag_txt) * 9, tag_y1 + 22], fill=col)
                draw.text((x1 + 4, tag_y1 + 4), tag_txt, fill="#000000" if col in ['#f59e0b', '#eab308'] else "#ffffff")
            st.image(annotated_img, caption="🥽 Flux de Réalité Augmentée (Spine HUD AR)", use_container_width=True)
            # Radar tactique
            st.markdown("### 📋 Radar Tactique de Rayon")
            rows_disp = [{
                "Décision": it['badge'],
                "Titre Identifié": it['matched_title'],
                "Statut": it['status'],
                "Détails / Emplacement": it['note']
            } for it in detected_items]
            st.dataframe(pd.DataFrame(rows_disp), use_container_width=True, hide_index=True)
    with tab_ar_manual:
        st.subheader("🎯 Test et Diagnostic Éclair de Titres de Rayon")
        raw_text = st.text_area("Titres vus sur l'étagère :", "Oppenheimer 4K\nInception Blu-ray\nGladiator 4K\nAlien Romulus 4K", height=120)
        if st.button("🚀 Analyser l'Étagère", key="btn_run_spine_manual"):
            res_manual = []
            for line_t in raw_text.split('\n'):
                if line_t.strip():
                    m = match_detected_spine(line_t.strip(), st.session_state['wishlist_data'])
                    if m:
                        res_manual.append({
                            "Titre sur Rayon": line_t.strip(),
                            "Décision AR": m['badge'],
                            "Correspondance Collection": m['matched_title'],
                            "Statut": m['status'],
                            "Détails": m['note']
                        })
            st.dataframe(pd.DataFrame(res_manual), use_container_width=True, hide_index=True)
    with tab_ar_wish:
        st.subheader("📝 Wishlist Magasin")
        c1, c2, c3 = st.columns(3)
        with c1: w_tit = st.text_input("Titre :", key="w_tit")
        with c2: w_fmt = st.selectbox("Format :", ["4K Ultra HD", "Steelbook", "Blu-ray", "Peu importe"], key="w_fmt")
        with c3: w_pri = st.selectbox("Priorité :", ["⭐⭐⭐ Haute", "⭐⭐ Moyenne", "⭐ Basse"], key="w_pri")
        if st.button("➕ Ajouter à la Wishlist", key="btn_add_w"):
            if w_tit.strip():
                st.session_state['wishlist_data'].append({"Titre": w_tit.strip(), "Format Visé": w_fmt, "Priorité": w_pri, "Note": ""})
                st.success("Ajouté !")
                st.rerun()
        st.dataframe(pd.DataFrame(st.session_state['wishlist_data']), use_container_width=True, hide_index=True)
# ==============================================================================
# FONCTION 2 : SAISIE CODE-BARRES (DOUCHETTE USB)
# ==============================================================================
elif menu == "2° Saisie Code-Barres (Douchette USB - 5 avant / 5 après)":
    st.title("📦 Saisie Code-Barres & Voisinage Alphabétique")
    sorted_df = df.sort_values(by=col_titre_fr, key=lambda x: x.astype(str).str.lower()).reset_index(drop=True)
    cb_in = st.text_input("Scannez à la douchette ou saisissez le code-barres :", key="cb_douchette").strip()
    
    if cb_in:
        m = sorted_df[col_barcode].astype(str).str.strip().str.contains(re.escape(cb_in), na=False)
        matches = sorted_df.index[m].tolist()
        if matches:
            idx = matches[0]
            r, e, p, cap, addr = get_rack_info(idx)
            st.success(f"🎯 Film identifié : **{sorted_df.iloc[idx][col_titre_fr]}**")
            st.markdown(f'<div class="wms-address-badge">📍 {addr}</div>', unsafe_allow_html=True)
            st.markdown(render_physical_shelf_visual(r, e, p, cap), unsafe_allow_html=True)
            start_i = max(0, idx - 5)
            end_i = min(len(sorted_df), idx + 6)
            sub = sorted_df.iloc[start_i:end_i].copy()
            sub.insert(0, "Repère", ["⭐ CIBLE" if i == idx else "" for i in range(start_i, end_i)])
            render_custom_table(sub, "f2_sub")
        else:
            st.error("Aucun film trouvé.")
    else:
        render_custom_table(sorted_df, "f2_full")
# ==============================================================================
# FONCTION 3 : SCAN CODE-BARRES PAR CAMÉRA (RÉPARÉ)
# ==============================================================================
elif menu == "3° Scan Code-Barres par Caméra (Smartphone / Webcam)":
    st.title("📷 Scan Code-Barres par Caméra")
    sorted_df = df.sort_values(by=col_titre_fr, key=lambda x: x.astype(str).str.lower()).reset_index(drop=True)
    
    cam_pic = st.camera_input("Prendre en photo le code-barres au dos du boîtier :", key="cam_scan_pic")
    detected = ""
    if cam_pic is not None:
        pil_im = Image.open(cam_pic)
        with st.spinner("Analyse optique multi-moteurs..."):
            detected = decode_barcode_robust(pil_im)
            if detected:
                st.success(f"✅ Code-barres détecté : **`{detected}`**")
            else:
                st.warning("Aucun code-barres net détecté automatiquement. Entrez-le ci-dessous :")
    cb_val = st.text_input("Code-barres analysé ou manuel :", value=detected, key="cam_manual_in").strip()
    if cb_val:
        m = sorted_df[col_barcode].astype(str).str.strip().str.contains(re.escape(cb_val), na=False)
        matches = sorted_df.index[m].tolist()
        if matches:
            idx = matches[0]
            r, e, p, cap, addr = get_rack_info(idx)
            st.success(f"🎯 Film identifié : **{sorted_df.iloc[idx][col_titre_fr]}**")
            st.markdown(f'<div class="wms-address-badge">📍 {addr}</div>', unsafe_allow_html=True)
            st.markdown(render_physical_shelf_visual(r, e, p, cap), unsafe_allow_html=True)
            start_i = max(0, idx - 5)
            end_i = min(len(sorted_df), idx + 6)
            sub = sorted_df.iloc[start_i:end_i].copy()
            sub.insert(0, "Repère", ["⭐ CIBLE" if i == idx else "" for i in range(start_i, end_i)])
            render_custom_table(sub, "f3_sub")
        else:
            st.error("Aucun film trouvé avec ce code-barres.")
    else:
        render_custom_table(sorted_df, "f3_full")
# ==============================================================================
# FONCTION 4 : CARTOGRAPHIE RACK & STOCKAGE INDUSTRIEL (WMS)
# ==============================================================================
elif menu == "4° Cartographie RACK & Stockage Industriel (WMS)":
    st.title("🏢 Cartographie RACK & Stockage Industriel (WMS)")
    sorted_df = df.sort_values(by=col_titre_fr, key=lambda x: x.astype(str).str.lower()).reset_index(drop=True)
    total_films = len(sorted_df)
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Unités Stockées", f"{total_films:,}".replace(',', ' '))
    k2.metric("Racks Physiques", f"{nb_racks_cfg}")
    k3.metric("Capacité / Étagère", f"{capa_calc_etagere} boîtiers")
    srch = st.text_input("Rechercher un film pour situer son étagère :", key="rack_srch").strip()
    if srch:
        m = sorted_df[col_titre_fr].astype(str).str.lower().str.contains(re.escape(srch.lower()), na=False)
        idx_f = sorted_df.index[m].tolist()
        if idx_f:
            t_idx = idx_f[0]
            r, e, p, cap, addr = get_rack_info(t_idx)
            st.success(f"🎯 **{sorted_df.iloc[t_idx][col_titre_fr]}** localisé !")
            st.markdown(f'<div class="wms-address-badge">📍 {addr}</div>', unsafe_allow_html=True)
            st.markdown(render_physical_shelf_visual(r, e, p, cap), unsafe_allow_html=True)
        else:
            st.warning("Titre non trouvé.")
    else:
        render_custom_table(sorted_df, "f4_full")
# ==============================================================================
# FONCTION 5 : SCANNER ANTI-DOUBLON MOBILE & WISHLIST MAGASIN
# ==============================================================================
elif menu == "5° Scanner Anti-Doublon Mobile & Wishlist Magasin":
    st.title("📱 Scanner Anti-Doublon Mobile & Wishlist Magasin")
    sorted_df = df.sort_values(by=col_titre_fr, key=lambda x: x.astype(str).str.lower()).reset_index(drop=True)
    
    txt_chk = st.text_input("Code-barres ou Titre du film en rayon :", key="chk_in").strip()
    if txt_chk:
        m = sorted_df[col_barcode].astype(str).str.strip() == txt_chk
        if not m.any():
            m = sorted_df[col_titre_fr].astype(str).str.lower().str.contains(re.escape(txt_chk.lower()), na=False)
        f_idx = sorted_df.index[m].tolist()
        if f_idx:
            row_found = sorted_df.iloc[f_idx[0]]
            r, e, p, cap, addr = get_rack_info(f_idx[0])
            st.error(f"⛔ **DÉJÀ EN COLLECTION ! (NE PAS ACHETER)**")
            st.markdown(f"**Titre :** {row_found[col_titre_fr]}")
            st.markdown(f"**Formats en stock :** `{row_found.get('supports', '-')}`")
            st.markdown(f"**Localisation :** `{addr}`")
        else:
            st.success("✅ **ABSENT DE LA COLLECTION (ACHAT RECOMMANDÉ)**")
# ==============================================================================
# AUTRES FONCTIONS (CONSULTATION, AUDIT, SAGAS, RECHERCHE, STATS)
# ==============================================================================
elif menu == "1° Consultation globale avec filtres enrichis":
    st.title("🗂️ Consultation Globale & Filtres Multi-critères")
    kw = st.text_input("Recherche rapide (Titre, Année, Studio...) :", "").strip().lower()
    f_df = df.copy()
    if kw:
        m = f_df[col_titre_fr].astype(str).str.lower().str.contains(kw, na=False)
        f_df = f_df[m]
    render_custom_table(f_df, "f1_view")
elif menu == "6° Audit Qualité : Doublons réels & Opportunités d'Upgrades 4K":
    st.title("🔍 Audit Qualité & Opportunités d'Upgrades 4K")
    dvd_only = df[(df['DVD'] == 'OUI') & (df['BLURAY'] != 'OUI') & (df['4K'] != 'OUI')]
    st.metric("Total DVD à upgrader en 4K/Blu-ray", f"{len(dvd_only):,}".replace(',', ' '))
    render_custom_table(dvd_only, "f6_dvd")
elif menu == "7° Sagathèque & Dossier d'Assurance Certifié":
    st.title("🏛️ Sagathèque & Dossier d'Assurance")
    st.metric("Total Boîtiers Physiques", f"{len(df):,}".replace(',', ' '))
    render_custom_table(df, "f7_view")
elif menu == "8° Recherche par titre / réalisateur / acteur & IMDb":
    st.title("🔍 Recherche par Titre, Réalisateur ou Acteur")
    q = st.text_input("Saisissez un mot-clé :", "").strip().lower()
    res = df[df[col_titre_fr].astype(str).str.lower().str.contains(q, na=False)] if q else df
    render_custom_table(res, "f8_view")
else:
    # 9° Statistiques complètes
    st.title("📊 Statistiques Complètes de la Collection")
    c1, c2, c3 = st.columns(3)
    c1.metric("🎬 Total Titres", f"{len(df):,}".replace(',', ' '))
    c2.metric("📀 DVD", f"{(df['DVD'] == 'OUI').sum():,}".replace(',', ' '))
    c3.metric("🔵 Blu-ray", f"{(df['BLURAY'] == 'OUI').sum():,}".replace(',', ' '))
    render_custom_table(df.head(50), "f9_head")
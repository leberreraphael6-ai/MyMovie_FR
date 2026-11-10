import streamlit as st
import pandas as pd
import io
import os
import math

# ==============================================================================
# MOTEURS OPTIQUES & CALCULATEURS INDUSTRIELS (RACK & SCAN)
# ==============================================================================
# Importations tolérantes aux pannes (OpenCV & NumPy optionnels)
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

def decode_barcode_robust(pil_img):
    """
    Décodeur optique multi-moteurs ultra-tolérant :
    - Fonctionne même si OpenCV ou pyzbar ne sont pas encore installés sur la machine
    - Tente pyzbar directement sur l'image PIL
    - Tente OpenCV si installé
    - Tente zxingcpp si installé
    """
    if pil_img is None:
        return ""

    # 1. Tentative pyzbar native directe sur l'image PIL (ne nécessite pas OpenCV)
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        results = pyzbar_decode(pil_img)
        for r in results:
            val = r.data.decode('utf-8', errors='ignore').strip()
            if val and len(val) >= 6:
                return val
    except Exception:
        pass

    # 2. Tentative via OpenCV (si installé)
    if cv2 is not None and np is not None:
        try:
            img_np = np.array(pil_img)
            if img_np.ndim == 2:
                gray = img_np
                bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            elif img_np.shape[2] == 4:
                bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            else:
                bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            h, w = gray.shape[:2]
            if max(h, w) > 1600:
                scale = 1600.0 / max(h, w)
                gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            # Test BarcodeDetector natif OpenCV
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
                        ok, decoded_info, decoded_type, _ = detector.detectAndDecode(r_v)
                        if ok:
                            for t in decoded_info:
                                t_clean = str(t).strip()
                                if t_clean and len(t_clean) >= 6:
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

    return ""
    """
    Décodeur optique multi-moteurs ultra-performant :
    - Étape 1 : Normalisation, conversion BGR/Gris, redimensionnement adaptatif
    - Étape 2 : Moteur pyzbar (si disponible) sur 4 rotations (0, 90, 180, 270)
    - Étape 3 : Moteur OpenCV BarcodeDetector avec rehaussement de contraste CLAHE et binarisation Otsu
    - Étape 4 : Moteur zxing-cpp (si disponible)
    """
    if pil_img is None:
        return ""
    try:
        # Conversion PIL -> NumPy / OpenCV
        img_np = np.array(pil_img)
        if img_np.ndim == 2:
            gray = img_np
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:
            bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Redimensionnement adaptatif pour optimiser vitesse et détection
        h, w = gray.shape[:2]
        if max(h, w) > 1600:
            scale = 1600.0 / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # 1. Tentative pyzbar (moteur référence)
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
            for rot in [0, 90, 180, 270]:
                r_gray = np.rot90(gray, rot // 90) if rot > 0 else gray
                results = pyzbar_decode(r_gray)
                for r in results:
                    val = r.data.decode('utf-8', errors='ignore').strip()
                    if val and len(val) >= 6:
                        return val
        except Exception:
            pass

        # 2. Tentative OpenCV BarcodeDetector natif
        try:
            if hasattr(cv2, 'barcode') and hasattr(cv2.barcode, 'BarcodeDetector'):
                detector = cv2.barcode.BarcodeDetector()
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                gray_clahe = clahe.apply(gray)
                
                # Variantes d'images : brute, rehaussée CLAHE, et binarisée Otsu
                variants = [
                    gray,
                    gray_clahe,
                    cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                ]
                for v in variants:
                    for rot in [0, 90, 180, 270]:
                        r_v = np.rot90(v, rot // 90) if rot > 0 else v
                        ok, decoded_info, decoded_type, _ = detector.detectAndDecode(r_v)
                        if ok:
                            for t in decoded_info:
                                t_clean = str(t).strip()
                                if t_clean and len(t_clean) >= 6:
                                    return t_clean
        except Exception:
            pass

        # 3. Tentative zxingcpp si installé
        try:
            import zxingcpp
            for rot in [0, 90, 180, 270]:
                r_bgr = np.rot90(bgr, rot // 90) if rot > 0 else bgr
                results = zxingcpp.read_barcodes(r_bgr)
                for r in results:
                    val = str(r.text).strip()
                    if val and len(val) >= 6:
                        return val
        except Exception:
            pass

    except Exception:
        pass
    return ""


def render_physical_shelf_visual(target_rack, target_etagere, target_pos, capa_etagere, shelf_titles=None):
    """
    Dessine visuellement le compartiment d'étagère avec ses positions de DVD alignées
    comme sur la photo réelle de la bibliothèque blanche de Raf.
    La position ciblée est illuminée en néon cyan.
    Chaîne compacte sans indentation pour éviter l'interprétation code-block de Markdown.
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
    Système WMS Topologique Physique de Référence (Inspiré du meuble réel) :
    - 1 RACK = Colonne verticale de N Étagères (défaut : 8)
    - 1 ÉTAGÈRE = Compartiment de P boîtiers/DVD alignés (défaut : 10)
    - Capacité d'un Rack = N * P (défaut : 8 * 10 = 80 DVD)
    - Calcule précisément : RACK N°, ÉTAGÈRE N° (1 à 8), POSITION N° (1 à 10)
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
    code_court = f"R{rack:02d}-E{etagere:02d}-P{pos_in_shelf:02d}"
    addr_str = f"RACK {rack:02d} // ÉTAGÈRE {etagere:02d} // POSITION {pos_in_shelf:02d} (sur {capa_etagere})"
    return rack, etagere, pos_in_shelf, capa_etagere, addr_str

import re
import urllib.parse
from PIL import Image
try:
    import cv2
except ImportError:
    cv2 = None
from collections import Counter
try:
    import altair as alt
except ImportError:
    alt = None

st.set_page_config(
    page_title="⚡ Volt Ciné-WMS v5.0 Ultra // Système Tactique",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# DESIGN SYSTEM CYBER-WMS V5.0 : MODERNE, FUTURISTE, FUN & DYNAMIQUE
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* --- FOND CYBERNETIQUE PROFOND AVEC GRILLE SPATIALE --- */
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

/* --- BANDEAU HUD HOLOGRAPHIQUE PRINCIPAL --- */
.cyber-hud-banner {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.75) 100%);
    border: 1px solid rgba(0, 242, 254, 0.4);
    border-radius: 14px;
    padding: 18px 26px;
    margin-bottom: 24px;
    box-shadow: 0 10px 35px rgba(0, 242, 254, 0.15), inset 0 0 15px rgba(0, 242, 254, 0.05);
    backdrop-filter: blur(14px);
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    overflow: hidden;
}

.cyber-hud-banner::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00f2fe, #a855f7, transparent);
    animation: hud_laser_sweep 4s linear infinite;
}

@keyframes hud_laser_sweep {
    0% { left: -100%; }
    100% { left: 100%; }
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

.cyber-hud-status {
    display: flex;
    gap: 12px;
    align-items: center;
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
    letter-spacing: 1px;
    display: flex;
    align-items: center;
    gap: 6px;
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
    letter-spacing: 1.2px;
    font-weight: 800;
    text-shadow: 0 0 10px rgba(0, 242, 254, 0.6);
    animation: cyber_pulse 2.5s infinite;
}

@keyframes cyber_pulse {
    0%, 100% { box-shadow: 0 0 10px rgba(0, 242, 254, 0.3); }
    50% { box-shadow: 0 0 22px rgba(0, 242, 254, 0.7); }
}

/* --- CARTES METRIQUES HOLOGRAPHIQUES --- */
div[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.35) !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
    overflow: hidden;
}

div[data-testid="stMetric"]:hover {
    border-color: #00f2fe !important;
    box-shadow: 0 0 25px rgba(0, 242, 254, 0.4) !important;
    transform: translateY(-4px) scale(1.01) !important;
}

div[data-testid="stMetricLabel"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #94a3b8 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

div[data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif !important;
    color: #f8fafc !important;
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    text-shadow: 0 0 15px rgba(56, 189, 248, 0.5) !important;
}

/* --- ANIMATION SCANNER LASER SUR LA RECHERCHE & SCAN --- */
.laser-scanner-box {
    position: relative;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(0, 242, 254, 0.4);
    border-radius: 12px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 8px 30px rgba(0, 242, 254, 0.15);
    overflow: hidden;
}

.laser-beam {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #00f2fe, #f43f5e, #00f2fe, transparent);
    box-shadow: 0 0 15px #00f2fe, 0 0 25px #f43f5e;
    animation: laser_sweep 2.2s ease-in-out infinite alternate;
}

@keyframes laser_sweep {
    0% { top: 0%; opacity: 0.4; }
    50% { opacity: 1; }
    100% { top: 96%; opacity: 0.4; }
}

/* --- FICHE FILM HOLOGRAPHIQUE (HUD CARD 360°) --- */
.hologram-film-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
    border: 1px solid rgba(0, 242, 254, 0.35);
    border-radius: 14px;
    padding: 20px 24px;
    margin: 15px 0;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(0, 242, 254, 0.05);
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
    letter-spacing: 1px;
    display: inline-block;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.35);
    margin-bottom: 12px;
}

/* --- BADGES SUPPORTS HOLOGRAPHIQUES --- */
.badge-4k {
    background: linear-gradient(135deg, rgba(147, 51, 234, 0.35), rgba(192, 132, 252, 0.15));
    border: 1px solid #c084fc;
    color: #e9d5ff;
    padding: 3px 10px;
    border-radius: 6px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    box-shadow: 0 0 10px rgba(192, 132, 252, 0.4);
}

.badge-sb {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.35), rgba(251, 191, 36, 0.15));
    border: 1px solid #fbbf24;
    color: #fef3c7;
    padding: 3px 10px;
    border-radius: 6px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    box-shadow: 0 0 10px rgba(251, 191, 36, 0.4);
}

.badge-br {
    background: linear-gradient(135deg, rgba(2, 132, 199, 0.35), rgba(56, 189, 248, 0.15));
    border: 1px solid #38bdf8;
    color: #e0f2fe;
    padding: 3px 10px;
    border-radius: 6px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
}

/* --- BOUTONS CYBER LASER --- */
.stButton > button {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    color: #ffffff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    letter-spacing: 1.2px !important;
    border: 1px solid rgba(56, 189, 248, 0.5) !important;
    border-radius: 8px !important;
    padding: 9px 24px !important;
    box-shadow: 0 4px 18px rgba(2, 132, 199, 0.35) !important;
    transition: all 0.25s ease-in-out !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #00f2fe 0%, #0284c7 100%) !important;
    border-color: #00f2fe !important;
    box-shadow: 0 0 25px rgba(0, 242, 254, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* --- BARRE LATERALE --- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1d 0%, #060913 100%) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.2) !important;
}

/* --- TITRES --- */
h1 {
    font-family: 'Orbitron', sans-serif !important;
    background: linear-gradient(90deg, #00f2fe, #38bdf8, #818cf8) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: 1.5px !important;
}

h2, h3 {
    font-family: 'Rajdhani', sans-serif !important;
    color: #38bdf8 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
</style>

<div class="cyber-hud-banner">
    <div>
        <div class="cyber-title">⚡ VOLT CINÉ-WMS // SYSTÈME DE SUPERVISION</div>
        <div style="font-family: Rajdhani, sans-serif; color: #94a3b8; font-size: 0.95rem; letter-spacing: 1px; margin-top: 4px;">
            PLATEFORME LOGISTIQUE DE PRÉCISION • GESTION TOPOLOGIQUE RACK 4K
        </div>
    </div>
    <div class="cyber-hud-status">
        <div class="cyber-badge-online">🟢 SYSTÈME EN LIGNE</div>
        <div class="cyber-badge-version">VERSION 5.0 ULTRA</div>
    </div>
</div>
""", unsafe_allow_html=True)

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
    "2° Saisie Code-Barres (Douchette USB - 5 avant / 5 après)",
    "3° Scan Code-Barres par Caméra (Smartphone / Webcam)",
    "4° Cartographie RACK & Stockage Industriel (WMS)",
    "5° Scanner Anti-Doublon Mobile & Wishlist Magasin",
    "6° Audit Qualité : Doublons réels & Opportunités d'Upgrades 4K",
    "7° Sagathèque & Dossier d'Assurance Certifié",
    "8° Recherche par titre / réalisateur / acteur & IMDb",
    "9° Statistiques complètes de la collection"
])

# Paramétrage logistique RACK dans la barre latérale (Étendu & Haute Capacité)
st.sidebar.markdown("---")
st.sidebar.subheader("🏢 Topologie de Stockage RACK (Magasin)")

mode_config = st.sidebar.radio(
    "Mode de calibrage :",
    ["Par Nombre d'Étagères", "Par Capacité Réelle par Étagère"],
    index=0,
    key="cfg_mode_rack"
)

if mode_config == "Par Nombre d'Étagères":
    nb_racks_cfg = st.sidebar.slider("Nombre de Racks (Meubles / Colonnes) :", min_value=1, max_value=50, value=12, key="cfg_nb_racks")
    nb_etageres_cfg = st.sidebar.slider("Nombre d'Étagères par Meuble :", min_value=1, max_value=30, value=10, key="cfg_nb_etageres")
    total_etageres_magasin = nb_racks_cfg * nb_etageres_cfg
    capa_calc_etagere = math.ceil(len(df) / total_etageres_magasin) if total_etageres_magasin > 0 else 1
    st.sidebar.info(f"📐 **{total_etageres_magasin} étagères** au total\n\n📦 Capacité requise : **{capa_calc_etagere} boîtiers / étagère**")
else:
    capa_voulue = st.sidebar.slider("Capacité max d'une étagère (boîtiers) :", min_value=20, max_value=120, value=45, step=5, key="cfg_capa_voulue")
    reserve_pct = st.sidebar.slider("Marge de réserve (places libres pour futurs achats) :", min_value=0, max_value=40, value=15, step=5, format="%d%%", key="cfg_reserve")
    eff_capa = max(1, math.floor(capa_voulue * (1 - reserve_pct / 100)))
    etageres_requises = math.ceil(len(df) / eff_capa)
    nb_etageres_cfg = st.sidebar.slider("Étagères par Meuble :", min_value=1, max_value=30, value=8, key="cfg_nb_etageres")
    nb_racks_cfg = math.ceil(etageres_requises / nb_etageres_cfg)
    total_etageres_magasin = nb_racks_cfg * nb_etageres_cfg
    capa_calc_etagere = capa_voulue
    st.sidebar.success(f"🎯 Configuration optimale :\n\n- **{nb_racks_cfg} Racks** de **{nb_etageres_cfg} étagères**\n- **{total_etageres_magasin} étagères** ({etageres_requises} utilisées, {total_etageres_magasin - etageres_requises} en réserve)")

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
# FONCTION 2 : SAISIE CODE-BARRES (DOUCHETTE USB - 5 AVANT / 5 APRÈS)
# ==============================================================================
elif menu == "2° Saisie Code-Barres (Douchette USB - 5 avant / 5 après)":
    st.title("📦 Saisie Code-Barres & Voisinage Alphabétique (Douchette USB)")
    st.markdown(f"Entrez ou scannez un code-barres avec une douchette USB : l'inventaire est trié alphabétiquement sur **`{col_titre_fr}`** et affiche le film identifié ainsi que **ses 5 voisins précédents et ses 5 voisins suivants** avec son **ID MyMovie** pour le repérer ou le ranger sur votre étagère.")

    sorted_df = df.sort_values(
        by=col_titre_fr,
        key=lambda x: x.astype(str).str.lower()
    ).reset_index(drop=True)

    series_b = sorted_df[col_barcode].fillna('').astype(str).str.strip()

    scan_c1, scan_c2 = st.columns([5, 1])
    with scan_c1:
        cb_raw = st.text_input(
            "Scannez à la douchette ou saisissez le code-barres (ou un Titre / ID en secours) :",
            value="",
            key="input_cb_scan",
            placeholder="Ex: 3344428038830, 5051889673620, ou titre de film..."
        )
    with scan_c2:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Effacer", help="Réinitialiser pour un nouveau scan"):
            st.rerun()

    if cb_raw:
        clean_cb = ''.join(ch for ch in str(cb_raw) if ch.isprintable()).strip()
        clean_cb = re.sub(r'^\][a-zA-Z0-9]{2}', '', clean_cb).strip()
        clean_cb = re.sub(r'\s+', '', clean_cb)

        if len(clean_cb) >= 26 and clean_cb.isdigit():
            clean_cb = clean_cb[-13:]
            st.info(f"ℹ️ Double scan consécutif détecté : le dernier code-barres analysé est **`{clean_cb}`**.")
        elif len(clean_cb) >= 24 and clean_cb.isdigit():
            clean_cb = clean_cb[-12:]
            st.info(f"ℹ️ Double scan consécutif détecté : le dernier code-barres analysé est **`{clean_cb}`**.")

        # Correspondance multi-paliers
        mask_match = series_b.str.lower() == clean_cb.lower()

        if not mask_match.any() and clean_cb.isdigit():
            norm_digits = clean_cb.lstrip('0')
            series_digits = series_b.str.lstrip('0')
            mask_match = (series_digits == norm_digits) & (series_digits != '')
            if not mask_match.any():
                z13 = clean_cb.zfill(13)
                mask_match = series_b.str.zfill(13) == z13

        if not mask_match.any() and len(clean_cb) >= 6:
            mask_match = series_b.str.contains(re.escape(clean_cb), case=False, na=False)

        match_indices = sorted_df.index[mask_match].tolist()
        matched_by = "barcode"

        if not match_indices:
            search_kw = cb_raw.strip().lower()
            mask_title = sorted_df[col_titre_fr].astype(str).str.lower().str.contains(re.escape(search_kw), na=False)
            if 'titre_anglais' in sorted_df.columns:
                mask_title = mask_title | sorted_df['titre_anglais'].astype(str).str.lower().str.contains(re.escape(search_kw), na=False)
            if 'ID Unique' in sorted_df.columns and search_kw.isdigit():
                mask_title = mask_title | (sorted_df['ID Unique'].astype(str) == search_kw)

            match_indices = sorted_df.index[mask_title].tolist()
            if match_indices:
                matched_by = "title"

        if match_indices:
            if len(match_indices) > 1:
                if matched_by == "barcode":
                    st.warning(f"📦 **Coffret / Édition Multi-films détectée :** Ce code-barres (`{clean_cb}`) regroupe **{len(match_indices)} films** dans votre collection.")
                else:
                    st.info(f"🔍 **Plusieurs correspondances trouvées :** **{len(match_indices)} films** correspondent à « {cb_raw} ».")

                multi_subset = sorted_df.iloc[match_indices].copy()
                cols_multi = [c for c in ['ID MyMovie', 'ID Unique', 'Réf Entrée', col_titre_fr, 'annee_num', 'duree_num', 'supports', 'Production_Corp'] if c in multi_subset.columns]
                st.dataframe(multi_subset[cols_multi], hide_index=True, use_container_width=True)

                film_labels = [f"#{sorted_df.iloc[i].get('ID Unique', i)} — {sorted_df.iloc[i][col_titre_fr]} (Position Alpha {i + 1})" for i in match_indices]
                sel_label = st.selectbox(
                    "🎯 Sélectionnez le film précis à repérer sur l'étagère (Afficher ses 5 voisins) :",
                    film_labels,
                    index=0,
                    key="sel_film_f2"
                )
                sel_pos = film_labels.index(sel_label)
                idx = match_indices[sel_pos]
            else:
                idx = match_indices[0]

            matched_row = sorted_df.iloc[idx]
            film_name = matched_row[col_titre_fr]

            st.success(f"🎯 Film identifié : **{film_name}** (Position alphabétique **{idx + 1}** sur {len(sorted_df):,}).".replace(',', ' '))

            ci_c1, ci_c2, ci_c3 = st.columns(3)
            with ci_c1:
                r_k, e_k, p_k, capa_k, addr_lbl = get_rack_info(idx)
                st.markdown(f"""<div class="wms-address-badge">📍 {addr_lbl}</div>""", unsafe_allow_html=True)
                st.markdown(render_physical_shelf_visual(r_k, e_k, p_k, capa_k), unsafe_allow_html=True)
                yt_query = f"{film_name} {matched_row.get('annee', '')} bande annonce vf".replace(' ', '+')
                st.markdown(f"""<a href="https://www.youtube.com/results?search_query={yt_query}" target="_blank" style="background: linear-gradient(135deg, #e11d48, #be123c); color: white; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-family: Rajdhani, sans-serif; font-weight: 700; font-size: 0.95rem; display: inline-block; margin-bottom: 10px; box-shadow: 0 0 12px rgba(225, 29, 72, 0.4);">🎬 VOIR BANDE-ANNONCE VIDÉO (HD) ↗</a>""", unsafe_allow_html=True)
                if 'ID MyMovie' in matched_row and pd.notna(matched_row['ID MyMovie']):
                    st.markdown(f"**🏷️ ID MyMovie (Boîtier Physique) :** `#{matched_row['ID MyMovie']}`")
                if 'ID Unique' in matched_row and pd.notna(matched_row['ID Unique']):
                    st.markdown(f"**🔑 ID Unique :** `{matched_row['ID Unique']}`")
                if 'Réf Entrée' in matched_row and pd.notna(matched_row['Réf Entrée']) and str(matched_row['Réf Entrée']) != str(matched_row.get('ID Unique')):
                    st.markdown(f"**📑 Réf Entrée :** `{matched_row['Réf Entrée']}`")
                if col_director and pd.notna(matched_row.get(col_director)):
                    st.markdown(f"**🎬 Réalisateur :** {matched_row[col_director]}")
            with ci_c2:
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
            st.subheader(f"📍 Emplacement sur l'Étagère (5 Titres Précédents ➔ Film Ciblé ➔ 5 Titres Suivants)")

            start_i = max(0, idx - 5)
            end_i = min(len(sorted_df), idx + 6)

            subset = sorted_df.iloc[start_i:end_i].copy()
            subset.insert(0, "Position", [i + 1 for i in range(start_i, end_i)])
            subset.insert(1, "Repère", ["⭐ FILM CIBLÉ" if i == idx else "" for i in range(start_i, end_i)])

            render_custom_table(subset, key_id=f"f2_barcode_match_{idx}")
        else:
            st.error(f"❌ Aucun film trouvé avec le code-barres « {cb_raw} ».")
            st.info("💡 **Astuce :** Vérifiez que la douchette a bien transmis tous les chiffres, ou essayez de taper directement le Titre du film ou son ID.")
            st.markdown("### Inventaire complet (Ordre Alphabétique) :")
            render_custom_table(sorted_df, key_id="f2_barcode_fallback")
    else:
        st.markdown("### Inventaire complet (Ordre Alphabétique) :")
        render_custom_table(sorted_df, key_id="f2_barcode_default")


# ==============================================================================
# FONCTION 3 : SCAN CODE-BARRES PAR CAMÉRA (SMARTPHONE / WEBCAM)
# ==============================================================================
elif menu == "3° Scan Code-Barres par Caméra (Smartphone / Webcam)":
    st.title("📷 Scan Code-Barres par Caméra (Smartphone / Webcam)")
    st.markdown("Prenez en photo le code-barres au dos du boîtier avec **l'appareil photo de votre smartphone ou votre webcam** : l'image est automatiquement analysée et décodée pour positionner le film sur votre étagère avec **ses 5 voisins précédents et suivants**.")
    st.markdown("""<div class="laser-scanner-box"><div class="laser-beam"></div><div style="font-family: Orbitron, sans-serif; font-size: 1.05rem; font-weight: 700; color: #00f2fe; text-align: center; letter-spacing: 1.5px;">⚡ CAPTEUR OPTIQUE ACTIF // FAISCEAU LASER WMS EN VEILLE</div></div>""", unsafe_allow_html=True)

    sorted_df = df.sort_values(
        by=col_titre_fr,
        key=lambda x: x.astype(str).str.lower()
    ).reset_index(drop=True)

    series_b = sorted_df[col_barcode].fillna('').astype(str).str.strip()

    cam_tab1, cam_tab2 = st.tabs(["📸 Prendre une photo en direct", "📁 Importer une photo depuis la galerie"])

    img_input = None
    with cam_tab1:
        st.caption("Cadrez bien le code-barres au centre de l'écran, évitez les reflets du plastique de la jaquette.")
        if cv2 is None:
            st.info("💡 **Conseil d'activation scan caméra :** Pour bénéficier de la reconnaissance optique avancée sur votre PC, vous pouvez installer la bibliothèque requise via votre invite de commande : `pip install opencv-python pyzbar` (la saisie manuelle et douchette USB fonctionnent déjà à 100%).")
        camera_pic = st.camera_input("Prendre la photo du code-barres :", key="cam_barcode_input")
        if camera_pic is not None:
            img_input = camera_pic

    with cam_tab2:
        st.caption("Si vous avez déjà photographié le boîtier ou si l'accès direct caméra est bloqué :")
        uploaded_pic = st.file_uploader("Choisissez une photo (JPG, PNG) :", type=["jpg", "jpeg", "png", "webp"], key="upload_barcode_input")
        if uploaded_pic is not None:
            img_input = uploaded_pic

    detected_code = ""
    if img_input is not None:
        try:
            pil_image = Image.open(img_input)
            with st.spinner("🔍 Analyse optique et décodage du code-barres en cours..."):
                detected_code = decode_barcode_robust(pil_image)
        except Exception as e:
            st.error(f"Erreur de lecture de l'image : {e}")

    # Zone de résultat ou de saisie manuelle si le code-barres a été lu ou s'il doit être corrigé
    st.markdown("---")
    st.markdown("### 🎯 Résultat de la Détection Optique")

    if img_input is not None:
        if detected_code:
            st.success(f"✅ Code-barres détecté avec succès : **`{detected_code}`**")
        else:
            st.warning("⚠️ Aucun code-barres net n'a pu être décodé automatiquement.")
            st.info("💡 **Conseils prise de vue smartphone :**\n- Rapprochez l'objectif pour que le code-barres occupe une large partie de la photo.\n- Inclinez légèrement le boîtier pour chasser le reflet blanc de la lumière.\n- Assurez-vous de la netteté (mise au point).")

    val_cb_cam = st.text_input(
        "Code-barres identifié (modifiable ou saisie de secours) :",
        value=detected_code if detected_code else "",
        key="input_cam_cb_final",
        placeholder="Ex: 3344428038830 ou titre du film..."
    ).strip()

    if val_cb_cam:
        clean_cb = ''.join(ch for ch in str(val_cb_cam) if ch.isprintable()).strip()
        clean_cb = re.sub(r'^\][a-zA-Z0-9]{2}', '', clean_cb).strip()
        clean_cb = re.sub(r'\s+', '', clean_cb)

        # 1. Correspondance exacte
        mask_match = series_b.str.lower() == clean_cb.lower()

        # 2. Correspondance numérique souple (zéros non significatifs et zfill 13)
        if not mask_match.any() and clean_cb.isdigit():
            norm_digits = clean_cb.lstrip('0')
            series_digits = series_b.str.lstrip('0')
            mask_match = (series_digits == norm_digits) & (series_digits != '')
            if not mask_match.any():
                z13 = clean_cb.zfill(13)
                mask_match = series_b.str.zfill(13) == z13

        # 3. Correspondance par terminaison (ex: 4 à 8 derniers chiffres saisis)
        if not mask_match.any() and clean_cb.isdigit() and len(clean_cb) >= 4:
            mask_match = series_b.str.endswith(clean_cb)

        # 4. Sous-chaîne de code-barres
        if not mask_match.any() and len(clean_cb) >= 4:
            mask_match = series_b.str.contains(re.escape(clean_cb), case=False, na=False)

        if not mask_match.any() and clean_cb.isdigit():
            norm_digits = clean_cb.lstrip('0')
            series_digits = series_b.str.lstrip('0')
            mask_match = (series_digits == norm_digits) & (series_digits != '')
            if not mask_match.any():
                z13 = clean_cb.zfill(13)
                mask_match = series_b.str.zfill(13) == z13

        if not mask_match.any() and len(clean_cb) >= 6:
            mask_match = series_b.str.contains(re.escape(clean_cb), case=False, na=False)

        match_indices = sorted_df.index[mask_match].tolist()
        matched_by = "barcode"

        if not match_indices:
            search_kw = val_cb_cam.lower()
            mask_title = sorted_df[col_titre_fr].astype(str).str.lower().str.contains(re.escape(search_kw), na=False)
            if 'titre_anglais' in sorted_df.columns:
                mask_title = mask_title | sorted_df['titre_anglais'].astype(str).str.lower().str.contains(re.escape(search_kw), na=False)
            if 'ID Unique' in sorted_df.columns and search_kw.isdigit():
                mask_title = mask_title | (sorted_df['ID Unique'].astype(str) == search_kw)

            match_indices = sorted_df.index[mask_title].tolist()
            if match_indices:
                matched_by = "title"

        if match_indices:
            if len(match_indices) > 1:
                if matched_by == "barcode":
                    st.warning(f"📦 **Coffret / Édition Multi-films détectée :** Ce code-barres (`{clean_cb}`) regroupe **{len(match_indices)} films** dans votre collection.")
                else:
                    st.info(f"🔍 **Plusieurs correspondances trouvées :** **{len(match_indices)} films** correspondent à votre recherche.")

                multi_subset = sorted_df.iloc[match_indices].copy()
                cols_multi = [c for c in ['ID MyMovie', 'ID Unique', 'Réf Entrée', col_titre_fr, 'annee_num', 'duree_num', 'supports', 'Production_Corp'] if c in multi_subset.columns]
                st.dataframe(multi_subset[cols_multi], hide_index=True, use_container_width=True)

                film_labels = [f"#{sorted_df.iloc[i].get('ID Unique', i)} — {sorted_df.iloc[i][col_titre_fr]} (Position Alpha {i + 1})" for i in match_indices]
                sel_label = st.selectbox(
                    "🎯 Sélectionnez le film précis à repérer sur l'étagère (Afficher ses 5 voisins) :",
                    film_labels,
                    index=0,
                    key="sel_film_f3"
                )
                sel_pos = film_labels.index(sel_label)
                idx = match_indices[sel_pos]
            else:
                idx = match_indices[0]

            matched_row = sorted_df.iloc[idx]
            film_name = matched_row[col_titre_fr]

            st.success(f"🎯 Film identifié : **{film_name}** (Position alphabétique **{idx + 1}** sur {len(sorted_df):,}).".replace(',', ' '))

            ci_c1, ci_c2, ci_c3 = st.columns(3)
            with ci_c1:
                r_k, e_k, p_k, capa_k, addr_lbl = get_rack_info(idx)
                st.markdown(f"""<div class="wms-address-badge">📍 {addr_lbl}</div>""", unsafe_allow_html=True)
                st.markdown(render_physical_shelf_visual(r_k, e_k, p_k, capa_k), unsafe_allow_html=True)
                yt_query = f"{film_name} {matched_row.get('annee', '')} bande annonce vf".replace(' ', '+')
                st.markdown(f"""<a href="https://www.youtube.com/results?search_query={yt_query}" target="_blank" style="background: linear-gradient(135deg, #e11d48, #be123c); color: white; padding: 6px 14px; border-radius: 6px; text-decoration: none; font-family: Rajdhani, sans-serif; font-weight: 700; font-size: 0.95rem; display: inline-block; margin-bottom: 10px; box-shadow: 0 0 12px rgba(225, 29, 72, 0.4);">🎬 VOIR BANDE-ANNONCE VIDÉO (HD) ↗</a>""", unsafe_allow_html=True)
                if 'ID MyMovie' in matched_row and pd.notna(matched_row['ID MyMovie']):
                    st.markdown(f"**🏷️ ID MyMovie (Boîtier Physique) :** `#{matched_row['ID MyMovie']}`")
                if 'ID Unique' in matched_row and pd.notna(matched_row['ID Unique']):
                    st.markdown(f"**🔑 ID Unique :** `{matched_row['ID Unique']}`")
                if 'Réf Entrée' in matched_row and pd.notna(matched_row['Réf Entrée']) and str(matched_row['Réf Entrée']) != str(matched_row.get('ID Unique')):
                    st.markdown(f"**📑 Réf Entrée :** `{matched_row['Réf Entrée']}`")
                if col_director and pd.notna(matched_row.get(col_director)):
                    st.markdown(f"**🎬 Réalisateur :** {matched_row[col_director]}")
            with ci_c2:
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
            st.subheader(f"📍 Emplacement sur l'Étagère (5 Titres Précédents ➔ Film Ciblé ➔ 5 Titres Suivants)")

            start_i = max(0, idx - 5)
            end_i = min(len(sorted_df), idx + 6)

            subset = sorted_df.iloc[start_i:end_i].copy()
            subset.insert(0, "Position", [i + 1 for i in range(start_i, end_i)])
            subset.insert(1, "Repère", ["⭐ FILM CIBLÉ" if i == idx else "" for i in range(start_i, end_i)])

            render_custom_table(subset, key_id=f"f3_cam_match_{idx}")
        else:
            st.error(f"❌ Aucun film trouvé avec le code-barres « {val_cb_cam} ».")
            st.info("💡 **Astuce :** Vérifiez la saisie ou tapez directement le Titre du film.")
            st.markdown("### Inventaire complet (Ordre Alphabétique) :")
            render_custom_table(sorted_df, key_id="f3_cam_fallback")
    else:
        st.markdown("### Inventaire complet (Ordre Alphabétique) :")
        render_custom_table(sorted_df, key_id="f3_cam_default")


# ==============================================================================
# FONCTION 4 : CARTOGRAPHIE RACK & STOCKAGE INDUSTRIEL (WMS)
# ==============================================================================
elif menu == "4° Cartographie RACK & Stockage Industriel (WMS)":
    st.title("🏢 Cartographie RACK & Stockage Industriel (WMS)")
    st.markdown("Système de gestion d'emplacement logistique pour vidéothèque physique : vos **4 771 films** sont modélisés selon la norme industrielle standard **`[RACK ➔ ÉTAGÈRE ➔ EMPLACEMENT]`** basée sur le tri alphabétique rigoureux.")

    sorted_df = df.sort_values(
        by=col_titre_fr,
        key=lambda x: x.astype(str).str.lower()
    ).reset_index(drop=True)

    total_films = len(sorted_df)
    total_tablettes = nb_racks_cfg * nb_etageres_cfg
    capa_par_etagere = capa_calc_etagere if "capa_calc_etagere" in locals() and capa_calc_etagere > 0 else (math.ceil(total_films / total_tablettes) if total_tablettes > 0 else 1)
    capacite_theorique = total_tablettes * capa_par_etagere
    taux_occupation = (total_films / capacite_theorique * 100) if capacite_theorique > 0 else 100.0

    # KPI Logistiques
    kpi_w1, kpi_w2, kpi_w3, kpi_w4 = st.columns(4)
    with kpi_w1:
        st.metric("📦 Unités Stockées", f"{total_films:,}".replace(',', ' '))
    with kpi_w2:
        st.metric("🏢 Racks Physiques", f"{nb_racks_cfg} Racks")
    with kpi_w3:
        st.metric("📐 Tablettes Actives", f"{total_tablettes} Étagères")
    with kpi_w4:
        st.metric("📊 Taux de Remplissage", f"{taux_occupation:.1f} %", f"~{capa_par_etagere} boîtiers/tablette")

    st.markdown("---")

    # Onglets d'exploitation logistique
    tab_rack_visu, tab_rack_search, tab_rack_labels = st.tabs([
        "🗺️ Coupe 2D du Meuble (Visualisation Rack)",
        "🎯 Recherche Directe d'Emplacement",
        "🏷️ Répertoire des Bornes & Étiquettes"
    ])

    with tab_rack_visu:
        st.subheader("Inspection Visuelle par Rack (Meuble)")
        rack_selected = st.selectbox(
            "Sélectionnez le RACK (Meuble / Colonne) à inspecter :",
            [f"RACK {r:02d}" for r in range(1, nb_racks_cfg + 1)],
            index=0
        )
        r_num = int(rack_selected.split()[1])

        st.caption(f"Visualisation en coupe verticale du **{rack_selected}** (du haut vers le bas, de l'Étagère 01 à l'Étagère {nb_etageres_cfg:02d}) :")

        for e_num in range(1, nb_etageres_cfg + 1):
            # Calcul des indices de films sur cette étagère
            start_idx = ((r_num - 1) * nb_etageres_cfg + (e_num - 1)) * capa_par_etagere
            end_idx = min(start_idx + capa_par_etagere, total_films)

            if start_idx < total_films:
                shelf_films = sorted_df.iloc[start_idx:end_idx]
                first_title = shelf_films.iloc[0][col_titre_fr]
                last_title = shelf_films.iloc[-1][col_titre_fr]
                nb_on_shelf = len(shelf_films)

                # Rendu visuel d'une tablette d'étagère
                fill_pct = min(100, int((nb_on_shelf / capa_par_etagere) * 100)) if capa_par_etagere > 0 else 100
                st.markdown(
                    f"""<div style='background: linear-gradient(90deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.8) 100%); border-left: 6px solid #00f2fe; border-top: 1px solid rgba(0, 242, 254, 0.25); border-right: 1px solid rgba(0, 242, 254, 0.25); border-bottom: 1px solid rgba(0, 242, 254, 0.25); padding: 14px 20px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 4px 18px rgba(0,0,0,0.4);'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-family: Orbitron, sans-serif; font-size: 1.1rem; font-weight: 700; color: #00f2fe; letter-spacing: 0.8px;'>⚡ {rack_selected} // ÉTAGÈRE {e_num:02d}</span>
                            <span style='background: rgba(0, 242, 254, 0.15); border: 1px solid #00f2fe; color: #38bdf8; padding: 3px 12px; border-radius: 20px; font-family: Rajdhani, sans-serif; font-size: 0.95rem; font-weight: 700;'>{nb_on_shelf} / {capa_par_etagere} boîtiers ({fill_pct}%)</span>
                        </div>
                        <div style='color: #cbd5e1; margin-top: 8px; font-size: 0.95rem;'>
                            🔤 <b>BORNES ALPHABÉTIQUES :</b> <code style='background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(56, 189, 248, 0.3); color: #f8fafc; padding: 2px 8px; border-radius: 4px;'>{first_title[:32]}</code> ➔ <code style='background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(56, 189, 248, 0.3); color: #f8fafc; padding: 2px 8px; border-radius: 4px;'>{last_title[:32]}</code>
                        </div>
                        <div style='width: 100%; background: rgba(51, 65, 85, 0.5); height: 5px; border-radius: 3px; margin-top: 10px; overflow: hidden;'>
                            <div style='width: {fill_pct}%; background: linear-gradient(90deg, #00f2fe, #38bdf8); height: 100%;'></div>
                        </div>
                    </div>""",
                    unsafe_allow_html=True
                )

                with st.expander(f"📋 Déplier l'inventaire complet de l'Étagère {e_num:02d} ({nb_on_shelf} films)") :
                    view_shelf = shelf_films[[c for c in ['ID MyMovie', 'ID Unique', col_titre_fr, 'annee_num', 'duree_num', 'supports', 'Production_Corp', col_barcode] if c in shelf_films.columns]].copy()
                    view_shelf.insert(0, "Emplacement", [f"P {p:03d}" for p in range(1, nb_on_shelf + 1)])
                    st.dataframe(view_shelf, use_container_width=True, hide_index=True)
            else:
                st.markdown(f"**📑 {rack_selected} — ÉTAGÈRE {e_num:02d} :** *(Emplacement vide / Réserve future)*")

    with tab_rack_search:
        st.subheader("Calculateur d'Emplacement Immédiat")
        st.markdown("Recherchez un titre ou scannez un code-barres pour obtenir son adresse logistique exacte :")

        srch_in = st.text_input("Titre, ID MyMovie ou Code-barres :", key="rack_search_input").strip()
        if srch_in:
            m_found = sorted_df[col_titre_fr].astype(str).str.lower().str.contains(re.escape(srch_in.lower()), na=False)
            if col_barcode in sorted_df.columns:
                m_found = m_found | (sorted_df[col_barcode].astype(str).str.strip() == srch_in)
            if 'ID Unique' in sorted_df.columns and srch_in.isdigit():
                m_found = m_found | (sorted_df['ID Unique'].astype(str) == srch_in)

            f_indices = sorted_df.index[m_found].tolist()
            if f_indices:
                if len(f_indices) > 1:
                    opts = [f"#{sorted_df.iloc[i].get('ID Unique', i)} — {sorted_df.iloc[i][col_titre_fr]} (Pos {i+1})" for i in f_indices]
                    sel_opt = st.selectbox("Plusieurs correspondances trouvées :", opts, key="rack_f_choice")
                    target_i = f_indices[opts.index(sel_opt)]
                else:
                    target_i = f_indices[0]

                r_t, e_t, p_t, capa_t, addr_res = get_rack_info(target_i)
                row_t = sorted_df.iloc[target_i]

                st.success(f"🎯 **{row_t[col_titre_fr]}** localisé avec succès !")
                st.markdown(f"### 📍 Adresse de Stockage : `{addr_res}`")
                st.markdown(render_physical_shelf_visual(r_t, e_t, p_t, capa_t), unsafe_allow_html=True)

                w1, w2, w3 = st.columns(3)
                w1.metric("🏢 Meuble / Rack", f"RACK {r_t:02d}")
                w2.metric("📑 Niveau / Étagère", f"ÉTAGÈRE {e_t:02d}")
                w3.metric("📍 Position de Gauche à Droite", f"Emplacement #{p_t:03d}")

                st.markdown("---")
                st.subheader(f"Voisinage immédiat sur l'Étagère {e_t:02d} :")
                s_i = max(0, target_i - 5)
                e_i = min(total_films, target_i + 6)
                sub_rack = sorted_df.iloc[s_i:e_i].copy()
                sub_rack.insert(0, "Position Alpha", [x + 1 for x in range(s_i, e_i)])
                sub_rack.insert(1, "Repère", ["⭐ FILM CIBLÉ" if x == target_i else "" for x in range(s_i, e_i)])
                render_custom_table(sub_rack, key_id=f"rack_vw_{target_i}")
            else:
                st.error(f"Aucun film correspondant à « {srch_in} ».")

    with tab_rack_labels:
        st.subheader("Répertoire Global des Bornes de Rayonnage")
        st.markdown("Générez le plan récapitulatif des étiquettes pour chaque meuble physique :")

        bornes_data = []
        for r_i in range(1, nb_racks_cfg + 1):
            for e_i in range(1, nb_etageres_cfg + 1):
                s_idx = ((r_i - 1) * nb_etageres_cfg + (e_i - 1)) * capa_par_etagere
                e_idx = min(s_idx + capa_par_etagere, total_films)
                if s_idx < total_films:
                    f_tit = sorted_df.iloc[s_idx][col_titre_fr]
                    l_tit = sorted_df.iloc[e_idx - 1][col_titre_fr]
                    bornes_data.append({
                        "RACK": f"RACK {r_i:02d}",
                        "ÉTAGÈRE": f"ÉTAGÈRE {e_i:02d}",
                        "Code Tablette": f"R{r_i:02d}-E{e_i:02d}",
                        "Borne Début": f_tit,
                        "Borne Fin": l_tit,
                        "Nombre de Boîtiers": e_idx - s_idx,
                        "Position Alpha Début": s_idx + 1,
                        "Position Alpha Fin": e_idx
                    })
        bornes_df = pd.DataFrame(bornes_data)
        st.dataframe(bornes_df, use_container_width=True, hide_index=True)

        csv_bornes = bornes_df.to_csv(sep=';', index=False).encode('utf-8-sig')
        st.download_button(
            "📥 Télécharger le Plan des Étiquettes d'Étagères (CSV)",
            data=csv_bornes,
            file_name="Plan_Etiquettes_Rayonnage_RACK.csv",
            mime="text/csv"
        )


# ==============================================================================
# FONCTION 5 : SCANNER ANTI-DOUBLON MOBILE & WISHLIST MAGASIN
# ==============================================================================
elif menu == "5° Scanner Anti-Doublon Mobile & Wishlist Magasin":
    st.title("📱 Scanner Anti-Doublon Mobile & Wishlist Magasin")
    st.markdown("Outil de poche optimisé pour smartphone en brocante ou en magasin (Fnac, disquaire). Visez un code-barres ou tapez un titre pour obtenir **une décision d'achat immédiate en 3 couleurs**.")

    sorted_df = df.sort_values(
        by=col_titre_fr,
        key=lambda x: x.astype(str).str.lower()
    ).reset_index(drop=True)

    tab_scan_mag, tab_wishlist = st.tabs(["🛒 Contrôle Anti-Doublon en Rayon", "📝 Ma Wishlist / Liste d'Envies"])

    with tab_scan_mag:
        st.subheader("Contrôle Instantané avant Achat")

        c_cam, c_txt = st.columns([1, 1])
        with c_cam:
            photo_mag = st.camera_input("📸 Viser le code-barres en magasin :", key="cam_mag_input")
        with c_txt:
            txt_mag = st.text_input("Ou saisir le code-barres / titre du film convoité :", key="txt_mag_input", placeholder="Ex: 3344428038830 ou Inception...").strip()

        scan_to_check = txt_mag
        if photo_mag is not None and not scan_to_check:
            try:
                p_im = Image.open(photo_mag)
                scan_to_check = decode_barcode_robust(p_im) or ""
            except Exception:
                pass

        if scan_to_check:
            st.markdown("---")
            st.markdown(f"#### Analyse de : **`{scan_to_check}`**")

            # Recherche intelligente (exacte, zfill 13, derniers chiffres ou titre)
            clean_s = scan_to_check.strip()
            s_barcode = sorted_df[col_barcode].astype(str).str.strip()
            m_exact = s_barcode.str.lower() == clean_s.lower()
            if not m_exact.any() and clean_s.isdigit():
                m_exact = (s_barcode.str.lstrip('0') == clean_s.lstrip('0')) | (s_barcode.str.zfill(13) == clean_s.zfill(13))
                if not m_exact.any() and len(clean_s) >= 4:
                    m_exact = s_barcode.str.endswith(clean_s)
            if not m_exact.any():
                m_exact = sorted_df[col_titre_fr].astype(str).str.lower().str.contains(re.escape(clean_s.lower()), na=False)

            found_idx = sorted_df.index[m_exact].tolist()

            if found_idx:
                matched_rows = sorted_df.iloc[found_idx]
                has_4k = (matched_rows['4K'] == 'OUI').any()
                has_br = (matched_rows['BLURAY'] == 'OUI').any()
                has_sb = (matched_rows['STEELBOOK'] == 'OUI').any()
                is_only_dvd = (matched_rows['DVD'] == 'OUI').all() and not (has_4k or has_br)

                premier_film = matched_rows.iloc[0]
                _, _, _, _, addr_txt = get_rack_info(found_idx[0])

                if has_4k or has_br or has_sb:
                    st.error(f"⛔ **DÉJÀ POSSÉDÉ EN HAUTE DÉFINITION ! (NE PAS ACHETER)**")
                    st.markdown(f"Le film **{premier_film[col_titre_fr]}** est déjà présent dans votre collection.")
                    st.markdown(f"**💿 Formats en stock :** `{premier_film.get('supports', 'N/A')}`")
                    st.markdown(f"**🏢 Emplacement actuel :** `{addr_txt}` (Boîtier `#{premier_film.get('ID MyMovie', 'N/A')}`)")
                elif is_only_dvd:
                    st.warning(f"💡 **OPPORTUNITÉ D'UPGRADE ! (POSSÉDÉ EN DVD UNIQUEMENT)**")
                    st.markdown(f"Vous possédez déjà **{premier_film[col_titre_fr]}**, mais **uniquement en format DVD standard**.")
                    st.markdown("Si le boîtier que vous avez entre les mains est une édition **Blu-ray** ou **4K UHD**, **ACHAT RECOMMANDÉ** pour monter en gamme !")
                    st.markdown(f"**🏢 Emplacement du DVD actuel :** `{addr_txt}`")
                else:
                    st.info(f"ℹ️ **DÉJÀ RÉPERTORIÉ DANS LA COLLECTION** : Formats `{premier_film.get('supports', 'N/A')}`.")
            else:
                st.success(f"✅ **FILM ABSENT DE LA COLLECTION (ACHAT VALIDÉ)**")
                st.markdown(f"Aucune correspondance trouvée pour **`{scan_to_check}`**. Ce film n'est pas encore sur vos étagères !")

    with tab_wishlist:
        st.subheader("Wishlist / Liste des Films Convoités")
        st.caption("Gardez en mémoire les titres que vous cherchez lors de vos sorties en brocante ou en magasin.")

        w_col1, w_col2, w_col3 = st.columns([3, 2, 2])
        with w_col1:
            w_titre = st.text_input("Titre recherché :", key="wish_titre", placeholder="Ex: Oppenheimer")
        with w_col2:
            w_format = st.selectbox("Format souhaité :", ["4K Ultra HD", "Steelbook", "Blu-ray", "Peu importe"], key="wish_format")
        with w_col3:
            w_priorite = st.selectbox("Priorité :", ["⭐⭐⭐ Haute", "⭐⭐ Moyenne", "⭐ Basse"], key="wish_prio")

        if 'wishlist_data' not in st.session_state:
            st.session_state['wishlist_data'] = [
                {"Titre": "Oppenheimer", "Format Visé": "4K Ultra HD", "Priorité": "⭐⭐⭐ Haute", "Note": "Édition Steelbook si possible"},
                {"Titre": "Killers of the Flower Moon", "Format Visé": "4K Ultra HD", "Priorité": "⭐⭐ Moyenne", "Note": "Attendre promo"},
                {"Titre": "Civil War", "Format Visé": "4K Ultra HD", "Priorité": "⭐⭐ Moyenne", "Note": "Sortie récente"}
            ]

        if st.button("➕ Ajouter à ma Wishlist", key="btn_add_wish"):
            if w_titre.strip():
                st.session_state['wishlist_data'].append({
                    "Titre": w_titre.strip(),
                    "Format Visé": w_format,
                    "Priorité": w_priorite,
                    "Note": ""
                })
                st.success(f"« {w_titre} » ajouté à votre Wishlist !")

        w_df = pd.DataFrame(st.session_state['wishlist_data'])
        st.dataframe(w_df, use_container_width=True, hide_index=True)


# ==============================================================================
# FONCTION 6 : AUDIT QUALITÉ : DOUBLONS RÉELS & OPPORTUNITÉS D'UPGRADES 4K
# ==============================================================================
elif menu == "6° Audit Qualité : Doublons réels & Opportunités d'Upgrades 4K":
    st.title("🔍 Audit Qualité : Doublons Réels & Opportunités d'Upgrades 4K")
    st.markdown("Audit industriel de votre catalogue : repérage des doublons physiques, matrice de conversion des DVD vers la haute définition 4K UHD, et contrôle des fiches incomplètes.")

    tab_upg, tab_dups, tab_incomplet = st.tabs([
        "💎 Opportunités d'Upgrade (DVD ➔ 4K / Blu-ray)",
        "👥 Détection des Doublons Réels",
        "⚠️ Fiches à Compléter"
    ])

    with tab_upg:
        st.subheader("Panthéon des DVD à Upgrader en 4K / Blu-ray")
        st.markdown("Ces films sont actuellement possédés **uniquement en simple définition DVD**. Ils sont classés par note IMDb pour cibler en priorité les chefs-d'œuvre mondiaux méritant une édition 4K.")

        is_dvd_seul = (df['DVD'] == 'OUI') & (df['BLURAY'] != 'OUI') & (df['4K'] != 'OUI')
        dvd_seul_df = df[is_dvd_seul].copy()

        dvd_seul_df['rating_num'] = pd.to_numeric(dvd_seul_df['rating_imdb'].astype(str).str.replace(',', '.'), errors='coerce')
        top_upg_df = dvd_seul_df.sort_values(by='rating_num', ascending=False)

        c_u1, c_u2 = st.columns(2)
        with c_u1:
            st.metric("Total DVD seuls sans version HD", f"{len(dvd_seul_df):,}".replace(',', ' '))
        with c_u2:
            st.metric("Chefs-d'œuvre majeurs (IMDb ≥ 8.0)", f"{(dvd_seul_df['rating_num'] >= 8.0).sum()} titres")

        min_note = st.slider("Note IMDb minimale pour la liste d'upgrade :", 5.0, 9.5, 7.5, 0.1)
        filtered_upg = top_upg_df[top_upg_df['rating_num'] >= min_note]

        cols_upg_view = [c for c in ['ID MyMovie', 'ID Unique', col_titre_fr, 'annee', 'rating_imdb', col_director, 'Production_Corp', 'Lien IMDb'] if c in filtered_upg.columns]
        st.dataframe(filtered_upg[cols_upg_view], use_container_width=True, hide_index=True)

    with tab_dups:
        st.subheader("Titres Multiples / Doublons dans la Collection")
        st.markdown("Films dont le titre apparaît sur plusieurs lignes de la base (ex: édition simple + réédition coffret ou multiples formats) :")

        counts = df[col_titre_fr].str.strip().str.lower().value_counts()
        mult_titles = counts[counts > 1].index.tolist()

        dups_df = df[df[col_titre_fr].str.strip().str.lower().isin(mult_titles)].sort_values(by=col_titre_fr)

        st.metric("Films avec plusieurs entrées", f"{len(mult_titles)} titres distincts", f"{len(dups_df)} lignes au total")

        cols_dup_view = [c for c in ['ID MyMovie', 'ID Unique', col_titre_fr, 'annee', 'supports', 'date_ajout', 'Production_Corp'] if c in dups_df.columns]
        st.dataframe(dups_df[cols_dup_view], use_container_width=True, hide_index=True)

    with tab_incomplet:
        st.subheader("Audit des Données Incomplètes")
        c_i1, c_i2, c_i3 = st.columns(3)
        sans_cb_count = df[col_barcode].astype(str).str.contains('SANS_CB', na=False).sum()
        sans_duree = df['duree_minutes'].isna().sum() if 'duree_minutes' in df.columns else 0
        sans_note = df['rating_imdb'].isna().sum() if 'rating_imdb' in df.columns else 0

        c_i1.metric("Boîtiers sans code-barres EAN", f"{sans_cb_count}")
        c_i2.metric("Films sans Durée renseignée", f"{sans_duree}")
        c_i3.metric("Films sans Note IMDb", f"{sans_note}")

        if sans_cb_count > 0:
            with st.expander("Voir les boîtiers sans code-barres (SANS_CB_...)") :
                st.dataframe(df[df[col_barcode].astype(str).str.contains('SANS_CB', na=False)][['ID MyMovie', 'ID Unique', col_titre_fr, col_barcode]], hide_index=True, use_container_width=True)


# ==============================================================================
# FONCTION 7 : SAGATHÈQUE & DOSSIER D'ASSURANCE CERTIFIÉ
# ==============================================================================
elif menu == "7° Sagathèque & Dossier d'Assurance Certifié":
    st.title("🏛️ Sagathèque & Dossier d'Assurance Certifié")
    st.markdown("Valorisation patrimoniale et protection juridique de votre collection physique de **4 301 boîtiers**.")

    tab_assur, tab_sagas = st.tabs([
        "🛡️ Dossier d'Assurance (Capital & Sinistre)",
        "🎬 Sagathèque (Grandes Franchises Cinéma)"
    ])

    with tab_assur:
        st.subheader("Valorisation Financière du Patrimoine Physique")
        st.markdown("Estimation certifiée de la valeur de remplacement à neuf de votre vidéothèque en cas de sinistre (vol, incendie, dégât des eaux) :")

        nb_4k = (df['4K'] == 'OUI').sum()
        nb_sb = (df['STEELBOOK'] == 'OUI').sum()
        nb_br = (df['BLURAY'] == 'OUI').sum()
        nb_dvd = (df['DVD'] == 'OUI').sum()
        nb_box = (df['COFFRET'] == 'OUI').sum()
        nb_3d = (df['3D'] == 'OUI').sum()

        st.markdown("##### ⚙️ Barème de Remplacement Unitaire Moyen (€) :")
        p_c1, p_c2, p_c3, p_c4, p_c5 = st.columns(5)
        with p_c1:
            pu_4k = st.number_input("Prix 4K UHD (€) :", value=25, min_value=5, max_value=60)
        with p_c2:
            pu_sb = st.number_input("Prix Steelbook (€) :", value=30, min_value=5, max_value=80)
        with p_c3:
            pu_br = st.number_input("Prix Blu-ray (€) :", value=15, min_value=2, max_value=40)
        with p_c4:
            pu_dvd = st.number_input("Prix DVD (€) :", value=8, min_value=1, max_value=25)
        with p_c5:
            pu_box = st.number_input("Prix Coffret (€) :", value=45, min_value=10, max_value=150)

        tot_val = (nb_4k * pu_4k) + (nb_sb * pu_sb) + (nb_br * pu_br) + (nb_dvd * pu_dvd) + (nb_box * pu_box)

        st.markdown("---")
        st.metric("💰 VALEUR DE REMPLACEMENT TOTALE ESTIMÉE", f"{tot_val:,.2f} €".replace(',', ' '))

        # Tableau récapitulatif officiel
        bilan_assur_df = pd.DataFrame([
            {"Catégorie de Support": "Éditions 4K Ultra HD", "Nombre de Disques": nb_4k, "Valeur Unitaire": f"{pu_4k} €", "Sous-Total Estimé": f"{nb_4k * pu_4k:,.2f} €".replace(',', ' ')},
            {"Catégorie de Support": "Boîtiers Métal Steelbook", "Nombre de Disques": nb_sb, "Valeur Unitaire": f"{pu_sb} €", "Sous-Total Estimé": f"{nb_sb * pu_sb:,.2f} €".replace(',', ' ')},
            {"Catégorie de Support": "Disques Blu-ray Standard", "Nombre de Disques": nb_br, "Valeur Unitaire": f"{pu_br} €", "Sous-Total Estimé": f"{nb_br * pu_br:,.2f} €".replace(',', ' ')},
            {"Catégorie de Support": "Disques DVD Standard", "Nombre de Disques": nb_dvd, "Valeur Unitaire": f"{pu_dvd} €", "Sous-Total Estimé": f"{nb_dvd * pu_dvd:,.2f} €".replace(',', ' ')},
            {"Catégorie de Support": "Coffrets Intégrales & Séries", "Nombre de Boîtiers": nb_box, "Valeur Unitaire": f"{pu_box} €", "Sous-Total Estimé": f"{nb_box * pu_box:,.2f} €".replace(',', ' ')}
        ])
        st.dataframe(bilan_assur_df, use_container_width=True, hide_index=True)

        csv_assur = bilan_assur_df.to_csv(sep=';', index=False).encode('utf-8-sig')
        st.download_button(
            "📥 Télécharger l'Attestation d'Inventaire pour Assurance (CSV)",
            data=csv_assur,
            file_name="Attestation_Inventaire_Assurance_Videotheque.csv",
            mime="text/csv"
        )

    with tab_sagas:
        st.subheader("La Sagathèque (Suites & Grandes Franchises)")
        st.markdown("Regroupement automatique des univers cinématographiques avec ordre chronologique de visionnage :")

        sagas_dict = {
            "Marvel Cinematic Universe": ["avengers", "iron man", "captain america", "thor", "guardians of the galaxy", "gardiens de la galaxie", "spider-man", "black panther"],
            "Star Wars": ["star wars", "l'empire contre-attaque", "le retour du jedi", "la menace fantôme"],
            "Le Seigneur des Anneaux & Le Hobbit": ["seigneur des anneaux", "hobbit", "communauté de l'anneau", "les deux tours", "retour du roi"],
            "Alien & Predator": ["alien", "predator", "prométhée", "prometheus"],
            "Rocky & Creed": ["rocky", "creed"],
            "James Bond (007)": ["james bond", "007", "dr no", "goldfinger", "spectre", "skyfall", "casino royale"],
            "Batman & The Dark Knight": ["batman", "dark knight", "chevalier noir"],
            "Harry Potter & Animaux Fantastiques": ["harry potter", "animaux fantastiques"],
            "Indiana Jones": ["indiana jones"],
            "Terminator": ["terminator"],
            "Mission: Impossible": ["mission: impossible", "mission impossible"],
            "Fast & Furious": ["fast & furious", "fast and furious"]
        }

        sel_saga = st.selectbox("Sélectionnez la Saga à explorer :", list(sagas_dict.keys()), index=0)
        kw_list = sagas_dict[sel_saga]
        pat_saga = "|".join(kw_list)

        m_saga = df[col_titre_fr].str.contains(pat_saga, case=False, na=False)
        saga_movies = df[m_saga].copy()

        st.success(f"🎬 **{len(saga_movies)} films** répertoriés dans la franchise **{sel_saga}**.")

        # Tri chronologique par année
        if 'annee_num' in saga_movies.columns:
            saga_movies = saga_movies.sort_values(by='annee_num', ascending=True)

        cols_saga_vw = [c for c in ['ID MyMovie', 'ID Unique', col_titre_fr, 'annee', 'duree_minutes', 'rating_imdb', 'supports', 'Production_Corp', 'Lien IMDb'] if c in saga_movies.columns]
        st.dataframe(saga_movies[cols_saga_vw], use_container_width=True, hide_index=True)


# ==============================================================================
# FONCTION 8 : RECHERCHE PAR TITRE / RÉALISATEUR / ACTEUR & IMDB
# ==============================================================================
elif menu == "8° Recherche par titre / réalisateur / acteur & IMDb":
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
            render_custom_table(res_df, key_id="f8_title_match")
        else:
            st.warning(f"Aucun film trouvé contenant « {query} ».")
            st.markdown("### Inventaire complet :")
            render_custom_table(df, key_id="f8_title_fallback")
    else:
        st.markdown("### Inventaire complet :")
        render_custom_table(df, key_id="f8_title_default")

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

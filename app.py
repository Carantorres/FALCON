import streamlit as st
import pandas as pd
import joblib
import re

# 1. Configuración de página WIDE
st.set_page_config(page_title="FALCON | Document AI", layout="wide", initial_sidebar_state="collapsed")

# 2. INYECCIÓN DE CSS Y DIBUJO DEL LOGO (SVG)
# Se eliminan los márgenes excesivos de Streamlit y se crean sombras corporativas
st.markdown("""
    <style>
    /* Resetear márgenes superiores */
    .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
    
    /* Ocultar elementos nativos de Streamlit para look SaaS */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Subtítulo elegante */
    .falcon-subtitle { 
        font-size: 1.05rem; 
        font-weight: 500; 
        color: #64748b; 
        margin-top: -0.5rem; 
        padding-bottom: 1.5rem; 
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 2rem;
        letter-spacing: 0.02em;
    }
    
    /* Estilo de Tarjetas */
    .st-emotion-cache-1jicfl2, .st-emotion-cache-1v0mbdj {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        padding: 1.5rem;
    }
    
    .panel-header { 
        font-size: 1.25rem; 
        font-weight: 700; 
        color: #0f172a; 
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# DIBUJO DEL LOGO EN SVG (Vectores puros, sin imágenes externas)
logo_svg = """
<div style="margin-bottom: 10px;">
    <svg width="280" height="60" viewBox="0 0 300 60" xmlns="http://www.w3.org/2000/svg">
        <g transform="translate(0, 5)">
            <!-- Ala / Cabeza de Halcón geométrica -->
            <path d="M 0 0 L 50 0 L 60 14 L 20 14 L 20 26 L 45 26 L 53 38 L 20 38 L 20 50 L 0 50 Z" fill="#335A7B"/>
            <path d="M 55 0 L 80 0 L 65 14 L 55 14 Z" fill="#5A83A6"/>
            <!-- Texto ALCON -->
            <text x="65" y="42" font-family="'Segoe UI', 'Inter', sans-serif" font-size="44" font-weight="900" fill="#1e293b" letter-spacing="-1">ALCON</text>
        </g>
    </svg>
</div>
"""
st.markdown(logo_svg, unsafe_allow_html=True)
st.markdown('<div class="falcon-subtitle">File Analysis and Learning for Classification and Organization Network</div>', unsafe_allow_html=True)

# 3. Cargar todos los modelos (Cacheado)
@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load('modelo_entrenado.pkl.gz') 
        vectorizer = joblib.load('vectorizador.pkl')
        features = joblib.load('features_cols.pkl')
        targets = joblib.load('target_cols.pkl')
        tenants = joblib.load('tenants_list.pkl')
        raw_map = joblib.load('tenant_project_map.pkl') 
        tenant_project_map = {str(k).strip().lower(): v for k, v in raw_map.items()}
        return model, vectorizer, features, targets, tenants, tenant_project_map
    except FileNotFoundError:
        return None, None, None, None, None, None

model, vectorizer, features, targets, tenants, tenant_project_map = load_ml_components()

if model is None:
    st.error("⚠️ Model files not found. Please ensure all required .pkl and .gz files are uploaded to GitHub.")
    st.stop()

# 4. ÁREA DE TRABAJO (2 COLUMNAS)
left_col, right_col = st.columns([1, 1], gap="large")

# COLUMNA IZQUIERDA: Inputs
with left_col:
    with st.container(border=True):
        st.markdown('<div class="panel-header">📂 Asset Details</div>', unsafe_allow_html=True)
        
        t_col, p_col = st.columns(2)
        with t_col:
            tenant_choice = st.selectbox("Tenant", options=tenants)
        with p_col:
            tenant_lower = tenant_choice.strip().lower()
            valid_projects = tenant_project_map.get(tenant_lower, [])
            project_options = ["-- New Project --"] + valid_projects
            proj_choice = st.selectbox("Project", options=project_options)

        if proj_choice == "-- New Project --":
            project_final_input = st.text_input("Enter New Project Name", placeholder="e.g., MSN 99999")
        else:
            project_final_input = proj_choice

        # TEXT AREA CON EL PLACEHOLDER CORREGIDO
        pdf_input = st.text_area(
            "Directory Structure / PDF Filenames", 
            height=300, 
            placeholder="Paste your PDF filenames here...\n\ne.g.,\n01_Removal_WO1956580.pdf\nMLG_LH_Assy_SN123.pdf\nAttachment_A_Scanned.pdf"
        )

# COLUMNA DERECHA: Resultados
with right_col:
    with st.container(border=True):
        st.markdown('<div class="panel-header">⚡ Inference Results</div>', unsafe_allow_html=True)
        
        if not pdf_input.strip():
            st.info("Awaiting input. Paste your PDF filenames on the left panel to trigger the AI classification.")
        else:
            project_lower = project_final_input.strip().lower()
            clean_text = re.sub(r'[^a-z0-9]', ' ', pdf_input.lower())
            
            if not clean_text.strip():
                st.warning("Please enter valid filename characters.")
            else:
                try:
                    # ML Inference
                    pdf_vec = vectorizer.transform([clean_text])
                    pdf_vec_df = pd.DataFrame(pdf_vec.toarray(), columns=[f"word_{w}" for w in vectorizer.get_feature_names_out()])
                    
                    cat_df = pd.DataFrame(0, index=[0], columns=[c for c in features if not c.startswith('word_')])
                    t_col_name = f'tenant_{tenant_lower}'
                    p_col_name = f'project_{project_lower}'
                    
                    if t_col_name in cat_df.columns: cat_df[t_col_name] = 1
                    if p_col_name in cat_df.columns: cat_df[p_col_name] = 1
                        
                    X_input = pd.concat([cat_df, pdf_vec_df], axis=1)[features] 
                    probas = model.predict_proba(X_input)
                    
                    # Formatting Results
                    results = []
                    for i, cat_col in enumerate(targets):
                        cat_name = cat_col.replace('cat_', '').replace('_present', '')
                        prob = probas[i][0][1] * 100
                        if prob > 2.0:
                            results.append({'category': cat_name, 'probability': prob})
                            
                    results = sorted(results, key=lambda x: x['probability'], reverse=True)
                    
                    if not results:
                        st.info("No historical correlation found for these files.")
                    else:
                        # Custom UI Render for results
                        for res in results[:8]:
                            cat = res['category']
                            prob = res['probability']
                            
                            st.markdown(f"""
                            <div style="margin-bottom: 1.2rem;">
                                <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 0.4rem;">
                                    <span style="font-size: 0.95rem; font-weight: 600; color: #1e293b;">{cat}</span>
                                    <span style="font-size: 0.9rem; font-weight: 700; color: {'#10b981' if prob > 80 else '#3b82f6' if prob > 40 else '#64748b'};">{prob:.1f}%</span>
                                </div>
                                <div style="width: 100%; background-color: #f1f5f9; border-radius: 99px; height: 8px;">
                                    <div style="width: {prob}%; background-color: {'#10b981' if prob > 80 else '#3b82f6' if prob > 40 else '#94a3b8'}; height: 100%; border-radius: 99px; transition: width 0.5s ease;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error("Error processing input data. Please refresh the page.")

import streamlit as st
import pandas as pd
import joblib
import re
import os

# 1. Configuración de página WIDE
st.set_page_config(page_title="FALCON | Document AI", layout="wide")

# CSS Rediseñado para un look más Enterprise
st.markdown("""
    <style>
    .falcon-subtitle { 
        font-size: 1.1rem; 
        font-weight: 500; 
        color: #475569; 
        margin-top: -0.5rem; 
        padding-bottom: 1.5rem; 
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 2rem;
    }
    .panel-header { 
        font-size: 1.25rem; 
        font-weight: 600; 
        color: #0F172A; 
        margin-bottom: 1rem;
    }
    /* Sutil ajuste para que el contenedor de texto de pdf no tenga tanto borde oscuro */
    .stTextArea textarea { border-color: #CBD5E1 !important; }
    </style>
""", unsafe_allow_html=True)

# --- HEADER REDISEÑADO ---
# Usamos columnas para darle un ancho máximo al header y que no se estire infinitamente
head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    if os.path.exists("Falcon.png"):
        st.image("Falcon.png", width=320) # Reduje un poco el ancho para mayor elegancia
    else:
        st.markdown('<div style="font-size: 3rem; font-weight: 800; color: #1E293B; margin-bottom:-10px;">FALCON</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="falcon-subtitle">File Analysis and Learning for Classification and Organization Network</div>', unsafe_allow_html=True)

# 2. Cargar todos los modelos
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
    st.error("Model files not found. Please ensure all required .pkl and .gz files are uploaded.")
    st.stop()

# 3. ÁREA DE TRABAJO EN TARJETAS (CARDS)
left_col, right_col = st.columns([1, 1], gap="large")

# COLUMNA IZQUIERDA: INPUTS ENCAPSULADOS EN UN CONTENEDOR
with left_col:
    with st.container(border=True):
        st.markdown('<div class="panel-header">Asset Details</div>', unsafe_allow_html=True)
        
        t_col, p_col = st.columns(2)
        
        with t_col:
            tenant_choice = st.selectbox("Tenant", options=tenants)
            
        with p_col:
            tenant_lower = tenant_choice.strip().lower()
            valid_projects = tenant_project_map.get(tenant_lower, [])
            project_options = ["-- New Project (Type Below) --"] + valid_projects
            proj_choice = st.selectbox("Project", options=project_options)

        if proj_choice == "-- New Project (Type Below) --":
            project_final_input = st.text_input("Enter New Project Name", placeholder="e.g., MSN 99999")
        else:
            project_final_input = proj_choice

        pdf_input = st.text_area(
            "PDF Filenames (Click outside to update results)", 
            height=280, 
            placeholder="Upload or paste directory list here...\n\nmts.pdf\nllp_sheet.pdf\nengine_log.pdf"
        )

# COLUMNA DERECHA: RESULTADOS ENCAPSULADOS
with right_col:
    with st.container(border=True):
        st.markdown('<div class="panel-header">Inference Results</div>', unsafe_allow_html=True)
        
        if not pdf_input.strip():
            # Estado vacío más limpio
            st.info("💡 Awaiting input. Paste your PDF filenames on the left panel to trigger the AI classification.")
        else:
            project_lower = project_final_input.strip().lower()
            clean_text = re.sub(r'[^a-z0-9]', ' ', pdf_input.lower())
            
            if not clean_text.strip():
                st.warning("Please enter valid filename characters.")
            else:
                try:
                    pdf_vec = vectorizer.transform([clean_text])
                    pdf_vec_df = pd.DataFrame(pdf_vec.toarray(), columns=[f"word_{w}" for w in vectorizer.get_feature_names_out()])
                    
                    cat_df = pd.DataFrame(0, index=[0], columns=[c for c in features if not c.startswith('word_')])
                    t_col_name = f'tenant_{tenant_lower}'
                    p_col_name = f'project_{project_lower}'
                    
                    if t_col_name in cat_df.columns: cat_df[t_col_name] = 1
                    if p_col_name in cat_df.columns: cat_df[p_col_name] = 1
                        
                    X_input = pd.concat([cat_df, pdf_vec_df], axis=1)[features] 
                    probas = model.predict_proba(X_input)
                    
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
                        for res in results[:10]:
                            cat = res['category']
                            prob = res['probability']
                            
                            # Diseño más limpio de los resultados
                            st.markdown(f"<span style='font-size: 0.95rem; font-weight: 600; color: #334155;'>{cat}</span> <span style='float:right; font-weight: 600; color: {'#16A34A' if prob > 80 else '#2563EB' if prob > 40 else '#64748B'};'>{prob:.1f}%</span>", unsafe_allow_html=True)
                            
                            st.markdown(f"""
                                <div style="width: 100%; background-color: #F1F5F9; border-radius: 99px; margin-bottom: 1.2rem; margin-top: 0.3rem;">
                                    <div style="width: {prob}%; background-color: {'#16A34A' if prob > 80 else '#3B82F6' if prob > 40 else '#94A3B8'}; height: 6px; border-radius: 99px;"></div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error("Error processing input data.")

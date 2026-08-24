import streamlit as st
import pandas as pd
import joblib
import re

# 1. Configuración de página limpia
st.set_page_config(page_title="FALCON | Document AI", layout="centered")

st.markdown("""
    <style>
    .falcon-title { font-size: 3.5rem; font-weight: 800; color: #1E293B; margin-bottom: 0; padding-bottom: 0; }
    .falcon-subtitle { font-size: 1.1rem; font-weight: 400; color: #64748B; margin-top: 0; padding-top: 0; margin-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="falcon-title">FALCON</div>', unsafe_allow_html=True)
st.markdown('<div class="falcon-subtitle">File Analysis and Learning for Classification and Organization Network</div>', unsafe_allow_html=True)

# 2. Cargar todos los modelos y listas (AHORA CON EL MAPA DE PROYECTOS)
@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load('modelo_entrenado.pkl.gz') # Recuerda usar tu archivo comprimido .gz
        vectorizer = joblib.load('vectorizador.pkl')
        features = joblib.load('features_cols.pkl')
        targets = joblib.load('target_cols.pkl')
        tenants = joblib.load('tenants_list.pkl')
        tenant_project_map = joblib.load('tenant_project_map.pkl') # <--- NUEVO
        return model, vectorizer, features, targets, tenants, tenant_project_map
    except FileNotFoundError:
        return None, None, None, None, None, None

model, vectorizer, features, targets, tenants, tenant_project_map = load_ml_components()

if model is None:
    st.error("Model files not found. Please ensure all required .pkl and .gz files are uploaded.")
    st.stop()

# 3. Inputs del usuario (Con menús desplegables dependientes)
st.markdown("### Asset Details")
col1, col2 = st.columns(2)

with col1:
    # 1. El usuario elige el Tenant
    tenant_choice = st.selectbox("Tenant", options=tenants)

with col2:
    # 2. Buscamos los proyectos específicos de ese Tenant en el mapa
    # Si el tenant no tiene proyectos, devuelve una lista vacía []
    tenant_lower = tenant_choice.strip().lower()
    valid_projects = tenant_project_map.get(tenant_lower, [])
    
    # 3. Mostramos solo esos proyectos en el segundo menú
    project_options = ["-- New Project (Type Below) --"] + valid_projects
    proj_choice = st.selectbox("Project", options=project_options)

# Si elige crear uno nuevo, mostramos el campo de texto
if proj_choice == "-- New Project (Type Below) --":
    project_final_input = st.text_input("Enter New Project Name", placeholder="e.g., MSN 99999")
else:
    project_final_input = proj_choice

pdf_input = st.text_area("PDF Filenames (Required for classification)", height=150, placeholder="mts.pdf\nllp_sheet.pdf\n(One per line)")

predict_button = st.button("Predict Categories", type="primary", use_container_width=True)

# 4. Lógica de predicción
if predict_button:
    if not pdf_input.strip():
        st.error("Error: PDF Filenames are required to perform a prediction.")
    else:
        with st.spinner("Analyzing historical metadata..."):
            project_lower = project_final_input.strip().lower()
            
            clean_text = re.sub(r'[^a-z0-9]', ' ', pdf_input.lower())
            pdf_vec = vectorizer.transform([clean_text])
            pdf_vec_df = pd.DataFrame(pdf_vec.toarray(), columns=[f"word_{w}" for w in vectorizer.get_feature_names_out()])
            
            cat_df = pd.DataFrame(0, index=[0], columns=[c for c in features if not c.startswith('word_')])
            t_col = f'tenant_{tenant_lower}'
            p_col = f'project_{project_lower}'
            
            if t_col in cat_df.columns:
                cat_df[t_col] = 1
            if p_col in cat_df.columns:
                cat_df[p_col] = 1
                
            X_input = pd.concat([cat_df, pdf_vec_df], axis=1)
            X_input = X_input[features] 
            
            probas = model.predict_proba(X_input)
            
            results = []
            for i, cat_col in enumerate(targets):
                cat_name = cat_col.replace('cat_', '').replace('_present', '')
                prob_present = probas[i][0][1] * 100
                if prob_present > 2.0:
                    results.append({'category': cat_name, 'probability': prob_present})
                    
            results = sorted(results, key=lambda x: x['probability'], reverse=True)
            
            st.markdown("---")
            st.markdown("### Prediction Results")
            
            if not results:
                st.info("No historical correlation found for these files.")
            else:
                for res in results[:10]:
                    cat = res['category']
                    prob = res['probability']
                    st.markdown(f"**{cat}** ({prob:.1f}%)")
                    st.progress(int(prob) / 100)

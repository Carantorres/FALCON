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

# 2. Cargar todos los modelos y listas
@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load('modelo_entrenado.pkl.gz')
        vectorizer = joblib.load('vectorizador.pkl')
        features = joblib.load('features_cols.pkl')
        targets = joblib.load('target_cols.pkl')
        tenants = joblib.load('tenants_list.pkl')
        projects = joblib.load('projects_list.pkl')
        return model, vectorizer, features, targets, tenants, projects
    except FileNotFoundError:
        return None, None, None, None, None, None

model, vectorizer, features, targets, tenants, projects = load_ml_components()

if model is None:
    st.error("Model files not found. Please ensure all 6 .pkl files are uploaded.")
    st.stop()

# 3. Inputs del usuario (Con menús desplegables)
st.markdown("### Asset Details")
col1, col2 = st.columns(2)

with col1:
    tenant_choice = st.selectbox("Tenant", options=tenants)

with col2:
    # Opción para elegir un proyecto existente o crear uno nuevo
    project_options = ["-- New Project (Type Below) --"] + projects
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
            # Normalizar a minúsculas
            tenant_lower = tenant_choice.strip().lower()
            project_lower = project_final_input.strip().lower()
            
            # 1. Transformar el texto con el Vectorizador de la IA
            clean_text = re.sub(r'[^a-z0-9]', ' ', pdf_input.lower())
            pdf_vec = vectorizer.transform([clean_text])
            pdf_vec_df = pd.DataFrame(pdf_vec.toarray(), columns=[f"word_{w}" for w in vectorizer.get_feature_names_out()])
            
            # 2. Preparar variables de Tenant y Project
            cat_df = pd.DataFrame(0, index=[0], columns=[c for c in features if not c.startswith('word_')])
            t_col = f'tenant_{tenant_lower}'
            p_col = f'project_{project_lower}'
            
            if t_col in cat_df.columns:
                cat_df[t_col] = 1
            if p_col in cat_df.columns:
                cat_df[p_col] = 1
                
            # 3. Unir todo en el formato exacto que el modelo espera
            X_input = pd.concat([cat_df, pdf_vec_df], axis=1)
            X_input = X_input[features] # Ordenar columnas idéntico al entrenamiento
            
            # 4. Predecir
            probas = model.predict_proba(X_input)
            
            results = []
            for i, cat_col in enumerate(targets):
                cat_name = cat_col.replace('cat_', '').replace('_present', '')
                prob_present = probas[i][0][1] * 100
                if prob_present > 2.0:
                    results.append({'category': cat_name, 'probability': prob_present})
                    
            results = sorted(results, key=lambda x: x['probability'], reverse=True)
            
            # 5. Mostrar resultados
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

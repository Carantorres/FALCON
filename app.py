import streamlit as st
import pandas as pd
import joblib
import re

# 1. Configuración de página (AHORA ES WIDE PARA USAR TODA LA PANTALLA)
st.set_page_config(page_title="FALCON | Document AI", layout="wide")

st.markdown("""
    <style>
    .falcon-title { font-size: 3.5rem; font-weight: 800; color: #1E293B; margin-bottom: 0; padding-bottom: 0; }
    .falcon-subtitle { font-size: 1.1rem; font-weight: 400; color: #64748B; margin-top: 0; padding-top: 0; margin-bottom: 2.5rem; }
    .results-header { font-size: 1.5rem; font-weight: 600; color: #1E293B; margin-bottom: 1.2rem; border-bottom: 2px solid #E2E8F0; padding-bottom: 0.5rem;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="falcon-title">FALCON</div>', unsafe_allow_html=True)
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

# 3. DIVIDIR LA PANTALLA EN DOS COLUMNAS (50% / 50%)
left_col, right_col = st.columns([1, 1], gap="large")

# COLUMNA IZQUIERDA: INPUTS
with left_col:
    st.markdown("### Asset Details")
    
    # Sub-columnas para Tenant y Project
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

    # Text area (Actualiza al hacer clic por fuera del cuadro)
    pdf_input = st.text_area(
        "PDF Filenames (Click outside to update results)", 
        height=300, 
        placeholder="mts.pdf\nllp_sheet.pdf\n(One per line)"
    )

# COLUMNA DERECHA: RESULTADOS (ACTUALIZACIÓN AUTOMÁTICA)
with right_col:
    st.markdown('<div class="results-header">Prediction Results</div>', unsafe_allow_html=True)
    
    if not pdf_input.strip():
        # Mensaje por defecto cuando está vacío
        st.info("👈 Start typing PDF filenames on the left to see real-time predictions.")
    else:
        project_lower = project_final_input.strip().lower()
        clean_text = re.sub(r'[^a-z0-9]', ' ', pdf_input.lower())
        
        # Validar que el texto limpio tenga algo
        if not clean_text.strip():
            st.info("👈 Please enter valid filename characters.")
        else:
            try:
                # 1. Transformar texto
                pdf_vec = vectorizer.transform([clean_text])
                pdf_vec_df = pd.DataFrame(pdf_vec.toarray(), columns=[f"word_{w}" for w in vectorizer.get_feature_names_out()])
                
                # 2. Asignar Tenant/Project
                cat_df = pd.DataFrame(0, index=[0], columns=[c for c in features if not c.startswith('word_')])
                t_col_name = f'tenant_{tenant_lower}'
                p_col_name = f'project_{project_lower}'
                
                if t_col_name in cat_df.columns:
                    cat_df[t_col_name] = 1
                if p_col_name in cat_df.columns:
                    cat_df[p_col_name] = 1
                    
                # 3. Unir y Predecir
                X_input = pd.concat([cat_df, pdf_vec_df], axis=1)
                X_input = X_input[features] 
                
                probas = model.predict_proba(X_input)
                
                # 4. Procesar Resultados
                results = []
                for i, cat_col in enumerate(targets):
                    cat_name = cat_col.replace('cat_', '').replace('_present', '')
                    prob_present = probas[i][0][1] * 100
                    if prob_present > 2.0:
                        results.append({'category': cat_name, 'probability': prob_present})
                        
                results = sorted(results, key=lambda x: x['probability'], reverse=True)
                
                # 5. Renderizar Resultados
                if not results:
                    st.info("No historical correlation found for these files.")
                else:
                    for res in results[:10]:
                        cat = res['category']
                        prob = res['probability']
                        st.markdown(f"**{cat}** ({prob:.1f}%)")
                        
                        # Colores dinámicos
                        color = "normal"
                        if prob > 80: color = "green"
                        elif prob < 40: color = "red"
                            
                        # Pequeño hack de HTML para barras con colores
                        st.markdown(f"""
                            <div style="width: 100%; background-color: #e2e8f0; border-radius: 4px; margin-bottom: 15px;">
                                <div style="width: {prob}%; background-color: {'#3b82f6' if prob >= 40 else '#94a3b8'}; height: 8px; border-radius: 4px;"></div>
                            </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error("Error processing input data.")

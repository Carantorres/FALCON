import streamlit as st
import pandas as pd
import joblib

# 1. Configuración de página limpia
st.set_page_config(page_title="FALCON | Document AI", layout="centered")

# Estilo minimalista solo para el título
st.markdown("""
    <style>
    .falcon-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .falcon-subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        color: #64748B;
        margin-top: 0;
        padding-top: 0;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Encabezado principal
st.markdown('<div class="falcon-title">FALCON</div>', unsafe_allow_html=True)
st.markdown('<div class="falcon-subtitle">File Analysis and Learning for Classification and Organization Network</div>', unsafe_allow_html=True)

# 3. Cargar el modelo
@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load('modelo_entrenado.pkl')
        features = joblib.load('features_cols.pkl')
        targets = joblib.load('target_cols.pkl')
        return model, features, targets
    except FileNotFoundError:
        return None, None, None

model, features, targets = load_ml_components()

if model is None:
    st.error("⚠️ Model files not found. Please upload 'modelo_entrenado.pkl', 'features_cols.pkl', and 'target_cols.pkl'.")
    st.stop()

# 4. Inputs del usuario (Diseño simple)
st.markdown("### Asset Details")
col1, col2 = st.columns(2)
with col1:
    tenant_input = st.text_input("Tenant Name (e.g., AAR, BBAM)")
with col2:
    project_input = st.text_input("Project Name (Optional)")
    
pdf_input = st.text_area("PDF Filenames", height=150, placeholder="LLP_Inspection_Report.pdf\nEASA_Form_1.pdf\n(One per line)")

predict_button = st.button("Predict Categories", type="primary", use_container_width=True)

# 5. Lógica de predicción
if predict_button:
    if not tenant_input and not pdf_input:
        st.warning("Please provide at least a Tenant or PDF filenames.")
    else:
        with st.spinner("Analyzing historical patterns..."):
            tenant = tenant_input.strip().lower()
            project = project_input.strip().lower()
            pdfs = [p.strip() for p in pdf_input.split('\n') if p.strip()]
            
            input_features = pd.DataFrame(0, index=[0], columns=features)
            
            if f'tenant_{tenant}' in features:
                input_features[f'tenant_{tenant}'] = 1
            if f'project_{project}' in features:
                input_features[f'project_{project}'] = 1
                
            input_features['total_pdfs_count'] = len(pdfs)
            for pdf in pdfs:
                pdf_lower = pdf.lower()
                for col in features:
                    if col.startswith('filename_contains_'):
                        keyword = col.replace('filename_contains_', '').replace('_', ' ').lower()
                        if keyword in pdf_lower:
                            input_features.at[0, col] += 1
                            
            probas = model.predict_proba(input_features)
            
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
                st.info("No highly probable categories found.")
            else:
                for res in results[:10]:
                    cat = res['category']
                    prob = res['probability']
                    st.markdown(f"**{cat}** ({prob:.1f}%)")
                    st.progress(int(prob) / 100)

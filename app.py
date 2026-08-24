import streamlit as st
import pandas as pd
import numpy as np
import time

# 1. CONFIGURACIÓN DE PÁGINA (Debe ser la primera línea)
st.set_page_config(
    page_title="FALCON | Aviation Document AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. INYECCIÓN DE CSS EMPRESARIAL
# Esto transforma el look predeterminado de Streamlit en un dashboard corporativo
st.markdown("""
    <style>
        /* Tipografía y colores base */
        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Headers FALCON */
        .falcon-title {
            font-size: 3rem;
            font-weight: 800;
            color: #0f172a; /* Slate 900 */
            letter-spacing: -0.05em;
            margin-bottom: 0rem;
            padding-bottom: 0rem;
            line-height: 1.1;
        }
        .falcon-subtitle {
            font-size: 1.25rem;
            font-weight: 500;
            color: #3b82f6; /* Blue 500 */
            margin-top: 0.5rem;
            margin-bottom: 0.2rem;
        }
        .falcon-acronym {
            font-size: 0.85rem;
            font-weight: 400;
            color: #64748b; /* Slate 500 */
            letter-spacing: 0.05em;
            margin-bottom: 2rem;
            text-transform: uppercase;
        }

        /* Estilo de Tarjetas/Paneles */
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .section-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1e293b;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
            margin-top: 2rem;
        }
        
        /* Árbol de directorios simulado */
        .directory-tree {
            background-color: #0f172a;
            color: #e2e8f0;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.85rem;
            padding: 1rem;
            border-radius: 4px;
            height: 300px;
            overflow-y: auto;
        }
        
        /* Ocultar elementos predeterminados de Streamlit para un look más limpio */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 3. SIDEBAR: HISTORIAL Y CONFIGURACIÓN
with st.sidebar:
    st.markdown("<h2 style='color: #0f172a; font-weight: 700;'>FALCON NETWORK</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.8rem; color: #64748b;'>SYSTEM VERSION 2.4.1</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<p style='font-weight: 600; font-size: 0.9rem;'>ANALYSIS HISTORY</p>", unsafe_allow_html=True)
    
    # Simulación de historial
    history_data = pd.DataFrame({
        "Date": ["2026-08-24 11:30", "2026-08-23 15:45", "2026-08-22 09:12"],
        "Tenant": ["AAR", "BBAM", "Lufthansa"],
        "Files": [142, 89, 412]
    })
    st.dataframe(history_data, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.markdown("<p style='font-weight: 600; font-size: 0.9rem;'>SYSTEM SETTINGS</p>", unsafe_allow_html=True)
    st.selectbox("Tenant Configuration", ["Auto-Detect", "AAR", "BBAM", "Lufthansa Technik", "United Airlines"])
    st.slider("Confidence Threshold", 0.0, 1.0, 0.75, 0.05)

# 4. ENCABEZADO PRINCIPAL
st.markdown("<div class='falcon-title'>FALCON</div>", unsafe_allow_html=True)
st.markdown("<div class='falcon-subtitle'>Intelligent Aviation Document Classification Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='falcon-acronym'>File Analysis and Learning for Classification and Organization Network</div>", unsafe_allow_html=True)

# 5. PANEL DE ESTADÍSTICAS (KPIs)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Global Model Accuracy", value="94.2%", delta="+1.1%")
with col2:
    st.metric(label="Files Processed (Today)", value="1,248")
with col3:
    st.metric(label="Average Processing Time", value="0.12s / file", delta="-0.03s", delta_color="inverse")
with col4:
    st.metric(label="Active Classifications", value="107 Categories")

# 6. ÁREA DE TRABAJO PRINCIPAL
st.markdown("<div class='section-header'>Workspace Analysis</div>", unsafe_allow_html=True)

work_col1, work_col2 = st.columns([1, 2])

with work_col1:
    st.markdown("<p style='font-weight: 600; color:#334155;'>1. Source Input</p>", unsafe_allow_html=True)
    # Streamlit file uploader modificado para aceptar múltiples archivos y simular carga de carpetas
    uploaded_files = st.file_uploader("Upload directory structure or files", accept_multiple_files=True)
    
    st.markdown("<p style='font-weight: 600; color:#334155; margin-top: 1rem;'>2. Directory Tree Mapping</p>", unsafe_allow_html=True)
    
    # Vista de árbol simulada
    tree_mockup = """[FALCON] Scanning metadata...
/ ROOT_ASSET_MSN_32663
├── / Engine_Logs
│   ├── LLP_Trace_Data_Rev2.pdf
│   └── Status_Engine_Update.pdf
├── / Landing_Gear
│   ├── MLG_Removal_Tag.pdf
│   └── NLG_Overhaul_Report.pdf
└── EASA_Form_1_Cert.pdf"""
    
    st.markdown(f"<div class='directory-tree'><pre>{tree_mockup}</pre></div>", unsafe_allow_html=True)
    
    analyze_btn = st.button("EXECUTE CLASSIFICATION", type="primary", use_container_width=True)

with work_col2:
    st.markdown("<p style='font-weight: 600; color:#334155;'>3. Real-Time Inference Matrix</p>", unsafe_allow_html=True)
    
    if analyze_btn:
        # Simulación de carga de progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
            status_text.text(f"Processing metadata... {i+1}%")
            
        status_text.text("Classification Complete.")
        
        # Simulación de resultados
        results_data = {
            "Filename": [
                "LLP_Trace_Data_Rev2.pdf", 
                "Status_Engine_Update.pdf", 
                "MLG_Removal_Tag.pdf", 
                "NLG_Overhaul_Report.pdf", 
                "EASA_Form_1_Cert.pdf"
            ],
            "Inferred Path Context": ["/Engine_Logs", "/Engine_Logs", "/Landing_Gear", "/Landing_Gear", "/ROOT"],
            "Predicted Category": ["LLP Sheet", "Status Report", "Removal Tag", "Overhaul Report", "Airworthiness Cert"],
            "Confidence": ["98.5%", "92.1%", "95.3%", "97.8%", "99.2%"]
        }
        df_results = pd.DataFrame(results_data)
        
        # Aplicar estilos corporativos a la tabla
        st.dataframe(
            df_results,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence Level",
                    help="Model inference certainty",
                    format="%f",
                    min_value=0,
                    max_value=100,
                )
            }
        )
        
        st.success("Metadata analysis successfully mapped to 5 target categories.")
    else:
        # Estado vacío
        st.info("System idle. Upload files and execute classification to populate the inference matrix.")
        
        # Tabla vacía para mantener la estructura visual
        empty_df = pd.DataFrame(columns=["Filename", "Inferred Path Context", "Predicted Category", "Confidence"])
        st.dataframe(empty_df, use_container_width=True, hide_index=True)
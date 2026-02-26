import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai

# 1. Configuración Estética de la Página
st.set_page_config(page_title="Corbel-Audit AI", layout="wide")

# Estilo profesional
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# 2. Carga de Inteligencia (Cerebro Numérico)
@st.cache_resource
def load_assets():
    model = joblib.load('modelo_corbel.pkl')
    sc_x = joblib.load('scaler_x.pkl')
    sc_y = joblib.load('scaler_y.pkl')
    return model, sc_x, sc_y

modelo, scaler_x, scaler_y = load_assets()

# 3. Interfaz de Usuario
st.title("🏗️ Agente Inteligente de Auditoría Estructural:")
st.title("Implementación de un Modelo Sustituto (MLP) y un LLM para la Evaluación de Capacidad de Carga en Ménsulas de Concreto Reforzado.")
st.write("Agente inteligente de auditoría estructural para la evaluación de capacidad de carga (Pu) en ménsulas de concreto reforzado, integrando un modelo sustituto MLP con inteligencia artificial narrativa.")

with st.sidebar:
    st.header("⚙️ Parámetros de Diseño")
    fc = st.slider("Resistencia Concreto ($f'c$ MPa)", 21, 35, 28)
    b = st.slider("Ancho de sección b (mm)", 200, 400, 300)
    d = st.slider("Peralte efectivo d (mm)", 300, 600, 500)
    a = st.slider("Brazo de palanca a (mm)", 100, 500, 250)
    rho = st.slider("Cuantía de acero ($\\rho$)", 0.0040, 0.0150, 0.0100, format="%.4f")
    st.markdown("---")
    api_key = st.text_input("Ingresa tu Gemini API Key", type="password")

# 4. Lógica de Auditoría Completa
if st.button("🚀 Iniciar Auditoría Completa"):
    if not api_key:
        st.error("Por favor, ingresa tu API Key para generar el reporte técnico.")
    else:
        # A. Predicción de Carga con la Red Neuronal (R2=0.94)
        input_data = pd.DataFrame([[fc, b, d, a, rho]], columns=['fc', 'b', 'd', 'a', 'rho'])
        X_scaled = scaler_x.transform(input_data)
        y_scaled = modelo.predict(X_scaled)
        pu_kn = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))[0][0]

        # B. Presentación de Resultados
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Capacidad de Carga Predicha ($P_u$)", f"{pu_kn:.2f} kN")
            ad_ratio = a/d
            st.info(f"Relación $a/d$: {ad_ratio:.2f}")
            if ad_ratio < 1.0:
                st.warning("⚠️ Elemento clasificado como Ménsula Corta.")

        with col2:
            st.subheader("📋 Informe Técnico del Agente IA")
            with st.spinner("Redactando diagnóstico experto..."):
                genai.configure(api_key=api_key)
                llm = genai.GenerativeModel('gemini-2.5-flash')
                
                # UTILIZANDO EL PROMPT DEL PUNTO 6
                prompt = f"""
                Eres un ingeniero especialista en diseño estructural.
                Debes auditar el siguiente elemento estructural:
                - Resistencia del Concreto ($f'c$): {fc} MPa
                - Geometría: Ancho={b}mm, Peralte efectivo={d}mm, Brazo={a}mm
                - Cuantía de refuerzo ($\\rho$): {rho*100:.2f}%
                - Capacidad de Carga Predicha por IA ($P_u$): {pu_kn:.2f} kN

                INSTRUCCIONES:
                1. Calcula mentalmente la relación $a/d$ y clasifica si es una ménsula corta.
                2. Evalúa si la carga de {pu_kn:.2f} kN es coherente para estas dimensiones.
                3. Redacta una recomendación de seguridad o refuerzo citando criterios del ACI 318.

                Formato: 3 párrafos técnicos pero claros, no los elabores como una nota, manéjalos como conclusiones del análisis realizado.
                """
                try:
                    response = llm.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error de conexión con la IA: {e}")

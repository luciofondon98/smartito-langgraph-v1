import streamlit as st
from main import graph, FlowState, ENDPOINTS_DOC, BUSINESS_CONTEXT
import logging
import time
from datetime import datetime

st.set_page_config(page_title="SMARTito Chatbot", page_icon="✈️", layout="wide")
st.title("SMARTito Chatbot ✈️")
st.markdown("Interactúa con tu API de métricas aéreas usando lenguaje natural.")

# Configurar logging para capturar en Streamlit
if "log_messages" not in st.session_state:
    st.session_state["log_messages"] = []

if "history" not in st.session_state:
    st.session_state["history"] = []

# Función para capturar logs
def capture_logs():
    class StreamlitHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            st.session_state["log_messages"].append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": log_entry,
                "level": record.levelname
            })
    
    # Configurar handler personalizado
    logger = logging.getLogger()
    streamlit_handler = StreamlitHandler()
    streamlit_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(streamlit_handler)

# Capturar logs
capture_logs()

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown("**Estado del sistema:**")
    
    # Mostrar logs en sidebar
    if st.session_state["log_messages"]:
        st.subheader("📊 Logs en tiempo real")
        for log in st.session_state["log_messages"][-10:]:  # Últimos 10 logs
            if "🤖" in log["message"]:
                st.success(log["message"])
            elif "❌" in log["message"]:
                st.error(log["message"])
            elif "🌐" in log["message"]:
                st.info(log["message"])
            else:
                st.text(log["message"])

# Área principal
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat")
    
    user_input = st.text_input("Haz tu pregunta:", key="user_input", placeholder="Ej: ¿Cuál fue el tráfico del website durante los últimos 4 días?")
    
    if st.button("🚀 Enviar", type="primary") and user_input:
        # Limpiar logs anteriores
        st.session_state["log_messages"] = []
        
        # Mostrar indicador de procesamiento
        with st.spinner("🤖 Procesando con agentes..."):
            state = FlowState(
                user_question=user_input,
                technical_query="",
                raw_data="",
                final_answer="",
                api_documentation=ENDPOINTS_DOC,
                business_context=BUSINESS_CONTEXT
            )
            
            # Ejecutar el flujo
            result = graph.invoke(state)
            
            # Agregar a historial
            st.session_state["history"].append(("user", user_input))
            st.session_state["history"].append(("bot", result["final_answer"]))
            
            # Mostrar resultado
            st.success("✅ Respuesta generada")
            st.rerun()

    # Mostrar historial de chat
    if st.session_state["history"]:
        st.subheader("📝 Historial de conversación")
        for role, msg in reversed(st.session_state["history"]):
            if role == "user":
                st.markdown(f"**👤 Tú:** {msg}")
            else:
                st.markdown(f"**🤖 SMARTito:** {msg}")

with col2:
    st.subheader("🔍 Flujo de Agentes")
    
    # Mostrar información del flujo
    st.markdown("""
    **Flujo actual:**
    1. 🔧 **Ingeniero de Insights** → Convierte pregunta en consulta técnica
    2. 🌐 **Agente de Datos** → Consulta la API SMARTito
    3. 📊 **Analista de Negocio** → Genera respuesta con contexto aéreo
    """)
    
    # Mostrar endpoints disponibles
    st.markdown("**📚 Endpoints disponibles:**")
    st.markdown("""
    - `/api/v1/realtime/` - Datos de conversión
    - `/api/v1/looks/` - Búsquedas por ruta  
    - `/health` - Estado de la API
    """)
    
    # Botón para limpiar historial
    if st.button("🗑️ Limpiar historial"):
        st.session_state["history"] = []
        st.session_state["log_messages"] = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>Powered by LangGraph + OpenAI + Streamlit</small>
</div>
""", unsafe_allow_html=True) 
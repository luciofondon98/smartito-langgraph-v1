import streamlit as st
import time
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from main import graph, FlowState, ENDPOINTS_DOC, BUSINESS_CONTEXT
import logging
import threading
import queue

# Configuración de la página
st.set_page_config(
    page_title="SMARTito Agent Flow Visualization", 
    page_icon="🔄", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔄 SMARTito Agent Flow Visualization")
st.markdown("Visualización en vivo del flujo de agentes colaborativos")

# Estado de la sesión para la visualización
if "flow_data" not in st.session_state:
    st.session_state.flow_data = {
        "agents": [],
        "connections": [],
        "current_step": 0,
        "is_running": False,
        "logs": [],
        "performance_metrics": {}
    }

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Función para capturar logs en tiempo real
class StreamlitFlowHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Detectar tipo de agente y acción
        agent_type = "System"
        action = "Info"
        
        if "[AGENTE]" in log_entry:
            if "Ingeniero de Insights" in log_entry:
                agent_type = "Engineer"
                action = "Processing"
            elif "Data Engineer" in log_entry:
                agent_type = "Data"
                action = "API Call"
            elif "Analista de Negocio" in log_entry:
                agent_type = "Analyst"
                action = "Analysis"
        
        st.session_state.flow_data["logs"].append({
            "timestamp": timestamp,
            "agent": agent_type,
            "action": action,
            "message": log_entry,
            "level": record.levelname
        })

# Configurar logging
logger = logging.getLogger()
streamlit_handler = StreamlitFlowHandler()
streamlit_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(streamlit_handler)

# Función para crear el gráfico del flujo de agentes
def create_agent_flow_chart():
    # Definir posiciones de los agentes
    agent_positions = {
        "Engineer": {"x": 0.2, "y": 0.5, "name": "🔧 Ingeniero de Insights"},
        "Data": {"x": 0.5, "y": 0.5, "name": "🌐 Data Engineer"},
        "Analyst": {"x": 0.8, "y": 0.5, "name": "📊 Analista de Negocio"}
    }
    
    # Crear figura
    fig = go.Figure()
    
    # Agregar nodos de agentes
    for agent, pos in agent_positions.items():
        # Determinar color basado en el estado actual
        color = "#e0e0e0"  # Gris por defecto
        if st.session_state.flow_data.get("current_agent") == agent:
            color = "#ff6b6b"  # Rojo para agente activo
        elif agent in [log["agent"] for log in st.session_state.flow_data["logs"][-10:]]:
            color = "#4ecdc4"  # Verde para agentes recientes
        
        fig.add_trace(go.Scatter(
            x=[pos["x"]],
            y=[pos["y"]],
            mode="markers+text",
            marker=dict(size=50, color=color, line=dict(width=2, color="black")),
            text=pos["name"],
            textposition="middle center",
            name=agent,
            showlegend=False
        ))
    
    # Agregar conexiones entre agentes
    connections = [
        ("Engineer", "Data"),
        ("Data", "Analyst")
    ]
    
    for start, end in connections:
        start_pos = agent_positions[start]
        end_pos = agent_positions[end]
        
        # Determinar si la conexión está activa
        line_color = "#cccccc"
        line_width = 2
        
        # Verificar si hay tráfico reciente en esta conexión
        recent_logs = st.session_state.flow_data["logs"][-20:]
        if any(log["agent"] in [start, end] for log in recent_logs):
            line_color = "#2196F3"
            line_width = 4
        
        fig.add_trace(go.Scatter(
            x=[start_pos["x"], end_pos["x"]],
            y=[start_pos["y"], end_pos["y"]],
            mode="lines",
            line=dict(color=line_color, width=line_width),
            showlegend=False,
            hoverinfo="skip"
        ))
    
    # Configurar layout
    fig.update_layout(
        title="Flujo de Agentes en Tiempo Real",
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        plot_bgcolor="white",
        height=400,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

# Función para crear gráfico de métricas de performance
def create_performance_chart():
    if not st.session_state.flow_data["logs"]:
        return None
    
    # Agrupar logs por agente
    agent_metrics = {}
    for log in st.session_state.flow_data["logs"]:
        agent = log["agent"]
        if agent not in agent_metrics:
            agent_metrics[agent] = {"count": 0, "actions": []}
        agent_metrics[agent]["count"] += 1
        agent_metrics[agent]["actions"].append(log["action"])
    
    # Crear gráfico de barras
    agents = list(agent_metrics.keys())
    counts = [agent_metrics[agent]["count"] for agent in agents]
    
    fig = px.bar(
        x=agents,
        y=counts,
        title="Actividad por Agente",
        labels={"x": "Agente", "y": "Acciones"},
        color=agents,
        color_discrete_map={
            "Engineer": "#ff6b6b",
            "Data": "#4ecdc4", 
            "Analyst": "#45b7d1",
            "System": "#96ceb4"
        }
    )
    
    fig.update_layout(height=300, showlegend=False)
    return fig

# Función para ejecutar el flujo en un hilo separado
def run_flow_thread(user_question, result_queue):
    try:
        # Reiniciar estado de visualización
        st.session_state.flow_data["logs"] = []
        st.session_state.flow_data["current_agent"] = "Engineer"
        
        state = FlowState(
            user_question=user_question,
            technical_query="",
            raw_data="",
            final_answer="",
            api_documentation=ENDPOINTS_DOC,
            business_context=BUSINESS_CONTEXT,
            conversation_history=st.session_state.conversation_history,
            last_query_context={}
        )
        
        # Ejecutar flujo
        result = graph.invoke(state)
        
        # Actualizar historial de conversación
        st.session_state.conversation_history = result.get("conversation_history", [])
        
        result_queue.put(("success", result))
        
    except Exception as e:
        result_queue.put(("error", str(e)))

# Interfaz principal
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎯 Control Panel")
    
    # Input para pregunta
    user_input = st.text_input(
        "Haz tu pregunta:", 
        key="user_input", 
        placeholder="Ej: ¿Cuál fue el tráfico del website durante los últimos 4 días?"
    )
    
    col1_1, col1_2 = st.columns(2)
    
    with col1_1:
        if st.button("🚀 Ejecutar Flujo", type="primary") and user_input:
            st.session_state.flow_data["is_running"] = True
            st.session_state.flow_data["logs"] = []
            
            # Ejecutar en hilo separado
            result_queue = queue.Queue()
            thread = threading.Thread(
                target=run_flow_thread, 
                args=(user_input, result_queue)
            )
            thread.start()
            
            # Mostrar progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simular progreso mientras se ejecuta
            for i in range(100):
                time.sleep(0.05)
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text("🔧 Ingeniero de Insights procesando...")
                elif i < 60:
                    status_text.text("🌐 Data Engineer consultando API...")
                else:
                    status_text.text("📊 Analista de Negocio generando respuesta...")
                
                # Verificar si terminó
                try:
                    result_type, result_data = result_queue.get_nowait()
                    if result_type == "success":
                        st.success("✅ Flujo completado exitosamente!")
                        st.session_state.flow_data["is_running"] = False
                        break
                    elif result_type == "error":
                        st.error(f"❌ Error: {result_data}")
                        st.session_state.flow_data["is_running"] = False
                        break
                except queue.Empty:
                    pass
            
            progress_bar.empty()
            status_text.empty()
    
    with col1_2:
        if st.button("🔄 Limpiar Logs"):
            st.session_state.flow_data["logs"] = []
            st.rerun()
    
    # Visualización del flujo
    st.subheader("🔄 Flujo de Agentes")
    flow_chart = create_agent_flow_chart()
    st.plotly_chart(flow_chart, use_container_width=True)
    
    # Métricas de performance
    st.subheader("📊 Métricas de Performance")
    perf_chart = create_performance_chart()
    if perf_chart:
        st.plotly_chart(perf_chart, use_container_width=True)

with col2:
    st.subheader("📝 Logs en Tiempo Real")
    
    # Mostrar logs más recientes
    if st.session_state.flow_data["logs"]:
        for log in reversed(st.session_state.flow_data["logs"][-20:]):
            # Determinar color basado en el agente
            color_map = {
                "Engineer": "🔧",
                "Data": "🌐", 
                "Analyst": "📊",
                "System": "⚙️"
            }
            
            icon = color_map.get(log["agent"], "📝")
            
            with st.container():
                st.markdown(f"**{icon} {log['timestamp']}** - {log['agent']}")
                st.text(log["message"][:100] + "..." if len(log["message"]) > 100 else log["message"])
                st.divider()
    else:
        st.info("No hay logs disponibles. Ejecuta un flujo para ver la actividad.")
    
    # Información del sistema
    st.subheader("ℹ️ Información del Sistema")
    st.markdown(f"""
    - **Agentes activos**: {len(set(log['agent'] for log in st.session_state.flow_data['logs']))}
    - **Total de logs**: {len(st.session_state.flow_data['logs'])}
    - **Conversaciones**: {len(st.session_state.conversation_history)}
    - **Estado**: {'🟢 Activo' if st.session_state.flow_data['is_running'] else '⚪ Inactivo'}
    """)

# Auto-refresh para logs en tiempo real
if st.session_state.flow_data["is_running"]:
    time.sleep(0.5)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>🔄 Visualización en vivo del flujo de agentes SMARTito</small>
</div>
""", unsafe_allow_html=True) 
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph import MessageGraph
from openai import OpenAI
import os
from dotenv import load_dotenv
import requests
import json

# Cargar variables de entorno desde .env
env_loaded = load_dotenv()

# Documentación de endpoints permitidos
ENDPOINTS_DOC = """
ENDPOINTS DISPONIBLES:
- /api/v1/realtime/  (parámetros: start_date, end_date, culture, device)
- /api/v1/looks/     (parámetros: start_date, end_date, hour_filter)
- /health            (sin parámetros)

DETALLES DE PARÁMETROS:
- start_date: Fecha de inicio en formato YYYY-MM-DD (requerido)
- end_date: Fecha de fin en formato YYYY-MM-DD (requerido)
- culture: Código de cultura (CL, AR, PE, CO, BR, UY, PY, EC, US) (requerido para realtime)
- device: Tipo de dispositivo (desktop o mobile) (requerido para realtime)
- hour_filter: Hora máxima a considerar 0-23 (opcional para looks, default: 23)

VALIDACIONES:
- Las fechas deben estar en formato YYYY-MM-DD
- La fecha de fin debe ser posterior a la fecha de inicio
- Culture debe ser uno de: CL, AR, PE, CO, BR, UY, PY, EC, US
- Device debe ser: desktop o mobile
- hour_filter debe estar entre 0 y 23

Ejemplos de JSON de consulta:
{"endpoint": "/api/v1/realtime/", "params": {"start_date": "2024-06-01", "end_date": "2024-06-04", "culture": "CL", "device": "desktop"}}
{"endpoint": "/api/v1/looks/", "params": {"start_date": "2024-06-01", "end_date": "2024-06-04", "hour_filter": 23}}
{"endpoint": "/health", "params": {}}

RESPUESTAS ESPERADAS:
- /api/v1/realtime/: Retorna datos de conversión con campos: date, culture, traffic, flight_dom_loaded_flight, payment_confirmation_loaded, conversion
- /api/v1/looks/: Retorna datos de búsquedas con campos: date, hour, origin, destination, looks, rtmarket
- /health: Retorna estado de salud con campos: status, timestamp, version, services

CÓDIGOS DE ERROR:
- 400: Error de validación en parámetros
- 503: Error en servicio externo (Amplitude API)
"""

ALLOWED_ENDPOINTS = {"/api/v1/realtime/", "/api/v1/looks/", "/health"}

# Estado compartido entre nodos
class FlowState(TypedDict):
    user_question: str
    technical_query: str  # Ahora será un JSON string
    raw_data: str
    final_answer: str

# Configuración del modelo (usa tu propia API key de OpenAI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-...pon-tu-api-key...")
llm = OpenAI(api_key=OPENAI_API_KEY)

# Nodo 1: Agente de Insights (reformula la pregunta en JSON estructurado)
def insights_rephrase(state: FlowState) -> FlowState:
    user_q = state["user_question"]
    prompt = (
        ENDPOINTS_DOC +
        "Dada la siguiente pregunta de usuario, responde SOLO con un JSON que indique el endpoint a consultar y los parámetros necesarios. "
        "Si la pregunta no corresponde a ningún endpoint, responde con un JSON vacío: {}. "
        f"Pregunta: '{user_q}'"
    )
    response = llm.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    technical_query = response.choices[0].message.content.strip()
    state["technical_query"] = technical_query
    return state

# Nodo 2: Agente de Datos (consulta la API real usando el JSON y valida endpoint)
def data_agent(state: FlowState) -> FlowState:
    try:
        query = json.loads(state["technical_query"])
        endpoint = query.get("endpoint")
        params = query.get("params", {})
        if not endpoint:
            state["raw_data"] = "No se pudo interpretar la pregunta o no corresponde a ningún endpoint disponible."
            return state
        if endpoint not in ALLOWED_ENDPOINTS:
            state["raw_data"] = f"Error: El endpoint '{endpoint}' no está permitido. Usa solo los endpoints documentados."
            return state
        # Construye la URL base de la API (ajusta según tu entorno)
        base_url = os.getenv("SMARTITO_API_URL", "http://localhost:8000")
        url = base_url.rstrip("/") + endpoint
        response = requests.get(url, params=params, timeout=10)
        if response.ok:
            # Si la respuesta es JSON, formatea bonito
            try:
                state["raw_data"] = json.dumps(response.json(), ensure_ascii=False, indent=2)
            except Exception:
                state["raw_data"] = response.text
        else:
            state["raw_data"] = f"Error consultando la API: {response.status_code} {response.text}"
    except Exception as e:
        state["raw_data"] = f"Error interpretando la consulta técnica: {e}"
    return state

# Nodo 3: Agente de Insights (convierte a lenguaje natural)
def insights_natural_answer(state: FlowState) -> FlowState:
    raw = state["raw_data"]
    user_q = state["user_question"]
    prompt = (
        "Eres un asistente de métricas aéreas. Responde en lenguaje natural para el usuario, usando estos datos de la API: "
        f"'{raw}'. Pregunta original: '{user_q}'"
    )
    response = llm.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    final_answer = response.choices[0].message.content.strip()
    state["final_answer"] = final_answer
    return state

# Construcción del grafo
workflow = StateGraph(FlowState)
workflow.add_node("rephrase", insights_rephrase)
workflow.add_node("data", data_agent)
workflow.add_node("natural_answer", insights_natural_answer)
workflow.set_entry_point("rephrase")
workflow.add_edge("rephrase", "data")
workflow.add_edge("data", "natural_answer")
workflow.add_edge("natural_answer", END)
graph = workflow.compile()

def main():
    user_question = "¿Cuál fue el tráfico del website durante los últimos 4 días?"
    state = FlowState(
        user_question=user_question,
        technical_query="",
        raw_data="",
        final_answer=""
    )
    result = graph.invoke(state)
    print("\nRespuesta final al usuario:")
    print(result["final_answer"])

if __name__ == "__main__":
    main() 
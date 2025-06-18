from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from openai import OpenAI
import os
from dotenv import load_dotenv
import requests
import json
import logging
import sys
from datetime import datetime
import time
import re
import pandas as pd
import numpy as np

# Configurar logging con codificación UTF-8 para Windows
if sys.platform == "win32":
    # Forzar UTF-8 en Windows
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smartito_agents.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno desde .env
load_dotenv()

def estimate_tokens(text: str) -> int:
    """Estimación aproximada de tokens (4 caracteres = ~1 token)"""
    return len(text) // 4

def log_token_usage(prompt: str, response: str, agent_name: str):
    """Log del uso estimado de tokens"""
    prompt_tokens = estimate_tokens(prompt)
    response_tokens = estimate_tokens(response)
    total_tokens = prompt_tokens + response_tokens
    
    logger.info(f"[TOKENS] {agent_name}: Prompt={prompt_tokens}, Response={response_tokens}, Total={total_tokens}")
    return total_tokens

def process_large_json_data(json_data: dict, endpoint: str) -> str:
    """
    Procesa datos JSON grandes y crea resúmenes estadísticos usando pandas
    """
    try:
        if endpoint == "/api/v1/looks/":
            return process_looks_data(json_data)
        elif endpoint == "/api/v1/realtime/":
            return process_realtime_data(json_data)
        else:
            return json.dumps(json_data, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error procesando datos JSON: {e}")
        return json.dumps(json_data, ensure_ascii=False, indent=2)

def process_looks_data(json_data: dict) -> str:
    """
    Procesa datos de looks y crea resúmenes estadísticos
    """
    try:
        # Extraer datos
        data = json_data.get("data", [])
        total_records = json_data.get("total_records", 0)
        date_range = json_data.get("date_range", {})
        hour_filter = json_data.get("hour_filter", 23)
        
        if not data:
            return "No hay datos de búsquedas disponibles"
        
        # Crear DataFrame
        df = pd.DataFrame(data)
        
        # Convertir tipos de datos
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
        df['looks'] = pd.to_numeric(df['looks'], errors='coerce')
        
        # Análisis estadístico
        summary = {
            "metadata": {
                "total_records": total_records,
                "date_range": date_range,
                "hour_filter": hour_filter,
                "processed_records": len(df)
            },
            "summary_stats": {
                "total_looks": int(df['looks'].sum()),
                "avg_looks_per_record": float(df['looks'].mean()),
                "max_looks": int(df['looks'].max()),
                "min_looks": int(df['looks'].min())
            },
            "top_routes": {},
            "hourly_analysis": df.groupby('hour')['looks'].sum().to_dict(),
            "daily_analysis": {},
            "top_origins": df.groupby('origin')['looks'].sum().nlargest(5).to_dict(),
            "top_destinations": df.groupby('destination')['looks'].sum().nlargest(5).to_dict()
        }
        
        # Manejar daily_analysis correctamente (convertir fechas a string)
        daily_analysis = df.groupby(df['date'].dt.date)['looks'].sum()
        summary["daily_analysis"] = {str(date): int(value) for date, value in daily_analysis.items()}
        
        # Manejar top_routes correctamente (MultiIndex)
        top_routes_grouped = df.groupby(['origin', 'destination'])['looks'].sum().nlargest(10)
        for (origin, destination), looks in top_routes_grouped.items():
            route_key = f"{origin} → {destination}"
            summary["top_routes"][route_key] = int(looks)
        
        return json.dumps(summary, ensure_ascii=False, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Error procesando datos de looks: {e}")
        return f"Error procesando datos: {str(e)}"

def process_realtime_data(json_data: dict) -> str:
    """
    Procesa datos de realtime y crea resúmenes estadísticos
    """
    try:
        # Si es una lista, convertir a DataFrame
        if isinstance(json_data, list):
            df = pd.DataFrame(json_data)
        else:
            # Si es un dict con 'data', extraer la lista
            data = json_data.get("data", json_data)
            df = pd.DataFrame(data)
        
        if df.empty:
            return "No hay datos de tráfico disponibles"
        
        # Convertir tipos de datos
        df['date'] = pd.to_datetime(df['date'])
        numeric_columns = ['traffic', 'flight_dom_loaded_flight', 'payment_confirmation_loaded', 'conversion']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Análisis estadístico
        summary = {
            "metadata": {
                "total_records": len(df),
                "date_range": {
                    "start": str(df['date'].min()),
                    "end": str(df['date'].max())
                }
            },
            "summary_stats": {}
        }
        
        # Agregar estadísticas por columna numérica
        for col in numeric_columns:
            if col in df.columns:
                summary["summary_stats"][col] = {
                    "total": float(df[col].sum()),
                    "average": float(df[col].mean()),
                    "max": float(df[col].max()),
                    "min": float(df[col].min())
                }
        
        # Análisis por cultura si existe
        if 'culture' in df.columns:
            summary["culture_analysis"] = df.groupby('culture')['traffic'].sum().to_dict()
        
        # Análisis por dispositivo si existe
        if 'device' in df.columns:
            summary["device_analysis"] = df.groupby('device')['traffic'].sum().to_dict()
        
        # Análisis diario - convertir fechas a string
        daily_analysis = df.groupby(df['date'].dt.date)['traffic'].sum()
        summary["daily_analysis"] = {str(date): float(value) for date, value in daily_analysis.items()}
        
        return json.dumps(summary, ensure_ascii=False, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Error procesando datos de realtime: {e}")
        return f"Error procesando datos: {str(e)}"

# Documentación de endpoints permitidos (optimizada para reducir tokens)
ENDPOINTS_DOC = """ENDPOINTS Y PARÁMETROS:
/api/v1/realtime/ - Datos de conversión web (culture, device requeridos)
/api/v1/looks/ - Datos de búsquedas de vuelos
/health - Estado del servicio

PARÁMETROS:
- start_date, end_date: YYYY-MM-DD (requeridos)
- culture: CL,AR,PE,CO,BR,UY,PY,EC,US (realtime)
- device: desktop/mobile (realtime)
- hour_filter: 0-23 (looks, opcional)

EJEMPLOS:
{"endpoint": "/api/v1/realtime/", "params": {"start_date": "2024-06-01", "end_date": "2024-06-04", "culture": "CL", "device": "desktop"}}
{"endpoint": "/api/v1/looks/", "params": {"start_date": "2024-06-01", "end_date": "2024-06-04", "hour_filter": 23}}
{"endpoint": "/health", "params": {}}"""

# Documentación de negocio para el agente final
BUSINESS_CONTEXT = """CONTEXTO DE NEGOCIO - SMARTito:
- Somos una aerolínea low-cost que opera en Latinoamérica
- Métricas clave: tráfico web, conversiones, búsquedas de vuelos
- Mercados principales: Chile (CL), Argentina (AR), Perú (PE), Colombia (CO), Brasil (BR)
- Dispositivos: desktop (mayor conversión) y mobile (mayor tráfico)
- Horarios pico: 9-18h para búsquedas, 24h para conversiones
- KPIs importantes: tasa de conversión, volumen de búsquedas, tráfico por dispositivo"""

ALLOWED_ENDPOINTS = {"/api/v1/realtime/", "/api/v1/looks/", "/health"}

# Estado compartido entre nodos
class FlowState(TypedDict):
    user_question: str
    technical_query: str  # Ahora será un JSON string
    raw_data: str
    final_answer: str
    api_documentation: str  # Documentación técnica de la API
    business_context: str   # Contexto de negocio

# Configuración del modelo (usa tu propia API key de OpenAI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-...pon-tu-api-key...")
llm = OpenAI(api_key=OPENAI_API_KEY)

def log_agent_action(agent_name: str, action: str, data: Any = None):
    """Función helper para logging consistente de agentes"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    logger.info(f"[AGENTE] [{timestamp}] {agent_name}: {action}")
    if data:
        logger.info(f"   [DATOS] {data}")

# Nodo 1: Ingeniero de Insights (reformula la pregunta en JSON estructurado)
def insights_rephrase(state: FlowState) -> FlowState:
    user_q = state["user_question"]
    api_doc = state["api_documentation"]
    log_agent_action("Ingeniero de Insights", f"Recibiendo pregunta: '{user_q}'")
    
    # Prompt técnico enfocado en la API
    prompt = (
        f"Eres un ingeniero de datos especializado en APIs de métricas. "
        f"Documentación técnica de la API:\n{api_doc}\n\n"
        f"Convierte la pregunta del usuario en una consulta JSON válida para la API. "
        f"Considera los parámetros obligatorios y las validaciones. "
        f"Responde SOLO con JSON válido, sin texto adicional. "
        f"Pregunta: '{user_q}'"
    )
    
    log_agent_action("Ingeniero de Insights", "Enviando prompt técnico a OpenAI")
    logger.info(f"[TOKENS] Estimando prompt: {estimate_tokens(prompt)} tokens")
    
    try:
        response = llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        technical_query = response.choices[0].message.content.strip()
        
        # Log del uso de tokens
        log_token_usage(prompt, technical_query, "Ingeniero de Insights")
        
        # Limpiar la respuesta de posibles caracteres extra
        technical_query = technical_query.strip()
        if technical_query.startswith("```json"):
            technical_query = technical_query[7:]
        if technical_query.endswith("```"):
            technical_query = technical_query[:-3]
        technical_query = technical_query.strip()
        
        log_agent_action("Ingeniero de Insights", f"Consulta técnica generada: {technical_query}")
        
        state["technical_query"] = technical_query
    except Exception as e:
        if "rate_limit_exceeded" in str(e):
            log_agent_action("Ingeniero de Insights", f"ERROR: Rate limit excedido - {e}")
            logger.error(f"[RATE_LIMIT] Detalles del error: {e}")
            state["technical_query"] = "{}"
            # Esperar un poco antes de continuar
            time.sleep(2)
        elif "model_not_found" in str(e):
            log_agent_action("Ingeniero de Insights", f"ERROR: Modelo no disponible - {e}")
            state["technical_query"] = "{}"
        else:
            log_agent_action("Ingeniero de Insights", f"ERROR en OpenAI: {e}")
            state["technical_query"] = "{}"
    
    return state

# Nodo 2: Data Engineer (consulta la API real y procesa datos JSON grandes)
def data_agent(state: FlowState) -> FlowState:
    technical_query = state["technical_query"]
    log_agent_action("Data Engineer", f"Recibiendo consulta técnica: {technical_query}")
    
    try:
        # Validar que no esté vacío
        if not technical_query or technical_query.strip() == "":
            log_agent_action("Data Engineer", "ERROR: Consulta técnica vacía")
            state["raw_data"] = "No se pudo interpretar la pregunta o no corresponde a ningún endpoint disponible."
            return state
        
        query = json.loads(technical_query)
        endpoint = query.get("endpoint")
        params = query.get("params", {})
        
        log_agent_action("Data Engineer", f"Endpoint extraído: {endpoint}")
        log_agent_action("Data Engineer", f"Parámetros extraídos: {params}")
        
        if not endpoint:
            log_agent_action("Data Engineer", "ERROR: No se pudo interpretar la pregunta")
            state["raw_data"] = "No se pudo interpretar la pregunta o no corresponde a ningún endpoint disponible."
            return state
            
        if endpoint not in ALLOWED_ENDPOINTS:
            log_agent_action("Data Engineer", f"ERROR: Endpoint '{endpoint}' no permitido")
            state["raw_data"] = f"Error: El endpoint '{endpoint}' no está permitido. Usa solo los endpoints documentados."
            return state
        
        # Construye la URL base de la API
        base_url = os.getenv("SMARTITO_API_URL", "http://localhost:8000")
        url = base_url.rstrip("/") + endpoint
        
        log_agent_action("Data Engineer", f"Consultando API: {url}")
        log_agent_action("Data Engineer", f"Parámetros de consulta: {params}")
        log_agent_action("Data Engineer", f"URL completa: {url}?{requests.compat.urlencode(params)}")
        
        # Aumentar timeout y agregar retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                log_agent_action("Data Engineer", f"Intento {attempt + 1}/{max_retries}")
                response = requests.get(url, params=params, timeout=30)
                break  # Si llega aquí, la conexión fue exitosa
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:  # Último intento
                    log_agent_action("Data Engineer", f"ERROR después de {max_retries} intentos: {e}")
                    state["raw_data"] = f"Error consultando la API después de {max_retries} intentos: {e}"
                    return state
                else:
                    log_agent_action("Data Engineer", f"Intento {attempt + 1} falló, reintentando...")
                    time.sleep(1)  # Esperar 1 segundo antes del siguiente intento
        
        if response.ok:
            log_agent_action("Data Engineer", f"SUCCESS: API respondió exitosamente (Status: {response.status_code})")
            
            try:
                json_response = response.json()
                log_agent_action("Data Engineer", f"Datos JSON recibidos ({len(str(json_response))} caracteres)")
                
                # Procesar datos JSON grandes con pandas
                log_agent_action("Data Engineer", "Procesando datos JSON con pandas...")
                processed_data = process_large_json_data(json_response, endpoint)
                
                # Log del tamaño antes y después del procesamiento
                original_size = len(str(json_response))
                processed_size = len(processed_data)
                reduction = ((original_size - processed_size) / original_size) * 100
                
                log_agent_action("Data Engineer", f"Optimización de datos: {original_size:,} → {processed_size:,} caracteres ({reduction:.1f}% reducción)")
                
                state["raw_data"] = processed_data
                log_agent_action("Data Engineer", f"SUCCESS: Datos procesados y optimizados")
                
            except Exception as e:
                log_agent_action("Data Engineer", f"ERROR procesando JSON: {e}")
                state["raw_data"] = response.text
                log_agent_action("Data Engineer", f"Respuesta de texto procesada ({len(response.text)} caracteres)")
        else:
            log_agent_action("Data Engineer", f"ERROR en API: {response.status_code}")
            state["raw_data"] = f"Error consultando la API: {response.status_code} {response.text}"
            
    except json.JSONDecodeError as e:
        log_agent_action("Data Engineer", f"ERROR parseando JSON: {e}")
        log_agent_action("Data Engineer", f"JSON recibido: '{technical_query}'")
        state["raw_data"] = f"Error interpretando la consulta técnica: JSON inválido - {e}"
    except Exception as e:
        log_agent_action("Data Engineer", f"ERROR inesperado: {e}")
        state["raw_data"] = f"Error interpretando la consulta técnica: {e}"
    
    return state

# Nodo 3: Analista de Negocio (convierte a lenguaje natural con contexto de negocio)
def insights_natural_answer(state: FlowState) -> FlowState:
    raw = state["raw_data"]
    user_q = state["user_question"]
    api_doc = state["api_documentation"]
    business_ctx = state["business_context"]
    
    log_agent_action("Analista de Negocio", f"Recibiendo datos crudos de la API")
    log_agent_action("Analista de Negocio", f"Pregunta original: '{user_q}'")
    
    # Prompt con contexto de negocio y documentación técnica
    prompt = (
        f"Eres un analista de negocio de SMARTito, una aerolínea low-cost latinoamericana. "
        f"Contexto de negocio:\n{business_ctx}\n\n"
        f"Documentación técnica de la API:\n{api_doc}\n\n"
        f"Analiza estos datos de la API: '{raw}' "
        f"y responde la pregunta del usuario en lenguaje natural, "
        f"interpretando los datos desde la perspectiva del negocio aéreo. "
        f"Pregunta: '{user_q}'"
    )
    
    log_agent_action("Analista de Negocio", "Enviando prompt con contexto de negocio a OpenAI")
    logger.info(f"[TOKENS] Estimando prompt: {estimate_tokens(prompt)} tokens")
    
    try:
        response = llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        final_answer = response.choices[0].message.content.strip()
        
        # Log del uso de tokens
        log_token_usage(prompt, final_answer, "Analista de Negocio")
        
        log_agent_action("Analista de Negocio", f"SUCCESS: Respuesta final generada: {final_answer[:100]}...")
        
        state["final_answer"] = final_answer
    except Exception as e:
        if "rate_limit_exceeded" in str(e):
            log_agent_action("Analista de Negocio", f"ERROR: Rate limit excedido - {e}")
            logger.error(f"[RATE_LIMIT] Detalles del error: {e}")
            state["final_answer"] = "Lo siento, he alcanzado el límite de consultas. Por favor, espera un momento y vuelve a intentar."
            time.sleep(2)
        elif "model_not_found" in str(e):
            log_agent_action("Analista de Negocio", f"ERROR: Modelo no disponible - {e}")
            state["final_answer"] = "Lo siento, hay un problema con el modelo de IA. Por favor, contacta al administrador."
        else:
            log_agent_action("Analista de Negocio", f"ERROR en OpenAI: {e}")
            state["final_answer"] = f"Lo siento, hubo un error generando la respuesta: {e}"
    
    
    return state

# Construcción del grafo
workflow = StateGraph(FlowState)
workflow.add_node("engineer", insights_rephrase)  # Ingeniero de Insights
workflow.add_node("data", data_agent)            # Data Engineer
workflow.add_node("analyst", insights_natural_answer)  # Analista de Negocio
workflow.set_entry_point("engineer")
workflow.add_edge("engineer", "data")
workflow.add_edge("data", "analyst")
workflow.add_edge("analyst", END)
graph = workflow.compile()

def main():
    user_question = "¿Puedes darme las rutas con mayor cantidad de looks durante el día 15 de junio de 2025?"
    
    logger.info("INICIANDO: Flujo de agentes SMARTito")
    logger.info(f"PREGUNTA: {user_question}")
    
    state = FlowState(
        user_question=user_question,
        technical_query="",
        raw_data="",
        final_answer="",
        api_documentation=ENDPOINTS_DOC,
        business_context=BUSINESS_CONTEXT
    )
    
    # Ejecutar el flujo
    result = graph.invoke(state)
    
    logger.info("COMPLETADO: Flujo exitosamente")
    logger.info(f"RESPUESTA: {result['final_answer']}")
    
    print("\n" + "="*50)
    print("RESPUESTA FINAL AL USUARIO:")
    print("="*50)
    print(result["final_answer"])
    print("="*50)

if __name__ == "__main__":
    main() 
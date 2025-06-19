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
    conversation_history: list  # Nueva: historial de conversaciones
    last_query_context: dict   # Nueva: contexto de la última consulta

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
    conversation_history = state["conversation_history"]
    last_query_context = state["last_query_context"]
    
    log_agent_action("Ingeniero de Insights", f"Recibiendo pregunta: '{user_q}'")
    
    # Analizar contexto de conversación
    context_analysis = analyze_conversation_context(user_q, conversation_history, last_query_context)
    
    if context_analysis["is_follow_up"]:
        log_agent_action("Ingeniero de Insights", f"Detectada pregunta de seguimiento. Pregunta mejorada: '{context_analysis['enhanced_question']}'")
        user_q = context_analysis["enhanced_question"]
        
        # Si hay fechas de comparación específicas, usarlas
        if "comparison_dates" in context_analysis:
            comparison_dates = context_analysis["comparison_dates"]
            current = comparison_dates["current"]
            previous = comparison_dates["previous"]
            
            # Si la última consulta fue de rutas, mantener el contexto de rutas
            if last_query_context.get("endpoint") == "/api/v1/looks/":
                log_agent_action("Ingeniero de Insights", f"Usando fechas de comparación para rutas: {current} vs {previous}")
                user_q = f"Compara las rutas con mayor cantidad de looks del {current['start']} al {current['end']} con las rutas del {previous['start']} al {previous['end']}"
    
    # Prompt técnico enfocado en la API con contexto mejorado
    prompt = f'''
Eres un ingeniero de datos especializado en APIs de métricas.

CONTEXTO IMPORTANTE:
- Si la pregunta anterior fue sobre rutas (/api/v1/looks/), mantén el contexto de rutas
- Si la pregunta anterior fue sobre tráfico web (/api/v1/realtime/), mantén el contexto de tráfico
- Usa las fechas correctas (2025, no 2024)
- Para rutas: usa /api/v1/looks/ con start_date, end_date, hour_filter
- Para tráfico: usa /api/v1/realtime/ con start_date, end_date, culture, device

Documentación técnica de la API:
{api_doc}

Si la pregunta es una comparación entre dos periodos, responde con un JSON así:
Para rutas:
{{
  "endpoint": "/api/v1/looks/",
  "params": [
    {{"start_date": "2025-06-15", "end_date": "2025-06-15", "hour_filter": 23}},
    {{"start_date": "2025-06-08", "end_date": "2025-06-08", "hour_filter": 23}}
  ]
}}

Para tráfico web:
{{
  "endpoint": "/api/v1/realtime/",
  "params": [
    {{"start_date": "2025-06-01", "end_date": "2025-06-07", "culture": "CL", "device": "desktop"}},
    {{"start_date": "2025-05-24", "end_date": "2025-05-31", "culture": "CL", "device": "desktop"}}
  ]
}}

Si es una sola consulta, responde como antes. Solo responde JSON válido, sin texto adicional.

Pregunta: '{user_q}'
'''
    
    log_agent_action("Ingeniero de Insights", "Enviando prompt técnico a OpenAI")
    logger.info(f"[TOKENS] Estimando prompt: {estimate_tokens(prompt)} tokens")
    
    try:
        response = llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
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
        
        # Si params es una lista, hacer múltiples consultas y devolver ambos resúmenes
        if isinstance(params, list):
            log_agent_action("Data Engineer", f"Detectada comparación: {len(params)} bloques de parámetros")
            results = []
            conversation_history = state.get("conversation_history", [])
            for i, param_set in enumerate(params):
                log_agent_action("Data Engineer", f"Consultando bloque {i+1}: {param_set}")
                # Buscar en historial
                summary = find_summary_in_history(endpoint, param_set, conversation_history)
                if summary:
                    log_agent_action("Data Engineer", f"Resumen encontrado en historial para bloque {i+1}")
                    results.append({
                        "params": param_set,
                        "summary": summary,
                        "from_history": True
                    })
                    continue
                # Si no está en historial, consultar API
                url = base_url.rstrip("/") + endpoint
                try:
                    response = requests.get(url, params=param_set, timeout=30)
                    if response.ok:
                        json_response = response.json()
                        processed_data = process_large_json_data(json_response, endpoint)
                        results.append({
                            "params": param_set,
                            "summary": processed_data,
                            "from_history": False
                        })
                        # Guardar en historial para futuras consultas
                        state["conversation_history"].append({
                            "endpoint": endpoint,
                            "params": param_set,
                            "summary": processed_data,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        results.append({
                            "params": param_set,
                            "error": f"Error consultando la API: {response.status_code} {response.text}",
                            "from_history": False
                        })
                except Exception as e:
                    results.append({
                        "params": param_set,
                        "error": str(e),
                        "from_history": False
                    })
            state["raw_data"] = json.dumps({"comparison": results}, ensure_ascii=False, indent=2)
            log_agent_action("Data Engineer", f"SUCCESS: Comparación procesada y optimizada (con memoria)")
            # Guardar contexto de comparación
            state["last_query_context"] = {
                "endpoint": endpoint,
                "params": params,
                "timestamp": datetime.now().isoformat()
            }
            return state
        
        # Construye la URL completa de la API
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
                
                # Guardar contexto de la consulta actual para memoria
                state["last_query_context"] = {
                    "endpoint": endpoint,
                    "params": params,
                    "timestamp": datetime.now().isoformat()
                }
                
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
    last_query_context = state.get("last_query_context", {})
    
    log_agent_action("Analista de Negocio", f"Recibiendo datos crudos de la API")
    log_agent_action("Analista de Negocio", f"Pregunta original: '{user_q}'")
    
    # Si el raw_data es una comparación, preparar el prompt para comparar ambos bloques
    try:
        raw_json = json.loads(raw)
        if isinstance(raw_json, dict) and "comparison" in raw_json:
            blocks = raw_json["comparison"]
            summaries = []
            
            # Determinar el tipo de datos basado en el contexto
            endpoint = last_query_context.get("endpoint", "")
            data_type = "rutas de vuelos" if endpoint == "/api/v1/looks/" else "tráfico web"
            
            for i, block in enumerate(blocks):
                if "summary" in block:
                    summaries.append(f"Bloque {i+1} (params: {block['params']}):\n{block['summary']}")
                elif "error" in block:
                    summaries.append(f"Bloque {i+1} (params: {block['params']}):\nERROR: {block['error']}")
            
            # Prompt específico para comparaciones
            if endpoint == "/api/v1/looks/":
                prompt = (
                    f"Eres un analista de negocio de SMARTito, una aerolínea low-cost latinoamericana. "
                    f"Contexto de negocio:\n{business_ctx}\n\n"
                    f"Estás analizando datos de RUTAS DE VUELOS (búsquedas de rutas específicas). "
                    f"Compara los siguientes resúmenes de datos de rutas:\n\n" + "\n\n".join(summaries) + 
                    f"\n\nPregunta del usuario: '{user_q}'\n\n"
                    f"Responde comparando las rutas más populares entre ambos periodos, "
                    f"identificando cambios en las rutas más buscadas, volúmenes de búsquedas, "
                    f"y proporcionando insights relevantes para el negocio aéreo."
                )
            else:
                prompt = (
                    f"Eres un analista de negocio de SMARTito, una aerolínea low-cost latinoamericana. "
                    f"Contexto de negocio:\n{business_ctx}\n\n"
                    f"Estás analizando datos de TRÁFICO WEB. "
                    f"Compara los siguientes resúmenes de datos de tráfico:\n\n" + "\n\n".join(summaries) + 
                    f"\n\nPregunta del usuario: '{user_q}'\n\n"
                    f"Responde comparando el tráfico web entre ambos periodos, "
                    f"identificando cambios en visitas, conversiones, y proporcionando insights relevantes."
                )
        else:
            # Prompt normal para consultas simples
            endpoint = last_query_context.get("endpoint", "")
            data_type = "rutas de vuelos" if endpoint == "/api/v1/looks/" else "tráfico web"
            
            prompt = (
                f"Eres un analista de negocio de SMARTito, una aerolínea low-cost latinoamericana. "
                f"Contexto de negocio:\n{business_ctx}\n\n"
                f"Estás analizando datos de {data_type.upper()}. "
                f"Analiza estos datos de la API: '{raw}' "
                f"y responde la pregunta del usuario en lenguaje natural, "
                f"interpretando los datos desde la perspectiva del negocio aéreo. "
                f"Pregunta: '{user_q}'"
            )
    except Exception as e:
        # Si falla el parseo, usar prompt normal
        log_agent_action("Analista de Negocio", f"ERROR parseando datos: {e}")
        endpoint = last_query_context.get("endpoint", "")
        data_type = "rutas de vuelos" if endpoint == "/api/v1/looks/" else "tráfico web"
        
        prompt = (
            f"Eres un analista de negocio de SMARTito, una aerolínea low-cost latinoamericana. "
            f"Contexto de negocio:\n{business_ctx}\n\n"
            f"Estás analizando datos de {data_type.upper()}. "
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
        
        # Guardar en el historial de conversación
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_question": state["user_question"],
            "technical_query": state["technical_query"],
            "final_answer": final_answer,
            "endpoint": state["last_query_context"].get("endpoint", ""),
            "params": state["last_query_context"].get("params", {})
        }
        
        state["conversation_history"].append(conversation_entry)
        log_agent_action("Analista de Negocio", f"Conversación guardada en historial (total: {len(state['conversation_history'])} entradas)")
        
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

def analyze_conversation_context(user_question: str, conversation_history: list, last_query_context: dict) -> dict:
    """
    Analiza el contexto de la conversación para detectar preguntas de seguimiento
    """
    context = {
        "is_follow_up": False,
        "referenced_context": None,
        "enhanced_question": user_question
    }
    
    # Palabras clave que indican preguntas de seguimiento
    follow_up_keywords = [
        "eso", "eso mismo", "lo mismo", "esa", "esa misma", "la misma",
        "comparar", "comparar con", "comparado", "comparado a",
        "respecto a", "respecto de", "en relación a", "en relación con",
        "versus", "vs", "contra", "frente a",
        "anterior", "anteriormente", "antes", "previo", "previo a",
        "pasado", "pasada", "último", "última", "reciente",
        "mismo", "misma", "igual", "similar"
    ]
    
    # Detectar si es una pregunta de seguimiento
    question_lower = user_question.lower()
    is_follow_up = any(keyword in question_lower for keyword in follow_up_keywords)
    
    if is_follow_up and conversation_history and last_query_context:
        context["is_follow_up"] = True
        context["referenced_context"] = last_query_context
        
        # Obtener el contexto de la última consulta
        last_endpoint = last_query_context.get("endpoint", "")
        last_params = last_query_context.get("params", {})
        
        # Mejorar la pregunta con contexto específico del endpoint
        if "comparar" in question_lower or "comparado" in question_lower:
            if last_endpoint == "/api/v1/looks/":
                # Si la última consulta fue de rutas, mantener el contexto de rutas
                start_date = last_params.get("start_date", "")
                end_date = last_params.get("end_date", "")
                
                if start_date and end_date:
                    try:
                        from datetime import datetime, timedelta
                        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                        days_diff = (end_dt - start_dt).days
                        
                        prev_start = start_dt - timedelta(days=days_diff + 1)
                        prev_end = start_dt - timedelta(days=1)
                        
                        enhanced_question = f"Compara las rutas con mayor cantidad de looks del {start_date} al {end_date} con las rutas del {prev_start.strftime('%Y-%m-%d')} al {prev_end.strftime('%Y-%m-%d')}"
                        context["enhanced_question"] = enhanced_question
                        context["comparison_dates"] = {
                            "current": {"start": start_date, "end": end_date},
                            "previous": {"start": prev_start.strftime('%Y-%m-%d'), "end": prev_end.strftime('%Y-%m-%d')}
                        }
                        logger.info(f"[CONTEXTO] Pregunta mejorada para rutas: {enhanced_question}")
                    except Exception as e:
                        logger.error(f"Error calculando fechas para rutas: {e}")
                        
            elif last_endpoint == "/api/v1/realtime/":
                # Si la última consulta fue de tráfico web, mantener el contexto de tráfico
                start_date = last_params.get("start_date", "")
                end_date = last_params.get("end_date", "")
                culture = last_params.get("culture", "")
                device = last_params.get("device", "")
                
                if start_date and end_date:
                    try:
                        from datetime import datetime, timedelta
                        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                        days_diff = (end_dt - start_dt).days
                        
                        prev_start = start_dt - timedelta(days=days_diff + 1)
                        prev_end = start_dt - timedelta(days=1)
                        
                        enhanced_question = f"Compara el tráfico web de {culture} en {device} del {start_date} al {end_date} con el tráfico del {prev_start.strftime('%Y-%m-%d')} al {prev_end.strftime('%Y-%m-%d')}"
                        context["enhanced_question"] = enhanced_question
                        context["comparison_dates"] = {
                            "current": {"start": start_date, "end": end_date},
                            "previous": {"start": prev_start.strftime("%Y-%m-%d"), "end": prev_end.strftime('%Y-%m-%d')}
                        }
                        logger.info(f"[CONTEXTO] Pregunta mejorada para tráfico: {enhanced_question}")
                    except Exception as e:
                        logger.error(f"Error calculando fechas para tráfico: {e}")
    
    return context

def test_conversation_memory():
    """Prueba el sistema de memoria con una conversación de seguimiento"""
    logger.info("🧠 INICIANDO: Prueba de memoria de conversación")
    
    # Primera pregunta
    user_question_1 = "¿Cuál fue el tráfico del website en Chile desktop durante la semana del 1 al 7 de junio de 2024?"
    
    state = FlowState(
        user_question=user_question_1,
        technical_query="",
        raw_data="",
        final_answer="",
        api_documentation=ENDPOINTS_DOC,
        business_context=BUSINESS_CONTEXT,
        conversation_history=[],
        last_query_context={}
    )
    
    logger.info(f"📝 PREGUNTA 1: {user_question_1}")
    result_1 = graph.invoke(state)
    logger.info(f"✅ RESPUESTA 1: {result_1['final_answer'][:100]}...")
    
    # Segunda pregunta (de seguimiento)
    user_question_2 = "Compara eso con la semana pasada"
    
    state_2 = FlowState(
        user_question=user_question_2,
        technical_query="",
        raw_data="",
        final_answer="",
        api_documentation=ENDPOINTS_DOC,
        business_context=BUSINESS_CONTEXT,
        conversation_history=result_1["conversation_history"],
        last_query_context=result_1["last_query_context"]
    )
    
    logger.info(f"📝 PREGUNTA 2: {user_question_2}")
    result_2 = graph.invoke(state_2)
    logger.info(f"✅ RESPUESTA 2: {result_2['final_answer'][:100]}...")
    
    logger.info("🧠 COMPLETADO: Prueba de memoria exitosa")

def find_summary_in_history(endpoint: str, params: dict, conversation_history: list) -> str:
    """
    Busca en el historial si ya existe un resumen para el endpoint y los parámetros dados.
    Devuelve el resumen si lo encuentra, o None si no existe.
    """
    for entry in conversation_history:
        if entry.get("endpoint") == endpoint and entry.get("params") == params:
            return entry.get("summary") or entry.get("raw_data") or entry.get("final_answer")
    return None

def test_routes_comparison():
    """Prueba específica para comparación de rutas manteniendo contexto"""
    logger.info("🧠 INICIANDO: Prueba de comparación de rutas")
    
    # Primera pregunta - rutas específicas
    user_question_1 = "Puedes darme las rutas con mayor looks el dia 15 de Junio de 2025?"
    
    state = FlowState(
        user_question=user_question_1,
        technical_query="",
        raw_data="",
        final_answer="",
        api_documentation=ENDPOINTS_DOC,
        business_context=BUSINESS_CONTEXT,
        conversation_history=[],
        last_query_context={}
    )
    
    logger.info(f"📝 PREGUNTA 1: {user_question_1}")
    result_1 = graph.invoke(state)
    logger.info(f"✅ RESPUESTA 1: {result_1['final_answer'][:100]}...")
    
    # Segunda pregunta - comparación de rutas (debería mantener contexto de rutas)
    user_question_2 = "Excelente. Puedes darme ahora la comparación de esas rutas vs la semana pasada?"
    
    state_2 = FlowState(
        user_question=user_question_2,
        technical_query="",
        raw_data="",
        final_answer="",
        api_documentation=ENDPOINTS_DOC,
        business_context=BUSINESS_CONTEXT,
        conversation_history=result_1["conversation_history"],
        last_query_context=result_1["last_query_context"]
    )
    
    logger.info(f"📝 PREGUNTA 2: {user_question_2}")
    result_2 = graph.invoke(state_2)
    logger.info(f"✅ RESPUESTA 2: {result_2['final_answer'][:100]}...")
    
    logger.info("🧠 COMPLETADO: Prueba de comparación de rutas")

def main():
    # Cambiar entre prueba de memoria y pregunta simple
    test_mode = "routes_comparison"  # Cambiar a "simple" para pregunta simple
    
    if test_mode == "routes_comparison":
        test_routes_comparison()
    elif test_mode == "memory":
        test_conversation_memory()
    else:
        user_question = "¿Puedes darme las rutas con mayor cantidad de looks durante el día 15 de junio de 2025?"
        
        logger.info("INICIANDO: Flujo de agentes SMARTito")
        logger.info(f"PREGUNTA: {user_question}")
        
        state = FlowState(
            user_question=user_question,
            technical_query="",
            raw_data="",
            final_answer="",
            api_documentation=ENDPOINTS_DOC,
            business_context=BUSINESS_CONTEXT,
            conversation_history=[],
            last_query_context={}
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
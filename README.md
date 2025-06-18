# LangGraph SMARTito MVP

Este es un MVP que demuestra un flujo de agentes colaborativos usando LangGraph y OpenAI para interactuar con una API de métricas aéreas.

## 🚀 Características

- **Flujo de agentes colaborativos**: Usando LangGraph para modelar la interacción entre agentes
- **Integración con API real**: Conecta con la API SMARTito v2 para métricas aéreas
- **Chatbot web**: Interfaz de usuario con Streamlit
- **Validación robusta**: Solo permite endpoints documentados
- **Respuestas en lenguaje natural**: Convierte datos técnicos en respuestas amigables

## 📋 Requisitos

- Python 3.9+
- Una API Key de OpenAI (puedes obtenerla en https://platform.openai.com/)
- API SMARTito v2 corriendo (opcional, para datos reales)

## 🛠️ Instalación y uso

1. **Crear un entorno virtual:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

2. **Instalar las dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar variables de entorno:**

   Edita el archivo `.env` con tu API Key de OpenAI:
   ```
   OPENAI_API_KEY=sk-tu-api-key-aqui
   SMARTITO_API_URL=http://localhost:8000  # URL de tu API SMARTito
   ```

4. **Ejecutar el chatbot:**

   ```bash
   streamlit run chatbot_app.py
   ```

5. **O ejecutar el script principal:**

   ```bash
   python main.py
   ```

## 📚 Endpoints Soportados

El chatbot puede consultar los siguientes endpoints de la API SMARTito:

- **`/api/v1/realtime/`** - Datos de conversión en tiempo real
  - Parámetros: `start_date`, `end_date`, `culture`, `device`
- **`/api/v1/looks/`** - Datos de búsquedas por ruta
  - Parámetros: `start_date`, `end_date`, `hour_filter`
- **`/health`** - Estado de salud de la API

## 🔄 Flujo de Agentes

1. **Agente de Insights**: Interpreta la pregunta del usuario y la convierte en una consulta técnica estructurada (JSON)
2. **Agente de Datos**: Ejecuta la consulta a la API usando los parámetros especificados
3. **Agente de Insights**: Convierte la respuesta técnica en lenguaje natural para el usuario

## 💡 Ejemplos de Preguntas

- "¿Cuál fue el tráfico del website durante los últimos 4 días?"
- "Muéstrame las búsquedas de vuelos de ayer"
- "¿Cuál es el estado de salud de la API?"
- "Dame los datos de conversión de Chile en desktop"

## 🔧 Desarrollo

### Estructura del Proyecto

```
LangGraph SMARTito/
├── main.py              # Flujo principal de LangGraph
├── chatbot_app.py       # Interfaz de chatbot con Streamlit
├── requirements.txt     # Dependencias del proyecto
├── .env                 # Variables de entorno (no subir a Git)
├── .gitignore          # Archivos a ignorar en Git
└── README.md           # Este archivo
```

### Personalización

- **Agregar nuevos endpoints**: Modifica `ENDPOINTS_DOC` y `ALLOWED_ENDPOINTS` en `main.py`
- **Mejorar prompts**: Ajusta los prompts en las funciones de los agentes
- **Validación adicional**: Agrega validaciones en `data_agent()`

## 🚀 Deployment

Para desplegar en producción:

1. Configura las variables de entorno en tu servidor
2. Instala las dependencias
3. Ejecuta con Streamlit o integra en tu aplicación web

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

## 🤝 Contribución

1. Fork el proyecto
2. Crear una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abrir un Pull Request 
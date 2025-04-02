# -*- coding: utf-8 -*-
import os                     # Proporciona funciones para interactuar con el sistema operativo
import re
import copy
import gradio as gr           # Gradio se utiliza para crear una interfaz web
import logging                # Para manejar los logs
import datetime               # Para registrar la fecha y hora de los logs
import speech_recognition as sr # Para la funcionalidad Speech-to-Text
from gtts import gTTS         # Para la funcionalidad Text-to-Speech
import tempfile               # Para crear archivos temporales para el audio TTS
import uuid                   # Para generar nombres de archivo únicos
from langchain_community.vectorstores import Chroma # Importar Chroma para cargar un almacén vectorial existente
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # Importar clases de OpenAI

# --- Configuración ---
# !!! IMPORTANTE: Cambia esto a la ruta real de tu base de datos Chroma !!!
CHROMA_DB_PATH = "chroma_db/" # Reemplaza con la ruta correcta
LOG_FILE = "app_log.txt"
TTS_LANG = 'es' # Idioma para Text-to-Speech (Español)

# Configurar el sistema de logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- Variables Globales ---
log_messages = []           # Almacena logs en memoria para mostrarlos en la UI
vectorstore = None          # Almacenará la base de datos vectorial cargada
current_model = "openai"    # Modelo por defecto
finetuning_model_name = ""  # Nombre del modelo de finetuning (si se usa)
openai_api_key_set = False  # Flag para saber si la API key está configurada
tts_enabled = False          # Flag para controlar la salida TTS (desactivado por defecto ahora en UI)
r = sr.Recognizer()         # Inicializar el reconocedor de voz

# --- Funciones de Logging ---
def add_log(message, level="INFO"):
    """Añade un mensaje al log y lo guarda en memoria"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {level} - {message}"
    log_messages.append(log_entry)

    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)

    # Opcional: Limitar el tamaño de los logs en memoria si es necesario
    # if len(log_messages) > 1000:
    #     log_messages.pop(0)

    return log_entry # Devuelve la entrada de log para posible uso inmediato

# --- Funciones Principales ---

def get_embeddings():
    """Obtiene el modelo de embeddings apropiado (solo OpenAI en esta versión)."""
    global current_model, openai_api_key_set
    if not openai_api_key_set:
        add_log("Advertencia: Intentando obtener embeddings sin configurar la API key de OpenAI.", "WARNING")
        return None

    add_log(f"Usando embeddings de OpenAI (modelo: text-embedding-3-small)")
    return OpenAIEmbeddings(model="text-embedding-3-small")

def load_vectorstore():
    """Carga la base de datos vectorial Chroma desde la ruta especificada."""
    global vectorstore, CHROMA_DB_PATH, openai_api_key_set
    add_log(f"Intentando cargar Chroma DB desde: {CHROMA_DB_PATH}")

    if not openai_api_key_set:
        error_msg = "❌ **Error:** La API key de OpenAI no ha sido configurada. Por favor, configúrala en la pestaña 'Configuración'."
        add_log(error_msg.replace("❌ **Error:** ", ""), "ERROR")
        return gr.update(value=error_msg, visible=True)

    if not os.path.exists(CHROMA_DB_PATH):
        error_msg = f"❌ **Error:** La ruta especificada para Chroma DB no existe: `{CHROMA_DB_PATH}`"
        add_log(error_msg.replace("❌ **Error:** ", "").replace("`",""), "ERROR")
        return gr.update(value=error_msg, visible=True)

    embeddings = get_embeddings()
    if embeddings is None:
        error_msg = "❌ **Error:** No se pudieron obtener los embeddings. Asegúrate de que la API key de OpenAI esté configurada."
        add_log(error_msg.replace("❌ **Error:** ", ""), "ERROR")
        return gr.update(value=error_msg, visible=True)

    try:
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings, collection_name="leyes_mexico_selectivo")
        success_msg = f"✅ **Éxito:** Base de datos Chroma cargada correctamente desde `{CHROMA_DB_PATH}`."
        add_log(success_msg.replace("✅ **Éxito:** ", "").replace("`",""))
        return gr.update(value=success_msg, visible=True)
    except Exception as e:
        error_msg = f"❌ **Error** al cargar Chroma DB desde `{CHROMA_DB_PATH}`: {str(e)}"
        add_log(error_msg.replace("❌ **Error** ", "").replace("`",""), "ERROR")
        vectorstore = None # Asegurarse de que esté None si falla la carga
        return gr.update(value=error_msg, visible=True)

def call_llm(question, context):
    """Llama al modelo LLM seleccionado (OpenAI o Finetuning)."""
    global current_model, finetuning_model_name, openai_api_key_set
    add_log(f"Enviando pregunta al modelo ({current_model}): {question[:50]}...")
    formatted_prompt = f"""Usa el siguiente contexto siguiento siempre las reglas. De manera secundaria usa la informacion que tengas sin el contexto de la ley federal de trabajo en México.
    Reglas:
    1. Responde en español.
    2. Respnode como un asistente legal.
    3. No uses emojis ni lenguaje coloquial.
    4. Si se pregunta por un articulo especifico, responder el resumen del articulo
    5. Si preguntan por cualquier tema que no sea le ley federal de trabajo, responder que no tienes la informacion.    
    6. Siempre responder con la ley actualizada 2025 si esta en el contexto.
    
    Contexto:
    {context}

    
    Pregunta: {question}
    

    """
    add_log("Esperando respuesta del modelo...")

    if not openai_api_key_set:
         add_log("Error: API key de OpenAI no configurada al intentar llamar al LLM.", "ERROR")
         return "Error: La API key de OpenAI no está configurada. Por favor, configúrala."

    try:
        if current_model == "openai":
            llm = ChatOpenAI(model="gpt-4o-mini") # Modelo base de OpenAI
            response = llm.invoke(formatted_prompt)
            result = response.content
        elif current_model == "finetuning":
            if not finetuning_model_name:
                add_log("Error: Modelo 'finetuning' seleccionado pero no se ha proporcionado un nombre de modelo.", "ERROR")
                return "Error: Falta el nombre del modelo de finetuning."
            add_log(f"Usando modelo de finetuning: {finetuning_model_name}")
            llm = ChatOpenAI(model=finetuning_model_name) # Modelo de finetuning
            response = llm.invoke(formatted_prompt)
            result = response.content
        else:
            add_log(f"Error: Modelo desconocido seleccionado: {current_model}", "ERROR")
            return f"Error: Modelo no soportado: {current_model}"

        add_log("Respuesta recibida del modelo.")
        return result
    except Exception as e:
        error_msg = f"Error al llamar al LLM ({current_model}): {str(e)}"
        add_log(error_msg, "ERROR")
        if "api key" in str(e).lower():
            return "Error: Problema con la API key de OpenAI. Verifícala en la configuración."
        return f"Error al contactar el modelo: {str(e)}"
def extract_target_article(question):
    """Intenta extraer un número de artículo de la pregunta."""
    # Busca patrones como "artículo 7", "articulo numero 5", "art. 12", etc.
    # Ignora mayúsculas/minúsculas y posibles typos simples en "articulo"
    match = re.search(r'art[íi]?[c]?[u]?[l]?[o]?\s*(?:numero|núm\.?|num\.?)?\s*(\d+)', question, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1)) # Devuelve el número como entero
        except ValueError:
            return None
    return None

def rag_chain(question):
    """Ejecuta la cadena RAG usando el vectorstore global cargado, aplicando filtros de metadatos si es necesario."""
    global vectorstore
    SEARCH_TYPE = "mmr"  # Tipo de búsqueda (Maximal Marginal Relevance)
    # Parámetros base de búsqueda
    BASE_SEARCH_KWARGS = {"k": 20, "fetch_k": 20} 

    if vectorstore is None:
        add_log("Error: Vectorstore no cargado. Intenta inicializar el chat.", "ERROR")
        return "Error: La base de datos vectorial no está cargada. Por favor, inicializa el chat primero desde 'Configuración'."

    add_log(f"Iniciando cadena RAG con la configuracion base, search_type : {SEARCH_TYPE}, search_kwargs base: {BASE_SEARCH_KWARGS}")

    target_article = extract_target_article(question) # Extrae el número de artículo deseado
    current_search_kwargs = copy.deepcopy(BASE_SEARCH_KWARGS) # Copia para no modificar el original
    filter_applied = False

    if target_article is not None:
        add_log(f"Se detectó solicitud del artículo específico: {target_article}. Aplicando filtro en la búsqueda...")

        metadata_filter = {'numero_articulo': target_article} 
        
        # Añade el filtro a los search_kwargs
        current_search_kwargs['filter'] = metadata_filter 
        filter_applied = True
        add_log(f"Search_kwargs con filtro: {current_search_kwargs}")
    else:
         add_log("No se detectó solicitud de artículo específico. Se realizará búsqueda general.")

    try:
        retriever = vectorstore.as_retriever(
            search_type=SEARCH_TYPE,
            search_kwargs=current_search_kwargs # Usa los kwargs con el filtro (si aplica)
        )
        
        add_log("Recuperando documentos relevantes (con filtro si aplica)...")
        # Ahora la recuperación ya considera el filtro de metadatos
        retrieved_docs = retriever.invoke(question) 
        
        add_log(f"Recuperados {len(retrieved_docs)} documentos relevantes.")
        
        if not retrieved_docs:
            log_message = f"No se encontraron documentos para el artículo {target_article} que coincidan con la consulta." if filter_applied else "No se encontraron documentos relevantes para la consulta."
            add_log(log_message, "WARNING")
            formatted_context = f"No se encontró información relevante para el artículo {target_article} en la base de datos." if filter_applied else "No se encontró información relevante en la base de datos para tu consulta."
        
        else:
            final_docs = retrieved_docs 
            formatted_context = "\n\n".join(doc.page_content for doc in final_docs)
            add_log(f"Contexto formateado (primeros chars): {formatted_context}...")

        return call_llm(question, formatted_context) # Obtener la respuesta del modelo

    except Exception as e:
        error_msg = f"Error durante la ejecución de RAG: {str(e)}"
        add_log(error_msg, "ERROR")
        # Podrías dar más detalles si el error es por el filtro
        if "filter" in str(e).lower():
             error_msg += " (Posible problema con la sintaxis del filtro de metadatos para tu Vector Store específico)"
             add_log(error_msg, "ERROR")
        return f"Error en RAG: {str(e)}"

# --- Funciones STT y TTS ---

def transcribe_audio(audio_filepath):
    """Transcribe un archivo de audio a texto usando SpeechRecognition."""
    global r
    if audio_filepath is None:
        return None

    add_log(f"Intentando transcribir audio desde: {audio_filepath}")
    try:
        with sr.AudioFile(audio_filepath) as source:
            audio_data = r.record(source) # Leer el archivo completo
        # Intentar reconocer usando Google Web Speech API
        text = r.recognize_google(audio_data, language='es-MX') # Especificar español de México
        add_log(f"Audio transcrito exitosamente: {text}")
        return text
    except sr.UnknownValueError:
        error_msg = "Speech Recognition no pudo entender el audio."
        add_log(error_msg, "WARNING")
        return f"Error: {error_msg}"
    except sr.RequestError as e:
        error_msg = f"No se pudo solicitar resultados del servicio de Google Speech Recognition; {e}"
        add_log(error_msg, "ERROR")
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Error inesperado durante la transcripción: {str(e)}"
        add_log(error_msg, "ERROR")
        return f"Error: {error_msg}"

def text_to_speech(text):
    """Convierte texto a un archivo de audio MP3 usando gTTS."""
    # Ya no necesitamos verificar tts_enabled aquí, se hace antes de llamar
    if not text or text.startswith("Error:") or text.startswith("⚠️"):
         add_log("No se generará TTS para mensajes de error o advertencia.", "INFO")
         return None
    try:
        add_log(f"Generando TTS para: {text[:50]}...")
        tts = gTTS(text=text, lang=TTS_LANG, slow=False)
        # Crear un archivo temporal seguro
        temp_dir = tempfile.gettempdir()
        # Generar nombre de archivo único para evitar colisiones
        filename = f"tts_output_{uuid.uuid4()}.mp3"
        filepath = os.path.join(temp_dir, filename)
        tts.save(filepath)
        add_log(f"Archivo TTS guardado en: {filepath}")
        return filepath
    except Exception as e:
        error_msg = f"Error al generar Text-to-Speech: {str(e)}"
        add_log(error_msg, "ERROR")
        return None

# --- Funciones de la Interfaz (Gradio) ---

def chat_response(message, audio_input, history):
    """Maneja la respuesta del chat, incluyendo STT y TTS (si está activado)."""
    global vectorstore, tts_enabled
    processed_message = None
    response_text = ""
    tts_audio_path = None

    # 1. Verificar si el vectorstore está cargado
    if vectorstore is None:
        response_text = "⚠️ **Atención:** La base de datos vectorial no está cargada. Por favor, ve a 'Configuración', aplica la configuración y luego haz clic en 'Inicializar Chat / Cargar Base de Datos'."
        add_log("Intento de chat sin vectorstore cargado.", "WARNING")
        history.append((message or "Entrada de audio", response_text))
        # Devolver valores para limpiar entradas, actualizar historial y SIN audio TTS visible
        return "", None, history, gr.update(value=None, visible=False) #

    # 2. Procesar entrada (Audio o Texto)
    if audio_input is not None:
        add_log("Procesando entrada de audio...")
        transcribed_text = transcribe_audio(audio_input)
        if transcribed_text and not transcribed_text.startswith("Error:"):
            processed_message = transcribed_text
            add_log(f"Mensaje procesado desde audio: {processed_message}")
        else:
            # Si la transcripción falla, mostrar error y no continuar
            response_text = transcribed_text or "Error: Falló la transcripción del audio."
            add_log(f"Fallo en STT: {response_text}", "ERROR")
            history.append(("Entrada de audio", response_text))
            # Limpiar, historial con error, SIN audio TTS visible
            return "", None, history, gr.update(value=None, visible=False) # <<< MODIFICADO
    elif message and message.strip():
        processed_message = message.strip()
        add_log(f"Procesando entrada de texto: {processed_message}")
    else:
        # Ni texto ni audio válido
        add_log("Intento de enviar mensaje vacío o solo audio fallido.", "INFO")
        # No añadir nada al historial, simplemente limpiar entradas y SIN audio TTS visible
        return "", None, history, gr.update(value=None, visible=False) # <<< MODIFICADO

    # 3. Si tenemos un mensaje procesado, obtener respuesta del RAG
    if processed_message:
        add_log(f"Enviando al RAG: {processed_message}")
        try:
            response_text = rag_chain(processed_message)
            add_log("Respuesta de RAG recibida.")
        except Exception as e:
            error_msg = f"Error al procesar la consulta de chat con RAG: {str(e)}"
            add_log(error_msg, "ERROR")
            response_text = f"❌ **Error interno:** {error_msg}"

    # 4. Generar TTS para la respuesta (SOLO SI ESTÁ HABILITADO y hay texto válido)
    if response_text and not response_text.startswith("Error:") and not response_text.startswith("❌") and not response_text.startswith("⚠️"):
        if tts_enabled:
            tts_audio_path = text_to_speech(response_text)
            if tts_audio_path:
                 add_log("TTS habilitado, generación de audio exitosa.")
            else:
                 add_log("TTS habilitado, pero falló la generación de audio.")
        else:
            add_log("TTS deshabilitado, omitiendo generación de audio.")
    else:
         add_log("No se intentará generar TTS debido a respuesta vacía, de error o advertencia.")


    # 5. Añadir al historial (siempre que haya habido una interacción válida)
    if processed_message and response_text:
        display_message = message if message else "Entrada de audio" # Usar texto o etiqueta de audio
        history.append((display_message, response_text))
        add_log("Respuesta añadida al historial.")


    # 6. Devolver resultados para actualizar la UI
    # Limpiar texto, limpiar audio, historial actualizado, y el componente de audio con su valor y visibilidad
    # bool(tts_audio_path) será True si hay una ruta, False si es None
    return "", None, history, gr.update(value=tts_audio_path, visible=bool(tts_audio_path)) # <<< MODIFICADO

def change_model(model_choice, api_key=None, finetuning_name=None):
    """Cambia el modelo actual y configura la API key si es necesario."""
    global current_model, finetuning_model_name, openai_api_key_set, vectorstore

    # Resetear estado al cambiar de modelo
    openai_api_key_set = False
    vectorstore = None # Forzar recarga del vectorstore con los nuevos embeddings/configuración
    status_message = ""
    status_level = "INFO"

    current_model = model_choice
    add_log(f"Cambiando modelo a: {model_choice}")

    if model_choice in ["openai", "finetuning"]:
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            openai_api_key_set = True
            add_log("API key de OpenAI configurada.")
            status_message += f"✅ Modelo seleccionado: **{model_choice}**. API Key configurada. "
        else:
            add_log("Advertencia: Modelo OpenAI/Finetuning seleccionado pero no se proporcionó API key.", "WARNING")
            status_message += f"⚠️ Modelo seleccionado: **{model_choice}**. ¡Falta API Key! "
            status_level = "WARNING"

    if model_choice == "finetuning":
        if finetuning_name:
            finetuning_model_name = finetuning_name
            add_log(f"Nombre del modelo de finetuning configurado: {finetuning_name}")
            status_message += f"Modelo Finetuning: **{finetuning_name}**. "
        else:
            add_log("Advertencia: Modelo 'finetuning' seleccionado pero no se proporcionó nombre.", "WARNING")
            status_message += "⚠️ ¡Falta el nombre del modelo Finetuning! "
            status_level = "WARNING"

    add_log(status_message.replace("*","").replace("\n\n", " "), status_level)

    return gr.update(value=status_message, visible=True)

def update_fields_visibility(model_choice):
    """Actualiza la visibilidad de los campos de API Key y Finetuning."""
    if model_choice == "openai":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    elif model_choice == "finetuning":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def toggle_tts(enable_tts):
    """Actualiza el estado global de TTS."""
    global tts_enabled
    tts_enabled = enable_tts
    status = "habilitada" if enable_tts else "deshabilitada"
    add_log(f"Salida de audio (TTS) {status}.")

def get_logs():
    """Obtiene los logs actuales para mostrar en la UI."""
    return "\n".join(reversed(log_messages))

# --- Interfaz Gráfica (Gradio) ---
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    font=["Arial", "sans-serif"],  # Usa fuentes comunes del sistema
    font_mono=["Consolas", "monospace"]
).set(
    button_primary_background_fill="#007bff",
    button_primary_background_fill_hover="#0056b3",
    button_secondary_background_fill="#f8f9fa",
    button_secondary_text_color="#343a40",
    input_background_fill="#ffffff",
)
def apply_config_and_initialize(model_choice_val, api_key_val, finetuning_model_val):
    """
    Aplica la configuración del modelo y luego inicializa el vectorstore.
    Devuelve los mensajes de estado para ambas operaciones.
    """
    # 1. Aplicar configuración del modelo
    model_status_msg = change_model(model_choice_val, api_key_val, finetuning_model_val)

    # Verificar si el cambio de modelo fue exitoso antes de inicializar
    # (Asumiendo que change_model devuelve algo que indica éxito/error,
    # aquí asumimos que si no lanza excepción, está bien, pero podrías
    # necesitar ajustar esto basado en cómo funciona tu change_model)
    # Por simplicidad, procedemos a inicializar independientemente del resultado exacto,
    # pero mostrando ambos estados. Si change_model falla, load_vectorstore
    # podría usar la configuración anterior o fallar también.

    # 2. Inicializar Chat / Cargar Base de Datos
    init_status_msg = load_vectorstore() # No necesita inputs directos de UI

    # Devolver ambos mensajes de estado para actualizar la UI
    # Asegúrate de que los mensajes sean informativos (ej. Markdown)
    return model_status_msg, init_status_msg

# --- Interfaz de Gradio Modificada ---
with gr.Blocks(theme=theme) as iface:
    gr.Markdown(
        """
        # 🤖 Asesor Legal (RAG y OpenAI)
        Consulta información actualizada sobre la Ley Federal del Trabajo de México usando texto o voz.
        """
    )

    with gr.Tabs():
        # --- Pestaña de Chat ---
        with gr.TabItem("💬 Chat Interactivo"):
            with gr.Column():
                chatbot = gr.Chatbot( 
                label="Conversación", 
                bubble_full_width=False, 
                height=350, 
                elem_id="chatbot_component" 
            ) 
            # Componente para mostrar la respuesta hablada 
            tts_output = gr.Audio( 
                label="🔊 Respuesta Hablada", 
                type="filepath", # Espera una ruta de archivo 
                autoplay=False,  # No reproducir automáticamente 
                interactive=False, # El usuario no carga audio aquí 
                visible=False 
            ) 

            with gr.Row(elem_id="chat_input_row"): 
                # Entrada de texto 
                msg_input = gr.Textbox( 
                    show_label=False, 
                    placeholder="Escribe tu pregunta aquí...", 
                    scale=5, # Ajustar escala 
                    container=False, 
                    elem_id="msg_input_component" 
                ) 
                
                submit_btn = gr.Button("✉️ Enviar", variant="primary", scale=1, min_width=100) 
                clear_btn = gr.Button("🗑️ Limpiar", variant="secondary", scale=1, min_width=100) 
            # Entrada de audio (Micrófono) 
            audio_input = gr.Audio( 
                sources=["microphone"], # Usar micrófono 
                type="filepath",      # Guardar como archivo temporal 
                label="🎤 Grabar Pregunta", 
                scale=1, # Ajustar escala 
                
                container=False 
            ) 



        # --- Pestaña de Configuración ---
        with gr.TabItem("⚙️ Configuración"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🧠 Selección de Modelo LLM")
                    model_choice = gr.Radio(
                        choices=["openai", "finetuning"], value="openai",
                        label="Modelo a usar para generar respuestas",
                        info="Los embeddings siempre serán de OpenAI (text-embedding-3-small)."
                    )
                    api_key_input = gr.Textbox(
                        placeholder="Ingresa tu API key de OpenAI (sk-...)",
                        label="🔑 OpenAI API Key", type="password", visible=True
                    )
                    finetuning_model_input = gr.Textbox(
                        placeholder="Nombre completo (ej: ft:gpt-3.5-turbo:...)",
                        label="🏷️ Nombre del Modelo Finetuning", visible=False
                    )
                    finetuning_info = gr.Markdown(
                        "**Nota:** Para usar un modelo *finetuning*, necesitas tu API key y el nombre completo del modelo.",
                        visible=False
                    )
                    # Mensaje de estado para la configuración del modelo
                    model_status = gr.Markdown(visible=False)

                    gr.Markdown("---")
                    gr.Markdown("### 🔊 Opciones de Salida")
                    tts_checkbox = gr.Checkbox(
                        label="Habilitar respuesta de audio (Text-to-Speech)",
                        value=False, # Valor inicial coherente con la variable global
                        info="Activa o desactiva la conversión de la respuesta a voz."
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 🚀 Aplicar e Inicializar")
                    # --- BOTÓN COMBINADO ---
                    apply_and_init_btn = gr.Button("💾 Aplicar Configuración e Inicializar Chat", variant="primary")
                    # Mensaje de estado para la inicialización del chat/DB
                    chat_init_status = gr.Markdown(visible=False)

                    gr.Markdown(
                        """
                        ---
                        ### ℹ️ Información Importante
                        1.  **Configura:** Selecciona modelo, ingresa API Key (si aplica), ajusta opciones (como TTS).
                        2.  **Aplica e Inicializa:** Haz clic en 'Aplicar Configuración e Inicializar Chat'. Espera a que ambos pasos se completen (verás mensajes de estado).
                        3.  **Chatea:** Ve a la pestaña 'Chat Interactivo'.
                        """
                    )

            # --- Eventos de Configuración ---
            # Evento para el botón combinado
            apply_and_init_btn.click(
                fn=apply_config_and_initialize, # Llama a la nueva función orquestadora
                inputs=[model_choice, api_key_input, finetuning_model_input], # Inputs necesarios para ambas funciones originales
                outputs=[model_status, chat_init_status] # Outputs para ambos mensajes de estado
            )

            # Eventos que se mantienen igual
            model_choice.change(
                fn=update_fields_visibility, inputs=[model_choice],
                outputs=[api_key_input, finetuning_model_input, finetuning_info]
            )
            tts_checkbox.change(
                fn=toggle_tts, inputs=[tts_checkbox], outputs=None
            )

        # --- Pestaña de Logs ---
        with gr.TabItem("📜 Logs del Sistema"):
             with gr.Column(): # Usar columna para mejor disposición
                 refresh_btn = gr.Button("🔄 Actualizar Logs")
                 logs_output = gr.Textbox(
                     label="📋 Registros de Eventos (más recientes primero)",
                     lines=15, interactive=False, max_lines=30,
                     autoscroll=False
                 )
           # Evento para logs
                 refresh_btn.click(get_logs, inputs=None, outputs=logs_output)
           # Cargar logs inicialmente
                 iface.load(get_logs, inputs=None, outputs=logs_output)


    # --- Conectar eventos de chat (sin cambios aquí) ---
    chat_inputs = [msg_input, audio_input, chatbot]
    chat_outputs = [msg_input, audio_input, chatbot, tts_output]

    submit_btn.click(
        fn=chat_response,
        inputs=chat_inputs,
        outputs=chat_outputs,
        api_name="enviar_mensaje_audio" # Mantener api_name si es necesario
    )
    msg_input.submit(
        fn=chat_response,
        inputs=chat_inputs,
        outputs=chat_outputs,
        api_name="enviar_mensaje_texto_enter" # Mantener api_name si es necesario
    )
    clear_btn.click(
        lambda: ("", None, [], gr.update(value=None, visible=False)),
        None,
        [msg_input, audio_input, chatbot, tts_output],
        queue=False
    )

# --- Inicio de la Aplicación ---
add_log("Aplicación iniciada")
# Asegúrate de tener instaladas las dependencias:
# pip install gradio langchain-openai langchain-community openai chromadb tiktoken speechrecognition gTTS pydub
# Puede requerir ffmpeg para pydub/speechrecognition: sudo apt update && sudo apt install ffmpeg (Linux) o choco install ffmpeg (Windows)

# Lanzar la aplicación Gradio
# iface.launch(share=True) # Descomenta para compartir si es necesario
# iface.launch() # Lanza localmente por defecto
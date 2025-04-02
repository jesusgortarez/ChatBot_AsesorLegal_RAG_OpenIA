# --- Importaciones de Librerías ---
import os                           # Proporciona funciones para interactuar con el sistema operativo (ej. variables de entorno)
import re                           # Módulo para trabajar con expresiones regulares (para buscar patrones en texto)
import copy                         # Para crear copias de objetos (útil para no modificar diccionarios originales)
import gradio as gr                 # Librería para crear interfaces web interactivas rápidamente
import logging                      # Para registrar eventos y errores de la aplicación
import datetime                     # Para obtener la fecha y hora actual (usado en los logs)
import speech_recognition as sr     # Librería para la conversión de voz a texto (STT)
from gtts import gTTS               # Librería de Google para la conversión de texto a voz (TTS)
import tempfile                     # Para crear archivos y directorios temporales (usado para guardar el audio TTS)
import uuid                         # Para generar identificadores únicos universales (usado para nombres de archivo TTS únicos)
from langchain_community.vectorstores import Chroma # Importa Chroma de Langchain para interactuar con la base de datos vectorial
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # Importa clases de Langchain para interactuar con OpenAI (embeddings y modelos de chat)

# --- Configuración Global ---
CHROMA_DB_PATH = "chroma_db/"       # Ruta donde se encuentra (o se guardará) la base de datos vectorial Chroma
LOG_FILE = "app_log.txt"            # Nombre del archivo donde se guardarán los logs de la aplicación
TTS_LANG = 'es'                     # Idioma para la síntesis de voz (Text-to-Speech), 'es' para español

# --- Configuración del Logging ---
# Configura el sistema de logging para guardar mensajes en un archivo
logging.basicConfig(
    filename=LOG_FILE,              # Nombre del archivo de log
    level=logging.INFO,             # Nivel mínimo de mensajes a registrar (INFO, WARNING, ERROR, etc.)
    format="%(asctime)s - %(levelname)s - %(message)s", # Formato de cada línea de log
    datefmt="%Y-%m-%d %H:%M:%S"      # Formato de la fecha y hora en los logs
)

# --- Variables Globales ---
log_messages = []                   # Lista para almacenar los mensajes de log en memoria (para mostrarlos en la UI)
vectorstore = None                  # Variable que contendrá la base de datos vectorial Chroma una vez cargada
current_model = "openai"            # Modelo LLM seleccionado por defecto ("openai" o "finetuning")
finetuning_model_name = ""          # Nombre específico del modelo de finetuning (si se selecciona esa opción)
openai_api_key_set = False          # Indicador (booleano) para saber si la API key de OpenAI ha sido configurada
tts_enabled = False                 # Indicador (booleano) para controlar si la salida de audio TTS está activa
r = sr.Recognizer()                 # Instancia del reconocedor de voz de SpeechRecognition

# --- Funciones de Logging ---
def add_log(message, level="INFO"):
    """
    Añade un mensaje al log (tanto al archivo como a la lista en memoria).

    Args:
        message (str): El mensaje a registrar.
        level (str): El nivel del log (INFO, WARNING, ERROR).

    Returns:
        str: La entrada de log formateada que se añadió.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Obtiene fecha y hora actual
    log_entry = f"{timestamp} - {level} - {message}" # Formatea la entrada del log
    log_messages.append(log_entry) # Añade a la lista en memoria

    # Registra en el archivo de log según el nivel
    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)

    # Nota: Se podría añadir lógica para limitar el tamaño de log_messages si crece mucho

    return log_entry # Devuelve la entrada para posible uso inmediato

# --- Funciones Principales de la Aplicación ---

def get_embeddings():
    """
    Obtiene el modelo de embeddings configurado (actualmente solo OpenAI).
    Verifica si la API key está configurada antes de intentar crear el objeto.

    Returns:
        OpenAIEmbeddings or None: El objeto de embeddings si la key está configurada, None si no.
    """
    global current_model, openai_api_key_set # Accede a variables globales
    if not openai_api_key_set:
        # Si la API key no está, registra una advertencia y devuelve None
        add_log("Advertencia: Intentando obtener embeddings sin configurar la API key de OpenAI.", "WARNING")
        return None

    # Si la key está, crea y devuelve el objeto de embeddings de OpenAI
    add_log(f"Usando embeddings de OpenAI (modelo: text-embedding-3-small)")
    # Utiliza un modelo específico de embeddings de OpenAI
    return OpenAIEmbeddings(model="text-embedding-3-small")

def load_vectorstore():
    """
    Carga la base de datos vectorial Chroma desde la ruta especificada en CHROMA_DB_PATH.
    Verifica la existencia de la ruta y la configuración de la API key.

    Returns:
        gradio.update: Un objeto de actualización para Gradio, mostrando un mensaje de éxito o error.
    """
    global vectorstore, CHROMA_DB_PATH, openai_api_key_set # Accede a variables globales
    add_log(f"Intentando cargar Chroma DB desde: {CHROMA_DB_PATH}")

    # Verifica si la API key está configurada (necesaria para los embeddings)
    if not openai_api_key_set:
        error_msg = "❌ **Error:** La API key de OpenAI no ha sido configurada. Por favor, configúrala en la pestaña 'Configuración'."
        add_log(error_msg.replace("❌ **Error:** ", ""), "ERROR")
        # Devuelve un mensaje de error visible en la UI de Gradio
        return gr.update(value=error_msg, visible=True)

    # Verifica si la ruta a la base de datos existe
    if not os.path.exists(CHROMA_DB_PATH):
        error_msg = f"❌ **Error:** La ruta especificada para Chroma DB no existe: `{CHROMA_DB_PATH}`"
        add_log(error_msg.replace("❌ **Error:** ", "").replace("`",""), "ERROR")
        # Devuelve un mensaje de error visible en la UI de Gradio
        return gr.update(value=error_msg, visible=True)

    # Obtiene el modelo de embeddings (necesario para cargar/consultar Chroma)
    embeddings = get_embeddings()
    if embeddings is None:
        # Si no se pudieron obtener los embeddings (probablemente por falta de API key)
        error_msg = "❌ **Error:** No se pudieron obtener los embeddings. Asegúrate de que la API key de OpenAI esté configurada."
        add_log(error_msg.replace("❌ **Error:** ", ""), "ERROR")
        return gr.update(value=error_msg, visible=True)

    # Intenta cargar la base de datos Chroma
    try:
        # Crea el objeto Chroma especificando el directorio, la función de embedding y el nombre de la colección
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings, collection_name="leyes_mexico_selectivo")
        success_msg = f"✅ **Éxito:** Base de datos Chroma cargada correctamente desde `{CHROMA_DB_PATH}`."
        add_log(success_msg.replace("✅ **Éxito:** ", "").replace("`",""))
        # Devuelve un mensaje de éxito visible en la UI
        return gr.update(value=success_msg, visible=True)
    except Exception as e:
        # Si ocurre cualquier error durante la carga
        error_msg = f"❌ **Error** al cargar Chroma DB desde `{CHROMA_DB_PATH}`: {str(e)}"
        add_log(error_msg.replace("❌ **Error** ", "").replace("`",""), "ERROR")
        vectorstore = None # Asegura que vectorstore sea None si la carga falla
        # Devuelve un mensaje de error visible en la UI
        return gr.update(value=error_msg, visible=True)

def call_llm(question, context):
    """
    Llama al modelo de lenguaje grande (LLM) seleccionado (OpenAI o Finetuning)
    con la pregunta del usuario y el contexto recuperado.

    Args:
        question (str): La pregunta original del usuario.
        context (str): El contexto relevante recuperado de la base de datos vectorial.

    Returns:
        str: La respuesta generada por el LLM o un mensaje de error.
    """
    global current_model, finetuning_model_name, openai_api_key_set # Accede a variables globales
    add_log(f"Enviando pregunta al modelo ({current_model}): {question[:50]}...") # Log truncado

    # --- Construcción del Prompt ---
    # Define el prompt que se enviará al LLM, incluyendo instrucciones y el contexto/pregunta
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

    # Verifica si la API key está configurada antes de llamar al LLM
    if not openai_api_key_set:
        add_log("Error: API key de OpenAI no configurada al intentar llamar al LLM.", "ERROR")
        return "Error: La API key de OpenAI no está configurada. Por favor, configúrala."

    # Intenta llamar al LLM seleccionado
    try:
        if current_model == "openai":
            # Usa el modelo base de OpenAI (gpt-4o-mini)
            llm = ChatOpenAI(model="gpt-4o-mini")
            response = llm.invoke(formatted_prompt) # Envía el prompt al modelo
            result = response.content # Extrae el contenido de la respuesta
        elif current_model == "finetuning":
            # Verifica si se ha proporcionado un nombre para el modelo finetuning
            if not finetuning_model_name:
                add_log("Error: Modelo 'finetuning' seleccionado pero no se ha proporcionado un nombre de modelo.", "ERROR")
                return "Error: Falta el nombre del modelo de finetuning."
            add_log(f"Usando modelo de finetuning: {finetuning_model_name}")
            # Usa el modelo de finetuning especificado
            llm = ChatOpenAI(model=finetuning_model_name)
            response = llm.invoke(formatted_prompt) # Envía el prompt al modelo
            result = response.content # Extrae el contenido de la respuesta
        else:
            # Si el modelo seleccionado no es válido
            add_log(f"Error: Modelo desconocido seleccionado: {current_model}", "ERROR")
            return f"Error: Modelo no soportado: {current_model}"

        add_log("Respuesta recibida del modelo.")
        return result # Devuelve la respuesta del LLM
    except Exception as e:
        # Si ocurre un error durante la llamada al LLM
        error_msg = f"Error al llamar al LLM ({current_model}): {str(e)}"
        add_log(error_msg, "ERROR")
        # Si el error menciona la API key, da un mensaje más específico
        if "api key" in str(e).lower():
            return "Error: Problema con la API key de OpenAI. Verifícala en la configuración."
        # Devuelve un mensaje de error genérico
        return f"Error al contactar el modelo: {str(e)}"

def extract_target_article(question):
    """
    Intenta extraer un número de artículo de la pregunta del usuario usando expresiones regulares.
    Busca patrones como "artículo X", "art. X", "articulo numero X", etc.

    Args:
        question (str): La pregunta del usuario.

    Returns:
        int or None: El número de artículo encontrado como entero, o None si no se encuentra.
    """
    # Expresión regular para buscar "art..." seguido opcionalmente de "numero" y un número (\d+)
    # re.IGNORECASE hace que la búsqueda no distinga mayúsculas/minúsculas
    match = re.search(r'art[íi]?[c]?[u]?[l]?[o]?\s*(?:numero|núm\.?|num\.?)?\s*(\d+)', question, re.IGNORECASE)
    if match:
        try:
            # Si encuentra una coincidencia, intenta convertir el grupo capturado (el número) a entero
            return int(match.group(1))
        except ValueError:
            # Si la conversión falla (aunque \d+ debería asegurar que sea un número)
            return None
    # Si no hay coincidencia
    return None

def rag_chain(question):
    """
    Ejecuta el proceso de Retrieval-Augmented Generation (RAG).
    1. Recupera documentos relevantes de la base de datos vectorial (filtrando por artículo si se detecta).
    2. Formatea el contexto con los documentos recuperados.
    3. Llama al LLM con la pregunta y el contexto formateado.

    Args:
        question (str): La pregunta del usuario.

    Returns:
        str: La respuesta generada por el LLM o un mensaje de error.
    """
    global vectorstore # Accede a la base de datos vectorial global
    SEARCH_TYPE = "mmr" # Tipo de búsqueda a usar (Maximal Marginal Relevance para diversidad)
    # Parámetros de búsqueda: k = número de documentos a devolver, fetch_k = número a recuperar inicialmente
    BASE_SEARCH_KWARGS = {"k": 20, "fetch_k": 20}

    # Verifica si la base de datos vectorial está cargada
    if vectorstore is None:
        add_log("Error: Vectorstore no cargado. Intenta inicializar el chat.", "ERROR")
        return "Error: La base de datos vectorial no está cargada. Por favor, inicializa el chat primero desde 'Configuración'."

    add_log(f"Iniciando cadena RAG con la configuracion base, search_type : {SEARCH_TYPE}, search_kwargs base: {BASE_SEARCH_KWARGS}")

    # Intenta extraer un número de artículo específico de la pregunta
    target_article = extract_target_article(question)
    # Crea una copia de los parámetros de búsqueda para no modificar el original
    current_search_kwargs = copy.deepcopy(BASE_SEARCH_KWARGS)
    filter_applied = False # Indicador para saber si se aplicó un filtro

    # Si se encontró un número de artículo
    if target_article is not None:
        add_log(f"Se detectó solicitud del artículo específico: {target_article}. Aplicando filtro en la búsqueda...")
        # Crea un diccionario de filtro para los metadatos (asumiendo que hay un campo 'numero_articulo')
        metadata_filter = {'numero_articulo': target_article}
        # Añade el filtro a los parámetros de búsqueda
        current_search_kwargs['filter'] = metadata_filter
        filter_applied = True
        add_log(f"Search_kwargs con filtro: {current_search_kwargs}")
    else:
        # Si no se encontró artículo, se hace búsqueda general
        add_log("No se detectó solicitud de artículo específico. Se realizará búsqueda general.")

    # Intenta realizar la recuperación y la llamada al LLM
    try:
        # Configura el retriever de Langchain usando el vectorstore y los parámetros de búsqueda (con filtro si aplica)
        retriever = vectorstore.as_retriever(
            search_type=SEARCH_TYPE,
            search_kwargs=current_search_kwargs # Usa los kwargs (posiblemente con filtro)
        )

        add_log("Recuperando documentos relevantes (con filtro si aplica)...")
        # Ejecuta la recuperación de documentos basada en la pregunta
        retrieved_docs = retriever.invoke(question)

        add_log(f"Recuperados {len(retrieved_docs)} documentos relevantes.")

        # Si no se encontraron documentos
        if not retrieved_docs:
            log_message = f"No se encontraron documentos para el artículo {target_article} que coincidan con la consulta." if filter_applied else "No se encontraron documentos relevantes para la consulta."
            add_log(log_message, "WARNING")
            # Prepara un contexto indicando que no se encontró información
            formatted_context = f"No se encontró información relevante para el artículo {target_article} en la base de datos." if filter_applied else "No se encontró información relevante en la base de datos para tu consulta."
        else:
            # Si se encontraron documentos, formatea el contexto uniendo el contenido de las páginas
            final_docs = retrieved_docs
            formatted_context = "\n\n".join(doc.page_content for doc in final_docs)
            add_log(f"Contexto formateado (primeros chars): {formatted_context[:100]}...") # Log truncado

        # Llama al LLM con la pregunta y el contexto obtenido
        return call_llm(question, formatted_context)

    except Exception as e:
        # Si ocurre un error durante el proceso RAG
        error_msg = f"Error durante la ejecución de RAG: {str(e)}"
        add_log(error_msg, "ERROR")
        # Si el error está relacionado con el filtro, añade una nota
        if "filter" in str(e).lower():
            error_msg += " (Posible problema con la sintaxis del filtro de metadatos para tu Vector Store específico)"
            add_log(error_msg, "ERROR")
        return f"Error en RAG: {str(e)}"

# --- Funciones de Speech-to-Text (STT) y Text-to-Speech (TTS) ---

def transcribe_audio(audio_filepath):
    """
    Transcribe un archivo de audio a texto usando la librería SpeechRecognition
    con el motor de reconocimiento de Google.

    Args:
        audio_filepath (str): La ruta al archivo de audio a transcribir.

    Returns:
        str or None: El texto transcrito o un mensaje de error/None si la entrada es None.
    """
    global r # Accede al reconocedor global
    if audio_filepath is None:
        # Si no hay ruta de archivo, no hay nada que transcribir
        return None

    add_log(f"Intentando transcribir audio desde: {audio_filepath}")
    try:
        # Abre el archivo de audio
        with sr.AudioFile(audio_filepath) as source:
            # Lee los datos del archivo de audio
            audio_data = r.record(source)
        # Intenta reconocer el audio usando la API de Google Web Speech
        # Especifica el idioma español (México) para mejorar la precisión
        text = r.recognize_google(audio_data, language='es-MX')
        add_log(f"Audio transcrito exitosamente: {text}")
        return text # Devuelve el texto reconocido
    except sr.UnknownValueError:
        # Si el reconocedor no pudo entender el audio
        error_msg = "Speech Recognition no pudo entender el audio."
        add_log(error_msg, "WARNING")
        return f"Error: {error_msg}"
    except sr.RequestError as e:
        # Si hubo un problema al contactar la API de Google
        error_msg = f"No se pudo solicitar resultados del servicio de Google Speech Recognition; {e}"
        add_log(error_msg, "ERROR")
        return f"Error: {error_msg}"
    except Exception as e:
        # Cualquier otro error inesperado
        error_msg = f"Error inesperado durante la transcripción: {str(e)}"
        add_log(error_msg, "ERROR")
        return f"Error: {error_msg}"

def text_to_speech(text):
    """
    Convierte un texto dado a un archivo de audio MP3 usando gTTS (Google Text-to-Speech).
    Guarda el archivo en un directorio temporal con un nombre único.

    Args:
        text (str): El texto a convertir en voz.

    Returns:
        str or None: La ruta al archivo MP3 generado o None si hubo un error o el texto no es válido.
    """
    # No genera audio para mensajes vacíos, de error o advertencia
    if not text or text.startswith("Error:") or text.startswith("⚠️"):
        add_log("No se generará TTS para mensajes de error o advertencia.", "INFO")
        return None
    try:
        add_log(f"Generando TTS para: {text[:50]}...") # Log truncado
        # Crea el objeto gTTS con el texto y el idioma especificado
        tts = gTTS(text=text, lang=TTS_LANG, slow=False) # slow=False para velocidad normal
        # Obtiene el directorio temporal del sistema
        temp_dir = tempfile.gettempdir()
        # Genera un nombre de archivo único usando UUID para evitar colisiones
        filename = f"tts_output_{uuid.uuid4()}.mp3"
        # Construye la ruta completa al archivo
        filepath = os.path.join(temp_dir, filename)
        # Guarda el audio generado en el archivo
        tts.save(filepath)
        add_log(f"Archivo TTS guardado en: {filepath}")
        return filepath # Devuelve la ruta al archivo generado
    except Exception as e:
        # Si ocurre algún error durante la generación del TTS
        error_msg = f"Error al generar Text-to-Speech: {str(e)}"
        add_log(error_msg, "ERROR")
        return None

# --- Funciones de la Interfaz de Usuario (Gradio Callbacks) ---

def chat_response(message, audio_input, history):
    """
    Callback principal para manejar la interacción del chat en Gradio.
    Procesa la entrada (texto o audio), llama a la cadena RAG y opcionalmente genera TTS.

    Args:
        message (str): El mensaje de texto ingresado por el usuario.
        audio_input (str): La ruta al archivo de audio grabado por el usuario (si lo hay).
        history (list): El historial de la conversación del chatbot de Gradio.

    Returns:
        tuple: Una tupla con los valores actualizados para los componentes de Gradio:
               (mensaje_input_limpio, audio_input_limpio, historial_actualizado, tts_output_actualizado)
    """
    global vectorstore, tts_enabled # Accede a variables globales
    processed_message = None # Variable para guardar el mensaje final a procesar (texto o transcripción)
    response_text = ""       # Variable para guardar la respuesta del RAG/LLM
    tts_audio_path = None    # Variable para guardar la ruta del archivo TTS generado

    # 1. Verificar si la base de datos vectorial está cargada
    if vectorstore is None:
        response_text = "⚠️ **Atención:** La base de datos vectorial no está cargada. Por favor, ve a 'Configuración', aplica la configuración y luego haz clic en 'Inicializar Chat / Cargar Base de Datos'."
        add_log("Intento de chat sin vectorstore cargado.", "WARNING")
        # Añade el mensaje de advertencia al historial
        history.append((message or "Entrada de audio", response_text))
        # Limpia campos de entrada, actualiza historial, oculta componente de audio TTS
        return "", None, history, gr.update(value=None, visible=False)

    # 2. Procesar la entrada del usuario (prioriza audio si existe)
    if audio_input is not None:
        add_log("Procesando entrada de audio...")
        # Transcribe el audio
        transcribed_text = transcribe_audio(audio_input)
        # Si la transcripción fue exitosa
        if transcribed_text and not transcribed_text.startswith("Error:"):
            processed_message = transcribed_text # Usa el texto transcrito como mensaje
            add_log(f"Mensaje procesado desde audio: {processed_message}")
        else:
            # Si la transcripción falló, muestra el error y detiene el proceso
            response_text = transcribed_text or "Error: Falló la transcripción del audio."
            add_log(f"Fallo en STT: {response_text}", "ERROR")
            history.append(("Entrada de audio", response_text)) # Añade el error al historial
            # Limpia campos, actualiza historial con error, oculta audio TTS
            return "", None, history, gr.update(value=None, visible=False)
    elif message and message.strip():
        # Si no hay audio pero sí hay texto, usa el texto
        processed_message = message.strip() # Elimina espacios en blanco al inicio/final
        add_log(f"Procesando entrada de texto: {processed_message}")
    else:
        # Si no hay ni audio válido ni texto
        add_log("Intento de enviar mensaje vacío o solo audio fallido.", "INFO")
        # No hace nada, solo limpia los campos y oculta audio TTS
        return "", None, history, gr.update(value=None, visible=False)

    # 3. Si se obtuvo un mensaje válido (texto o transcripción), ejecutar RAG
    if processed_message:
        add_log(f"Enviando al RAG: {processed_message}")
        try:
            # Llama a la función RAG para obtener la respuesta
            response_text = rag_chain(processed_message)
            add_log("Respuesta de RAG recibida.")
        except Exception as e:
            # Si ocurre un error durante RAG
            error_msg = f"Error al procesar la consulta de chat con RAG: {str(e)}"
            add_log(error_msg, "ERROR")
            response_text = f"❌ **Error interno:** {error_msg}" # Muestra un error interno

    # 4. Generar Text-to-Speech (TTS) si está habilitado y la respuesta es válida
    # No genera TTS para errores o advertencias
    if response_text and not response_text.startswith("Error:") and not response_text.startswith("❌") and not response_text.startswith("⚠️"):
        if tts_enabled: # Verifica si el TTS está habilitado globalmente
            tts_audio_path = text_to_speech(response_text) # Intenta generar el audio
            if tts_audio_path:
                add_log("TTS habilitado, generación de audio exitosa.")
            else:
                add_log("TTS habilitado, pero falló la generación de audio.")
        else:
            add_log("TTS deshabilitado, omitiendo generación de audio.")
    else:
        add_log("No se intentará generar TTS debido a respuesta vacía, de error o advertencia.")

    # 5. Añadir la interacción al historial del chatbot
    if processed_message and response_text:
        # Usa el mensaje original o "Entrada de audio" como etiqueta de la entrada del usuario
        display_message = message if message else "Entrada de audio"
        history.append((display_message, response_text)) # Añade el par (usuario, bot) al historial
        add_log("Respuesta añadida al historial.")

    # 6. Devolver los valores actualizados para la interfaz de Gradio
    # Limpia el campo de texto, limpia el campo de audio, devuelve el historial actualizado,
    # y actualiza el componente de audio TTS (lo hace visible solo si tts_audio_path tiene una ruta)
    return "", None, history, gr.update(value=tts_audio_path, visible=bool(tts_audio_path))

def change_model(model_choice, api_key=None, finetuning_name=None):
    """
    Callback para cambiar el modelo LLM seleccionado y configurar la API key.
    Resetea el vectorstore para forzar su recarga con la nueva configuración.

    Args:
        model_choice (str): El modelo seleccionado ("openai" o "finetuning").
        api_key (str, optional): La API key de OpenAI ingresada. Defaults to None.
        finetuning_name (str, optional): El nombre del modelo finetuning ingresado. Defaults to None.

    Returns:
        gradio.update: Objeto de actualización para mostrar el estado de la configuración del modelo.
    """
    global current_model, finetuning_model_name, openai_api_key_set, vectorstore # Accede a globales

    # Resetea el estado relacionado con OpenAI y el vectorstore al cambiar de modelo
    openai_api_key_set = False
    vectorstore = None # Fuerza la recarga si se inicializa de nuevo
    status_message = "" # Mensaje para mostrar en la UI
    status_level = "INFO" # Nivel de log para el mensaje

    # Actualiza el modelo actual seleccionado
    current_model = model_choice
    add_log(f"Cambiando modelo a: {model_choice}")

    # Si el modelo requiere API key de OpenAI (openai o finetuning)
    if model_choice in ["openai", "finetuning"]:
        if api_key:
            # Si se proporcionó una API key, la establece como variable de entorno
            os.environ["OPENAI_API_KEY"] = api_key
            openai_api_key_set = True # Marca que la key está configurada
            add_log("API key de OpenAI configurada.")
            status_message += f"✅ Modelo seleccionado: **{model_choice}**. API Key configurada. "
        else:
            # Si no se proporcionó API key
            add_log("Advertencia: Modelo OpenAI/Finetuning seleccionado pero no se proporcionó API key.", "WARNING")
            status_message += f"⚠️ Modelo seleccionado: **{model_choice}**. ¡Falta API Key! "
            status_level = "WARNING"

    # Si se seleccionó el modelo finetuning, verifica el nombre
    if model_choice == "finetuning":
        if finetuning_name:
            # Si se proporcionó un nombre, lo guarda
            finetuning_model_name = finetuning_name
            add_log(f"Nombre del modelo de finetuning configurado: {finetuning_name}")
            status_message += f"Modelo Finetuning: **{finetuning_name}**. "
        else:
            # Si no se proporcionó nombre
            add_log("Advertencia: Modelo 'finetuning' seleccionado pero no se proporcionó nombre.", "WARNING")
            status_message += "⚠️ ¡Falta el nombre del modelo Finetuning! "
            status_level = "WARNING"

    # Registra el mensaje de estado final
    add_log(status_message.replace("*","").replace("\n\n", " "), status_level)

    # Devuelve el mensaje para actualizar el componente Markdown en la UI
    return gr.update(value=status_message, visible=True)

def update_fields_visibility(model_choice):
    """
    Callback para mostrar u ocultar los campos de API Key y nombre de Finetuning
    basado en la selección del modelo.

    Args:
        model_choice (str): El modelo seleccionado.

    Returns:
        tuple: Una tupla de objetos gr.update para controlar la visibilidad de los campos.
               (visibilidad_api_key, visibilidad_finetuning_name, visibilidad_finetuning_info)
    """
    if model_choice == "openai":
        # OpenAI base solo necesita API key
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    elif model_choice == "finetuning":
        # Finetuning necesita API key y nombre del modelo
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    else:
        # Otros modelos (si se añadieran) podrían no necesitar nada
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def toggle_tts(enable_tts):
    """
    Callback para actualizar el estado global que controla si el TTS está habilitado.

    Args:
        enable_tts (bool): El valor del checkbox de habilitar TTS.
    """
    global tts_enabled # Accede a la variable global
    tts_enabled = enable_tts # Actualiza el estado
    status = "habilitada" if enable_tts else "deshabilitada"
    add_log(f"Salida de audio (TTS) {status}.")

def get_logs():
    """
    Callback para obtener los mensajes de log almacenados en memoria y mostrarlos en la UI.
    Muestra los logs más recientes primero.

    Returns:
        str: Una cadena con todos los mensajes de log, separados por saltos de línea.
    """
    # Une los mensajes de la lista `log_messages` (invertida) con saltos de línea
    return "\n".join(reversed(log_messages))

# --- Definición de la Interfaz Gráfica con Gradio ---

# Define un tema visual para la interfaz
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    font=["Arial", "sans-serif"],
    font_mono=["Consolas", "monospace"]
).set( # Permite personalizar colores específicos
    button_primary_background_fill="#007bff",
    button_primary_background_fill_hover="#0056b3",
    button_secondary_background_fill="#f8f9fa",
    button_secondary_text_color="#343a40",
    input_background_fill="#ffffff",
)

def apply_config_and_initialize(model_choice_val, api_key_val, finetuning_model_val):
    """
    Función orquestadora que se llama al presionar el botón "Aplicar e Inicializar".
    Primero aplica la configuración del modelo y luego intenta cargar el vectorstore.

    Args:
        model_choice_val (str): Valor del radio button de selección de modelo.
        api_key_val (str): Valor del campo de texto de API Key.
        finetuning_model_val (str): Valor del campo de texto de nombre de finetuning.

    Returns:
        tuple: Una tupla con los mensajes de estado de ambas operaciones para la UI.
               (mensaje_estado_modelo, mensaje_estado_inicializacion)
    """
    # 1. Aplica la configuración del modelo (llama a change_model)
    model_status_msg = change_model(model_choice_val, api_key_val, finetuning_model_val)

    # 2. Intenta inicializar/cargar la base de datos vectorial (llama a load_vectorstore)
    #    load_vectorstore usará las variables globales actualizadas por change_model
    init_status_msg = load_vectorstore()

    # Devuelve ambos mensajes de estado para actualizar los componentes Markdown correspondientes
    return model_status_msg, init_status_msg

# --- Construcción de la Interfaz de Gradio ---
# gr.Blocks permite una disposición más personalizada de los componentes
with gr.Blocks(theme=theme) as iface:
    # Título principal de la aplicación
    gr.Markdown(
        """
        # 🤖 Asesor Legal (RAG y OpenAI)
        Consulta información actualizada sobre la Ley Federal del Trabajo de México usando texto o voz.
        """
    )

    # Pestañas para organizar la interfaz
    with gr.Tabs():
        # --- Pestaña de Chat ---
        with gr.TabItem("💬 Chat Interactivo"):
            with gr.Column(): # Organiza verticalmente
                # Componente de Chatbot para mostrar la conversación
                chatbot = gr.Chatbot(
                    label="Conversación",
                    bubble_full_width=False, # Las burbujas no ocupan todo el ancho
                    height=350,             # Altura fija del chatbot
                    elem_id="chatbot_component" # ID para posible CSS/JS personalizado
                )
                # Componente de Audio para reproducir la respuesta TTS
                tts_output = gr.Audio(
                    label="🔊 Respuesta Hablada",
                    type="filepath",        # Espera una ruta de archivo como salida
                    autoplay=False,         # No reproducir automáticamente
                    interactive=False,      # El usuario no puede cargar audio aquí
                    visible=False           # Inicialmente oculto
                )

            # Fila para los controles de entrada de chat
            with gr.Row(elem_id="chat_input_row"):
                # Campo de texto para la entrada del usuario
                msg_input = gr.Textbox(
                    show_label=False,       # No mostrar etiqueta
                    placeholder="Escribe tu pregunta aquí...", # Texto de ayuda
                    scale=5,                # Ocupa más espacio horizontal en la fila
                    container=False,        # No poner en un contenedor visual extra
                    elem_id="msg_input_component" # ID
                )
                # Botón para enviar el mensaje de texto
                submit_btn = gr.Button("✉️ Enviar", variant="primary", scale=1, min_width=100)
                # Botón para limpiar la conversación y entradas
                clear_btn = gr.Button("🗑️ Limpiar", variant="secondary", scale=1, min_width=100)
            # Componente de Audio para grabar la entrada de voz del usuario
            audio_input = gr.Audio(
                sources=["microphone"],     # Usa el micrófono como fuente
                type="filepath",            # Guarda la grabación como un archivo temporal
                label="🎤 Grabar Pregunta",
                scale=1,                    # Ajusta escala si es necesario en su contenedor
                container=False             # Sin contenedor extra
            )

        # --- Pestaña de Configuración ---
        with gr.TabItem("⚙️ Configuración"):
            with gr.Row(): # Organiza en dos columnas
                # Columna izquierda para configuración de modelo y TTS
                with gr.Column(scale=1):
                    gr.Markdown("### 🧠 Selección de Modelo LLM")
                    # Radio buttons para elegir el modelo
                    model_choice = gr.Radio(
                        choices=["openai", "finetuning"], value="openai", # Opciones y valor por defecto
                        label="Modelo a usar para generar respuestas",
                        info="Los embeddings siempre serán de OpenAI (text-embedding-3-small)." # Texto informativo
                    )
                    # Campo para la API Key (tipo password para ocultar)
                    api_key_input = gr.Textbox(
                        placeholder="Ingresa tu API key de OpenAI (sk-...)",
                        label="🔑 OpenAI API Key", type="password", visible=True # Visible por defecto
                    )
                    # Campo para el nombre del modelo finetuning
                    finetuning_model_input = gr.Textbox(
                        placeholder="Nombre completo (ej: ft:gpt-3.5-turbo:...)",
                        label="🏷️ Nombre del Modelo Finetuning", visible=False # Oculto por defecto
                    )
                    # Nota informativa sobre finetuning
                    finetuning_info = gr.Markdown(
                        "**Nota:** Para usar un modelo *finetuning*, necesitas tu API key y el nombre completo del modelo.",
                        visible=False # Oculto por defecto
                    )
                    # Componente para mostrar el estado de la configuración del modelo
                    model_status = gr.Markdown(visible=False) # Oculto inicialmente

                    gr.Markdown("---") # Separador visual
                    gr.Markdown("### 🔊 Opciones de Salida")
                    # Checkbox para habilitar/deshabilitar TTS
                    tts_checkbox = gr.Checkbox(
                        label="Habilitar respuesta de audio (Text-to-Speech)",
                        value=False, # Deshabilitado por defecto (coherente con variable global)
                        info="Activa o desactiva la conversión de la respuesta a voz."
                    )

                # Columna derecha para el botón de inicialización e información
                with gr.Column(scale=1):
                    gr.Markdown("### 🚀 Aplicar e Inicializar")
                    # Botón combinado para aplicar configuración e inicializar
                    apply_and_init_btn = gr.Button("💾 Aplicar Configuración e Inicializar Chat", variant="primary")
                    # Componente para mostrar el estado de la inicialización del chat/DB
                    chat_init_status = gr.Markdown(visible=False) # Oculto inicialmente

                    gr.Markdown( # Bloque informativo con instrucciones
                        """
                        ---
                        ### ℹ️ Información Importante
                        1.  **Configura:** Selecciona modelo, ingresa API Key (si aplica), ajusta opciones (como TTS).
                        2.  **Aplica e Inicializa:** Haz clic en 'Aplicar Configuración e Inicializar Chat'. Espera a que ambos pasos se completen (verás mensajes de estado).
                        3.  **Chatea:** Ve a la pestaña 'Chat Interactivo'.
                        """
                    )

            # --- Conexión de Eventos de Configuración ---
            # Evento para el botón combinado "Aplicar e Inicializar"
            apply_and_init_btn.click(
                fn=apply_config_and_initialize, # Llama a la función orquestadora
                inputs=[model_choice, api_key_input, finetuning_model_input], # Entradas necesarias
                outputs=[model_status, chat_init_status] # Salidas para actualizar los mensajes de estado
            )

            # Evento que se dispara cuando cambia la selección del modelo
            model_choice.change(
                fn=update_fields_visibility, # Llama a la función para actualizar visibilidad
                inputs=[model_choice], # La entrada es la nueva selección
                outputs=[api_key_input, finetuning_model_input, finetuning_info] # Actualiza la visibilidad de estos campos
            )
            # Evento que se dispara cuando cambia el estado del checkbox TTS
            tts_checkbox.change(
                fn=toggle_tts, # Llama a la función para actualizar el estado global
                inputs=[tts_checkbox], # La entrada es el nuevo estado del checkbox
                outputs=None # No actualiza directamente ningún componente de UI (solo variable global)
            )

        # --- Pestaña de Logs ---
        with gr.TabItem("📜 Logs del Sistema"):
              with gr.Column(): # Organiza verticalmente
                  # Botón para refrescar los logs mostrados
                  refresh_btn = gr.Button("🔄 Actualizar Logs")
                  # Campo de texto (multilínea y no editable) para mostrar los logs
                  logs_output = gr.Textbox(
                      label="📋 Registros de Eventos (más recientes primero)",
                      lines=15, interactive=False, max_lines=30, # Configuración de tamaño y edición
                      autoscroll=False # No hacer scroll automático
                  )
            # Evento para el botón de actualizar logs
                  refresh_btn.click(get_logs, inputs=None, outputs=logs_output) # Llama a get_logs y actualiza el Textbox
            # Carga inicial de los logs cuando la interfaz se carga
                  iface.load(get_logs, inputs=None, outputs=logs_output) # Llama a get_logs al inicio


    # --- Conexión de Eventos de Chat ---
    # Define las entradas y salidas comunes para las acciones de chat
    chat_inputs = [msg_input, audio_input, chatbot] # Mensaje, Audio, Historial
    chat_outputs = [msg_input, audio_input, chatbot, tts_output] # Limpia msg, limpia audio, actualiza historial, actualiza audio TTS

    # Evento al hacer clic en el botón "Enviar"
    submit_btn.click(
        fn=chat_response,           # Llama a la función principal de chat
        inputs=chat_inputs,         # Pasa las entradas definidas
        outputs=chat_outputs,       # Recibe las salidas definidas
        api_name="enviar_mensaje_audio" # Nombre opcional para la API (si se usa)
    )
    # Evento al presionar Enter en el campo de texto
    msg_input.submit(
        fn=chat_response,           # Llama a la misma función de chat
        inputs=chat_inputs,
        outputs=chat_outputs,
        api_name="enviar_mensaje_texto_enter"
    )
    # Evento al hacer clic en el botón "Limpiar"
    clear_btn.click(
        # Usa una función lambda para devolver los valores de reseteo
        lambda: ("", None, [], gr.update(value=None, visible=False)),
        None,                       # Sin entradas explícitas para la lambda
        [msg_input, audio_input, chatbot, tts_output], # Componentes a limpiar/resetear
        queue=False                 # No poner esta acción en cola (ejecución inmediata)
    )

# --- Inicio de la Aplicación ---
add_log("Aplicación iniciada") # Registra el inicio de la aplicación

# Nota sobre dependencias (puede ser útil si se comparte el script)
# Asegúrate de tener instaladas las dependencias:
# pip install gradio langchain-openai langchain-community openai chromadb tiktoken speechrecognition gTTS pydub
# Puede requerir ffmpeg para pydub/speechrecognition: sudo apt update && sudo apt install ffmpeg (Linux) o choco install ffmpeg (Windows)

# Lanza la interfaz de Gradio
# iface.launch(share=True) # Descomentar para obtener un enlace público temporal (útil para compartir)
# iface.launch() # Lanza la interfaz localmente (accesible en http://127.0.0.1:7860 por defecto)

from fastapi import FastAPI
import gradio as gr

from ChatBot_AsesorLegal_RAG import iface

app = FastAPI()

@app.get('/')
async def root():
    return 'Gradio app is running at /gradio', 200

app = gr.mount_gradio_app(app, iface, path='/gradio')

#uvicorn run:app --host 0.0.0.0 --port 5000


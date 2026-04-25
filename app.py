import gradio as gr
import torch
import librosa
from transformers import pipeline

# Use pipeline instead (lighter + safer)
pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

def transcribe(audio):
    audio, sr = librosa.load(audio, sr=16000)
    result = pipe(audio)
    return result["text"]

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Speech to Text",
)

iface.launch()

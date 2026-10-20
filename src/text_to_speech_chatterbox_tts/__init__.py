import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Cargando modelo TTS...")

modelo = ChatterboxMultilingualTTS.from_pretrained(
    device="cpu"
)
texto = "Bienvenidos a inteligencia artificial generativa vamos a crear imagene, audios , texto y si el pc aguanta video"

print("Generando audio...")

audio = modelo.generate(
    text=texto,
    language_id="es",
    exaggeration=0.7,
    cfg_weight=0.4,
    temperature=0.8,
    repetition_penalty=1.2,
    min_p=0.05,
    top_p=1.0
)
ta.save(
    "audio.wav",
    audio,
    modelo.sr

)
print("audio guardado.")

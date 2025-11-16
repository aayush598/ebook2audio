#######################################################################################
#
# MIT License
#
# Copyright (c) [2025] [leonelhs@gmail.com]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#######################################################################################

# This file implements an API endpoint for the Hindi Kokoro Text-to-Speech (TTS) system.
# It provides functionality to generate TTS audio from input Hindi text using the Kokoro voice model.


# Source code is based on or inspired by several projects.
# For more details and proper attribution, please refer to the following resources:
#
# - [Kokoro] - [https://github.com/hexgrad/kokoro]
# - [Misaki] - [https://github.com/hexgrad/misaki]
# - [Kokoro-82M] - [https://huggingface.co/hexgrad/Kokoro-82M]
# - [Kokoro-onnx] - [https://github.com/thewh1teagle/kokoro-onnx]



import os
import gradio as gr
from misaki import espeak
from misaki.espeak import EspeakG2P
from kokoro_onnx import Kokoro
from huggingface_hub import snapshot_download

KOKORO_REPO_ID = "leonelhs/kokoro-thewh1teagle"

VOICES = {
    '🚺 Alpha':'hf_alpha',
    '🚺 Beta':'hf_beta',
    '🚹 Omega':'hm_omega',
    '🚹 Psi':'hm_psi'
}

snapshot = snapshot_download(repo_id=KOKORO_REPO_ID)

# Misaki G2P with espeak-ng fallback
fallback = espeak.EspeakFallback(british=False)
g2p = EspeakG2P(language="hi")

# Kokoro
model_path = os.path.join(snapshot, "kokoro-v1.0.onnx")
voices_path = os.path.join(snapshot, "voices-v1.0.bin")
kokoro = Kokoro(model_path, voices_path)

def predict(text, voice='hf_alpha', speed=1):
    """
        Generate speech audio from hindi text input.

        Parameters:
            text (string): The text to be converted into speech.
            voice (string): The selected male of female voice profile (specific voice ID).
            speed (float): The speaking rate multiplier (e.g., 1.0 = normal speed, 0.8 = slower, 1.2 = faster).

        Returns:
            path: File path to the generated audio speech.
    """

    phonemes, _ = g2p(text)
    samples, sample_rate = kokoro.create(phonemes, voice, speed, is_phonemes=True)
    return sample_rate, samples

app = gr.Interface(
    predict,
    [
        gr.Textbox(label='Input Text'),
        gr.Dropdown(list(VOICES.items()), value='hf_alpha', label='Voice'),
        gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label='Speed')
    ],
    gr.Audio(label='Output Audio', interactive=False, streaming=False, autoplay=True),
    description="Kokoro TTS 🇮🇳 API Endpoint",
)

app.launch(share=False, debug=True, show_error=True, mcp_server=True)
app.queue()


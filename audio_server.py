import numpy as np
from flask import Flask, request, jsonify
import soundfile as sf
import resampy
import io
import base64
from pydub import AudioSegment
from qai_hub_models.models.yamnet import Model, YAMNetConfig

app = Flask(__name__)

# Load YamNet model from QAI Hub
model = Model.from_pretrained()

def convert_audio_to_wav(audio_bytes, input_format='m4a'):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=input_format)
    wav_io = io.BytesIO()
    audio.export(wav_io, format='wav')
    wav_io.seek(0)
    return wav_io

def process_audio(audio, sr):
    """Process audio to match YamNet input requirements."""
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample to 16kHz if necessary
    if sr != 16000:
        audio = resampy.resample(audio, sr, 16000)
    
    # Normalize audio
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    return audio.astype(np.float32)

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    try:
        if request.files and 'audio' in request.files:
            file = request.files['audio']
            if file.filename.lower().endswith('.m4a'):
                wav_io = convert_audio_to_wav(file.read())
                audio, sr = sf.read(wav_io)
            else:
                audio, sr = sf.read(file)
        elif request.is_json:
            data = request.get_json()
            if 'audio' not in data:
                return jsonify({'predictions': []}), 400
            audio_bytes = base64.b64decode(data['audio'])
            filename = data.get('filename', '').lower()
            if filename.endswith('.m4a'):
                wav_io = convert_audio_to_wav(audio_bytes)
                audio, sr = sf.read(wav_io)
            else:
                audio, sr = sf.read(io.BytesIO(audio_bytes))
        else:
            return jsonify({'predictions': []}), 400

        # Process audio
        processed_audio = process_audio(audio, sr)
        
        # Run inference using QAI Hub model
        predictions = model({"waveform": processed_audio})
        
        # Get top 5 predictions
        scores = predictions['scores']
        top_5_indices = np.argsort(scores)[-5:][::-1]
        
        results = [
            {
                'label': predictions['labels'][idx],
                'confidence': float(scores[idx] * 100)
            }
            for idx in top_5_indices
        ]
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return jsonify({'predictions': [], 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
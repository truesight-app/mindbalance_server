import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, jsonify
import soundfile as sf
import resampy
import io
import base64
from pydub import AudioSegment
import csv

app = Flask(__name__)

# Load YAMNet model
model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load class map
class_names = []
with open('yamnet_class_map.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_names.append(row['display_name'])

def convert_audio_to_wav(audio_bytes, input_format='m4a'):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=input_format)
    wav_io = io.BytesIO()
    audio.export(wav_io, format='wav')
    wav_io.seek(0)
    return wav_io

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
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Ensure audio is float32
        audio = audio.astype(np.float32)
        
        # Resample to 16kHz if necessary
        if sr != 16000:
            audio = resampy.resample(audio, sr, 16000)
        
        # Run inference
        scores, embeddings, mel_spec = model(audio)
        
        # Get top 5 predictions
        class_scores = tf.reduce_mean(scores, axis=0)
        top_5_indices = tf.argsort(class_scores, direction='DESCENDING')[:5]
        
        predictions = [
            {
                'label': class_names[idx],
                'confidence': float(class_scores[idx].numpy() * 100)
            }
            for idx in top_5_indices.numpy()
        ]
        
        return jsonify({'predictions': predictions})
    
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return jsonify({'predictions': [], 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
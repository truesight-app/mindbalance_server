import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
import soundfile as sf
import resampy
import io
import base64
from pydub import AudioSegment
import csv
import librosa

app = Flask(__name__)

# Load ONNX model
session = ort.InferenceSession('YamNet.onnx')
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]
print("Model input name:", input_name)
print("Model output names:", output_names)

# Audio processing parameters
SAMPLE_RATE = 16000
N_MEL_BANDS = 96  # Changed from 64 to 96
WINDOW_SIZE = 400
HOP_SIZE = 160

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

def process_audio(audio, sr):
    # Convert to mono and resample if needed
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        audio = resampy.resample(audio, sr, SAMPLE_RATE)
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MEL_BANDS,
        n_fft=WINDOW_SIZE,
        hop_length=HOP_SIZE,
        fmin=125,
        fmax=7500
    )
    
    # Convert to log mel spectrogram
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / log_mel_spec.std()
    
    # Reshape for model (batch_size, channels, mel_bands, time_steps)
    return log_mel_spec.T.reshape(1, 1, 96, -1)[:, :, :, :64]

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
        
        # Process audio into mel spectrogram
        processed_audio = process_audio(audio, sr)
        
        # Run inference with ONNX
        input_data = {input_name: processed_audio.astype(np.float32)}
        outputs = session.run(output_names, input_data)
        scores = outputs[0]  # Get just the scores from the output
        
        # Get top 5 predictions
        class_scores = np.mean(scores, axis=0)
        top_5_indices = np.argsort(class_scores)[-5:][::-1]
        
        predictions = [
            {
                'label': class_names[idx],
                'confidence': float(class_scores[idx] * 100)
            }
            for idx in top_5_indices
        ]
        
        return jsonify({'predictions': predictions})
    
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return jsonify({'predictions': [], 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
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
session = ort.InferenceSession('YamNet.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

def convert_audio_to_wav(audio_bytes, input_format='m4a'):
    """Convert audio bytes to WAV format"""
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=input_format)
    wav_io = io.BytesIO()
    audio.export(wav_io, format='wav')
    wav_io.seek(0)
    return wav_io

def process_audio(audio, sr):
    """Process audio according to YAMNet requirements"""
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample to 16kHz if necessary
    if sr != 16000:
        audio = resampy.resample(audio, sr, 16000)
    
    # Normalize the waveform to be in [-1, 1]
    audio = audio / (np.max(np.abs(audio)) + 1e-10)
    
    # Parameters for mel spectrogram
    window_length_seconds = 0.025
    hop_length_seconds = 0.010
    n_fft = int(round(window_length_seconds * 16000))
    hop_length = int(round(hop_length_seconds * 16000))
    
    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=64,
        fmin=125.0,
        fmax=7500.0,
        power=1.0
    )
    
    # Convert to log mel spectrogram
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-5)
    
    # Prepare input shape (1, 1, 96, 64)
    if log_mel_spec.shape[1] < 96:
        pad_width = ((0, 0), (0, 96 - log_mel_spec.shape[1]))
        log_mel_spec = np.pad(log_mel_spec, pad_width, mode='constant')
    else:
        log_mel_spec = log_mel_spec[:, :96]
    
    # Reshape to model's expected input shape (1, 1, 96, 64)
    return log_mel_spec.T.reshape(1, 1, 96, 64)

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
        model_input = process_audio(audio, sr)
        
        # Run inference with ONNX
        outputs = session.run(None, {input_name: model_input.astype(np.float32)})
        scores = outputs[0]
        
        # Load class names
        class_names = []
        with open('yamnet_class_map.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row['display_name'])
        
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
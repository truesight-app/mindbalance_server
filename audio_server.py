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

# Audio processing parameters for YamNet
SAMPLE_RATE = 16000
WINDOW_LENGTH = 0.025  # 25ms
HOP_LENGTH = 0.010    # 10ms
N_MEL_BANDS = 64

def log_mel_spectrogram(waveform):
    """Convert waveform to log mel spectrogram following YamNet preprocessing."""
    # Compute STFT
    n_fft = int(SAMPLE_RATE * WINDOW_LENGTH)
    hop_length = int(SAMPLE_RATE * HOP_LENGTH)
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=N_MEL_BANDS,
        fmin=125,
        fmax=7500
    )
    
    # Convert to log mel spectrogram
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    
    # YamNet expects (1, 1, 96, 64) input shape
    # Pad or trim to 96 frames
    target_length = 96
    current_length = log_mel.shape[1]
    
    if current_length < target_length:
        pad_width = target_length - current_length
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)))
    elif current_length > target_length:
        log_mel = log_mel[:, :target_length]
    
    # Reshape to model's expected input shape (1, 1, 96, 64)
    return log_mel.T.reshape(1, 1, 96, 64)

def process_audio(audio, sr):
    """Process audio to match YamNet input requirements."""
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample to 16kHz if necessary
    if sr != SAMPLE_RATE:
        audio = resampy.resample(audio, sr, SAMPLE_RATE)
    
    # Normalize audio
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    return log_mel_spectrogram(audio)

def convert_audio_to_wav(audio_bytes, input_format='m4a'):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=input_format)
    wav_io = io.BytesIO()
    audio.export(wav_io, format='wav')
    wav_io.seek(0)
    return wav_io

# Load class map
class_names = []
with open('yamnet_class_map.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_names.append(row['display_name'])

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

        # Process audio to match model input requirements
        processed_audio = process_audio(audio, sr)
        
        # Run inference
        outputs = session.run(None, {input_name: processed_audio.astype(np.float32)})
        scores = outputs[0]
        
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
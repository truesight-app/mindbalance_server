import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
import soundfile as sf
import resampy
import io
import base64
from pydub import AudioSegment
import csv

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
    audio = audio.astype(np.float32)
    
    # Calculate spectrogram
    frame_length = int(0.025 * 16000)  # 25ms
    frame_step = int(0.010 * 16000)    # 10ms
    
    # Pad the signal to make sure we get enough frames
    pad_len = frame_length
    padded = np.pad(audio, (pad_len, pad_len), mode='reflect')
    
    # Frame the signal
    frames = []
    for i in range(0, len(padded) - frame_length, frame_step):
        frames.append(padded[i:i + frame_length])
    frames = np.array(frames)
    
    # Apply Hanning window
    window = np.hanning(frame_length)
    frames = frames * window
    
    # Compute FFT
    fft = np.abs(np.fft.rfft(frames, axis=1))
    
    # Prepare for model input (1, 1, 96, 64)
    # Take first 96 frames and reshape
    if len(fft) < 96:
        pad_frames = np.zeros((96 - len(fft), fft.shape[1]))
        fft = np.vstack([fft, pad_frames])
    else:
        fft = fft[:96]
    
    # Reshape to model's expected input shape
    model_input = fft.reshape(1, 1, 96, -1)
    if model_input.shape[3] > 64:
        model_input = model_input[:, :, :, :64]
    elif model_input.shape[3] < 64:
        pad_width = ((0, 0), (0, 0), (0, 0), (0, 64 - model_input.shape[3]))
        model_input = np.pad(model_input, pad_width, mode='constant')
    
    return model_input

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
        print(scores)
        
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
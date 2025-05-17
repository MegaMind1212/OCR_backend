from flask import Flask, request, jsonify
from flask_cors import CORS
from deepgram import Deepgram
import os
import logging
from werkzeug.utils import secure_filename
import ffmpeg
import tempfile

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "32683910a62128aacbef2a8e7710d94c77a46101")
deepgram = Deepgram(DEEPGRAM_API_KEY)

ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.mp4'}

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def root():
    return jsonify({"message": "Backend is running"})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    logger.info('Received /transcribe request')
    
    if 'audio' not in request.files:
        logger.error('No audio file uploaded')
        return jsonify({'error': 'No audio file uploaded'}), 400

    file = request.files['audio']
    filename = secure_filename(file.filename)
    
    if not allowed_file(filename):
        logger.error(f'Unsupported file type: {filename}')
        return jsonify({'error': 'Unsupported file type. Please upload an MP3, WAV, or MP4 file.'}), 400

    logger.info(f'Uploaded file: {filename}, size: {file.stream.seek(0, os.SEEK_END)} bytes')
    file.stream.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
        file.save(temp_file.name)
        temp_file_path = temp_file.name

    output_audio_path = temp_file_path
    if filename.endswith('.mp4'):
        output_audio_path = temp_file_path.replace('.mp4', '.wav')
        try:
            ffmpeg.input(temp_file_path).output(output_audio_path, format='wav', acodec='pcm_s16le', ar=16000).run(overwrite_output=True, quiet=True)
            logger.info(f'Extracted audio from MP4 to WAV: {output_audio_path}')
        except ffmpeg.Error as e:
            logger.error(f'FFmpeg error: {e.stderr.decode()}')
            os.unlink(temp_file_path)
            return jsonify({'error': f'FFmpeg error: {e.stderr.decode()}'}), 500
        finally:
            os.unlink(temp_file_path)

    mime_types = {
        '.mp3': 'audio/mp3',
        '.wav': 'audio/wav'
    }
    extension = os.path.splitext(output_audio_path)[1].lower()
    mimetype = mime_types.get(extension)
    logger.info(f'Detected MIME type: {mimetype}')

    try:
        logger.info('Attempting transcription with Deepgram...')
        with open(output_audio_path, 'rb') as audio_file:
            source = {
                'buffer': audio_file.read(),
                'mimetype': mimetype
            }

        response = deepgram.transcription.sync_prerecorded(
            source,
            {
                'model': 'nova-2-general',
                'diarize': True,
                'smart_format': True,
                'punctuate': True,
                'language': 'en'
            }
        )

        transcription = ''
        if (response.get('results') and 
            response['results'].get('channels') and 
            response['results']['channels'][0].get('alternatives') and 
            response['results']['channels'][0]['alternatives'][0].get('transcript')):
            transcription = response['results']['channels'][0]['alternatives'][0]['transcript']
            logger.info(f'Deepgram transcription successful: {transcription}')
        else:
            logger.error('Deepgram transcription returned no result')
            return jsonify({'error': 'Deepgram transcription returned no result'}), 500

        return jsonify({'text': transcription})

    except Exception as e:
        logger.error(f'Deepgram transcription failed: {str(e)}')
        return jsonify({'error': f'Deepgram transcription failed: {str(e)}'}), 500
    finally:
        if os.path.exists(output_audio_path):
            os.unlink(output_audio_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
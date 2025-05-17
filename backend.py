from flask import Flask, request, jsonify
from flask_cors import CORS
from deepgram import Deepgram
import os
import logging
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Deepgram API key from environment variable
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    logger.error("Deepgram API key not found in environment variables")
    raise ValueError("Deepgram API key not found in environment variables")

deepgram = Deepgram(DEEPGRAM_API_KEY)

# Supported file extensions
ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.mp4'}

# Check if file extension is allowed
def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    logger.info("Root endpoint accessed")
    return jsonify({"message": "Welcome to the OCR Backend! Use /transcribe to transcribe audio."})

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

    # Save the file temporarily to /tmp
    temp_file_path = f"/tmp/{filename}"
    try:
        file.save(temp_file_path)
        file_size = os.path.getsize(temp_file_path)
        logger.info(f'Uploaded file: {filename}, size: {file_size} bytes')

        # Determine MIME type
        extension = os.path.splitext(filename)[1].lower()
        mime_types = {
            '.mp3': 'audio/mp3',
            '.wav': 'audio/wav',
            '.mp4': 'video/mp4'
        }
        mimetype = mime_types.get(extension)
        logger.info(f'Detected MIME type: {mimetype}')

        # Read file content into memory
        with open(temp_file_path, 'rb') as f:
            audio_data = f.read()
        logger.info(f'Read audio data, size: {len(audio_data)} bytes')

        # Transcribe with Deepgram (synchronous call)
        logger.info('Attempting transcription with Deepgram...')
        source = {
            'buffer': audio_data,
            'mimetype': mimetype
        }

        response = deepgram.transcription.prerecorded(
            source,
            {
                'model': 'nova-2-general',
                'diarize': True,
                'smart_format': True,
                'punctuate': True,
                'language': 'en'  # Set to 'de' for German audio if needed
            }
        )

        # Extract transcription
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
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f'Cleaned up temporary file: {temp_file_path}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
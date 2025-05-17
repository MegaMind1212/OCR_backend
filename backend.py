from flask import Flask, request, jsonify
from flask_cors import CORS
from deepgram import Deepgram
import os
import logging
from werkzeug.utils import secure_filename
import asyncio

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Deepgram API key
DEEPGRAM_API_KEY = '32683910a62128aacbef2a8e7710d94c77a46101'  # Your Deepgram API key
deepgram = Deepgram(DEEPGRAM_API_KEY)

# Supported file extensions
ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.mp4'}

# Check if file extension is allowed
def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/transcribe', methods=['POST'])
async def transcribe():
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
    file.stream.seek(0)  # Reset file pointer to start

    # Determine MIME type
    extension = os.path.splitext(filename)[1].lower()
    mime_types = {
        '.mp3': 'audio/mp3',
        '.wav': 'audio/wav',
        '.mp4': 'video/mp4'
    }
    mimetype = mime_types.get(extension)
    logger.info(f'Detected MIME type: {mimetype}')

    try:
        logger.info('Attempting transcription with Deepgram...')
        # Read file content into memory
        source = {
            'buffer': file.read(),
            'mimetype': mimetype
        }

        # Transcribe with Deepgram
        response = await deepgram.transcription.prerecorded(
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
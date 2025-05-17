import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepgram import Deepgram
import ffmpeg

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get Deepgram API key from environment variable
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    logger.error("Deepgram API key not found in environment variables")

deepgram = Deepgram(DEEPGRAM_API_KEY)

@app.route('/')
def home():
    logger.info("Root endpoint accessed")
    return jsonify({"message": "Welcome to the OCR Backend! Use /transcribe to transcribe audio."})

@app.route('/transcribe', methods=['POST'])
async def transcribe():
    logger.info("Received request at /transcribe endpoint")
    try:
        # Check if audio file is in the request
        if 'audio' not in request.files:
            logger.error("No audio file provided in request")
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        logger.info(f"Received audio file: {audio_file.filename}")

        # Ensure the file is not empty
        if audio_file.filename == '':
            logger.error("Empty audio file provided")
            return jsonify({"error": "Empty audio file provided"}), 400

        # Save the uploaded file temporarily
        temp_file_path = f"/tmp/{audio_file.filename}"
        audio_file.save(temp_file_path)
        logger.info(f"Saved audio file to {temp_file_path}")

        # Convert audio to WAV format if necessary (Deepgram supports WAV)
        try:
            output_path = "/tmp/output.wav"
            ffmpeg.input(temp_file_path).output(output_path, acodec='pcm_s16le', ar='16000', ac=1).run(overwrite_output=True)
            logger.info(f"Converted audio to WAV: {output_path}")
        except Exception as e:
            logger.error(f"Error converting audio with ffmpeg: {str(e)}")
            return jsonify({"error": f"Error converting audio: {str(e)}"}), 500

        # Read the converted audio file
        with open(output_path, 'rb') as f:
            audio_data = f.read()
        logger.info(f"Read audio data, size: {len(audio_data)} bytes")

        # Prepare Deepgram request
        source = {'buffer': audio_data, 'mimetype': 'audio/wav'}
        options = {
            "punctuate": True,
            "model": "general",
            "language": "en",
            "tier": "enhanced"
        }

        # Transcribe audio using Deepgram
        logger.info("Sending audio to Deepgram for transcription")
        response = await deepgram.transcription.sync_prerecorded(source, options)
        logger.info("Received response from Deepgram")

        # Extract transcription
        transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
        if not transcript:
            logger.warning("No transcription available")
            return jsonify({"text": "No transcription available"})

        logger.info(f"Transcription successful: {transcript}")
        return jsonify({"text": transcript})

    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

    finally:
        # Clean up temporary files
        for temp_file in [temp_file_path, output_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info(f"Cleaned up temporary file: {temp_file}")

if __name__ == '__main__':
    app.run(debug=True)
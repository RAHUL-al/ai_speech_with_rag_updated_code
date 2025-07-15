# tasks.py
from celery_app import celery
from ai_speech_module import Topic

@celery.task
def process_chunk_celery(chunk_filename, essay_id):
    topic = Topic()

    transcribed_text = topic.speech_to_text(chunk_filename)
    emotion = topic._detect_emotion_sync(chunk_filename)
    fluency = topic._fluency_score_sync(transcribed_text)
    pronunciation = topic._pronunciation_score_sync(chunk_filename)
    vad_segments = topic._silvero_vad_sync(chunk_filename)

    return {
        "essay_id": essay_id,
        "chunk_filename": chunk_filename,
        "text": transcribed_text,
        "emotion": emotion,
        "fluency": fluency,
        "pronunciation": pronunciation,
        "vad_segments": vad_segments,
    }

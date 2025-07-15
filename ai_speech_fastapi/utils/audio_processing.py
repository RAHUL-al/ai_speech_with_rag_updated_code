from pydub import AudioSegment
import os

def save_audio_chunk(data: bytes, path: str):
    audio = AudioSegment(
        data=data,
        sample_width=2,
        frame_rate=16000,
        channels=1
    )
    audio.export(path, format="wav")

def merge_chunks(chunk_files, final_output):
    combined = AudioSegment.empty()
    for file in chunk_files:
        audio = AudioSegment.from_file(file, format="wav")
        combined += audio
    combined.export(final_output, format="wav")

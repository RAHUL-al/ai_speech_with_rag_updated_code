from dotenv import load_dotenv
import soundfile as sf
import requests
import os
import torchaudio
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from datetime import datetime
import asyncio
import threading
from gector.modeling import GECToR
from gector.predict import predict, load_verb_dict
from kokoro import KPipeline
from IPython.display import display, Audio
from pydub import AudioSegment
import re
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch
from langchain_ollama import OllamaLLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from firebase_admin import credentials, firestore, initialize_app
from firebase import db
import asyncio
import json
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger("essay")
logging.basicConfig(level=logging.INFO)


async def async_with_timeout(coro, timeout: int, name=""):
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"[TIMEOUT] Task '{name}' timed out after {timeout} seconds")
        return f"[TIMEOUT] Task '{name}' took too long"
    except Exception as e:
        logger.exception(f"[ERROR] Task '{name}' failed: {e}")
        return f"[ERROR] Task '{name}' failed"


load_dotenv()

class Topic:
    def __init__(self):
        self.total_fluency = 0.0
        self.total_pronunciation = 0.0
        self.total_emotion = {}
        self.chunk_count = 0
        self.model_name = OllamaLLM(model="mistral")

    def reset_realtime_stats(self):
        self.total_fluency = 0.0
        self.total_pronunciation = 0.0
        self.total_emotion = {}
        self.chunk_count = 0

    def update_realtime_stats(self, fluency, pronunciation, emotion):
        try:
            self.total_fluency += float(fluency)
        except:
            pass
        try:
            self.total_pronunciation += float(pronunciation)
        except:
            pass
        self.total_emotion[emotion] = self.total_emotion.get(emotion, 0) + 1
        self.chunk_count += 1

    def get_average_realtime_scores(self):
        if self.chunk_count == 0:
            return {"fluency": 0, "pronunciation": 0, "emotion": "unknown"}
        avg_fluency = round(self.total_fluency / self.chunk_count, 2)
        avg_pronunciation = round(self.total_pronunciation / self.chunk_count, 2)
        dominant_emotion = max(self.total_emotion.items(), key=lambda x: x[1])[0] if self.total_emotion else "unknown"
        return {
            "fluency": avg_fluency,
            "pronunciation": avg_pronunciation,
            "emotion": dominant_emotion
        }

    def topic_data_model_for_Qwen(self, username: str, prompt: str) -> str:
        try:
            model_name = OllamaLLM(model="mistral")
            response = model_name.invoke(prompt)
            threading.Thread(target=self.text_to_speech, args=(response, username), daemon=True).start()
            return response
        except Exception as e:
            print(f"[mistral API Error] {e}")
            return "mistral model failed to generate a response."



    async def detect_emotion(self, audio_path):
        return await asyncio.to_thread(self._detect_emotion_sync, audio_path)

    def _detect_emotion_sync(self, audio_path):
        model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        model = AutoModelForAudioClassification.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        speech_array, sampling_rate = torchaudio.load(audio_path)

        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            speech_array = resampler(speech_array)

        inputs = feature_extractor(speech_array.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_class_id = torch.argmax(logits).item()
        labels = model.config.id2label

        return labels.get(predicted_class_id, "unknown")

    async def fluency_scoring(self, text):
        return await asyncio.to_thread(self._fluency_score_sync, text)

    def _fluency_score_sync(self, text):
        model_name = "prithivida/parrot_fluency_model"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        inputs = tokenizer(text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            score = probs[0][1].item()

        return f"{score:.2f}"

    async def silvero_vad(self, audio_path):
        return await asyncio.to_thread(self._pronunciation_score_sync, audio_path)
    
    def _silvero_vad_sync(self,audio_path):
        model_path="silero_vad/silero-vad/src/silero_vad/data/silero_vad.jit"

        model = torch.jit.load(model_path)
        model.eval()

        wav, sr = torchaudio.load(audio_path)

        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)

        wav = wav / wav.abs().max()
        wav = wav.squeeze()

        SAMPLE_RATE = 16000
        WINDOW_SIZE = 512
        HOP_SIZE = 160
        THRESHOLD = 0.3

        speech_segments = []

        for i in range(0, len(wav) - WINDOW_SIZE + 1, HOP_SIZE):
            chunk = wav[i:i + WINDOW_SIZE].unsqueeze(0)
            
            with torch.no_grad():
                try:
                    prob = model(chunk, SAMPLE_RATE).item()
                except Exception as e:
                    print(f"Skipping chunk due to error: {e}")
                    continue

            label = "Speech" if prob > THRESHOLD else "Silence"
            time = i / SAMPLE_RATE
            speech_segments.append((time, label, prob))

        return [seg for seg in speech_segments if seg[1] == "Speech"]

    async def pronunciation_scoring(self, audio_path):
        return await asyncio.to_thread(self._pronunciation_score_sync, audio_path)

    

    def _pronunciation_score_sync(self, audio_path):
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "hafidikhsan/wav2vec2-large-xlsr-53-english-pronunciation-evaluation-aod-real"
        )

        speech_array, sampling_rate = torchaudio.load(audio_path)

        if speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)

        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            speech_array = resampler(speech_array)

        inputs = feature_extractor(speech_array.squeeze(), sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        scores = torch.softmax(logits, dim=-1)
        return f"{scores[0][1].item():.2f}"
    

    async def grammar_checking(self, spoken_text, original_text):
        return await asyncio.to_thread(self._grammar_check_sync, spoken_text, original_text)

    def _grammar_check_sync(self, spoken_text, original_text):
        model_id = "gotutiyan/gector-roberta-base-5k"
        model = GECToR.from_pretrained(model_id)

        if torch.cuda.is_available():
            model = model.cuda()

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        encode, decode = load_verb_dict('data/verb-form-vocab.txt')

        srcs = [spoken_text.strip()]

        corrected = predict(
            model, tokenizer, srcs, encode, decode,
            keep_confidence=0.0,
            min_error_prob=0.0,
            n_iteration=5,
            batch_size=1,
        )

        corrected_sent = corrected[0] if corrected else spoken_text
        grammar_score = self._compare_sentences(original_text, corrected_sent)

        return grammar_score

    def _compare_sentences(self, reference, corrected):
        ref_words = reference.strip().lower().split()
        cor_words = corrected.strip().lower().split()
        matches = sum(1 for r, c in zip(ref_words, cor_words) if r == c)
        score = (matches / max(len(ref_words), 1)) * 10
        return round(score, 2)
    


    async def overall_scoring_by_id(self, essay_id: str):
        try:
            essay_doc = db.collection("essays").document(essay_id).get()
            if not essay_doc.exists:
                return {"error": f"No essay found with id {essay_id}"}

            essay_data = essay_doc.to_dict()
            original_text = essay_data.get("content", "")
            username = essay_data.get("username")
            if not username:
                return {"error": "Username missing in essay data"}

            date_str = datetime.now().strftime("%Y-%m-%d")
            audio_path = os.path.join("audio_folder", username, date_str, f"{username}_output.wav")

            if not os.path.exists(audio_path):
                return {"error": f"Audio file not found at {audio_path}"}

            spoken_text = self.speech_to_text(audio_path)

            # Run each task with timeout to isolate failures
            pronunciation_task = async_with_timeout(self.pronunciation_scoring(audio_path), 20, name="pronunciation")
            fluency_task = async_with_timeout(self.fluency_scoring(spoken_text), 20, name="fluency")
            emotion_task = async_with_timeout(self.detect_emotion(audio_path), 20, name="emotion")
            grammar_task = async_with_timeout(self.grammar_checking(spoken_text, original_text), 25, name="grammar")

            pronunciation, fluency, emotion, grammar = await asyncio.gather(
                pronunciation_task, fluency_task, emotion_task, grammar_task
            )

            prompt = f"""
                You are an AI English teacher evaluating a student's spoken response based on their performance. You will be provided with the student's spoken text and a reference essay.

                Here are the available scores (only use the ones that are valid and available, ignore or skip any missing or erroneous ones like '[ERROR]' or None):
                - Pronunciation: {pronunciation}
                - Grammar: {grammar}
                - Fluency: {fluency}
                - Emotion: {emotion}

                Reference Essay:
                \"\"\"{original_text}\"\"\"

                Spoken Text:
                \"\"\"{spoken_text}\"\"\"

                Based on the above, return an honest but encouraging evaluation in **JSON format** with the following keys:
                - "understanding": Describe how well the spoken text reflects understanding of the reference essay.
                - "topic_grip": Comment on how well the speaker stayed on topic and conveyed key points.
                - "suggestions": A list of 3 teacher-style suggestions to improve the student's speaking and comprehension.

                Important:
                - Do not include invalid, missing, or placeholder values in the response.
                - Do not talk about improving the AI itself; focus on guiding a human student.
                - Keep the tone supportive and constructive, like a real teacher giving oral feedback.
            """


            summary_response = self.topic_data_model_for_Qwen(username, prompt)

            try:
                result = json.loads(str(summary_response))
            except Exception as e:
                logger.warning(f"Could not parse JSON: {e}")
                result = {"raw_response": str(summary_response)}

            result.update({
                "pronunciation": pronunciation,
                "grammar": grammar,
                "fluency": fluency,
                "emotion": emotion
            })

            return result

        except Exception as e:
            logger.exception(f"[ERROR] overall_scoring_by_id failed: {e}")
            return {"error": "Internal Server Error"}



    def speech_to_text(self, audio_path: str) -> str:
        API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
        headers = {
            "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}",
            "Content-Type": "audio/wav"
        }

        if not os.path.exists(audio_path):
            print(f"[Error] Audio file not found: {audio_path}")
            return ""

        try:
            with open(audio_path, "rb") as f:
                data = f.read()

            response = requests.post(API_URL, headers=headers, data=data)
            response.raise_for_status()

            result = response.json()
            text = result.get("text", "").strip()
            print(f"Transcribed [{os.path.basename(audio_path)}]: {text}")
            return text

        except Exception as e:
            print(f"[Error] Failed to transcribe {audio_path}: {e}")
            return ""
        
    def text_to_speech(self, text_data, username):
        output_dir = os.path.join("text_to_speech_audio_folder", username)
        os.makedirs(output_dir, exist_ok=True)

        pipeline = KPipeline(lang_code='a')
        text = text_data
        max_chars = 400
        sentences = re.split(r'(?<=[.?!])\s+', text.strip())

        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())

        generated_files = []
        for i, chunk in enumerate(chunks):
            print(f"\n chunk {i+1}: {chunk}\n")
            generator = pipeline(chunk, voice='af_heart')
            for j, (gs, ps, audio) in enumerate(generator):
                filename = os.path.join(output_dir, f'chunk_{i+1}_part_{j+1}.wav')
                display(Audio(data=audio, rate=24000, autoplay=False))
                sf.write(filename, audio, 24000)
                generated_files.append(filename)

        combined = AudioSegment.empty()
        for file in generated_files:
            audio = AudioSegment.from_wav(file)
            combined += audio

        output_path = os.path.join(output_dir, f"{username}_output.wav")
        combined.export(output_path, format="wav")
        print(f"Exported final audio to {output_path}")

        for file in generated_files:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting file: {file} - {e}")

        return output_path
"""
Speech-to-Text transcription module for ClipVid.
Supports modular providers: Groq (Whisper-large-v3), OpenAI (Whisper-1), Gemini, and Offline Demo.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
import io
from pathlib import Path

from modules.config import GROQ_API_KEY, MODELS
from modules.utils import format_seconds

class TranscriptSegment(BaseModel):
    """Represents a single timestamped section of transcribed speech."""
    start: float
    end: float
    text: str

    @property
    def formatted_start(self) -> str:
        return format_seconds(self.start)

    @property
    def formatted_end(self) -> str:
        return format_seconds(self.end)


class TranscriptResult(BaseModel):
    """Complete speech-to-text output."""
    full_text: str
    segments: List[TranscriptSegment]
    duration_seconds: float
    language: str = "en"
    word_count: int = 0

    def model_post_init(self, __context):
        if not self.word_count and self.full_text:
            self.word_count = len(self.full_text.split())


class TranscriptionError(Exception):
    """Base exception for transcription failures."""
    pass


class BaseTranscriber(ABC):
    """Abstract interface for speech-to-text providers."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptResult:
        """Transcribe an audio file into text with timestamp segments."""
        pass


class GroqWhisperTranscriber(BaseTranscriber):
    """
    Fast, cloud-based Whisper-large-v3 transcription via Groq Cloud API.
    Provides segment timestamps and high accuracy.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GROQ_API_KEY
        if not self.api_key:
            raise TranscriptionError(
                "Groq API Key is required for Groq Whisper transcription. "
                "Please set GROQ_API_KEY in .env or provide it in the sidebar."
            )

    def transcribe(self, audio_path: str) -> TranscriptResult:
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            
            with open(audio_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), file.read()),
                    model=MODELS["groq"]["stt"],
                    response_format="verbose_json",
                    temperature=0.0
                )
            
            raw_text = transcription.text.strip()
            if not raw_text:
                raise TranscriptionError(
                    "Speech recognition completed, but no spoken words were detected in the audio."
                )

            segments: List[TranscriptSegment] = []
            if hasattr(transcription, "segments") and transcription.segments:
                for seg in transcription.segments:
                    # seg can be dict or object
                    start = getattr(seg, 'start', None) if not isinstance(seg, dict) else seg.get('start')
                    end = getattr(seg, 'end', None) if not isinstance(seg, dict) else seg.get('end')
                    text = getattr(seg, 'text', '') if not isinstance(seg, dict) else seg.get('text', '')
                    segments.append(TranscriptSegment(
                        start=float(start or 0.0),
                        end=float(end or 0.0),
                        text=str(text).strip()
                    ))
            else:
                segments.append(TranscriptSegment(
                    start=0.0,
                    end=float(getattr(transcription, "duration", 0.0) or 0.0),
                    text=raw_text
                ))

            duration = float(getattr(transcription, "duration", 0.0) or 0.0)
            language = getattr(transcription, "language", "en") or "en"

            return TranscriptResult(
                full_text=raw_text,
                segments=segments,
                duration_seconds=duration,
                language=language,
                word_count=len(raw_text.split())
            )

        except TranscriptionError:
            raise
        except Exception as e:
            err_msg = str(e)
            if "rate_limit_exceeded" in err_msg.lower():
                raise TranscriptionError("Groq API rate limit exceeded. Please wait a moment or switch provider.")
            elif "invalid_api_key" in err_msg.lower() or "authentication" in err_msg.lower():
                raise TranscriptionError("Invalid Groq API key. Please check your credentials in the sidebar or .env.")
            raise TranscriptionError(f"Groq Whisper transcription failed: {err_msg}")


class OfflineMockTranscriber(BaseTranscriber):
    """
    Demonstration and offline fallback transcriber.
    Ensures tests and offline demonstrations run reliably without network dependencies.
    """
    def __init__(self, sample_topic: str = "AI Video Summarization"):
        self.sample_topic = sample_topic

    def transcribe(self, audio_path: str) -> TranscriptResult:
        import time
        time.sleep(0.5)
        
        segments = [
            TranscriptSegment(
                start=0.0,
                end=8.5,
                text="Welcome everyone. Today we are discussing AI-based video summarization using Generative AI."
            ),
            TranscriptSegment(
                start=8.5,
                end=21.2,
                text="The exponential growth of online video content on platforms like YouTube, Coursera, and Zoom has created an urgent need for automated content extraction."
            ),
            TranscriptSegment(
                start=21.2,
                end=37.0,
                text="Our system extracts audio tracks directly from video containers using FFmpeg and processes speech into text using Whisper speech recognition models."
            ),
            TranscriptSegment(
                start=37.0,
                end=54.8,
                text="For lengthy recordings, we apply a hierarchical map-reduce chunking technique to bypass LLM context window restrictions and prevent hallucinations."
            ),
            TranscriptSegment(
                start=54.8,
                end=72.3,
                text="Finally, the Generative AI model synthesizes executive summaries, key bullet points, thematic topics, and timestamp markers to save students and professionals hours of viewing time."
            ),
            TranscriptSegment(
                start=72.3,
                end=84.0,
                text="Thank you for listening, and we now open the presentation for questions and viva evaluation."
            )
        ]

        full_text = " ".join(s.text for s in segments)
        return TranscriptResult(
            full_text=full_text,
            segments=segments,
            duration_seconds=84.0,
            language="en",
            word_count=len(full_text.split())
        )


def get_transcriber(provider_name: str = "groq", api_key: Optional[str] = None) -> BaseTranscriber:
    """Factory method to instantiate the speech-to-text provider."""
    provider = provider_name.lower().strip()
    if provider == "offline" or provider == "mock":
        return OfflineMockTranscriber()
    return GroqWhisperTranscriber(api_key=api_key)

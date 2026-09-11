"""
Transcript chunking module for ClipVid.
Implements sentence-aware and timestamp-preserving chunking for long transcripts.
"""
from typing import List, Optional
from pydantic import BaseModel
from modules.transcription import TranscriptResult, TranscriptSegment
from modules.config import DEFAULT_CHUNK_SIZE_WORDS, DEFAULT_CHUNK_OVERLAP_WORDS, LONG_TRANSCRIPT_THRESHOLD_WORDS
from modules.utils import format_seconds

class TranscriptChunk(BaseModel):
    """A segment of transcript partitioned for LLM processing."""
    chunk_index: int
    text: str
    start_time: float
    end_time: float
    word_count: int

    @property
    def time_range(self) -> str:
        return f"{format_seconds(self.start_time)} - {format_seconds(self.end_time)}"


def should_chunk(transcript: TranscriptResult, threshold_words: int = LONG_TRANSCRIPT_THRESHOLD_WORDS) -> bool:
    """Returns True if the transcript length warrants map-reduce chunking."""
    return transcript.word_count > threshold_words


def chunk_transcript(
    transcript: TranscriptResult,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS
) -> List[TranscriptChunk]:
    """
    Partitions a transcript into overlapping chunks.
    Preserves timestamp metadata when segment data is present.
    """
    if not transcript.segments:
        # Fallback to pure word-based chunking if no segments exist
        words = transcript.full_text.split()
        if not words:
            return []
        
        chunks: List[TranscriptChunk] = []
        start_idx = 0
        chunk_idx = 0
        step = max(1, chunk_size_words - overlap_words)

        while start_idx < len(words):
            end_idx = min(len(words), start_idx + chunk_size_words)
            chunk_words = words[start_idx:end_idx]
            chunks.append(TranscriptChunk(
                chunk_index=chunk_idx,
                text=" ".join(chunk_words),
                start_time=0.0,
                end_time=transcript.duration_seconds,
                word_count=len(chunk_words)
            ))
            chunk_idx += 1
            start_idx += step
            if end_idx >= len(words):
                break

        return chunks

    # Segment-aware chunking: group segments until chunk_size_words is reached
    chunks: List[TranscriptChunk] = []
    current_segments: List[TranscriptSegment] = []
    current_words = 0
    chunk_idx = 0

    for seg in transcript.segments:
        seg_words = len(seg.text.split())
        
        # If adding this segment exceeds chunk_size and we already have words
        if current_words + seg_words > chunk_size_words and current_segments:
            chunk_text = " ".join(s.text for s in current_segments)
            chunks.append(TranscriptChunk(
                chunk_index=chunk_idx,
                text=chunk_text,
                start_time=current_segments[0].start,
                end_time=current_segments[-1].end,
                word_count=len(chunk_text.split())
            ))
            chunk_idx += 1

            # Retain overlap from end of current_segments if desired
            if overlap_words > 0 and len(current_segments) > 1:
                overlap_segs = []
                acc = 0
                for s in reversed(current_segments):
                    acc += len(s.text.split())
                    overlap_segs.insert(0, s)
                    if acc >= overlap_words:
                        break
                current_segments = overlap_segs
                current_words = sum(len(s.text.split()) for s in current_segments)
            else:
                current_segments = []
                current_words = 0

        current_segments.append(seg)
        current_words += seg_words

    # Append remaining segments
    if current_segments:
        chunk_text = " ".join(s.text for s in current_segments)
        chunks.append(TranscriptChunk(
            chunk_index=chunk_idx,
            text=chunk_text,
            start_time=current_segments[0].start,
            end_time=current_segments[-1].end,
            word_count=len(chunk_text.split())
        ))

    return chunks

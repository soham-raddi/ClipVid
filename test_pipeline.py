"""
Comprehensive Test Suite for ClipVid
Tests:
1. Video validation and format checking
2. Audio extraction and error handling (silent video, corrupt container)
3. Speech-to-text data model and offline transcription
4. Transcript chunking & overlap logic
5. Summarizer output schema validation
6. Academic evaluation metrics computation
"""
import os
import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from modules.config import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB
from modules.utils import (
    validate_video_file, format_seconds, format_file_size,
    generate_full_report
)
from modules.transcription import (
    TranscriptResult, TranscriptSegment, get_transcriber,
    OfflineMockTranscriber
)
from modules.chunking import chunk_transcript, should_chunk, TranscriptChunk
from modules.summarizer import (
    VideoSummaryOutput, TimestampSection, get_summarizer,
    OfflineMockSummarizer
)
from modules.evaluation import evaluate_summary, calculate_flesch_reading_ease
from modules.video_processor import (
    VideoProcessor, NoAudioTrackError, CorruptedVideoError
)
from sample_videos.create_samples import generate_samples

class TestClipVidPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate sample videos for testing if they don't exist yet
        generate_samples()
        cls.sample_dir = BASE_DIR / "sample_videos"
        cls.processor = VideoProcessor()

    def test_01_validation_logic(self):
        """Test file extension and size validation."""
        # Valid MP4
        valid, err = validate_video_file("lecture.mp4", 10 * 1024 * 1024)
        self.assertTrue(valid)
        self.assertIsNone(err)

        # Valid MKV
        valid, err = validate_video_file("recording.mkv", 50 * 1024 * 1024)
        self.assertTrue(valid)

        # Invalid format (.xyz)
        valid, err = validate_video_file("document.xyz", 1024)
        self.assertFalse(valid)
        self.assertIn("Unsupported video format", err)

        # Empty file
        valid, err = validate_video_file("empty.mp4", 0)
        self.assertFalse(valid)
        self.assertIn("empty", err.lower())

        # Oversized file
        oversize = (MAX_FILE_SIZE_MB + 10) * 1024 * 1024
        valid, err = validate_video_file("giant.mp4", oversize)
        self.assertFalse(valid)
        self.assertIn("exceeds the maximum", err.lower())

    def test_02_formatting_utilities(self):
        """Test time and size formatting."""
        self.assertEqual(format_seconds(0), "00:00")
        self.assertEqual(format_seconds(65), "01:05")
        self.assertEqual(format_seconds(3665), "01:01:05")

        self.assertEqual(format_file_size(500), "500.0 B")
        self.assertEqual(format_file_size(1024 * 1024), "1.0 MB")

    def test_03_video_inspection_and_audio_extraction(self):
        """Test inspecting valid video and extracting audio."""
        edu_video = self.sample_dir / "sample_education.mp4"
        meta = self.processor.inspect_video(edu_video)
        self.assertTrue(meta["has_audio"])
        self.assertGreater(meta["duration_seconds"], 0)

        # Extract audio
        audio_path = self.processor.extract_audio(edu_video, output_format="mp3")
        self.assertTrue(os.path.exists(audio_path))
        self.assertGreater(os.path.getsize(audio_path), 0)
        # Cleanup
        os.remove(audio_path)

    def test_04_silent_video_error_handling(self):
        """Test that a video without an audio track raises NoAudioTrackError."""
        silent_video = self.sample_dir / "sample_silent.mp4"
        with self.assertRaises(NoAudioTrackError):
            self.processor.extract_audio(silent_video)

    def test_05_corrupted_video_error_handling(self):
        """Test that corrupted video raises CorruptedVideoError."""
        corrupt_video = self.sample_dir / "sample_corrupted.mp4"
        with self.assertRaises(CorruptedVideoError):
            self.processor.inspect_video(corrupt_video)

    def test_06_transcription_offline_provider(self):
        """Test offline mock transcription generates valid segments."""
        transcriber = get_transcriber("offline")
        res = transcriber.transcribe("dummy_path.mp3")
        self.assertIsInstance(res, TranscriptResult)
        self.assertGreater(len(res.segments), 0)
        self.assertIn("AI-based video summarization", res.full_text)
        self.assertGreater(res.word_count, 10)

    def test_07_chunking_logic(self):
        """Test sentence and segment-preserving chunking."""
        segments = [
            TranscriptSegment(start=0.0, end=10.0, text="Word " * 50),
            TranscriptSegment(start=10.0, end=20.0, text="Word " * 50),
            TranscriptSegment(start=20.0, end=30.0, text="Word " * 50),
            TranscriptSegment(start=30.0, end=40.0, text="Word " * 50),
        ]
        t_res = TranscriptResult(
            full_text=" ".join(s.text for s in segments),
            segments=segments,
            duration_seconds=40.0,
            word_count=200
        )
        
        chunks = chunk_transcript(t_res, chunk_size_words=80, overlap_words=20)
        self.assertGreater(len(chunks), 1)
        for chk in chunks:
            self.assertIsInstance(chk, TranscriptChunk)
            self.assertGreater(chk.word_count, 0)

    def test_08_summarizer_offline(self):
        """Test summarization generates complete structured summary schema."""
        transcriber = get_transcriber("offline")
        transcript = transcriber.transcribe("dummy.mp3")
        
        summarizer = get_summarizer("offline")
        summary_out: VideoSummaryOutput = summarizer.generate_summary(transcript)

        self.assertTrue(summary_out.suggested_title)
        self.assertTrue(summary_out.short_summary)
        self.assertTrue(summary_out.detailed_summary)
        self.assertGreater(len(summary_out.key_points), 0)
        self.assertGreater(len(summary_out.topics), 0)
        self.assertGreater(len(summary_out.important_timestamps), 0)

    def test_09_academic_evaluation_metrics(self):
        """Test evaluation metrics calculations."""
        t_text = (
            "Artificial intelligence and natural language processing enable automated systems "
            "to summarize long documents, lectures, and webinars. By extracting the key spoken "
            "words and synthesizing them into structured reports, students and researchers can "
            "save significant study time. In traditional classroom environments, reviewing an "
            "entire recorded lecture can be challenging and time-consuming. Automated video "
            "summarization provides immediate access to core concepts, key topics, and timestamped "
            "sections, vastly accelerating comprehension and retention across higher education."
        )
        s_data = {
            "short_summary": "AI enables automated video summarization to save study time.",
            "detailed_summary": (
                "Artificial intelligence and natural language processing summarize webinars "
                "into structured reports, allowing researchers to save significant study time."
            ),
            "key_points": [
                "Automates speech extraction and summarization.",
                "Saves study time for researchers."
            ]
        }
        metrics = evaluate_summary(t_text, s_data)
        self.assertIn("compression_ratio", metrics)
        self.assertIn("lexical_coverage", metrics)
        self.assertIn("reading_ease", metrics)
        self.assertIn("time_saved", metrics)
        self.assertGreater(metrics["compression_ratio"], 0.0)

        # Flesch reading ease
        score, label = calculate_flesch_reading_ease("This is a simple sentence. It is easy to read.")
        self.assertGreater(score, 60.0)

    def test_10_full_report_generation(self):
        """Test text/markdown report generator."""
        s_data = {
            "suggested_title": "Test Title",
            "short_summary": "Short summary",
            "detailed_summary": "Detailed summary",
            "key_points": ["Point A", "Point B"],
            "topics": ["AI", "Video"],
            "important_timestamps": [{"timestamp": "00:00", "description": "Intro"}]
        }
        report = generate_full_report(
            summary_data=s_data,
            transcript_text="Sample transcript",
            video_info={"filename": "test.mp4", "duration_seconds": 10, "size_bytes": 1024}
        )
        self.assertIn("CLIPVID: AI-BASED VIDEO SUMMARIZATION REPORT", report)
        self.assertIn("Test Title", report)
        self.assertIn("1. SHORT SUMMARY", report)

    def test_11_youtube_url_validation(self):
        """Test YouTube URL detection and ID parsing."""
        from modules.youtube import is_valid_youtube_url, extract_video_id
        
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "http://youtube.com/watch?v=dQw4w9WgXcQ"
        ]
        for url in valid_urls:
            self.assertTrue(is_valid_youtube_url(url))
            self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")

        invalid_urls = [
            "https://vimeo.com/123456",
            "https://example.com/video.mp4",
            "not_a_url",
            ""
        ]
        for url in invalid_urls:
            self.assertFalse(is_valid_youtube_url(url))

if __name__ == "__main__":
    unittest.main()

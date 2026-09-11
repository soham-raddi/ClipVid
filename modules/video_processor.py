"""
Video and audio processing module for ClipVid.
Handles video inspection, audio track extraction, and format conversion.
"""
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import imageio_ffmpeg
from moviepy.editor import VideoFileClip

from modules.config import TEMP_DIR, AUDIO_SAMPLE_RATE
from modules.utils import format_seconds, format_file_size

class VideoProcessingError(Exception):
    """Base exception for video processing errors."""
    pass

class NoAudioTrackError(VideoProcessingError):
    """Raised when the uploaded video does not contain any audio stream."""
    pass

class CorruptedVideoError(VideoProcessingError):
    """Raised when the video file cannot be decoded or is corrupted."""
    pass


class VideoProcessor:
    """Handles video validation, metadata inspection, and audio extraction."""

    def __init__(self, temp_dir: Path = TEMP_DIR):
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    def inspect_video(self, video_path: str | Path) -> Dict[str, Any]:
        """
        Inspect video file and return metadata including duration, resolution,
        and audio track presence.
        """
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        file_size = os.path.getsize(video_path)
        if file_size == 0:
            raise CorruptedVideoError("The video file is empty (0 bytes).")

        try:
            with VideoFileClip(video_path) as clip:
                duration = clip.duration or 0.0
                has_audio = clip.audio is not None
                w, h = clip.size if clip.size else (0, 0)
                fps = clip.fps or 0.0

            return {
                "filename": os.path.basename(video_path),
                "filepath": video_path,
                "size_bytes": file_size,
                "size_formatted": format_file_size(file_size),
                "duration_seconds": duration,
                "duration_formatted": format_seconds(duration),
                "resolution": f"{w}x{h}",
                "fps": round(fps, 2),
                "has_audio": has_audio
            }
        except Exception as e:
            # Fallback to probe using ffmpeg directly if moviepy raises an exception
            try:
                cmd = [self.ffmpeg_path, "-i", video_path]
                res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, errors="replace")
                output = res.stderr
                has_audio = "Audio:" in output
                
                # Check for critical errors
                if "Invalid data found when processing input" in output:
                    raise CorruptedVideoError(f"Corrupted or invalid video file: {str(e)}")
                    
                return {
                    "filename": os.path.basename(video_path),
                    "filepath": video_path,
                    "size_bytes": file_size,
                    "size_formatted": format_file_size(file_size),
                    "duration_seconds": 0.0,
                    "duration_formatted": "Unknown",
                    "resolution": "Unknown",
                    "fps": 0.0,
                    "has_audio": has_audio
                }
            except CorruptedVideoError:
                raise
            except Exception:
                raise CorruptedVideoError(f"Could not read video file: {str(e)}")

    def extract_audio(
        self,
        video_path: str | Path,
        output_format: str = "mp3"
    ) -> str:
        """
        Extract the audio track from the video and save to a temporary audio file.
        Optimized for speech-to-text models (16kHz mono).
        
        Returns:
            str: Path to the generated audio file.
        """
        video_path = str(video_path)
        meta = self.inspect_video(video_path)
        
        if not meta["has_audio"]:
            raise NoAudioTrackError(
                "The selected video does not contain an audio track. "
                "Speech recognition requires an audible audio stream."
            )

        video_stem = Path(video_path).stem
        output_filename = f"{video_stem}_extracted.{output_format.lower()}"
        output_path = str(self.temp_dir / output_filename)

        # Remove existing file if present
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass

        # Use imageio-ffmpeg for fast and reliable extraction to 16kHz mono
        # -y : overwrite
        # -vn : disable video recording
        # -ac 1 : mono channel (best for whisper/stt)
        # -ar 16000 : 16000 Hz sample rate
        try:
            if output_format.lower() == "mp3":
                codec_args = ["-acodec", "libmp3lame", "-b:a", "128k"]
            else:
                codec_args = ["-acodec", "pcm_s16le"]  # WAV

            cmd = [
                self.ffmpeg_path,
                "-y",
                "-i", video_path,
                "-vn",
                *codec_args,
                "-ar", str(AUDIO_SAMPLE_RATE),
                "-ac", "1",
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace"
            )

            if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                # Fallback to MoviePy audio extraction if ffmpeg subprocess had issues
                with VideoFileClip(video_path) as clip:
                    if clip.audio is None:
                        raise NoAudioTrackError("Video has no audio track.")
                    clip.audio.write_audiofile(
                        output_path,
                        fps=AUDIO_SAMPLE_RATE,
                        nbytes=2,
                        codec='pcm_s16le' if output_format.lower() == 'wav' else 'libmp3lame',
                        verbose=False,
                        logger=None
                    )

            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise VideoProcessingError("Audio extraction failed; output audio file is empty.")

            return output_path

        except NoAudioTrackError:
            raise
        except Exception as e:
            raise VideoProcessingError(f"Failed to extract audio track: {str(e)}")

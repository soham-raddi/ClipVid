"""
YouTube video processing module for ClipVid.
Handles URL validation, metadata fetching, and audio track downloading using yt-dlp.
"""
import re
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import imageio_ffmpeg
import yt_dlp

from modules.config import TEMP_DIR, AUDIO_SAMPLE_RATE
from modules.utils import format_seconds, format_file_size

YOUTUBE_REGEX = re.compile(
    r'^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/embed/)([\w-]{11})'
)

class YouTubeError(Exception):
    """Base exception for YouTube operations."""
    pass

class InvalidYouTubeURLError(YouTubeError):
    """Raised when URL is not a recognized YouTube format."""
    pass

class YouTubeDownloadError(YouTubeError):
    """Raised when video stream or audio track download fails."""
    pass


def is_valid_youtube_url(url: str) -> bool:
    """Validate whether a string matches standard YouTube URL patterns."""
    if not url or not isinstance(url, str):
        return False
    return bool(YOUTUBE_REGEX.search(url.strip()))


def extract_video_id(url: str) -> Optional[str]:
    """Extract the 11-character YouTube video ID."""
    match = YOUTUBE_REGEX.search(url.strip())
    if match:
        return match.group(4)
    return None


def get_youtube_metadata(url: str) -> Dict[str, Any]:
    """
    Fetch YouTube video title, duration, author, and thumbnail without downloading media.
    """
    if not is_valid_youtube_url(url):
        raise InvalidYouTubeURLError("The provided link is not a valid YouTube video URL.")

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url.strip(), download=False)
            duration = info.get('duration', 0) or 0
            return {
                "title": info.get('title', 'YouTube Video'),
                "channel": info.get('uploader', info.get('channel', 'Unknown Channel')),
                "duration_seconds": duration,
                "duration_formatted": format_seconds(duration),
                "thumbnail_url": info.get('thumbnail', ''),
                "view_count": info.get('view_count', 0),
                "webpage_url": info.get('webpage_url', url.strip()),
                "video_id": info.get('id', extract_video_id(url))
            }
    except Exception as e:
        raise YouTubeDownloadError(f"Could not retrieve YouTube video metadata: {str(e)}")


def download_youtube_audio(url: str) -> str:
    """
    Download audio track from a YouTube video and convert to 16kHz mono MP3.
    Returns path to the downloaded audio file in TEMP_DIR.
    """
    if not is_valid_youtube_url(url):
        raise InvalidYouTubeURLError("Invalid YouTube URL provided.")

    video_id = extract_video_id(url) or f"yt_{int(time.time())}"
    output_stem = f"yt_{video_id}_{int(time.time())}"
    output_template = str(TEMP_DIR / f"{output_stem}.%(ext)s")
    final_audio_path = str(TEMP_DIR / f"{output_stem}.mp3")

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'ffmpeg_location': ffmpeg_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '64',
        }],
        'postprocessor_args': [
            '-ar', str(AUDIO_SAMPLE_RATE),
            '-b:a', '64k',
            '-ac', '1'  # Convert to mono channel for speech-to-text
        ],
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url.strip()])

        if os.path.exists(final_audio_path) and os.path.getsize(final_audio_path) > 0:
            return final_audio_path
        
        # Check if downloaded under alternate extension
        for f in TEMP_DIR.glob(f"{output_stem}.*"):
            if f.is_file() and f.stat().st_size > 0:
                return str(f)

        raise YouTubeDownloadError("Audio extraction completed but target audio file was not found.")

    except Exception as e:
        raise YouTubeDownloadError(f"Failed to download audio from YouTube: {str(e)}")

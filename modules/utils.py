"""
Utility helper functions for ClipVid.
"""
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional
from modules.config import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB, TEMP_DIR

def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS string."""
    if seconds is None or seconds < 0:
        return "00:00"
    seconds = int(round(seconds))
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

def format_file_size(bytes_size: int) -> str:
    """Format bytes into readable string (KB, MB, GB)."""
    if bytes_size is None:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def validate_video_file(file_name: str, file_size_bytes: int) -> tuple[bool, Optional[str]]:
    """
    Validate uploaded video file extension and size.
    Returns (is_valid, error_message).
    """
    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, (
            f"Unsupported video format '{ext}'. "
            f"Please upload one of: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        return False, (
            f"File size ({format_file_size(file_size_bytes)}) exceeds the maximum "
            f"allowed limit of {MAX_FILE_SIZE_MB} MB."
        )
    
    if file_size_bytes == 0:
        return False, "The uploaded file is empty (0 bytes)."
        
    return True, None

def safe_delete_file(file_path: Optional[str | Path]) -> None:
    """Safely delete a file if it exists, without raising an exception."""
    if not file_path:
        return
    try:
        p = Path(file_path)
        if p.exists() and p.is_file():
            p.unlink(missing_ok=True)
    except Exception:
        pass

def cleanup_old_temp_files(max_age_seconds: int = 3600) -> None:
    """Clean up temporary files older than max_age_seconds."""
    now = time.time()
    if not TEMP_DIR.exists():
        return
    for item in TEMP_DIR.iterdir():
        try:
            if item.is_file() and (now - item.stat().st_mtime > max_age_seconds):
                item.unlink(missing_ok=True)
            elif item.is_dir() and (now - item.stat().st_mtime > max_age_seconds):
                shutil.rmtree(item, ignore_errors=True)
        except Exception:
            pass

def generate_full_report(
    summary_data: Dict[str, Any],
    transcript_text: str,
    video_info: Dict[str, Any],
    eval_metrics: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate an academic-ready Markdown / Text report containing all metadata,
    summaries, key points, topics, timestamps, and evaluation metrics.
    """
    title = summary_data.get("suggested_title", "Video Summary Report")
    short_summary = summary_data.get("short_summary", "")
    detailed_summary = summary_data.get("detailed_summary", "")
    key_points = summary_data.get("key_points", [])
    topics = summary_data.get("topics", [])
    timestamps = summary_data.get("important_timestamps", [])

    lines = [
        "=" * 70,
        f"CLIPVID: AI-BASED VIDEO SUMMARIZATION REPORT",
        "=" * 70,
        f"Suggested Title : {title}",
        f"Original Video  : {video_info.get('filename', 'Unknown')}",
        f"Video Duration  : {format_seconds(video_info.get('duration_seconds', 0))}",
        f"Video Size      : {format_file_size(video_info.get('size_bytes', 0))}",
        f"Generated At    : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "1. SHORT SUMMARY",
        "-" * 40,
        short_summary.strip(),
        "",
        "2. DETAILED SUMMARY",
        "-" * 40,
        detailed_summary.strip(),
        "",
        "3. KEY TAKEAWAYS & POINTS",
        "-" * 40,
    ]
    
    for i, pt in enumerate(key_points, 1):
        lines.append(f"  {i}. {pt}")
    
    lines.extend([
        "",
        "4. TOPICS IDENTIFIED",
        "-" * 40,
        ", ".join(topics) if topics else "None identified",
        "",
        "5. IMPORTANT VIDEO SECTIONS & TIMESTAMPS",
        "-" * 40,
    ])

    if timestamps:
        for item in timestamps:
            ts = item.get("timestamp", "--:--")
            desc = item.get("description", "")
            lines.append(f"  [{ts}] {desc}")
    else:
        lines.append("  (No discrete timestamp sections available)")

    if eval_metrics:
        lines.extend([
            "",
            "6. ACADEMIC EVALUATION METRICS",
            "-" * 40,
            f"  Transcript Word Count : {eval_metrics.get('transcript_words', 0)} words",
            f"  Summary Word Count    : {eval_metrics.get('summary_words', 0)} words",
            f"  Compression Ratio     : {eval_metrics.get('compression_ratio', 0.0):.1f}% reduction",
            f"  Lexical Overlap       : {eval_metrics.get('lexical_overlap', 0.0):.1f}%",
            f"  Flesch Reading Ease   : {eval_metrics.get('reading_ease', 0.0):.1f} ({eval_metrics.get('reading_ease_label', 'Standard')})",
            f"  Est. Reading Time     : {eval_metrics.get('summary_reading_time', '0 min')} (Saved: {eval_metrics.get('time_saved', '0 min')})",
        ])

    lines.extend([
        "",
        "7. FULL TRANSCRIPT",
        "-" * 40,
        transcript_text.strip(),
        "",
        "=" * 70,
        "End of Report | Generated by ClipVid Mini-Project",
        "=" * 70
    ])

    return "\n".join(lines)

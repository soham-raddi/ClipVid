"""
Utility script to generate sample test videos for evaluation and viva demonstration.
Generates:
1. sample_education.mp4 - Video with audio track
2. sample_silent.mp4 - Video with NO audio track (tests edge case error handling)
3. sample_corrupted.mp4 - Corrupted video file (tests corruption validation)
4. sample_unsupported.xyz - Unsupported file type (tests format validation)
"""
import os
import subprocess
from pathlib import Path
import imageio_ffmpeg

SAMPLE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

def generate_samples():
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"Using FFmpeg at: {ffmpeg_exe}")

    # 1. Educational demo video with audio tone (12 seconds)
    demo_video = SAMPLE_DIR / "sample_education.mp4"
    if not demo_video.exists():
        print("Generating sample_education.mp4...")
        cmd = [
            ffmpeg_exe, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=12:size=640x360:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            str(demo_video)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Created: {demo_video}")

    # 2. Silent video with NO audio track (tests NoAudioTrackError)
    silent_video = SAMPLE_DIR / "sample_silent.mp4"
    if not silent_video.exists():
        print("Generating sample_silent.mp4...")
        cmd = [
            ffmpeg_exe, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=6:size=640x360:rate=24",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-an",  # No audio!
            str(silent_video)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Created: {silent_video}")

    # 3. Corrupted video (tests CorruptedVideoError)
    corrupted_video = SAMPLE_DIR / "sample_corrupted.mp4"
    if not corrupted_video.exists():
        with open(corrupted_video, "wb") as f:
            f.write(b"CORRUPTED_HEADER_DATA_NOT_A_REAL_MP4_CONTAINER_1234567890")
        print(f"Created: {corrupted_video}")

    # 4. Unsupported file extension
    unsupported_file = SAMPLE_DIR / "sample_unsupported.xyz"
    if not unsupported_file.exists():
        with open(unsupported_file, "w") as f:
            f.write("This is an unsupported text/xyz format.")
        print(f"Created: {unsupported_file}")

    print("All sample videos ready for evaluation!")

if __name__ == "__main__":
    generate_samples()

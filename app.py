"""
ClipVid - AI-Based Video Summarization Using Generative AI
Main Streamlit Web Application
"""
import os
import sys
import time
from pathlib import Path
import streamlit as st

# Configure base paths
APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "logo.png"

# Configure page settings
st.set_page_config(
    page_title="ClipVid - Video Intelligence & Summarization",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add parent directory to path for imports
sys.path.insert(0, str(APP_DIR))

from modules.config import (
    SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB, GROQ_API_KEY, TEMP_DIR
)
from modules.utils import (
    validate_video_file, format_seconds, format_file_size,
    safe_delete_file, cleanup_old_temp_files, generate_full_report
)
from modules.video_processor import VideoProcessor, NoAudioTrackError, CorruptedVideoError
from modules.transcription import get_transcriber, TranscriptionError, TranscriptResult
from modules.summarizer import get_summarizer, SummarizationError, VideoSummaryOutput
from modules.evaluation import evaluate_summary

# Clean up temp files older than 1 hour on app load
cleanup_old_temp_files()

# Custom CSS for high-grade professional web aesthetics (No AI emojis)
st.markdown("""
<style>
    /* Global Typography and Card Spacing */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 8px;
    }
    .brand-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #f8fafc;
        margin: 0;
        line-height: 1.2;
    }
    .brand-tagline {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 4px;
        margin-bottom: 20px;
    }
    .meta-pill {
        display: inline-block;
        background: #1e293b;
        color: #94a3b8;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 6px;
    }
    
    /* Topic pills */
    .topic-pill {
        display: inline-block;
        background: #0f172a;
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 14px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 4px 6px 4px 0;
    }

    /* Timestamp badge */
    .timestamp-badge {
        background: #0f172a;
        color: #38bdf8;
        border: 1px solid #1e293b;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.85rem;
    }

    /* Download button container */
    .stDownloadButton button {
        width: 100%;
        border-radius: 6px;
        font-weight: 500;
    }

    /* Clean subtle dividers */
    hr {
        border-color: #1e293b !important;
        margin: 1.5rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR CONFIGURATION
# -------------------------------------------------------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=56)
    st.markdown("### ClipVid Settings")

    st.markdown("---")
    st.markdown("#### Processing Parameters")
    chunk_size = st.slider(
        "Transcript Chunk Size (words)",
        min_value=300,
        max_value=1200,
        value=600,
        step=50,
        help="Chunk size for map-reduce processing of longer transcripts."
    )
    overlap_words = st.slider(
        "Chunk Overlap (words)",
        min_value=20,
        max_value=100,
        value=50,
        step=10,
        help="Sliding window overlap between consecutive chunks to preserve segment context."
    )

    st.markdown("---")
    st.markdown("#### Test Media Presets")
    sample_choice = st.selectbox(
        "Select Verification Sample",
        options=[
            "None (Custom upload / URL)",
            "Educational Demo (Clear audio & speech)",
            "Silent Video (Missing audio stream test)",
            "Corrupted Video (Container integrity test)",
            "Unsupported Format (.xyz rejection test)"
        ]
    )

# -------------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------------
st.markdown("""
<div class="brand-header">
    <div>
        <h1 class="brand-title">AI-Based Video Summarization</h1>
        <div class="brand-tagline">
            Automated speech extraction, timestamp segmentation, and generative synthesis using Whisper and Llama models.
        </div>
        <div>
            <span class="meta-pill">Speech-to-Text: Whisper-large-v3</span>
            <span class="meta-pill">Generative AI: Llama-3.3-70b</span>
            <span class="meta-pill">Digital Signal Processing: FFmpeg 16kHz</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Import YouTube handlers
from modules.youtube import (
    is_valid_youtube_url, get_youtube_metadata,
    download_youtube_audio, YouTubeError
)

# Video Input Source: File Upload vs YouTube URL
input_mode = st.radio(
    "Select Video Source",
    options=["Upload Local Video", "YouTube Video Link"],
    horizontal=True
)

uploaded_file = None
active_video_path = None
active_video_name = None
active_video_size = 0
is_youtube = False
youtube_url = ""
video_meta = {}

sample_dir = APP_DIR / "sample_videos"

if input_mode == "YouTube Video Link" and sample_choice == "None (Custom upload / URL)":
    is_youtube = True
    youtube_url = st.text_input(
        "YouTube Video Link",
        placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
        help="Paste any public YouTube video link for automated transcription and summarization."
    )

    if youtube_url:
        if not is_valid_youtube_url(youtube_url):
            st.error("Invalid YouTube URL: Please enter a valid YouTube video link.")
            st.stop()

        with st.spinner("Fetching YouTube video details..."):
            try:
                yt_info = get_youtube_metadata(youtube_url)
                video_meta = {
                    "filename": yt_info["title"],
                    "size_bytes": 0,
                    "size_formatted": "YouTube Stream",
                    "duration_seconds": yt_info["duration_seconds"],
                    "duration_formatted": yt_info["duration_formatted"],
                    "resolution": "YouTube Web",
                    "has_audio": True,
                    "channel": yt_info["channel"],
                    "url": yt_info["webpage_url"]
                }
                active_video_name = yt_info["title"]
            except Exception as e:
                st.error(f"Failed to fetch YouTube metadata: {str(e)}")
                st.stop()

        col_preview, col_meta = st.columns([3, 2])
        with col_preview:
            st.markdown("### Video Preview")
            st.video(youtube_url)

        with col_meta:
            st.markdown("### Video Metadata")
            st.markdown(f"**Title:** `{video_meta['filename']}`")
            st.markdown(f"**Channel:** {video_meta.get('channel', 'Unknown')}")
            st.markdown(f"**Duration:** {video_meta['duration_formatted']}")
            st.markdown(f"**Source:** YouTube Stream")
            st.markdown(f"**Audio Stream:** Present")

else:
    # Local video handling (Upload or Sample Preset)
    if sample_choice != "None (Custom upload / URL)":
        if "Educational Demo" in sample_choice:
            sample_path = sample_dir / "sample_education.mp4"
        elif "Silent Video" in sample_choice:
            sample_path = sample_dir / "sample_silent.mp4"
        elif "Corrupted Video" in sample_choice:
            sample_path = sample_dir / "sample_corrupted.mp4"
        else:
            sample_path = sample_dir / "sample_unsupported.xyz"
            
        if sample_path.exists():
            active_video_path = str(sample_path)
            active_video_name = sample_path.name
            active_video_size = os.path.getsize(sample_path)
            st.info(f"Loaded verification preset: **{active_video_name}** ({format_file_size(active_video_size)})")
        else:
            st.warning(f"Preset file `{sample_path.name}` not found. Run `sample_videos/create_samples.py`.")
    else:
        uploaded_file = st.file_uploader(
            "Upload a video file for summarization",
            type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
            help=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}. Maximum size: {MAX_FILE_SIZE_MB}MB."
        )
        if uploaded_file:
            active_video_name = uploaded_file.name
            active_video_size = uploaded_file.size

    # Validate file format and size
    if active_video_name:
        is_valid, validation_err = validate_video_file(active_video_name, active_video_size)
        if not is_valid:
            st.error(f"Validation Error: {validation_err}")
            st.stop()

    # Display Video Preview and Metadata for local video
    if active_video_name and not is_youtube:
        if uploaded_file and not active_video_path:
            temp_input = TEMP_DIR / f"upload_{int(time.time())}_{active_video_name}"
            with open(temp_input, "wb") as f:
                f.write(uploaded_file.getbuffer())
            active_video_path = str(temp_input)

        processor = VideoProcessor()
        
        col_preview, col_meta = st.columns([3, 2])
        with col_preview:
            st.markdown("### Video Preview")
            try:
                st.video(active_video_path)
            except Exception:
                st.warning("Preview unavailable for this container format.")

        with col_meta:
            st.markdown("### Video Metadata")
            try:
                video_meta = processor.inspect_video(active_video_path)
                st.markdown(f"**Filename:** `{video_meta['filename']}`")
                st.markdown(f"**File Size:** {video_meta['size_formatted']}")
                st.markdown(f"**Duration:** {video_meta['duration_formatted']}")
                st.markdown(f"**Resolution:** {video_meta['resolution']}")
                st.markdown(f"**Audio Stream:** {'Present' if video_meta['has_audio'] else 'Not Detected'}")
            except CorruptedVideoError as cve:
                st.error(f"Container Error: {str(cve)}")
                st.stop()
            except Exception as e:
                video_meta = {
                    "filename": active_video_name,
                    "size_bytes": active_video_size,
                    "size_formatted": format_file_size(active_video_size),
                    "duration_seconds": 0.0,
                    "duration_formatted": "Unknown",
                    "resolution": "Unknown",
                    "has_audio": True
                }
                st.warning(f"Metadata extraction failed: {str(e)}")

if active_video_name:
    st.markdown("---")
    
    # Process Button
    start_btn = st.button("Process Video & Generate Summary", type="primary", use_container_width=True)

    if start_btn:
        progress_bar = st.progress(0, text="Initializing processing pipeline...")
        status_box = st.empty()

        audio_path = None
        try:
            # 1. Video & Audio Extraction
            if is_youtube:
                status_box.info("Step 1/4: Downloading audio stream from YouTube...")
                progress_bar.progress(0.2, text="Extracting 16kHz mono audio from YouTube...")
                try:
                    audio_path = download_youtube_audio(youtube_url)
                except YouTubeError as yte:
                    progress_bar.empty()
                    status_box.empty()
                    st.error(f"YouTube Download Error: {str(yte)}")
                    st.stop()
            else:
                status_box.info("Step 1/4: Extracting audio track using FFmpeg...")
                progress_bar.progress(0.2, text="Extracting 16kHz mono audio track...")
                try:
                    audio_path = processor.extract_audio(active_video_path, output_format="mp3")
                except NoAudioTrackError as nate:
                    progress_bar.empty()
                    status_box.empty()
                    st.error(f"Audio Extraction Failed: {str(nate)}")
                    st.info("Suggestion: Please upload a video containing audible spoken dialogue or voiceover.")
                    st.stop()
                except CorruptedVideoError as cve:
                    progress_bar.empty()
                    status_box.empty()
                    st.error(f"Media Container Error: {str(cve)}")
                    st.stop()

            # Check for API Key in environment
            if not GROQ_API_KEY:
                progress_bar.empty()
                status_box.empty()
                st.error("Configuration Error: `GROQ_API_KEY` is not configured in your `.env` file.")
                st.info("Please add `GROQ_API_KEY=your_key` to your local `.env` file and reload.")
                st.stop()

            # 2. Speech-to-Text Transcription
            status_box.info("Step 2/4: Transcribing speech to text...")
            progress_bar.progress(0.4, text="Transcribing audio with Groq Whisper-large-v3...")
            
            transcriber = get_transcriber("groq", api_key=GROQ_API_KEY)
            transcript_result: TranscriptResult = transcriber.transcribe(audio_path)

            if not transcript_result.full_text.strip():
                progress_bar.empty()
                status_box.empty()
                st.error("Empty Transcript: No spoken dialogue was detected in the audio track.")
                st.stop()

            # 3. LLM Summarization
            status_box.info("Step 3/4: Synthesizing summary using Groq Llama-3.3...")
            progress_bar.progress(0.65, text="Applying generative synthesis and chunked processing...")
            
            summarizer = get_summarizer("groq", api_key=GROQ_API_KEY)
            
            def progress_update(ratio: float, msg: str):
                progress_bar.progress(ratio, text=msg)

            summary_output: VideoSummaryOutput = summarizer.generate_summary(
                transcript=transcript_result,
                chunk_size_words=chunk_size,
                overlap_words=overlap_words,
                progress_callback=progress_update
            )

            # 4. Evaluation Metrics
            status_box.info("Step 4/4: Computing academic evaluation metrics...")
            progress_bar.progress(0.95, text="Calculating compression and readability metrics...")
            eval_metrics = evaluate_summary(transcript_result.full_text, summary_output.model_dump())

            progress_bar.progress(1.0, text="Summarization complete.")
            time.sleep(0.5)
            progress_bar.empty()
            status_box.empty()

            # Store in session state for tab rendering and downloads
            st.session_state["results"] = {
                "summary": summary_output.model_dump(),
                "transcript": transcript_result,
                "video_meta": video_meta,
                "eval_metrics": eval_metrics
            }

        except TranscriptionError as te:
            progress_bar.empty()
            status_box.empty()
            st.error(f"Transcription Error: {str(te)}")
            st.info("Please verify `GROQ_API_KEY` in your `.env` file.")
        except SummarizationError as se:
            progress_bar.empty()
            status_box.empty()
            st.error(f"Summarization Error: {str(se)}")
            st.info("Please verify `GROQ_API_KEY` and rate limits in your `.env` file.")
        except Exception as e:
            progress_bar.empty()
            status_box.empty()
            st.error(f"Unexpected Processing Error: {str(e)}")
        finally:
            # Clean up temporary audio file
            if audio_path:
                safe_delete_file(audio_path)

# -------------------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------------------
if "results" in st.session_state:
    results = st.session_state["results"]
    summary_data = results["summary"]
    transcript: TranscriptResult = results["transcript"]
    video_meta = results["video_meta"]
    eval_metrics = results["eval_metrics"]

    st.success("Summary generated successfully.")

    # Suggested Title
    st.markdown(f"## {summary_data.get('suggested_title', 'Video Summary')}")
    
    # Render topics as clean badges without emojis
    topics = summary_data.get("topics", [])
    if topics:
        topics_html = " ".join([f"<span class='topic-pill'>{t}</span>" for t in topics])
        st.markdown(topics_html, unsafe_allow_html=True)
        st.markdown("")

    # Create Tabs for organized presentation
    tab_summary, tab_details, tab_transcript, tab_eval = st.tabs([
        "Summary & Key Points",
        "Detailed Breakdown & Timestamps",
        "Full Transcript",
        "Academic Evaluation"
    ])

    # Tab 1: Executive Summary & Key Points
    with tab_summary:
        st.markdown("### Executive Summary")
        st.info(summary_data.get("short_summary", "No executive summary available."))

        st.markdown("### Key Takeaways")
        key_points = summary_data.get("key_points", [])
        if key_points:
            for i, pt in enumerate(key_points, 1):
                st.markdown(f"**{i}.** {pt}")
        else:
            st.write("No distinct key points identified.")

    # Tab 2: Detailed Summary & Timestamps
    with tab_details:
        st.markdown("### Detailed Summary")
        st.write(summary_data.get("detailed_summary", "No detailed summary available."))

        st.markdown("### Section Timestamps")
        timestamps = summary_data.get("important_timestamps", [])
        if timestamps:
            for item in timestamps:
                ts = item.get("timestamp", "--:--")
                desc = item.get("description", "")
                st.markdown(
                    f"<div style='margin-bottom: 8px;'>"
                    f"<span class='timestamp-badge'>{ts}</span> &nbsp; {desc}"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("Discrete section timestamps were not identified for this recording.")

    # Tab 3: Interactive Transcript
    with tab_transcript:
        st.markdown("### Transcript")
        search_query = st.text_input("Search transcript content:", placeholder="Filter by keyword...")

        view_mode = st.radio("Display Mode:", ["Timestamped Segments", "Raw Text"], horizontal=True)

        if view_mode == "Timestamped Segments" and transcript.segments:
            for seg in transcript.segments:
                if not search_query or search_query.lower() in seg.text.lower():
                    st.markdown(
                        f"**`{seg.formatted_start} - {seg.formatted_end}`**: {seg.text}"
                    )
        else:
            filtered_text = transcript.full_text
            st.text_area("Full Transcript Text", value=filtered_text, height=350)

    # Tab 4: Academic Evaluation
    with tab_eval:
        st.markdown("### Quantitative Evaluation Metrics")
        st.caption("Standard academic metrics for compression, readability, and content preservation.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                label="Compression Ratio",
                value=f"{eval_metrics['compression_ratio']}%",
                help="Percentage reduction in content length from transcript to summary."
            )
        with c2:
            st.metric(
                label="Time Saved",
                value=eval_metrics["time_saved"],
                help="Estimated viewing/speaking time saved compared to reading the summary."
            )
        with c3:
            st.metric(
                label="Lexical Coverage",
                value=f"{eval_metrics['lexical_coverage']}%",
                help="Percentage of core topic vocabulary preserved in the summary."
            )
        with c4:
            st.metric(
                label="Readability",
                value=f"{eval_metrics['reading_ease']}/100",
                help=f"Flesch Reading Ease: {eval_metrics['reading_ease_label']}"
            )

        st.markdown("---")
        with st.expander("Evaluation Metric Definitions & Viva Guide"):
            st.markdown(f"""
            - **Compression Ratio ({eval_metrics['compression_ratio']}%)**:  
              {eval_metrics['metrics_explanation']['compression_ratio']}
            - **Lexical Coverage ({eval_metrics['lexical_coverage']}%)**:  
              {eval_metrics['metrics_explanation']['lexical_coverage']}
            - **Factual Groundedness ({eval_metrics['groundedness_score']}%)**:  
              {eval_metrics['metrics_explanation']['groundedness_score']}
            - **Readability ({eval_metrics['reading_ease']} - {eval_metrics['reading_ease_label']})**:  
              {eval_metrics['metrics_explanation']['reading_ease']}
            """)

    # ---------------------------------------------------------
    # DOWNLOAD OPTIONS
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### Export Artifacts")
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    # 1. Transcript TXT
    with col_dl1:
        st.download_button(
            label="Download Transcript (TXT)",
            data=transcript.full_text,
            file_name=f"{Path(video_meta['filename']).stem}_transcript.txt",
            mime="text/plain"
        )
    
    # 2. Summary TXT
    with col_dl2:
        summary_txt = (
            f"TITLE: {summary_data.get('suggested_title')}\n\n"
            f"SHORT SUMMARY:\n{summary_data.get('short_summary')}\n\n"
            f"DETAILED SUMMARY:\n{summary_data.get('detailed_summary')}\n\n"
            f"KEY POINTS:\n" + "\n".join([f"- {p}" for p in summary_data.get("key_points", [])])
        )
        st.download_button(
            label="Download Summary (TXT)",
            data=summary_txt,
            file_name=f"{Path(video_meta['filename']).stem}_summary.txt",
            mime="text/plain"
        )

    # 3. Complete Academic Report
    with col_dl3:
        full_report_text = generate_full_report(
            summary_data=summary_data,
            transcript_text=transcript.full_text,
            video_info=video_meta,
            eval_metrics=eval_metrics
        )
        st.download_button(
            label="Download Evaluation Report (TXT)",
            data=full_report_text,
            file_name=f"{Path(video_meta['filename']).stem}_academic_report.txt",
            mime="text/plain"
        )

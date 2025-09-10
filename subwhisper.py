#!/usr/bin/env python3
"""SubWhisper: Automatic subtitle generator using OpenAI Whisper.

This module provides functionality to automatically generate subtitles for videos
using OpenAI's Whisper speech-to-text model with extensive logging and error handling
for real-world deployment.
"""
# pylint: disable=logging-fstring-interpolation

import argparse
import glob
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Union

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import whisper

# Create a temp directory for extracted audio
TEMP_DIR: Optional[str] = None

# Thread lock for temporary directory operations
_TEMP_DIR_LOCK = threading.Lock()


# Type definitions
class WhisperSegment(TypedDict):
    """Type definition for Whisper transcription segment."""

    start: float
    end: float
    text: str


class WhisperResult(TypedDict):
    """Type definition for Whisper transcription result."""

    segments: List[WhisperSegment]
    language: str


# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("subwhisper.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# Try to find ffmpeg in common locations if not in PATH
def find_ffmpeg() -> Optional[str]:
    """Find FFmpeg executable in the system with comprehensive logging.

    Returns:
        Path to FFmpeg executable if found, None otherwise.

    Raises:
        No exceptions are raised; all errors are logged and handled gracefully.
    """
    logger.info("Searching for FFmpeg executable...")

    # Common locations for FFmpeg
    ffmpeg_locations = [
        # Windows Program Files and common installation locations
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        # Winget installation location
        os.path.expanduser(
            r"~\AppData\Local\Microsoft\WinGet\Packages"
            + r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
            + r"\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"
        ),
        # Default command (if in PATH)
        "ffmpeg",
    ]

    # Check if ffmpeg is in PATH first
    try:
        # Use appropriate command based on OS
        if sys.platform == "win32":
            check_cmd = ["where", "ffmpeg"]
        else:
            check_cmd = ["which", "ffmpeg"]

        logger.debug(f"Checking PATH for FFmpeg using command: {' '.join(check_cmd)}")
        result = subprocess.run(
            check_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            ffmpeg_path = result.stdout.strip().split("\n")[0]
            logger.info(f"FFmpeg found in PATH: {ffmpeg_path}")
            return ffmpeg_path
        logger.debug(f"FFmpeg not found in PATH, stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.warning("Timeout expired while searching for FFmpeg in PATH")
    except Exception as e:
        logger.warning(f"Error while searching for FFmpeg in PATH: {e}")

    # Check common locations
    logger.debug("Checking common FFmpeg installation locations...")
    for i, location in enumerate(ffmpeg_locations, 1):
        try:
            if os.path.isfile(location):
                logger.info(
                    f"FFmpeg found at location {i}/{len(ffmpeg_locations)}: {location}"
                )
                return location

            logger.debug(
                f"FFmpeg not found at location {i}/{len(ffmpeg_locations)}: {location}"
            )
        except Exception as e:
            logger.debug(f"Error checking FFmpeg location {location}: {e}")

    # If we get here, we couldn't find ffmpeg
    logger.error("FFmpeg executable not found in any known locations")
    return None


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT format timestamp (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def extract_audio(
    video_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
) -> str:
    """Extract audio from video file using ffmpeg with comprehensive error handling.

    Args:
        video_path: Path to the input video file.
        output_path: Optional output path for extracted audio.
            If None, creates temp file.

    Returns:
        Path to the extracted audio file.

    Raises:
        RuntimeError: If FFmpeg execution fails or audio extraction fails.
        FileNotFoundError: If FFmpeg executable is not found.
        ValueError: If input video path is invalid.
    """
    global TEMP_DIR  # pylint: disable=global-statement

    # Validate input
    if not video_path:
        raise ValueError("Video path cannot be empty")

    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise ValueError(f"Video file does not exist: {video_path}")

    logger.info(f"Starting audio extraction from: {video_path}")

    # Thread-safe temporary directory creation
    if output_path is None:
        with _TEMP_DIR_LOCK:
            if TEMP_DIR is None:
                TEMP_DIR = tempfile.mkdtemp(prefix="subwhisper_")
                logger.info(f"Created temporary directory: {TEMP_DIR}")

        # Create a safe filename based on the input video name
        basename_no_ext = video_path_obj.stem
        output_path = os.path.join(TEMP_DIR, f"{basename_no_ext}.wav")
        logger.debug(f"Generated output path: {output_path}")
    else:
        logger.debug(f"Using provided output path: {output_path}")

    try:
        # Find ffmpeg executable
        ffmpeg_path = find_ffmpeg()
        if ffmpeg_path is None:
            error_msg = (
                "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        command = [
            ffmpeg_path,
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
            "-y",
        ]

        logger.info(f"Executing FFmpeg command: {' '.join(command)}")
        start_time = time.time()

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3600,  # 1 hour timeout for large files
            check=False,
        )

        extraction_time = time.time() - start_time
        logger.info(f"Audio extraction completed in {extraction_time:.2f} seconds")

        if result.returncode != 0:
            error_msg = (
                result.stderr.strip() if result.stderr else "Unknown FFmpeg error"
            )
            logger.error(
                f"FFmpeg failed with return code {result.returncode}: {error_msg}"
            )
            raise RuntimeError(f"FFmpeg error (code {result.returncode}): {error_msg}")

        # Verify output file was created and has reasonable size
        output_path_obj = Path(output_path)
        if not output_path_obj.exists():
            raise RuntimeError(
                f"Audio extraction failed: output file not created at {output_path}"
            )

        file_size = output_path_obj.stat().st_size
        if file_size < 1000:  # Less than 1KB is suspicious
            logger.warning(
                f"Extracted audio file is very small ({file_size} bytes), "
                "may indicate extraction issues"
            )
        else:
            logger.info(
                f"Successfully extracted audio file ({file_size} bytes): {output_path}"
            )

        return str(output_path)

    except subprocess.TimeoutExpired:
        error_msg = f"Audio extraction timed out after 1 hour for file: {video_path}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from None
    except FileNotFoundError:
        logger.error("FFmpeg executable not found")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during audio extraction: {e}")
        raise RuntimeError(f"Audio extraction failed: {e}") from e


def generate_srt(
    segments: List[WhisperSegment],
    output_file: Union[str, Path],
    max_segment_length: Optional[int] = None,
) -> None:
    """Generate an SRT file from the Whisper segments.

    Optional splitting of long segments is supported.

    Args:
        segments: List of segment dictionaries containing start, end, and text.
        output_file: Path to the output SRT file.
        max_segment_length: Optional maximum character length for subtitle segments.

    Raises:
        RuntimeError: If SRT generation fails.
        ValueError: If segments data is invalid.
    """
    if not segments:
        logger.warning("No segments provided for SRT generation")
        segments = []

    logger.info(f"Generating SRT file with {len(segments)} segments: {output_file}")
    if max_segment_length:
        logger.info(f"Using maximum segment length: {max_segment_length} characters")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            subtitle_index = 1

            for segment in segments:
                start = segment["start"]
                end = segment["end"]
                text = segment["text"].strip()

                # Split long segments if max length is specified
                if max_segment_length and len(text) > max_segment_length:
                    words = text.split()
                    current_line: List[str] = []
                    current_length = 0

                    for word in words:
                        # Add word length plus space
                        if (
                            current_length + len(word) + 1 <= max_segment_length
                            or not current_line
                        ):
                            current_line.append(word)
                            current_length += len(word) + 1
                        else:
                            # Calculate timing for partial segment
                            segment_length = end - start
                            chars_ratio = current_length / len(text)
                            segment_duration = segment_length * chars_ratio

                            sub_end = start + segment_duration

                            # Write the current line
                            f.write(f"{subtitle_index}\n")
                            f.write(
                                f"{format_timestamp(start)} --> "
                                f"{format_timestamp(sub_end)}\n"
                            )
                            f.write(f"{' '.join(current_line)}\n\n")

                            # Setup for next segment
                            subtitle_index += 1
                            start = sub_end
                            current_line = [word]
                            current_length = len(word) + 1

                    # Write the last segment if there's anything left
                    if current_line:
                        f.write(f"{subtitle_index}\n")
                        f.write(
                            f"{format_timestamp(start)} --> {format_timestamp(end)}\n"
                        )
                        f.write(f"{' '.join(current_line)}\n\n")
                        subtitle_index += 1
                else:
                    # Write normal segment
                    f.write(f"{subtitle_index}\n")
                    f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
                    f.write(f"{text}\n\n")
                    subtitle_index += 1

        logger.info(f"Successfully generated SRT file: {output_file}")

        # Verify the generated file
        output_path_obj = Path(output_file)
        file_size = output_path_obj.stat().st_size
        logger.debug(f"Generated SRT file size: {file_size} bytes")

    except Exception as e:
        logger.error(f"Failed to generate SRT file {output_file}: {e}")
        raise RuntimeError(f"SRT generation failed: {e}") from e


def process_video(video_path: Union[str, Path], args: argparse.Namespace) -> bool:
    """Process a single video file with comprehensive error handling and logging.

    Args:
        video_path: Path to the video file to process.
        args: Command line arguments namespace.

    Returns:
        True if processing was successful, False otherwise.
    """
    video_path_obj = Path(video_path)
    logger.info(f"Starting video processing: {video_path_obj}")

    try:
        # Validate video file
        if not video_path_obj.exists():
            logger.error(f"Video file not found: {video_path_obj}")
            return False

        file_size = video_path_obj.stat().st_size
        logger.info(
            f"Video file size: {file_size} bytes ({file_size / (1024 * 1024):.2f} MB)"
        )

        # Extract audio
        logger.info("Beginning audio extraction phase...")
        audio_path = extract_audio(str(video_path_obj))

        # Set output path if not provided
        if args.output is None:
            output_path = video_path_obj.with_suffix(".srt")
            logger.debug(f"Using default output path: {output_path}")
        else:
            # For batch processing, create output in the specified directory
            if args.batch:
                output_dir = Path(args.output)
                output_path = output_dir / video_path_obj.with_suffix(".srt").name
                logger.debug(f"Batch mode output path: {output_path}")
            else:
                output_path = Path(args.output)
                logger.debug(f"Using specified output path: {output_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory ready: {output_path.parent}")

        # Transcribe audio
        logger.info("Beginning transcription phase...")
        result = transcribe_audio(audio_path, args)
        if result is None:
            logger.error("Transcription failed, aborting video processing")
            return False

        logger.info(
            f"Transcription completed, found {len(result.get('segments', []))} segments"
        )

        # Generate SRT file
        logger.info("Beginning SRT generation phase...")
        generate_srt(result["segments"], str(output_path), args.max_segment_length)

        # Apply post-processing if specified
        if args.post_process:
            logger.info("Beginning post-processing phase...")

            # Replace placeholders in the command
            process_command = args.post_process
            process_command = process_command.replace("INPUT_FILE", str(output_path))

            # Add support for basename placeholder (for Docker)
            basename = output_path.name
            process_command = process_command.replace("INPUT_FILE_BASENAME", basename)

            logger.debug(f"Post-processing command: {process_command}")

            try:
                subprocess_result = subprocess.run(
                    process_command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=1800,  # 30 minute timeout for post-processing
                    check=False,
                )

                if subprocess_result.returncode != 0:
                    error_msg = (
                        subprocess_result.stderr.strip()
                        if subprocess_result.stderr
                        else "Unknown post-processing error"
                    )
                    logger.warning(
                        f"Post-processing failed "
                        f"(code {subprocess_result.returncode}): "
                        f"{error_msg}"
                    )
                else:
                    logger.info("Post-processing completed successfully")
                    if subprocess_result.stdout.strip():
                        logger.debug(
                            f"Post-processing output: "
                            f"{subprocess_result.stdout.strip()}"
                        )
            except subprocess.TimeoutExpired:
                logger.error("Post-processing timed out after 30 minutes")
            except Exception as e:
                logger.warning(f"Error during post-processing: {e}")

        logger.info(f"Successfully completed processing video: {video_path_obj}")
        return True

    except Exception as e:
        logger.error(f"Failed to process video {video_path_obj}: {e}")
        return False


def transcribe_audio(  # pylint: disable=too-many-return-statements
    audio_path: Union[str, Path], args: argparse.Namespace
) -> Optional[WhisperResult]:
    """Transcribe audio using Whisper with comprehensive error handling and logging.

    Args:
        audio_path: Path to the audio file to transcribe.
        args: Command line arguments containing model and language settings.

    Returns:
        Transcription result dictionary if successful, None otherwise.

    Raises:
        No exceptions are raised; all errors are logged and handled gracefully.
    """
    audio_path_obj = Path(audio_path)
    logger.info(f"Starting audio transcription: {audio_path_obj}")

    try:
        # Validate audio file
        if not audio_path_obj.exists():
            logger.error(f"Audio file not found: {audio_path_obj}")
            return None

        file_size = audio_path_obj.stat().st_size
        logger.info(
            f"Audio file size: {file_size} bytes ({file_size / (1024 * 1024):.2f} MB)"
        )

        # Load Whisper model
        logger.info(f"Loading Whisper model: {args.model}")
        try:
            model = whisper.load_model(args.model)
            logger.info(f"Successfully loaded Whisper model: {args.model}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{args.model}': {e}")
            return None

        # Setup transcription options
        transcribe_options: Dict[str, Union[bool, str]] = {
            "verbose": False,  # We'll handle our own progress reporting
            "fp16": False,  # Disable fp16 which can cause issues on CPU
        }

        if args.language:
            transcribe_options["language"] = args.language
            logger.info(f"Using specified language: {args.language}")
        else:
            logger.info("Using automatic language detection")

        logger.debug(f"Transcription options: {transcribe_options}")

        # Start transcription
        logger.info("Beginning audio transcription...")
        start_time = time.time()

        try:
            # Try to load audio with numpy directly for better control
            try:

                logger.debug("Loading audio with scipy for preprocessing...")
                sample_rate, audio_data = wav.read(str(audio_path_obj))
                logger.debug(
                    f"Loaded audio: sample_rate={sample_rate}, "
                    f"shape={audio_data.shape}, dtype={audio_data.dtype}"
                )

                # Convert to float32 and normalize
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                else:
                    logger.debug(f"Audio data already in format: {audio_data.dtype}")

                # Convert to mono if stereo
                if audio_data.ndim > 1:
                    logger.debug(
                        f"Converting stereo audio to mono (shape: {audio_data.shape})"
                    )
                    audio_data = np.mean(audio_data, axis=1)

                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    logger.info(f"Resampling audio from {sample_rate}Hz to 16kHz")

                    audio_data = scipy.signal.resample(
                        audio_data, int(len(audio_data) * 16000 / sample_rate)
                    )

                logger.debug(f"Final audio shape: {audio_data.shape}")

                # Perform transcription with the loaded audio
                result = model.transcribe(audio_data, **transcribe_options)

            except Exception as audio_error:
                logger.warning(
                    f"Error loading audio with scipy: {audio_error}. "
                    "Falling back to whisper's audio loading"
                )

                # Fall back to whisper's default loading
                result = model.transcribe(str(audio_path_obj), **transcribe_options)

            # Calculate and log processing time
            elapsed_time = time.time() - start_time
            logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")

            # Validate and log results
            if result and "segments" in result:
                segment_count = len(result["segments"])
                detected_language = result.get("language", "unknown")
                logger.info(
                    f"Transcription successful: {segment_count} segments detected, "
                    f"language: {detected_language}"
                )

                # Log some basic statistics
                if segment_count > 0:
                    total_duration = (
                        result["segments"][-1]["end"] if result["segments"] else 0
                    )
                    logger.info(
                        f"Total transcribed duration: {total_duration:.2f} seconds"
                    )

                return result  # type: ignore[no-any-return]

            logger.error("Transcription returned invalid result")
            return None

        except KeyboardInterrupt:
            logger.warning("Transcription cancelled by user")
            return None
        except Exception as transcription_error:
            logger.error(f"Transcription failed: {transcription_error}")
            return None

    except Exception as e:
        logger.error(f"Unexpected error during transcription: {e}")
        return None


def cleanup_and_exit(exit_code: int = 0) -> None:
    """Clean up temporary files and exit with the given code.

    Args:
        exit_code: Exit code to use when exiting the application.
    """
    logger.info(f"Application cleanup initiated with exit code: {exit_code}")

    if TEMP_DIR and os.path.exists(TEMP_DIR):
        logger.info(f"Cleaning up temporary directory: {TEMP_DIR}")
        try:
            # Get directory size before cleanup for logging
            temp_path = Path(TEMP_DIR)
            total_size = sum(
                f.stat().st_size for f in temp_path.rglob("*") if f.is_file()
            )
            logger.debug(f"Removing {total_size} bytes from temporary directory")

            with _TEMP_DIR_LOCK:
                shutil.rmtree(TEMP_DIR)
            logger.info("Temporary directory cleaned up successfully")
        except Exception as e:
            logger.error(f"Failed to clean up temporary directory {TEMP_DIR}: {e}")
    else:
        logger.debug("No temporary directory to clean up")

    logger.info(f"SubWhisper application exiting with code: {exit_code}")
    sys.exit(exit_code)


def main() -> None:
    """Main function that handles argument parsing and orchestrates video processing."""
    parser = argparse.ArgumentParser(
        description="SubWhisper: Generate subtitles from video files "
        "using OpenAI's Whisper"
    )

    # Input options
    parser.add_argument(
        "video_path", help="Path to the video file or directory (if using --batch)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all video files in the specified directory",
    )
    parser.add_argument(
        "--extensions",
        default="mp4,mkv,avi,mov,webm",
        help="Comma-separated list of video extensions to process in batch mode",
    )

    # Whisper model options
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model size (tiny, base, small, medium, large)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (optional, auto-detected if not specified)",
    )

    # Output options
    parser.add_argument(
        "--output",
        default=None,
        help="Output SRT file path or directory (for batch processing)",
    )
    parser.add_argument(
        "--max-segment-length",
        type=int,
        default=None,
        help="Maximum character length for subtitle segments",
    )

    # Post-processing options group
    post_group = parser.add_argument_group("Post-processing options")
    post_group.add_argument(
        "--post-process",
        default=None,
        help="Command to run on generated subtitle file "
        "(use INPUT_FILE as placeholder)",
    )

    # Subtitle Edit CLI preset options
    se_group = parser.add_argument_group("Subtitle Edit presets (Docker required)")
    se_group.add_argument(
        "--fix-common-errors",
        action="store_true",
        help="Apply common error fixes to the generated subtitles",
    )
    se_group.add_argument(
        "--remove-hi", action="store_true", help="Remove text for hearing impaired"
    )
    se_group.add_argument(
        "--auto-split-long-lines",
        action="store_true",
        help="Automatically split long lines",
    )
    se_group.add_argument(
        "--fix-punctuation", action="store_true", help="Fix punctuation issues"
    )
    se_group.add_argument(
        "--ocr-fix", action="store_true", help="Apply OCR fixes (common OCR errors)"
    )
    se_group.add_argument(
        "--convert-to",
        choices=["srt", "ass", "stl", "smi", "vtt"],
        help="Convert subtitle to specified format",
    )

    # Parse args
    args = parser.parse_args()

    # Generate Docker post-processing command based on presets
    docker_cmd = generate_docker_post_process_cmd(args)
    if docker_cmd:
        if args.post_process:
            print(
                "Warning: Both custom post-process command and presets "
                "specified. Using presets."
            )
        args.post_process = docker_cmd

    try:
        # Print banner and log startup
        banner = "=" * 60
        title = "SubWhisper - Automatic Subtitle Generator"
        print(banner)
        print(title)
        print(banner)

        logger.info("SubWhisper application started")
        logger.info(f"Command line arguments: {vars(args)}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {sys.platform}")

        # Log system information for debugging
        try:
            import torch  # pylint: disable=import-outside-toplevel

            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
        except Exception as e:
            logger.debug(f"Could not get PyTorch info: {e}")

        # Check for batch processing
        if args.batch:
            logger.info("Starting batch processing mode")
            video_dir = Path(args.video_path)

            if not video_dir.is_dir():
                logger.error(f"Specified path is not a directory: {video_dir}")
                print(f"Error: {video_dir} is not a directory")
                return

            logger.info(f"Processing videos from directory: {video_dir}")

            # Create output directory if specified
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Output directory: {output_dir}")

            # Get list of video files
            extensions = [ext.strip().lower() for ext in args.extensions.split(",")]
            logger.info(f"Searching for video files with extensions: {extensions}")

            video_files = []
            for ext in extensions:
                pattern = f"{video_dir}/**/*.{ext}"
                found_files = glob.glob(pattern, recursive=True)
                video_files.extend(found_files)
                logger.debug(f"Found {len(found_files)} files with extension .{ext}")

                # Also check uppercase extensions
                pattern_upper = f"{video_dir}/**/*.{ext.upper()}"
                found_files_upper = glob.glob(pattern_upper, recursive=True)
                video_files.extend(found_files_upper)
                logger.debug(
                    f"Found {len(found_files_upper)} files with extension "
                    f".{ext.upper()}"
                )

            # Remove duplicates
            video_files = list(set(video_files))

            if not video_files:
                logger.warning(
                    f"No video files found with extensions: {args.extensions}"
                )
                print(
                    f"SubWhisper: No video files found with extensions: "
                    f"{args.extensions}"
                )
                return

            logger.info(f"Found {len(video_files)} video files to process")
            print(f"SubWhisper: Found {len(video_files)} video files to process")

            # Process each video
            successful = 0
            failed = 0
            start_time = time.time()

            for i, video_file in enumerate(video_files):
                logger.info(
                    f"Processing batch file {i + 1}/{len(video_files)}: {video_file}"
                )
                print(
                    f"\nSubWhisper: Processing file {i+1}/{len(video_files)}: "
                    f"{video_file}"
                )

                if process_video(video_file, args):
                    successful += 1
                    logger.info(
                        f"Successfully processed file {i + 1}/{len(video_files)}"
                    )
                else:
                    failed += 1
                    logger.error(f"Failed to process file {i + 1}/{len(video_files)}")

            total_time = time.time() - start_time
            logger.info(
                f"Batch processing complete: {successful} succeeded, "
                f"{failed} failed in {total_time:.2f} seconds"
            )
            print(
                f"\nSubWhisper: Batch processing complete: {successful} succeeded, "
                f"{failed} failed"
            )
            print(f"Total processing time: {total_time:.2f} seconds")

        else:
            # Process single video
            logger.info("Starting single video processing mode")
            start_time = time.time()

            success = process_video(args.video_path, args)
            processing_time = time.time() - start_time

            if success:
                logger.info(
                    f"Single video processing completed successfully in "
                    f"{processing_time:.2f} seconds"
                )
                print("\nSubWhisper: Processing complete!")
                if not args.post_process:
                    print(
                        "You can now open the SRT file in Subtitle Edit for any "
                        "additional formatting or timing adjustments."
                    )
            else:
                logger.error(
                    f"Single video processing failed after "
                    f"{processing_time:.2f} seconds"
                )
                print("\nSubWhisper: Processing failed. Check the logs for details.")

    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user (KeyboardInterrupt)")
        print("\nSubWhisper: Operation cancelled by user")
        cleanup_and_exit(1)
    except Exception as e:
        logger.error(f"Unexpected application error: {e}", exc_info=True)
        print(f"SubWhisper Error: {str(e)}")
        cleanup_and_exit(1)
    finally:
        # Clean up temporary files
        cleanup_and_exit(0)


def generate_docker_post_process_cmd(args: argparse.Namespace) -> Optional[str]:
    """Generate Docker post-processing command based on preset arguments"""
    if not (
        args.fix_common_errors
        or args.remove_hi
        or args.auto_split_long_lines
        or args.fix_punctuation
        or args.ocr_fix
        or args.convert_to
    ):
        return None

    # Start building Docker command
    docker_cmd = (
        'docker run --rm -v "$(pwd)":/subtitles seconv:1.0 '
        "/subtitles/INPUT_FILE_BASENAME"
    )

    # Set output format (default is subrip/srt)
    output_format = args.convert_to if args.convert_to else "subrip"
    docker_cmd += f" {output_format}"

    # Add operations
    if args.fix_common_errors:
        docker_cmd += " /fixcommonerrors"
    if args.remove_hi:
        docker_cmd += " /removetextforhi"
    if args.auto_split_long_lines:
        docker_cmd += " /splitlonglines"
    if args.fix_punctuation:
        docker_cmd += " /fixpunctuation"
    if args.ocr_fix:
        docker_cmd += " /ocrfix"

    return docker_cmd


if __name__ == "__main__":
    main()

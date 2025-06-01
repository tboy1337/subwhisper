#!/usr/bin/env python3
import os
import argparse
import subprocess
import torch
import whisper
import glob
import shutil
import tempfile
import sys
import time
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm

# Create a temp directory for extracted audio
TEMP_DIR = None

# Try to find ffmpeg in common locations if not in PATH
def find_ffmpeg():
    """Find FFmpeg executable in the system"""
    # Common locations for FFmpeg
    ffmpeg_locations = [
        # Windows Program Files and common installation locations
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        # Winget installation location
        os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"),
        # Default command (if in PATH)
        "ffmpeg"
    ]
    
    # Check if ffmpeg is in PATH first
    try:
        # Use appropriate command based on OS
        if sys.platform == "win32":
            check_cmd = ["where", "ffmpeg"]
        else:
            check_cmd = ["which", "ffmpeg"]
            
        result = subprocess.run(check_cmd, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True,
                               check=False)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]
    except Exception:
        pass
    
    # Check common locations
    for location in ffmpeg_locations:
        if os.path.isfile(location):
            return location
    
    # If we get here, we couldn't find ffmpeg
    return None

def format_timestamp(seconds):
    """Convert seconds to SRT format timestamp (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def extract_audio(video_path, output_path=None):
    """Extract audio from video file using ffmpeg"""
    global TEMP_DIR
    
    if output_path is None:
        if TEMP_DIR is None:
            TEMP_DIR = tempfile.mkdtemp(prefix="subwhisper_")
        
        # Create a safe filename based on the input video name
        basename = os.path.basename(video_path)
        basename_no_ext = os.path.splitext(basename)[0]
        output_path = os.path.join(TEMP_DIR, f"{basename_no_ext}.wav")
    
    try:
        # Find ffmpeg executable
        ffmpeg_path = find_ffmpeg()
        if ffmpeg_path is None:
            print("Error: FFmpeg not found. Please make sure FFmpeg is installed and in your PATH.")
            cleanup_and_exit(1)
            
        command = [
            ffmpeg_path, "-i", video_path, 
            "-vn", "-acodec", "pcm_s16le", 
            "-ar", "16000", "-ac", "1", 
            output_path, "-y"
        ]
        
        print(f"SubWhisper: Extracting audio from {video_path}...")
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            error_msg = result.stderr.strip()
            raise RuntimeError(f"FFmpeg error: {error_msg}")
            
        return output_path
    
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please make sure FFmpeg is installed and in your PATH.")
        cleanup_and_exit(1)
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        cleanup_and_exit(1)

def generate_srt(segments, output_file, max_segment_length=None):
    """Generate an SRT file from the Whisper segments with optional splitting of long segments"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            subtitle_index = 1
            
            for i, segment in enumerate(segments):
                start = segment['start']
                end = segment['end']
                text = segment['text'].strip()
                
                # Split long segments if max length is specified
                if max_segment_length and len(text) > max_segment_length:
                    words = text.split()
                    current_line = []
                    current_length = 0
                    
                    for word in words:
                        # Add word length plus space
                        if current_length + len(word) + 1 <= max_segment_length or not current_line:
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
                            f.write(f"{format_timestamp(start)} --> {format_timestamp(sub_end)}\n")
                            f.write(f"{' '.join(current_line)}\n\n")
                            
                            # Setup for next segment
                            subtitle_index += 1
                            start = sub_end
                            current_line = [word]
                            current_length = len(word) + 1
                    
                    # Write the last segment if there's anything left
                    if current_line:
                        f.write(f"{subtitle_index}\n")
                        f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
                        f.write(f"{' '.join(current_line)}\n\n")
                        subtitle_index += 1
                else:
                    # Write normal segment
                    f.write(f"{subtitle_index}\n")
                    f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
                    f.write(f"{text}\n\n")
                    subtitle_index += 1
        
        print(f"SubWhisper: SRT file saved to {output_file}")
    except Exception as e:
        print(f"Error generating SRT file: {str(e)}")
        cleanup_and_exit(1)

def process_video(video_path, args):
    """Process a single video file"""
    try:
        # Check if video exists
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Error: Video file '{video_path}' not found")
            return False
        
        # Extract audio
        audio_path = extract_audio(str(video_path))
        
        # Set output path if not provided
        if args.output is None:
            output_path = video_path.with_suffix(".srt")
        else:
            # For batch processing, create output in the specified directory
            if args.batch:
                output_dir = Path(args.output)
                output_path = output_dir / video_path.with_suffix(".srt").name
            else:
                output_path = Path(args.output)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Transcribe audio
        result = transcribe_audio(audio_path, args)
        if result is None:
            return False
        
        # Generate SRT file
        generate_srt(result["segments"], str(output_path), args.max_segment_length)
        
        # Apply post-processing if specified
        if args.post_process:
            print(f"SubWhisper: Applying post-processing to subtitle file...")
            
            # Replace placeholders in the command
            process_command = args.post_process
            process_command = process_command.replace("INPUT_FILE", str(output_path))
            
            # Add support for basename placeholder (for Docker)
            basename = output_path.name
            process_command = process_command.replace("INPUT_FILE_BASENAME", basename)
            
            try:
                result = subprocess.run(process_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                if result.returncode != 0:
                    error_msg = result.stderr.strip()
                    print(f"Warning: Post-processing failed: {error_msg}")
                else:
                    print(f"SubWhisper: Post-processing completed successfully")
                    if result.stdout.strip():
                        print(result.stdout.strip())
            except Exception as e:
                print(f"Warning: Error during post-processing: {str(e)}")
        
        return True
    
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return False

def transcribe_audio(audio_path, args):
    """Transcribe audio using Whisper with progress indication"""
    try:
        # Load Whisper model
        print(f"SubWhisper: Loading Whisper model ({args.model})...")
        model = whisper.load_model(args.model)
        
        print("SubWhisper: Transcribing audio... This may take a while depending on the file size and model used.")
        
        # Setup transcription options
        transcribe_options = {
            "verbose": False,  # We'll handle our own progress reporting
            "fp16": False      # Disable fp16 which can cause issues on CPU
        }
        
        if args.language:
            transcribe_options["language"] = args.language
        
        # Create a simple progress indicator for the transcription process
        spinner = ['|', '/', '-', '\\']
        i = 0
        
        def progress_callback():
            nonlocal i
            sys.stdout.write(f"\rSubWhisper: Processing... {spinner[i % len(spinner)]}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)
            return True
        
        # Start progress thread
        print("SubWhisper: Starting transcription...")
        start_time = time.time()
        
        # Setup a simple progress indicator
        try:
            while True:
                progress_callback()
                # Break if more than a short time has passed (whisper will start processing)
                if time.time() - start_time > 2:
                    break
            
            # Try to load audio with numpy directly
            try:
                import numpy as np
                import scipy.io.wavfile as wav
                
                print("\rSubWhisper: Loading audio with scipy...")
                sample_rate, audio_data = wav.read(audio_path)
                
                # Convert to float32 and normalize
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                
                # Convert to mono if stereo
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    print("\rSubWhisper: Resampling audio to 16kHz...")
                    from scipy import signal
                    audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
                
                # Perform transcription with the loaded audio
                result = model.transcribe(audio_data, **transcribe_options)
            except Exception as audio_error:
                print(f"\rSubWhisper: Error loading audio with scipy: {str(audio_error)}. Trying with whisper...")
                
                # Fall back to whisper's default loading
                result = model.transcribe(audio_path, **transcribe_options)
            
            # Clear the progress indicator
            sys.stdout.write("\r" + " " * 40 + "\r")
            sys.stdout.flush()
            
            # Display time taken
            elapsed_time = time.time() - start_time
            print(f"SubWhisper: Transcription completed in {elapsed_time:.2f} seconds.")
            
            return result
            
        except KeyboardInterrupt:
            print("\nSubWhisper: Transcription cancelled by user.")
            return None
    
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return None

def cleanup_and_exit(exit_code=0):
    """Clean up temporary files and exit with the given code"""
    global TEMP_DIR
    
    if TEMP_DIR and os.path.exists(TEMP_DIR):
        print(f"SubWhisper: Cleaning up temporary files...")
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {TEMP_DIR}: {str(e)}")
    
    sys.exit(exit_code)

def main():
    parser = argparse.ArgumentParser(description="SubWhisper: Generate subtitles from video files using OpenAI's Whisper")
    
    # Input options
    parser.add_argument("video_path", help="Path to the video file or directory (if using --batch)")
    parser.add_argument("--batch", action="store_true", help="Process all video files in the specified directory")
    parser.add_argument("--extensions", default="mp4,mkv,avi,mov,webm", help="Comma-separated list of video extensions to process in batch mode")
    
    # Whisper model options
    parser.add_argument("--model", default="small", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--language", default=None, help="Language code (optional, auto-detected if not specified)")
    
    # Output options
    parser.add_argument("--output", default=None, help="Output SRT file path or directory (for batch processing)")
    parser.add_argument("--max-segment-length", type=int, default=None, help="Maximum character length for subtitle segments")
    
    # Post-processing options group
    post_group = parser.add_argument_group('Post-processing options')
    post_group.add_argument("--post-process", default=None, help="Command to run on generated subtitle file (use INPUT_FILE as placeholder)")
    
    # Subtitle Edit CLI preset options
    se_group = parser.add_argument_group('Subtitle Edit presets (Docker required)')
    se_group.add_argument("--fix-common-errors", action="store_true", help="Apply common error fixes to the generated subtitles")
    se_group.add_argument("--remove-hi", action="store_true", help="Remove text for hearing impaired")
    se_group.add_argument("--auto-split-long-lines", action="store_true", help="Automatically split long lines")
    se_group.add_argument("--fix-punctuation", action="store_true", help="Fix punctuation issues")
    se_group.add_argument("--ocr-fix", action="store_true", help="Apply OCR fixes (common OCR errors)")
    se_group.add_argument("--convert-to", choices=["srt", "ass", "stl", "smi", "vtt"], help="Convert subtitle to specified format")
    
    # Parse args
    args = parser.parse_args()
    
    # Generate Docker post-processing command based on presets
    docker_cmd = generate_docker_post_process_cmd(args)
    if docker_cmd:
        if args.post_process:
            print("Warning: Both custom post-process command and presets specified. Using presets.")
        args.post_process = docker_cmd
    
    try:
        # Print banner
        print("=" * 60)
        print("SubWhisper - Automatic Subtitle Generator")
        print("=" * 60)
        
        # Check for batch processing
        if args.batch:
            video_dir = Path(args.video_path)
            if not video_dir.is_dir():
                print(f"Error: {video_dir} is not a directory")
                return
            
            # Create output directory if specified
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get list of video files
            extensions = args.extensions.split(',')
            video_files = []
            for ext in extensions:
                pattern = f"{video_dir}/**/*.{ext}"
                video_files.extend(glob.glob(pattern, recursive=True))
                pattern = f"{video_dir}/**/*.{ext.upper()}"
                video_files.extend(glob.glob(pattern, recursive=True))
            
            if not video_files:
                print(f"SubWhisper: No video files found with extensions: {args.extensions}")
                return
            
            print(f"SubWhisper: Found {len(video_files)} video files to process")
            
            # Process each video
            successful = 0
            failed = 0
            
            for i, video_file in enumerate(video_files):
                print(f"\nSubWhisper: Processing file {i+1}/{len(video_files)}: {video_file}")
                if process_video(video_file, args):
                    successful += 1
                else:
                    failed += 1
            
            print(f"\nSubWhisper: Batch processing complete: {successful} succeeded, {failed} failed")
            
        else:
            # Process single video
            process_video(args.video_path, args)
            print("\nSubWhisper: Processing complete!")
            if not args.post_process:
                print("You can now open the SRT file in Subtitle Edit for any additional formatting or timing adjustments.")
        
    except KeyboardInterrupt:
        print("\nSubWhisper: Operation cancelled by user")
    except Exception as e:
        print(f"SubWhisper Error: {str(e)}")
    finally:
        # Clean up temporary files
        cleanup_and_exit()

def generate_docker_post_process_cmd(args):
    """Generate Docker post-processing command based on preset arguments"""
    if not (args.fix_common_errors or args.remove_hi or args.auto_split_long_lines or 
            args.fix_punctuation or args.ocr_fix or args.convert_to):
        return None
    
    # Start building Docker command
    docker_cmd = 'docker run --rm -v "$(pwd)":/subtitles seconv:1.0 /subtitles/INPUT_FILE_BASENAME'
    
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
# SubWhisper

A powerful tool that uses OpenAI's Whisper speech-to-text model to automatically generate subtitles for video files.

## Prerequisites

1. Python 3.7 or later
2. FFmpeg installed on your system

### Installing FFmpeg

FFmpeg can be installed using various package managers or directly from its website:

#### Windows
- Download the latest static build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
- Extract the files and add the `bin` folder to your system's PATH.
- Use a package manager:
  - **Winget**:
    ```batch
    winget install ffmpeg
    ```
  - **Chocolatey**:
    ```batch
    choco install ffmpeg-full
    ```

#### macOS
- Use Homebrew:
  ```bash
  brew install ffmpeg
  ```

#### Linux
- Use your distribution's package manager:
  - Ubuntu/Debian:
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```
  - Fedora:
    ```bash
    sudo dnf install ffmpeg
    ```
  - Arch:
    ```bash
    sudo pacman -S ffmpeg
    ```

## Installation

1. Clone or download this repository
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Single Video)

```bash
python subwhisper.py path/to/your/video.mp4
```

### Batch Processing

Process all video files in a directory:

```bash
python subwhisper.py path/to/video/directory --batch --output path/to/output/directory
```

### Command Line Arguments

#### Input Options:
- `video_path`: Path to the video file or directory (if using --batch)
- `--batch`: Process all video files in the specified directory
- `--extensions`: Comma-separated list of video extensions to process in batch mode (default: "mp4,mkv,avi,mov,webm")

#### Whisper Model Options:
- `--model`: Whisper model size to use (default: "small")
  - Options: "tiny", "base", "small", "medium", "large"
  - Larger models are more accurate but require more RAM and processing time
- `--language`: Language code for transcription (optional, auto-detected if not specified)
  - Example: "en" for English, "fr" for French, etc.

#### Output Options:
- `--output`: Output SRT file path or directory (for batch processing)
- `--max-segment-length`: Maximum character length for subtitle segments. Longer segments will be split intelligently.

### Examples

Process a single video with medium model and English language:
```bash
python subwhisper.py video.mp4 --model medium --language en
```

Batch process all video files and split long subtitles:
```bash
python subwhisper.py videos_folder --batch --output subtitles_folder --max-segment-length 80
```

## Features

- Automatic speech-to-text transcription with OpenAI's Whisper
- Batch processing for multiple videos
- Smart cleanup of temporary files
- Progress indication during transcription
- Subtitle segment length control
- Error handling and graceful failure recovery
- Support for different languages and model sizes

## Working with Subtitle Edit

The script generates standard SRT subtitle files, which you can then open and edit with [Subtitle Edit](https://www.nikse.dk/subtitleedit/) for:

- Fine-tuning subtitle timing
- Fixing any transcription errors
- Styling or formatting the subtitles
- Translating to other languages

## Automated Post-Processing

SubWhisper can also automatically post-process subtitles using command-line tools:

### Using Subtitle Edit CLI

You can automatically post-process subtitles with Subtitle Edit CLI for tasks like fixing common errors, removing hearing-impaired text, or adjusting timing.

```bash
# Install Subtitle Edit CLI
# Windows: Download from https://github.com/SubtitleEdit/subtitleedit-cli
# Linux/macOS: Use the Docker version

# Example: Fix common errors and clean up subtitles
python subwhisper.py video.mp4 --post-process "seconv INPUT_FILE subrip /fixcommonerrors"

# Example: Remove text for hearing impaired
python subwhisper.py video.mp4 --post-process "seconv INPUT_FILE subrip /removetextforhi"
```

### Using Other CLI Tools

SubWhisper supports using any command-line subtitle processor by passing the appropriate command:

```bash
# Using subedit (a BASH-based subtitle editor)
python subwhisper.py video.mp4 --post-process "subedit -i INPUT_FILE -c -k '()' -m"

# Custom command (replace INPUT_FILE with your subtitle path)
python subwhisper.py video.mp4 --post-process "your-subtitle-tool --input INPUT_FILE --output INPUT_FILE --option1 value1"
```

The placeholder `INPUT_FILE` will be automatically replaced with the path to the generated subtitle file.

## Performance Notes

- The first run will download the selected Whisper model, which may take some time
- Processing long videos with larger models ("medium" or "large") requires significant RAM and may take a while
- A GPU will significantly speed up the processing time if available
- For batch processing of many files, consider using the "tiny" or "base" model for faster processing

## Error Handling

The script includes robust error handling:
- FFmpeg errors are properly caught and reported
- Temporary files are cleaned up even if processing is interrupted
- Each video in batch mode is processed independently, so one failure won't stop the entire batch
- Detailed error messages help diagnose issues 
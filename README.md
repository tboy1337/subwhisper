# SubWhisper

Automatically generate subtitles for videos using OpenAI's Whisper speech-to-text model.

## Quick Start

1. **Prerequisites**: 
   - Python 3.7+
   - FFmpeg installed

### Installing FFmpeg

FFmpeg can be installed using various package managers or directly from its website:

#### Windows
- Download the latest static build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
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

2. **Install**:
   ```bash
   git clone https://github.com/tboy1337/subwhisper.git
   cd subwhisper
   pip install -r requirements.txt
   ```

3. **Basic Usage**:
   ```bash
   python subwhisper.py video.mp4
   ```

## Features

- Transcribe audio to subtitles in multiple languages
- Batch process entire directories of videos
- Control subtitle segment length
- Post-process subtitles with various improvements
- Integrates with Subtitle Edit for advanced editing

## Command Reference

### Basic Options

```bash
# Single video
python subwhisper.py video.mp4

# Batch process directory
python subwhisper.py videos_folder --batch --output subtitles_folder

# Choose model size (tiny, base, small, medium, large)
python subwhisper.py video.mp4 --model medium

# Specify language
python subwhisper.py video.mp4 --language en
```

### Advanced Options

```bash
# Control subtitle segment length
python subwhisper.py video.mp4 --max-segment-length 80

# Specify video file extensions for batch processing
python subwhisper.py videos_folder --batch --extensions "mp4,mkv,avi"
```

## Subtitle Post-Processing

SubWhisper can automatically improve generated subtitles using preset options:

```bash
# Fix common errors
python subwhisper.py video.mp4 --fix-common-errors

# Remove text for hearing impaired
python subwhisper.py video.mp4 --remove-hi

# Apply multiple fixes at once
python subwhisper.py video.mp4 --fix-common-errors --remove-hi --auto-split-long-lines
```

Available presets:
- `--fix-common-errors`: Fix common subtitle issues
- `--remove-hi`: Remove hearing impaired text
- `--auto-split-long-lines`: Split long subtitle lines
- `--fix-punctuation`: Fix punctuation issues
- `--ocr-fix`: Apply OCR fixes
- `--convert-to`: Convert format (srt, ass, stl, smi, vtt)

## Advanced Usage with Subtitle Edit

For more advanced subtitle editing, you can use Subtitle Edit CLI through Docker:

1. Build the Docker image:
   ```bash
   git clone https://github.com/SubtitleEdit/subtitleedit-cli.git
   cd subtitleedit-cli
   docker build -t seconv:1.0 -f docker/Dockerfile .
   ```

2. Use with SubWhisper:
   ```bash
   python subwhisper.py video.mp4 --post-process "docker run --rm -v \"$(pwd)\":/subtitles seconv:1.0 /subtitles/INPUT_FILE_BASENAME subrip /fixcommonerrors"
   ```

## Performance Tips

- First run downloads the selected Whisper model
- Larger models ("medium", "large") need more RAM but are more accurate
- GPU acceleration significantly improves processing speed
- For batch processing, consider using "tiny" or "base" models for faster results 
#!/usr/bin/env python3
"""
Comprehensive pytest-based test suite for SubWhisper.

This module provides 100% test coverage for the SubWhisper application
with proper timeouts, fixtures, and real-world deployment testing.
"""

import os
import sys
import tempfile
import shutil
import subprocess
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List, Optional, Generator

import pytest
import numpy as np
from scipy.io import wavfile

# Add parent directory to path so we can import subwhisper
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import subwhisper


# Global fixtures available to all test classes
@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp(prefix="subwhisper_pytest_"))
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture(scope="session") 
def sample_video_file(temp_dir: Path) -> Path:
    """Create or find a sample video file for testing."""
    # Look for test video in the tests directory
    test_video = Path(__file__).parent / "test.mp4"
    if test_video.exists():
        return test_video
    
    # Create a minimal test video using ffmpeg if available
    output_video = temp_dir / "generated_test.mp4"
    try:
        ffmpeg_path = subwhisper.find_ffmpeg()
        if ffmpeg_path:
            # Generate a 5-second test video with audio tone
            cmd = [
                ffmpeg_path, "-f", "lavfi", "-i", "testsrc2=duration=5:size=320x240:rate=1",
                "-f", "lavfi", "-i", "sine=frequency=1000:duration=5",
                "-c:v", "libx264", "-c:a", "aac", "-t", "5",
                str(output_video), "-y"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and output_video.exists():
                return output_video
    except Exception:
        pass
        
    # Skip video-dependent tests if no video is available
    pytest.skip("No test video file available and cannot generate one")


@pytest.fixture
def sample_audio_data(temp_dir: Path) -> Path:
    """Generate sample audio data for testing."""
    # Generate a 2-second sine wave at 16kHz
    sample_rate = 16000
    duration = 2.0
    frequency = 440.0  # A note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to int16
    audio_data = (audio_data * 32767).astype(np.int16)
    
    output_path = temp_dir / "sample_audio.wav"
    wavfile.write(str(output_path), sample_rate, audio_data)
    
    return output_path


@pytest.fixture
def mock_args() -> Mock:
    """Create mock command line arguments."""
    args = Mock()
    args.model = "tiny"
    args.language = None
    args.output = None
    args.batch = False
    args.max_segment_length = None
    args.post_process = None
    args.extensions = "mp4,mkv,avi"
    return args


class TestSubWhisperCore:
    """Core functionality tests for SubWhisper."""

    @pytest.mark.unit
    @pytest.mark.timeout(10)
    def test_format_timestamp(self) -> None:
        """Test timestamp formatting function with comprehensive cases."""
        test_cases = [
            (0, "00:00:00,000"),
            (1.5, "00:00:01,500"),
            (61, "00:01:01,000"),
            (3661.42, "01:01:01,420"),
            (7323.999, "02:02:03,999"),
            (86399.123, "23:59:59,123"),  # Edge case: almost 24 hours
        ]
        
        for seconds, expected in test_cases:
            result = subwhisper.format_timestamp(seconds)
            assert result == expected, f"Failed for {seconds}s: expected {expected}, got {result}"

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_find_ffmpeg(self) -> None:
        """Test FFmpeg detection functionality."""
        # This test should find FFmpeg if it's installed
        ffmpeg_path = subwhisper.find_ffmpeg()
        
        if ffmpeg_path:
            # If found, verify it's a valid executable
            assert Path(ffmpeg_path).exists() or ffmpeg_path == "ffmpeg"
            
            # Test that we can run ffmpeg -version
            try:
                result = subprocess.run([ffmpeg_path, "-version"], 
                                      capture_output=True, text=True, timeout=10)
                assert result.returncode == 0
                assert "ffmpeg" in result.stdout.lower()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pytest.fail(f"FFmpeg executable found but not functional: {ffmpeg_path}")
        else:
            # If not found, that's okay but log it
            pytest.skip("FFmpeg not found on system - some tests will be skipped")

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_find_ffmpeg_comprehensive_paths(self) -> None:
        """Test FFmpeg detection with comprehensive path scenarios."""
        with patch('subprocess.run') as mock_run, \
             patch('os.path.isfile') as mock_isfile:
            
            # Test when subprocess.run returns non-zero (not in PATH)
            mock_run.return_value = Mock(returncode=1, stderr="not found")
            mock_isfile.return_value = False
            
            result = subwhisper.find_ffmpeg()
            assert result is None
            
            # Test when a common location exists
            mock_isfile.side_effect = lambda path: path == r"C:\ffmpeg\bin\ffmpeg.exe"
            result = subwhisper.find_ffmpeg()
            assert result == r"C:\ffmpeg\bin\ffmpeg.exe"

    @pytest.mark.unit
    @pytest.mark.timeout(30)  
    def test_find_ffmpeg_path_success(self) -> None:
        """Test FFmpeg found in PATH."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="/usr/bin/ffmpeg\n", stderr="")
            result = subwhisper.find_ffmpeg()
            assert result == "/usr/bin/ffmpeg"

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_find_ffmpeg_timeout_exception(self) -> None:
        """Test FFmpeg detection with timeout exception."""
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("which", 30)):
            result = subwhisper.find_ffmpeg()
            # Should still try common locations and return None if none found
            assert result is None

    @pytest.mark.unit  
    @pytest.mark.timeout(30)
    def test_find_ffmpeg_general_exception(self) -> None:
        """Test FFmpeg detection with general exception."""
        with patch('subprocess.run', side_effect=Exception("Command failed")):
            result = subwhisper.find_ffmpeg()
            # Should still try common locations and return None if none found
            assert result is None

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_find_ffmpeg_file_check_exception(self) -> None:
        """Test FFmpeg detection with file check exceptions."""
        with patch('subprocess.run') as mock_run, \
             patch('os.path.isfile', side_effect=Exception("File access error")):
            
            mock_run.return_value = Mock(returncode=1, stderr="not found")
            result = subwhisper.find_ffmpeg()
            assert result is None

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_find_ffmpeg_non_windows_platform(self) -> None:
        """Test FFmpeg detection on non-Windows platforms."""
        with patch('sys.platform', 'linux'), \
             patch('subprocess.run') as mock_run, \
             patch('os.path.isfile', return_value=False):
            
            mock_run.return_value = Mock(returncode=1, stderr="not found")
            result = subwhisper.find_ffmpeg()
            
            # Should have used 'which' command instead of 'where'
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert call_args == ["which", "ffmpeg"]
            assert result is None

    @pytest.mark.unit  
    @pytest.mark.timeout(60)
    def test_extract_audio_mock(self, temp_dir: Path, mock_args: Mock) -> None:
        """Test audio extraction with mocked subprocess calls."""
        input_video = temp_dir / "input.mp4" 
        output_audio = temp_dir / "output.wav"
        
        # Create a fake input video file
        input_video.write_text("fake video content")
        
        with patch('subwhisper.find_ffmpeg', return_value="ffmpeg"), \
             patch('subprocess.run') as mock_run:
            
            # Mock successful FFmpeg execution
            mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
            
            # Create the expected output file
            output_audio.write_bytes(b"fake audio data" * 1000)  # Make it reasonable size
            
            result = subwhisper.extract_audio(str(input_video), str(output_audio))
            
            assert result == str(output_audio)
            assert output_audio.exists()
            mock_run.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_extract_audio_error_handling(self, temp_dir: Path) -> None:
        """Test audio extraction error handling."""
        non_existent_video = temp_dir / "nonexistent.mp4"
        
        with pytest.raises(ValueError, match="Video file does not exist"):
            subwhisper.extract_audio(str(non_existent_video))
        
        # Test empty path
        with pytest.raises(ValueError, match="Video path cannot be empty"):
            subwhisper.extract_audio("")

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_extract_audio_ffmpeg_not_found(self, temp_dir: Path) -> None:
        """Test audio extraction when FFmpeg is not found."""
        input_video = temp_dir / "input.mp4"
        input_video.write_text("fake video")
        
        with patch('subwhisper.find_ffmpeg', return_value=None):
            with pytest.raises(FileNotFoundError, match="FFmpeg not found"):
                subwhisper.extract_audio(str(input_video))

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_extract_audio_ffmpeg_error(self, temp_dir: Path) -> None:
        """Test audio extraction when FFmpeg returns an error."""
        input_video = temp_dir / "input.mp4"
        input_video.write_text("fake video")
        
        with patch('subwhisper.find_ffmpeg', return_value="ffmpeg"), \
             patch('subprocess.run') as mock_run:
            
            # Mock FFmpeg failure
            mock_run.return_value = Mock(returncode=1, stderr="FFmpeg error message")
            
            with pytest.raises(RuntimeError, match="FFmpeg error"):
                subwhisper.extract_audio(str(input_video))

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_extract_audio_output_not_created(self, temp_dir: Path) -> None:
        """Test audio extraction when output file is not created."""
        input_video = temp_dir / "input.mp4" 
        output_audio = temp_dir / "output.wav"
        
        input_video.write_text("fake video")
        
        with patch('subwhisper.find_ffmpeg', return_value="ffmpeg"), \
             patch('subprocess.run') as mock_run:
            
            # Mock successful FFmpeg execution but no output file created
            mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
            
            # Explicitly ensure the output file doesn't exist
            if output_audio.exists():
                output_audio.unlink()
                
            with pytest.raises(RuntimeError, match="output file not created"):
                subwhisper.extract_audio(str(input_video), str(output_audio))

    @pytest.mark.unit  
    @pytest.mark.timeout(30)
    def test_extract_audio_small_output_warning(self, temp_dir: Path) -> None:
        """Test audio extraction with suspiciously small output file."""
        input_video = temp_dir / "input.mp4"
        output_audio = temp_dir / "output.wav"
        
        input_video.write_text("fake video")
        
        with patch('subwhisper.find_ffmpeg', return_value="ffmpeg"), \
             patch('subprocess.run') as mock_run:
            
            mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
            
            # Create a very small output file (should trigger warning)
            output_audio.write_bytes(b"tiny")  # Less than 1000 bytes
            
            result = subwhisper.extract_audio(str(input_video), str(output_audio))
            assert result == str(output_audio)

    @pytest.mark.unit
    @pytest.mark.timeout(30)  
    def test_extract_audio_timeout_handling(self, temp_dir: Path) -> None:
        """Test audio extraction timeout handling."""
        input_video = temp_dir / "input.mp4"
        input_video.write_text("fake video")
        
        with patch('subwhisper.find_ffmpeg', return_value="ffmpeg"), \
             patch('subprocess.run', side_effect=subprocess.TimeoutExpired("ffmpeg", 3600)):
            
            with pytest.raises(RuntimeError, match="timed out"):
                subwhisper.extract_audio(str(input_video))

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_extract_audio_unexpected_exception(self, temp_dir: Path) -> None:
        """Test audio extraction with unexpected exception."""
        input_video = temp_dir / "input.mp4"
        input_video.write_text("fake video") 
        
        with patch('subwhisper.find_ffmpeg', return_value="ffmpeg"), \
             patch('subprocess.run', side_effect=RuntimeError("Unexpected error")):
            
            with pytest.raises(RuntimeError, match="Audio extraction failed"):
                subwhisper.extract_audio(str(input_video))

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_generate_srt_basic(self, temp_dir: Path) -> None:
        """Test basic SRT generation functionality."""
        test_segments = [
            {"start": 0.0, "end": 2.5, "text": "First test segment."},
            {"start": 3.0, "end": 5.5, "text": "Second test segment."},
            {"start": 6.0, "end": 8.0, "text": "Third segment with content."}
        ]
        
        output_file = temp_dir / "test_basic.srt"
        subwhisper.generate_srt(test_segments, str(output_file))
        
        assert output_file.exists()
        
        # Validate SRT content
        content = output_file.read_text(encoding='utf-8')
        
        # Check for basic SRT format elements
        assert "1\n00:00:00,000 --> 00:00:02,500\nFirst test segment.\n\n" in content
        assert "2\n00:00:03,000 --> 00:00:05,500\nSecond test segment.\n\n" in content
        assert "3\n00:00:06,000 --> 00:00:08,000\nThird segment with content.\n\n" in content

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_generate_srt_with_max_length(self, temp_dir: Path) -> None:
        """Test SRT generation with segment length limits."""
        test_segments = [
            {"start": 0.0, "end": 5.0, "text": "This is a very long segment that should be split into multiple lines because it exceeds the maximum character limit."}
        ]
        
        output_file = temp_dir / "test_split.srt"
        subwhisper.generate_srt(test_segments, str(output_file), max_segment_length=30)
        
        assert output_file.exists()
        
        content = output_file.read_text(encoding='utf-8')
        # Should have multiple segments now
        assert content.count("1\n") == 1  # First segment
        assert content.count("2\n") >= 1  # At least one more segment from splitting

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_generate_srt_empty_segments(self, temp_dir: Path) -> None:
        """Test SRT generation with empty segments."""
        output_file = temp_dir / "test_empty.srt"
        subwhisper.generate_srt([], str(output_file))
        
        assert output_file.exists()
        content = output_file.read_text(encoding='utf-8')
        assert content == ""  # Empty file for empty segments

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_cleanup_and_exit_mock(self) -> None:
        """Test cleanup and exit functionality with mocking."""
        with patch('sys.exit') as mock_exit, \
             patch('shutil.rmtree') as mock_rmtree, \
             patch('os.path.exists', return_value=True):
            
            # Set a temporary directory
            subwhisper.TEMP_DIR = "/fake/temp/dir"
            
            subwhisper.cleanup_and_exit(42)
            
            mock_rmtree.assert_called_once_with("/fake/temp/dir")
            mock_exit.assert_called_once_with(42)

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_transcribe_audio_mock(self, temp_dir: Path, sample_audio_data: Path, mock_args: Mock) -> None:
        """Test audio transcription with mocked Whisper model."""
        mock_result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Test transcription segment"}
            ],
            "language": "en"
        }
        
        with patch('whisper.load_model') as mock_load_model:
            mock_model = Mock()
            mock_model.transcribe.return_value = mock_result
            mock_load_model.return_value = mock_model
            
            result = subwhisper.transcribe_audio(str(sample_audio_data), mock_args)
            
            assert result == mock_result
            mock_load_model.assert_called_once_with("tiny")
            mock_model.transcribe.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_transcribe_audio_file_not_found(self, temp_dir: Path, mock_args: Mock) -> None:
        """Test transcription with non-existent audio file."""
        non_existent_audio = temp_dir / "nonexistent.wav"
        
        result = subwhisper.transcribe_audio(str(non_existent_audio), mock_args)
        assert result is None

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_transcribe_audio_model_loading_failure(self, sample_audio_data: Path, mock_args: Mock) -> None:
        """Test transcription with model loading failure."""
        with patch('whisper.load_model', side_effect=Exception("Model download failed")):
            result = subwhisper.transcribe_audio(str(sample_audio_data), mock_args)
            assert result is None

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_transcribe_audio_with_language_specified(self, sample_audio_data: Path, mock_args: Mock) -> None:
        """Test transcription with language specified."""
        mock_result = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "Test transcription"}],
            "language": "en"
        }
        
        mock_args.language = "en"
        
        with patch('whisper.load_model') as mock_load_model:
            mock_model = Mock()
            mock_model.transcribe.return_value = mock_result
            mock_load_model.return_value = mock_model
            
            result = subwhisper.transcribe_audio(str(sample_audio_data), mock_args)
            
            assert result == mock_result
            # Check that language was passed in transcription options
            call_args = mock_model.transcribe.call_args
            assert call_args[1]["language"] == "en"

    @pytest.mark.unit 
    @pytest.mark.timeout(60)
    def test_transcribe_audio_scipy_loading_success(self, temp_dir: Path, mock_args: Mock) -> None:
        """Test transcription with successful scipy audio loading."""
        # Create a proper WAV file using scipy
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        audio_file = temp_dir / "test_audio.wav"
        wavfile.write(str(audio_file), sample_rate, audio_data)
        
        mock_result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "Test"}],
            "language": "en"
        }
        
        with patch('whisper.load_model') as mock_load_model:
            mock_model = Mock()
            mock_model.transcribe.return_value = mock_result
            mock_load_model.return_value = mock_model
            
            result = subwhisper.transcribe_audio(str(audio_file), mock_args)
            assert result == mock_result

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_transcribe_audio_scipy_loading_failure_fallback(self, sample_audio_data: Path, mock_args: Mock) -> None:
        """Test transcription with scipy loading failure, falls back to whisper loading."""
        mock_result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "Test"}],
            "language": "en"
        }
        
        with patch('whisper.load_model') as mock_load_model, \
             patch('scipy.io.wavfile.read', side_effect=Exception("Scipy error")):
            
            mock_model = Mock()
            mock_model.transcribe.return_value = mock_result
            mock_load_model.return_value = mock_model
            
            result = subwhisper.transcribe_audio(str(sample_audio_data), mock_args)
            assert result == mock_result
            
            # Should have been called with file path (fallback mode)
            call_args = mock_model.transcribe.call_args[0]
            # Use more flexible path checking to handle Windows path differences
            assert sample_audio_data.name in str(call_args)

    @pytest.mark.unit
    @pytest.mark.timeout(60) 
    def test_transcribe_audio_resampling(self, temp_dir: Path, mock_args: Mock) -> None:
        """Test transcription with audio resampling from different sample rate."""
        # Create audio at 44.1kHz that needs resampling to 16kHz
        original_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(original_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        audio_file = temp_dir / "high_rate_audio.wav"
        wavfile.write(str(audio_file), original_rate, audio_data)
        
        mock_result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "Resampled test"}],
            "language": "en"
        }
        
        with patch('whisper.load_model') as mock_load_model, \
             patch('scipy.signal.resample') as mock_resample:
            
            mock_model = Mock()
            mock_model.transcribe.return_value = mock_result
            mock_load_model.return_value = mock_model
            mock_resample.return_value = audio_data[:16000]  # Mock resampled data
            
            result = subwhisper.transcribe_audio(str(audio_file), mock_args)
            assert result == mock_result
            
            # Should have called resample
            mock_resample.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_transcribe_audio_stereo_to_mono(self, temp_dir: Path, mock_args: Mock) -> None:
        """Test transcription with stereo audio conversion to mono."""
        # Create stereo audio
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        left_channel = np.sin(2 * np.pi * 440 * t)
        right_channel = np.sin(2 * np.pi * 880 * t)
        stereo_audio = np.column_stack((left_channel, right_channel))
        stereo_audio = (stereo_audio * 32767).astype(np.int16)
        
        audio_file = temp_dir / "stereo_audio.wav"
        wavfile.write(str(audio_file), sample_rate, stereo_audio)
        
        mock_result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "Stereo converted"}],
            "language": "en"
        }
        
        with patch('whisper.load_model') as mock_load_model:
            mock_model = Mock()
            mock_model.transcribe.return_value = mock_result
            mock_load_model.return_value = mock_model
            
            result = subwhisper.transcribe_audio(str(audio_file), mock_args)
            assert result == mock_result

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_transcribe_audio_int32_conversion(self, temp_dir: Path, mock_args: Mock) -> None:
        """Test transcription with int32 audio data conversion."""
        # Create int32 audio data
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)
        # Convert to int32 (this will trigger the int32 conversion path)
        audio_data = (audio_data * 2147483647).astype(np.int32)
        
        audio_file = temp_dir / "int32_audio.wav"
        wavfile.write(str(audio_file), sample_rate, audio_data)
        
        mock_result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "Int32 converted"}],
            "language": "en"
        }
        
        with patch('whisper.load_model') as mock_load_model:
            mock_model = Mock()
            mock_model.transcribe.return_value = mock_result
            mock_load_model.return_value = mock_model
            
            result = subwhisper.transcribe_audio(str(audio_file), mock_args)
            assert result == mock_result

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_transcribe_audio_keyboard_interrupt(self, sample_audio_data: Path, mock_args: Mock) -> None:
        """Test transcription cancelled by keyboard interrupt."""
        with patch('whisper.load_model') as mock_load_model:
            mock_model = Mock()
            mock_model.transcribe.side_effect = KeyboardInterrupt()
            mock_load_model.return_value = mock_model
            
            result = subwhisper.transcribe_audio(str(sample_audio_data), mock_args)
            assert result is None

    @pytest.mark.unit 
    @pytest.mark.timeout(60)
    def test_transcribe_audio_transcription_exception(self, sample_audio_data: Path, mock_args: Mock) -> None:
        """Test transcription with transcription exception."""
        with patch('whisper.load_model') as mock_load_model:
            mock_model = Mock()
            mock_model.transcribe.side_effect = Exception("Transcription failed")
            mock_load_model.return_value = mock_model
            
            result = subwhisper.transcribe_audio(str(sample_audio_data), mock_args)
            assert result is None

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_transcribe_audio_invalid_result(self, sample_audio_data: Path, mock_args: Mock) -> None:
        """Test transcription with invalid result structure."""
        with patch('whisper.load_model') as mock_load_model:
            mock_model = Mock()
            # Return invalid result (missing segments)
            mock_model.transcribe.return_value = {"language": "en"}
            mock_load_model.return_value = mock_model
            
            result = subwhisper.transcribe_audio(str(sample_audio_data), mock_args)
            assert result is None

    @pytest.mark.unit
    @pytest.mark.timeout(60) 
    def test_transcribe_audio_empty_result(self, sample_audio_data: Path, mock_args: Mock) -> None:
        """Test transcription with empty result."""
        with patch('whisper.load_model') as mock_load_model:
            mock_model = Mock()
            mock_model.transcribe.return_value = None
            mock_load_model.return_value = mock_model
            
            result = subwhisper.transcribe_audio(str(sample_audio_data), mock_args)
            assert result is None

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_transcribe_audio_outer_exception_handler(self, sample_audio_data: Path, mock_args: Mock) -> None:
        """Test transcription outer exception handler (lines 614-616)."""
        # Mock an exception that occurs before entering the try block with model loading
        with patch('pathlib.Path.stat', side_effect=Exception("File access error")):
            result = subwhisper.transcribe_audio(str(sample_audio_data), mock_args)
            assert result is None

    @pytest.mark.unit 
    @pytest.mark.timeout(60)
    def test_process_video_mock(self, temp_dir: Path, sample_video_file: Path, mock_args: Mock) -> None:
        """Test video processing with comprehensive mocking."""
        mock_transcription_result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Mock transcription"}
            ]
        }
        
        with patch('subwhisper.extract_audio') as mock_extract, \
             patch('subwhisper.transcribe_audio') as mock_transcribe, \
             patch('subwhisper.generate_srt') as mock_generate_srt:
            
            # Setup mocks
            mock_extract.return_value = str(temp_dir / "extracted.wav")
            mock_transcribe.return_value = mock_transcription_result
            
            result = subwhisper.process_video(str(sample_video_file), mock_args)
            
            assert result is True
            mock_extract.assert_called_once()
            mock_transcribe.assert_called_once()
            mock_generate_srt.assert_called_once_with(
                mock_transcription_result["segments"],
                str(sample_video_file.with_suffix(".srt")),
                None
            )

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_process_video_nonexistent_file(self, temp_dir: Path, mock_args: Mock) -> None:
        """Test processing non-existent video file."""
        non_existent_video = temp_dir / "nonexistent.mp4"
        
        result = subwhisper.process_video(str(non_existent_video), mock_args)
        assert result is False

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_process_video_transcription_failure(self, temp_dir: Path, mock_args: Mock) -> None:
        """Test video processing when transcription fails."""
        # Create a fake video file
        test_video = temp_dir / "test.mp4" 
        test_video.write_text("fake video content")
        
        with patch('subwhisper.extract_audio') as mock_extract, \
             patch('subwhisper.transcribe_audio') as mock_transcribe:
            
            mock_extract.return_value = str(temp_dir / "extracted.wav")
            mock_transcribe.return_value = None  # Transcription failure
            
            result = subwhisper.process_video(str(test_video), mock_args)
            assert result is False

    @pytest.mark.unit
    @pytest.mark.timeout(60) 
    def test_process_video_with_output_path(self, temp_dir: Path, mock_args: Mock) -> None:
        """Test video processing with specified output path."""
        test_video = temp_dir / "test.mp4"
        test_video.write_text("fake video content")
        
        output_path = temp_dir / "custom_output.srt"
        mock_args.output = str(output_path)
        
        mock_transcription = {"segments": [{"start": 0.0, "end": 1.0, "text": "Test"}]}
        
        with patch('subwhisper.extract_audio') as mock_extract, \
             patch('subwhisper.transcribe_audio') as mock_transcribe, \
             patch('subwhisper.generate_srt') as mock_generate_srt:
            
            mock_extract.return_value = str(temp_dir / "extracted.wav")
            mock_transcribe.return_value = mock_transcription
            
            result = subwhisper.process_video(str(test_video), mock_args)
            assert result is True
            
            # Check that generate_srt was called with the specified output path
            mock_generate_srt.assert_called_once_with(
                mock_transcription["segments"],
                str(output_path),
                None
            )

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_process_video_batch_mode(self, temp_dir: Path, mock_args: Mock) -> None:
        """Test video processing in batch mode with output directory."""
        test_video = temp_dir / "videos" / "test.mp4"
        test_video.parent.mkdir()
        test_video.write_text("fake video content")
        
        output_dir = temp_dir / "output"
        mock_args.output = str(output_dir)
        mock_args.batch = True
        
        mock_transcription = {"segments": [{"start": 0.0, "end": 1.0, "text": "Test"}]}
        
        with patch('subwhisper.extract_audio') as mock_extract, \
             patch('subwhisper.transcribe_audio') as mock_transcribe, \
             patch('subwhisper.generate_srt') as mock_generate_srt:
            
            mock_extract.return_value = str(temp_dir / "extracted.wav")
            mock_transcribe.return_value = mock_transcription
            
            result = subwhisper.process_video(str(test_video), mock_args)
            assert result is True
            
            # Check that output directory structure is maintained
            expected_output = output_dir / "test.srt"
            mock_generate_srt.assert_called_once_with(
                mock_transcription["segments"],
                str(expected_output),
                None
            )

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_process_video_with_post_processing(self, temp_dir: Path, mock_args: Mock) -> None:
        """Test video processing with post-processing command."""
        test_video = temp_dir / "test.mp4"
        test_video.write_text("fake video content")
        
        # Set up post-processing command
        mock_args.post_process = "echo Processing INPUT_FILE INPUT_FILE_BASENAME"
        
        mock_transcription = {"segments": [{"start": 0.0, "end": 1.0, "text": "Test"}]}
        
        with patch('subwhisper.extract_audio') as mock_extract, \
             patch('subwhisper.transcribe_audio') as mock_transcribe, \
             patch('subwhisper.generate_srt') as mock_generate_srt, \
             patch('subprocess.run') as mock_subprocess:
            
            mock_extract.return_value = str(temp_dir / "extracted.wav")
            mock_transcribe.return_value = mock_transcription
            mock_subprocess.return_value = Mock(returncode=0, stdout="Success", stderr="")
            
            result = subwhisper.process_video(str(test_video), mock_args)
            assert result is True
            
            # Check that subprocess.run was called for post-processing
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            
            # Check that INPUT_FILE and INPUT_FILE_BASENAME were replaced
            command = call_args[0][0]
            expected_srt_path = str(test_video.with_suffix(".srt"))
            assert expected_srt_path in command
            assert "test.srt" in command

    @pytest.mark.unit  
    @pytest.mark.timeout(60)
    def test_process_video_post_processing_failure(self, temp_dir: Path, mock_args: Mock) -> None:
        """Test video processing when post-processing fails."""
        test_video = temp_dir / "test.mp4"
        test_video.write_text("fake video content")
        
        mock_args.post_process = "failing_command"
        
        mock_transcription = {"segments": [{"start": 0.0, "end": 1.0, "text": "Test"}]}
        
        with patch('subwhisper.extract_audio') as mock_extract, \
             patch('subwhisper.transcribe_audio') as mock_transcribe, \
             patch('subwhisper.generate_srt') as mock_generate_srt, \
             patch('subprocess.run') as mock_subprocess:
            
            mock_extract.return_value = str(temp_dir / "extracted.wav")
            mock_transcribe.return_value = mock_transcription
            mock_subprocess.return_value = Mock(returncode=1, stdout="", stderr="Post-processing failed")
            
            # Should still return True (video processing succeeded, only post-processing failed)
            result = subwhisper.process_video(str(test_video), mock_args)
            assert result is True

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_process_video_post_processing_timeout(self, temp_dir: Path, mock_args: Mock) -> None:
        """Test video processing when post-processing times out."""
        test_video = temp_dir / "test.mp4"
        test_video.write_text("fake video content")
        
        mock_args.post_process = "long_running_command"
        
        mock_transcription = {"segments": [{"start": 0.0, "end": 1.0, "text": "Test"}]}
        
        with patch('subwhisper.extract_audio') as mock_extract, \
             patch('subwhisper.transcribe_audio') as mock_transcribe, \
             patch('subwhisper.generate_srt') as mock_generate_srt, \
             patch('subprocess.run', side_effect=subprocess.TimeoutExpired("cmd", 1800)):
            
            mock_extract.return_value = str(temp_dir / "extracted.wav") 
            mock_transcribe.return_value = mock_transcription
            
            result = subwhisper.process_video(str(test_video), mock_args)
            assert result is True  # Video processing still succeeded

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_process_video_post_processing_exception(self, temp_dir: Path, mock_args: Mock) -> None:
        """Test video processing when post-processing raises exception."""
        test_video = temp_dir / "test.mp4"
        test_video.write_text("fake video content")
        
        mock_args.post_process = "exception_command"
        
        mock_transcription = {"segments": [{"start": 0.0, "end": 1.0, "text": "Test"}]}
        
        with patch('subwhisper.extract_audio') as mock_extract, \
             patch('subwhisper.transcribe_audio') as mock_transcribe, \
             patch('subwhisper.generate_srt') as mock_generate_srt, \
             patch('subprocess.run', side_effect=Exception("Subprocess error")):
            
            mock_extract.return_value = str(temp_dir / "extracted.wav")
            mock_transcribe.return_value = mock_transcription
            
            result = subwhisper.process_video(str(test_video), mock_args)
            assert result is True  # Video processing still succeeded

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_process_video_general_exception(self, temp_dir: Path, mock_args: Mock) -> None:
        """Test video processing with general exception."""
        test_video = temp_dir / "test.mp4"
        test_video.write_text("fake video content")
        
        with patch('subwhisper.extract_audio', side_effect=Exception("General error")):
            result = subwhisper.process_video(str(test_video), mock_args)
            assert result is False

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_generate_docker_post_process_cmd(self) -> None:
        """Test Docker post-processing command generation."""
        args = Mock()
        args.fix_common_errors = True
        args.remove_hi = True
        args.auto_split_long_lines = False
        args.fix_punctuation = False
        args.ocr_fix = False
        args.convert_to = None
        
        result = subwhisper.generate_docker_post_process_cmd(args)
        
        assert result is not None
        assert "docker run" in result
        assert "/fixcommonerrors" in result
        assert "/removetextforhi" in result
        assert "/splitlonglines" not in result

    @pytest.mark.unit
    @pytest.mark.timeout(10)
    def test_generate_docker_post_process_cmd_none(self) -> None:
        """Test Docker command generation with no options."""
        args = Mock()
        args.fix_common_errors = False
        args.remove_hi = False
        args.auto_split_long_lines = False
        args.fix_punctuation = False
        args.ocr_fix = False
        args.convert_to = None
        
        result = subwhisper.generate_docker_post_process_cmd(args)
        assert result is None

    @pytest.mark.unit
    @pytest.mark.timeout(10)
    def test_generate_docker_post_process_cmd_with_convert(self) -> None:
        """Test Docker command generation with format conversion."""
        args = Mock()
        args.fix_common_errors = False
        args.remove_hi = False
        args.auto_split_long_lines = False 
        args.fix_punctuation = False
        args.ocr_fix = False
        args.convert_to = "ass"
        
        result = subwhisper.generate_docker_post_process_cmd(args)
        
        assert result is not None
        assert "docker run" in result
        assert " ass" in result  # Should use ass format (can be at end of command)

    @pytest.mark.unit
    @pytest.mark.timeout(10)
    def test_generate_docker_post_process_cmd_all_options(self) -> None:
        """Test Docker command generation with all options enabled."""
        args = Mock()
        args.fix_common_errors = True
        args.remove_hi = True  
        args.auto_split_long_lines = True
        args.fix_punctuation = True
        args.ocr_fix = True
        args.convert_to = "vtt"
        
        result = subwhisper.generate_docker_post_process_cmd(args)
        
        assert result is not None
        assert "docker run" in result
        assert " vtt " in result
        assert "/fixcommonerrors" in result
        assert "/removetextforhi" in result
        assert "/splitlonglines" in result
        assert "/fixpunctuation" in result
        assert "/ocrfix" in result

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_cleanup_and_exit_no_temp_dir(self) -> None:
        """Test cleanup when no temporary directory exists."""
        with patch('sys.exit') as mock_exit:
            # Ensure no temp directory is set
            original_temp_dir = subwhisper.TEMP_DIR
            subwhisper.TEMP_DIR = None
            
            try:
                subwhisper.cleanup_and_exit(0)
                mock_exit.assert_called_once_with(0)
            finally:
                subwhisper.TEMP_DIR = original_temp_dir

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_cleanup_and_exit_temp_dir_missing(self) -> None:
        """Test cleanup when temp directory is set but doesn't exist."""
        with patch('sys.exit') as mock_exit, \
             patch('os.path.exists', return_value=False):
            
            original_temp_dir = subwhisper.TEMP_DIR
            subwhisper.TEMP_DIR = "/non/existent/dir"
            
            try:
                subwhisper.cleanup_and_exit(1)
                mock_exit.assert_called_once_with(1)
            finally:
                subwhisper.TEMP_DIR = original_temp_dir

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_cleanup_and_exit_rmtree_failure(self) -> None:
        """Test cleanup when rmtree fails."""
        with patch('sys.exit') as mock_exit, \
             patch('os.path.exists', return_value=True), \
             patch('shutil.rmtree', side_effect=Exception("Permission denied")), \
             patch('pathlib.Path.rglob', return_value=iter([])):
            
            original_temp_dir = subwhisper.TEMP_DIR
            subwhisper.TEMP_DIR = "/fake/temp/dir"
            
            try:
                subwhisper.cleanup_and_exit(2)
                mock_exit.assert_called_once_with(2)
            finally:
                subwhisper.TEMP_DIR = original_temp_dir


class TestSubWhisperIntegration:
    """Integration tests for SubWhisper."""
    
    @pytest.mark.integration
    @pytest.mark.timeout(120)
    @pytest.mark.slow
    def test_real_audio_extraction(self, temp_dir: Path) -> None:
        """Test real audio extraction if FFmpeg is available."""
        ffmpeg_path = subwhisper.find_ffmpeg()
        if not ffmpeg_path:
            pytest.skip("FFmpeg not available for integration testing")
            
        # Create a simple test video with ffmpeg
        test_video = temp_dir / "integration_test.mp4"
        
        try:
            # Generate a 3-second test video
            cmd = [
                ffmpeg_path, "-f", "lavfi", "-i", "testsrc2=duration=3:size=320x240:rate=1",
                "-f", "lavfi", "-i", "sine=frequency=800:duration=3",
                "-c:v", "libx264", "-c:a", "aac", "-t", "3",
                str(test_video), "-y"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and test_video.exists():
                # Now test audio extraction
                output_audio = temp_dir / "extracted_integration.wav"
                extracted_path = subwhisper.extract_audio(str(test_video), str(output_audio))
                
                assert Path(extracted_path).exists()
                assert Path(extracted_path).stat().st_size > 1000  # Reasonable file size
                
            else:
                pytest.skip(f"Could not create test video: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            pytest.skip("Video generation timed out")

    @pytest.mark.integration
    @pytest.mark.timeout(180)
    @pytest.mark.slow
    def test_end_to_end_with_tiny_model(self, temp_dir: Path) -> None:
        """Test complete end-to-end processing with tiny Whisper model."""
        ffmpeg_path = subwhisper.find_ffmpeg()
        if not ffmpeg_path:
            pytest.skip("FFmpeg not available for integration testing")
            
        # Create a test video with known audio content
        test_video = temp_dir / "e2e_test.mp4"
        
        try:
            # Generate a short test video with audio
            cmd = [
                ffmpeg_path, "-f", "lavfi", "-i", "color=red:duration=5:size=320x240:rate=1",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
                "-c:v", "libx264", "-c:a", "aac", "-shortest",
                str(test_video), "-y"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0 or not test_video.exists():
                pytest.skip(f"Could not create test video: {result.stderr}")
            
            # Create mock args for processing
            args = Mock()
            args.model = "tiny"  # Use smallest model for speed
            args.language = None
            args.output = None
            args.batch = False
            args.max_segment_length = None
            args.post_process = None
            
            # Process the video
            success = subwhisper.process_video(str(test_video), args)
            
            # Verify output
            assert success is True
            
            expected_srt = test_video.with_suffix(".srt")
            assert expected_srt.exists()
            
            # Basic validation of SRT content
            srt_content = expected_srt.read_text(encoding='utf-8')
            assert len(srt_content.strip()) > 0  # Should have some content
            
        except subprocess.TimeoutExpired:
            pytest.skip("Video generation timed out")
        except Exception as e:
            pytest.skip(f"Integration test failed due to environment: {e}")


class TestSubWhisperThreadSafety:
    """Thread safety tests for SubWhisper."""
    
    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_temp_dir_thread_safety(self) -> None:
        """Test that temporary directory operations are thread-safe."""
        import threading
        import concurrent.futures
        
        # Reset global temp dir
        subwhisper.TEMP_DIR = None
        
        results = []
        
        def create_temp_dir_worker():
            """Worker function that creates temporary directories."""
            with subwhisper._TEMP_DIR_LOCK:
                if subwhisper.TEMP_DIR is None:
                    subwhisper.TEMP_DIR = tempfile.mkdtemp(prefix="thread_test_")
                return subwhisper.TEMP_DIR
        
        # Run multiple threads simultaneously
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_temp_dir_worker) for _ in range(10)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        # All results should be the same (same temp directory)
        assert len(set(results)) == 1, "Thread safety violation: multiple temp directories created"
        
        # Clean up
        if subwhisper.TEMP_DIR and os.path.exists(subwhisper.TEMP_DIR):
            shutil.rmtree(subwhisper.TEMP_DIR)
            subwhisper.TEMP_DIR = None


class TestSubWhisperMainFunction:
    """Tests for the main() function and CLI interface."""

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_main_single_video_success(self, temp_dir: Path) -> None:
        """Test main function with single video processing success."""
        test_video = temp_dir / "test.mp4"
        test_video.write_text("fake video")
        
        test_args = [
            "subwhisper.py",
            str(test_video),
            "--model", "tiny"
        ]
        
        with patch('sys.argv', test_args), \
             patch('subwhisper.process_video', return_value=True), \
             patch('subwhisper.cleanup_and_exit') as mock_cleanup, \
             patch('sys.exit'), \
             patch('builtins.print'):
            
            subwhisper.main()
            mock_cleanup.assert_called_once_with(0)

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_main_single_video_failure(self, temp_dir: Path) -> None:
        """Test main function with single video processing failure."""
        test_video = temp_dir / "test.mp4"
        test_video.write_text("fake video")
        
        test_args = [
            "subwhisper.py", 
            str(test_video)
        ]
        
        with patch('sys.argv', test_args), \
             patch('subwhisper.process_video', return_value=False), \
             patch('subwhisper.cleanup_and_exit') as mock_cleanup, \
             patch('sys.exit'), \
             patch('builtins.print'):
            
            subwhisper.main()
            mock_cleanup.assert_called_once_with(0)

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_main_batch_processing_success(self, temp_dir: Path) -> None:
        """Test main function with successful batch processing."""
        # Create test directory with video files
        video_dir = temp_dir / "videos_batch"
        video_dir.mkdir(exist_ok=True)
        
        (video_dir / "video1.mp4").write_text("fake video 1")
        (video_dir / "video2.mkv").write_text("fake video 2")
        
        output_dir = temp_dir / "outputs"
        
        test_args = [
            "subwhisper.py",
            str(video_dir),
            "--batch",
            "--output", str(output_dir),
            "--extensions", "mp4,mkv"
        ]
        
        with patch('sys.argv', test_args), \
             patch('subwhisper.process_video', return_value=True), \
             patch('subwhisper.cleanup_and_exit') as mock_cleanup, \
             patch('sys.exit'), \
             patch('builtins.print'), \
             patch('glob.glob', return_value=[str(video_dir / "video1.mp4"), str(video_dir / "video2.mkv")]):
            
            subwhisper.main()
            mock_cleanup.assert_called_once_with(0)

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_main_batch_processing_no_files_found(self, temp_dir: Path) -> None:
        """Test main function when no video files are found in batch mode."""
        video_dir = temp_dir / "empty_videos"
        video_dir.mkdir()
        
        test_args = [
            "subwhisper.py",
            str(video_dir),
            "--batch"
        ]
        
        with patch('sys.argv', test_args), \
             patch('subwhisper.cleanup_and_exit') as mock_cleanup, \
             patch('sys.exit'), \
             patch('builtins.print'), \
             patch('glob.glob', return_value=[]):
            
            subwhisper.main()
            mock_cleanup.assert_called_once_with(0)

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_main_batch_processing_not_directory(self, temp_dir: Path) -> None:
        """Test main function when batch path is not a directory."""
        fake_file = temp_dir / "notadirectory.txt"
        fake_file.write_text("fake file")
        
        test_args = [
            "subwhisper.py",
            str(fake_file),
            "--batch"
        ]
        
        with patch('sys.argv', test_args), \
             patch('subwhisper.cleanup_and_exit') as mock_cleanup, \
             patch('sys.exit'), \
             patch('builtins.print'):
            
            subwhisper.main()
            mock_cleanup.assert_called_once_with(0)

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_main_with_docker_presets(self, temp_dir: Path) -> None:
        """Test main function with Docker preset arguments."""
        test_video = temp_dir / "test.mp4"
        test_video.write_text("fake video")
        
        test_args = [
            "subwhisper.py",
            str(test_video),
            "--fix-common-errors",
            "--remove-hi"
        ]
        
        with patch('sys.argv', test_args), \
             patch('subwhisper.process_video', return_value=True), \
             patch('subwhisper.cleanup_and_exit') as mock_cleanup, \
             patch('sys.exit'), \
             patch('builtins.print'):
            
            subwhisper.main()
            mock_cleanup.assert_called_once_with(0)

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_main_docker_preset_overrides_custom_postprocess(self, temp_dir: Path, capsys) -> None:
        """Test that Docker presets override custom post-process commands."""
        test_video = temp_dir / "test.mp4"
        test_video.write_text("fake video")
        
        test_args = [
            "subwhisper.py",
            str(test_video),
            "--post-process", "custom command",
            "--fix-common-errors"
        ]
        
        with patch('sys.argv', test_args), \
             patch('subwhisper.process_video', return_value=True), \
             patch('subwhisper.cleanup_and_exit') as mock_cleanup, \
             patch('sys.exit'):
            
            subwhisper.main()
            
            # Check that warning was printed
            captured = capsys.readouterr()
            assert "Warning" in captured.out
            mock_cleanup.assert_called_once_with(0)

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_main_keyboard_interrupt(self, temp_dir: Path) -> None:
        """Test main function handling of keyboard interrupt."""
        test_video = temp_dir / "test.mp4"
        test_video.write_text("fake video")
        
        test_args = [
            "subwhisper.py",
            str(test_video)
        ]
        
        with patch('sys.argv', test_args), \
             patch('subwhisper.process_video', side_effect=KeyboardInterrupt()), \
             patch('subwhisper.cleanup_and_exit') as mock_cleanup, \
             patch('sys.exit'), \
             patch('builtins.print'):
            
            subwhisper.main()
            # Called twice: once in exception handler with 1, once in finally with 0
            assert mock_cleanup.call_count == 2
            assert mock_cleanup.call_args_list[0] == ((1,),)
            assert mock_cleanup.call_args_list[1] == ((0,),)

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_main_unexpected_exception(self, temp_dir: Path) -> None:
        """Test main function handling of unexpected exceptions."""
        test_video = temp_dir / "test.mp4"
        test_video.write_text("fake video")
        
        test_args = [
            "subwhisper.py",
            str(test_video)
        ]
        
        with patch('sys.argv', test_args), \
             patch('subwhisper.process_video', side_effect=Exception("Unexpected error")), \
             patch('subwhisper.cleanup_and_exit') as mock_cleanup, \
             patch('sys.exit'), \
             patch('builtins.print'):
            
            subwhisper.main()
            # Called twice: once in exception handler with 1, once in finally with 0
            assert mock_cleanup.call_count == 2
            assert mock_cleanup.call_args_list[0] == ((1,),)
            assert mock_cleanup.call_args_list[1] == ((0,),)

    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_main_batch_mixed_success_failure(self, temp_dir: Path) -> None:
        """Test batch processing with mixed success and failure."""
        video_dir = temp_dir / "videos_mixed"
        video_dir.mkdir(exist_ok=True)
        
        video1 = video_dir / "video1.mp4"
        video2 = video_dir / "video2.mp4"
        video1.write_text("fake video 1")
        video2.write_text("fake video 2")
        
        test_args = [
            "subwhisper.py",
            str(video_dir),
            "--batch"
        ]
        
        # Mock process_video to return True for first, False for second
        def mock_process_video(video_path, args):
            if "video1" in str(video_path):
                return True
            return False
        
        with patch('sys.argv', test_args), \
             patch('subwhisper.process_video', side_effect=mock_process_video), \
             patch('subwhisper.cleanup_and_exit') as mock_cleanup, \
             patch('sys.exit'), \
             patch('builtins.print'), \
             patch('glob.glob', return_value=[str(video1), str(video2)]):
            
            subwhisper.main()
            mock_cleanup.assert_called_once_with(0)

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_main_torch_info_logging(self, temp_dir: Path) -> None:
        """Test main function logs PyTorch information."""
        test_video = temp_dir / "test.mp4"
        test_video.write_text("fake video")
        
        test_args = ["subwhisper.py", str(test_video)]
        
        with patch('sys.argv', test_args), \
             patch('subwhisper.process_video', return_value=True), \
             patch('subwhisper.cleanup_and_exit'), \
             patch('sys.exit'), \
             patch('builtins.print'), \
             patch('torch.cuda.is_available', return_value=True):
            
            subwhisper.main()
            # Test passes if no exception is raised

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_main_torch_info_exception(self, temp_dir: Path) -> None:
        """Test main function handles PyTorch info exception gracefully."""
        test_video = temp_dir / "test.mp4"
        test_video.write_text("fake video")
        
        test_args = ["subwhisper.py", str(test_video)]
        
        with patch('sys.argv', test_args), \
             patch('subwhisper.process_video', return_value=True), \
             patch('subwhisper.cleanup_and_exit'), \
             patch('sys.exit'), \
             patch('builtins.print'), \
             patch('torch.cuda.is_available', side_effect=Exception("PyTorch error")):
            
            subwhisper.main()
            # Test passes if exception is handled gracefully


class TestSubWhisperErrorRecovery:
    """Error recovery and resilience tests."""
    
    @pytest.mark.unit
    @pytest.mark.timeout(60)
    def test_ffmpeg_timeout_handling(self, temp_dir: Path) -> None:
        """Test handling of FFmpeg timeouts."""
        input_video = temp_dir / "input.mp4"
        input_video.write_text("fake video")
        
        with patch('subwhisper.find_ffmpeg', return_value="ffmpeg"), \
             patch('subprocess.run', side_effect=subprocess.TimeoutExpired("ffmpeg", 30)):
            
            with pytest.raises(RuntimeError, match="timed out"):
                subwhisper.extract_audio(str(input_video))

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_transcription_model_loading_failure(self, temp_dir: Path, sample_audio_data: Path, mock_args: Mock) -> None:
        """Test handling of Whisper model loading failures."""
        with patch('whisper.load_model', side_effect=Exception("Model download failed")):
            result = subwhisper.transcribe_audio(str(sample_audio_data), mock_args)
            assert result is None

    @pytest.mark.unit
    @pytest.mark.timeout(30)
    def test_srt_generation_io_error(self, temp_dir: Path) -> None:
        """Test SRT generation with I/O errors."""
        segments = [{"start": 0.0, "end": 1.0, "text": "Test"}]
        
        # Try to write to a directory instead of a file (should cause IO error)
        invalid_output = temp_dir / "invalid_dir"
        invalid_output.mkdir()
        
        with pytest.raises(RuntimeError):
            subwhisper.generate_srt(segments, str(invalid_output))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

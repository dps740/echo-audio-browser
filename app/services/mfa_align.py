"""Montreal Forced Aligner (MFA) integration for accurate word timestamps."""

import subprocess
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class MFAConfig:
    """MFA configuration."""
    dictionary: str = "english_mfa"
    acoustic_model: str = "english_mfa"
    clean: bool = True
    # MFA expects 16kHz mono WAV
    sample_rate: int = 16000
    channels: int = 1


def convert_to_wav(input_path: str, output_path: str, config: MFAConfig = None) -> bool:
    """
    Convert audio to WAV format suitable for MFA.
    
    Args:
        input_path: Path to input audio (MP3, M4A, etc.)
        output_path: Path for output WAV
        config: MFA configuration (for sample rate, channels)
    
    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = MFAConfig()
    
    cmd = [
        "ffmpeg", "-y",  # Overwrite output
        "-i", input_path,
        "-ar", str(config.sample_rate),  # Sample rate
        "-ac", str(config.channels),  # Mono
        "-acodec", "pcm_s16le",  # 16-bit PCM
        output_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout for long files
        )
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        print(f"WAV conversion failed: {e}")
        return False


def prepare_mfa_input(
    wav_path: str,
    transcript_text: str,
    work_dir: str
) -> tuple[str, str]:
    """
    Prepare input directory structure for MFA.
    
    MFA expects:
        input_dir/
            speaker_name/
                audio.wav
                audio.txt
    
    Args:
        wav_path: Path to WAV file
        transcript_text: Plain text transcript
        work_dir: Working directory for MFA files
    
    Returns:
        (input_dir, output_dir) paths
    """
    input_dir = os.path.join(work_dir, "mfa_input", "speaker")
    output_dir = os.path.join(work_dir, "mfa_output")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base name without extension
    base_name = Path(wav_path).stem
    
    # Copy WAV to input dir
    wav_dest = os.path.join(input_dir, f"{base_name}.wav")
    shutil.copy(wav_path, wav_dest)
    
    # Write transcript text file (must match WAV name)
    txt_dest = os.path.join(input_dir, f"{base_name}.txt")
    with open(txt_dest, 'w', encoding='utf-8') as f:
        # Clean transcript for MFA
        # Remove special characters that MFA doesn't handle well
        clean_text = transcript_text
        clean_text = clean_text.replace('\n', ' ')
        clean_text = clean_text.replace('\r', ' ')
        # Normalize whitespace
        clean_text = ' '.join(clean_text.split())
        f.write(clean_text)
    
    return os.path.dirname(input_dir), output_dir


def run_mfa_align(
    input_dir: str,
    output_dir: str,
    config: MFAConfig = None,
    progress_callback: callable = None
) -> Optional[str]:
    """
    Run MFA alignment.
    
    Args:
        input_dir: MFA input directory (containing speaker subdirs)
        output_dir: Output directory for TextGrid files
        config: MFA configuration
        progress_callback: Optional callback for progress updates
    
    Returns:
        Path to TextGrid file if successful, None otherwise
    """
    if config is None:
        config = MFAConfig()
    
    cmd = [
        "mfa", "align",
        input_dir,
        config.dictionary,
        config.acoustic_model,
        output_dir,
    ]
    
    if config.clean:
        cmd.append("--clean")
    
    try:
        if progress_callback:
            progress_callback("Starting MFA alignment...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout for long files
        )
        
        if result.returncode != 0:
            print(f"MFA alignment failed: {result.stderr}")
            return None
        
        # Find the TextGrid file
        textgrid_files = list(Path(output_dir).rglob("*.TextGrid"))
        if not textgrid_files:
            print("No TextGrid file found after alignment")
            return None
        
        return str(textgrid_files[0])
        
    except subprocess.TimeoutExpired:
        print("MFA alignment timed out")
        return None
    except Exception as e:
        print(f"MFA alignment error: {e}")
        return None


def align_audio_transcript(
    audio_path: str,
    transcript_text: str,
    output_dir: str = None,
    config: MFAConfig = None,
    progress_callback: callable = None
) -> Optional[str]:
    """
    Full pipeline: Convert audio → Prepare input → Run MFA → Return TextGrid path.
    
    Args:
        audio_path: Path to audio file (MP3, WAV, etc.)
        transcript_text: Plain text transcript
        output_dir: Where to store output (defaults to temp)
        config: MFA configuration
        progress_callback: Optional callback(status_msg) for progress updates
    
    Returns:
        Path to TextGrid file if successful, None otherwise
    """
    if config is None:
        config = MFAConfig()
    
    # Use temp dir if no output specified
    use_temp = output_dir is None
    work_dir = output_dir or tempfile.mkdtemp(prefix="mfa_")
    
    try:
        wav_path = audio_path
        
        # Convert to WAV if needed
        if not audio_path.lower().endswith('.wav'):
            if progress_callback:
                progress_callback("Converting to WAV...")
            
            wav_path = os.path.join(work_dir, "audio.wav")
            if not convert_to_wav(audio_path, wav_path, config):
                print("WAV conversion failed")
                return None
        
        # Prepare MFA input
        if progress_callback:
            progress_callback("Preparing MFA input...")
        
        mfa_input, mfa_output = prepare_mfa_input(wav_path, transcript_text, work_dir)
        
        # Run alignment
        if progress_callback:
            progress_callback("Running MFA alignment...")
        
        textgrid_path = run_mfa_align(mfa_input, mfa_output, config, progress_callback)
        
        return textgrid_path
        
    finally:
        # Clean up temp dir if we created one
        if use_temp and os.path.exists(work_dir):
            try:
                shutil.rmtree(work_dir)
            except:
                pass


def check_mfa_installed() -> bool:
    """Check if MFA is installed and accessible."""
    try:
        result = subprocess.run(
            ["mfa", "version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except:
        return False


def check_mfa_models() -> dict:
    """Check if required MFA models are downloaded."""
    models = {
        "dictionary": False,
        "acoustic": False
    }
    
    try:
        # Check dictionary
        result = subprocess.run(
            ["mfa", "model", "list", "dictionary"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if "english_mfa" in result.stdout.lower():
            models["dictionary"] = True
        
        # Check acoustic model
        result = subprocess.run(
            ["mfa", "model", "list", "acoustic"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if "english_mfa" in result.stdout.lower():
            models["acoustic"] = True
            
    except:
        pass
    
    return models

import logging
import os
import subprocess
import tempfile
from pathlib import Path

class TextToSpeech:
    def __init__(self, voice: str = "adam", volume: float = 1.0, force_engine: str = None):
        """
        Initialize TTS with Kokoro as primary, BarkTTS as fallback.

        Args:
            voice: Voice name
            volume: Volume (0.0-1.0)
            force_engine: Force specific engine ("kokoro" or "bark"), or None for auto-fallback
        """
        self.voice = voice
        self.volume = max(0.0, min(1.0, volume))  # Clamp volume to 0.0-1.0
        self.tts_engine = None  # Will be "kokoro" or "bark"
        self.tts = None
        self.bark_processor = None
        self.bark_model = None

        if force_engine == "bark":
            # Force BarkTTS
            logging.info(f"Forcing BarkTTS engine")
            if self._init_bark():
                logging.info(f"✓ BarkTTS initialized successfully")
                return
            logging.error("❌ Failed to initialize BarkTTS")
        elif force_engine == "kokoro":
            # Force Kokoro
            logging.info(f"Forcing Kokoro engine with voice: {voice}, volume: {self.volume}")
            if self._init_kokoro():
                logging.info(f"✓ Kokoro TTS initialized successfully")
                return
            logging.error("❌ Failed to initialize Kokoro TTS")
        else:
            # Try Kokoro first (primary), then fall back
            logging.info(f"Attempting to initialize Kokoro TTS (primary) with voice: {voice}, volume: {self.volume}")
            if self._init_kokoro():
                logging.info(f"✓ Kokoro TTS initialized successfully")
                return

            # Fall back to BarkTTS
            logging.warning("Kokoro TTS failed, falling back to BarkTTS (secondary)")
            if self._init_bark():
                logging.info(f"✓ BarkTTS initialized successfully")
                return

            # No TTS available
            logging.error("❌ Failed to initialize any TTS engine (Kokoro and Bark both failed)")

    def _init_kokoro(self) -> bool:
        """Try to initialize Kokoro TTS. Returns True on success."""
        try:
            from kokoro_onnx import Kokoro
            import numpy as np
            from pathlib import Path
            self.np = np

            # Use downloaded models from ~/.local/share/kokoro/
            models_dir = Path.home() / '.local' / 'share' / 'kokoro'
            model_path = str(models_dir / 'kokoro-v1.0.onnx')
            voices_path = str(models_dir / 'voices-v1.0.bin')

            self.tts = Kokoro(model_path=model_path, voices_path=voices_path)
            self.tts_engine = "kokoro"
            return True
        except Exception as e:
            logging.debug(f"Kokoro initialization failed: {e}")
            return False

    def _init_bark(self) -> bool:
        """Try to initialize BarkTTS. Returns True on success."""
        try:
            from transformers import AutoProcessor, BarkModel
            import numpy as np
            import torch

            self.np = np
            self.torch = torch

            # Load Bark model and processor
            logging.info("Loading BarkTTS model (this may take a moment)...")
            self.bark_processor = AutoProcessor.from_pretrained("suno/bark-small")
            self.bark_model = BarkModel.from_pretrained("suno/bark-small")

            # Move to GPU if available
            if torch.cuda.is_available():
                self.bark_model = self.bark_model.to("cuda")
                logging.info("BarkTTS using GPU acceleration")

            self.tts_engine = "bark"
            return True
        except Exception as e:
            logging.debug(f"BarkTTS initialization failed: {e}")
            return False

    def set_voice(self, voice: str):
        """Change voice dynamically"""
        self.voice = voice
        logging.info(f"Voice changed to: {voice}")

    def set_volume(self, volume: float):
        """Set volume (0.0-1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        logging.info(f"Volume changed to: {self.volume}")

    def speak(self, text: str):
        """Speak text using available TTS engine (Kokoro or Bark)"""
        if not text:
            logging.debug("No text provided to speak.")
            return

        if not self.tts_engine:
            logging.error("No TTS engine initialized, cannot speak.")
            return

        # Route to appropriate TTS engine
        if self.tts_engine == "kokoro":
            self._speak_kokoro(text)
        elif self.tts_engine == "bark":
            self._speak_bark(text)

    def _speak_kokoro(self, text: str):
        """Speak using Kokoro TTS"""
        logging.info(f"Speaking with Kokoro ({self.voice}): {text[:100]}...")
        try:
            # Generate audio using Kokoro
            audio_array, sample_rate = self.tts.create(text, voice=self.voice, speed=1.0)

            # Apply volume adjustment
            audio_array = audio_array * self.volume

            # Save and play
            self._save_and_play_audio(audio_array, sample_rate, "Kokoro")
            logging.debug("Successfully spoke text with Kokoro.")
        except Exception as e:
            logging.error(f"Kokoro TTS error: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def _speak_bark(self, text: str):
        """Speak using BarkTTS"""
        logging.info(f"Speaking with BarkTTS ({self.voice}): {text[:100]}...")
        try:
            # Process text input
            inputs = self.bark_processor(text, voice_preset=self.voice)

            # Move inputs to same device as model
            if self.torch.cuda.is_available():
                inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}

            # Generate audio
            with self.torch.no_grad():
                audio_array = self.bark_model.generate(**inputs)

            # Convert to numpy and get sample rate
            audio_array = audio_array.cpu().numpy().squeeze()
            sample_rate = self.bark_model.generation_config.sample_rate

            # Apply volume adjustment
            audio_array = audio_array * self.volume

            # Save and play
            self._save_and_play_audio(audio_array, sample_rate, "BarkTTS")
            logging.debug("Successfully spoke text with BarkTTS.")
        except Exception as e:
            logging.error(f"BarkTTS error: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def _save_and_play_audio(self, audio_array, sample_rate: int, engine_name: str):
        """Save audio to temp file and play it"""
        import scipy.io.wavfile as wavfile

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            wavfile.write(tmp_path, sample_rate, (audio_array * 32767).astype(self.np.int16))

        logging.info(f"Generated audio saved to {tmp_path}, playing...")

        # Try different audio players
        played = False
        players = [
            (["aplay", tmp_path], "aplay"),
            (["paplay", tmp_path], "paplay"),
            (["ffplay", "-nodisp", "-autoexit", tmp_path], "ffplay")
        ]

        for cmd, player_name in players:
            try:
                subprocess.run(cmd, check=True, timeout=120,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                played = True
                logging.info(f"Audio played with {player_name}")
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                logging.debug(f"{player_name} error: {e}")
                continue

        if not played:
            logging.error("No audio player found (aplay, paplay, or ffplay). Cannot play audio.")

        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

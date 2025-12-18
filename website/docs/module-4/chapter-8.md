---
sidebar_position: 8
title: Chapter 8 - Speech Recognition Integration
---

# Chapter 8 - Speech Recognition Integration

In this chapter, we explore the integration of speech recognition systems with robotic platforms, enabling natural human-robot interaction through spoken language. Speech recognition technology allows robots to understand verbal commands, engage in conversations, and respond appropriately to human speech in real-world environments.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the fundamentals of automatic speech recognition (ASR) systems
- Integrate speech recognition with robotic platforms for human-robot interaction
- Implement real-time speech processing for robotic applications
- Handle challenges in speech recognition for robotics (noise, accents, etc.)
- Design robust speech interfaces for robotic systems

## Introduction to Speech Recognition for Robotics

Speech recognition in robotics enables natural interaction between humans and robots through spoken language. Unlike traditional speech recognition systems designed for quiet environments, robotic applications must handle challenging acoustic conditions, including background noise, reverberation, and varying distances between speakers and microphones.

### Key Challenges in Robotic Speech Recognition

1. **Acoustic Environment**: Robots operate in noisy environments with varying acoustic properties
2. **Real-time Processing**: Speech recognition must be fast enough for natural interaction
3. **Robustness**: Systems must handle different accents, speaking styles, and environmental conditions
4. **Integration**: Speech recognition must be tightly integrated with other robotic systems
5. **Privacy and Security**: Processing of speech data must respect user privacy

## Speech Recognition Architecture for Robotics

Modern speech recognition systems for robotics typically follow a pipeline architecture:

```python
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import threading
import queue
import time
import asyncio

class SpeechRecognitionPipeline:
    """Complete speech recognition pipeline for robotic applications"""

    def __init__(self, model_config):
        self.model_config = model_config
        self.audio_processor = AudioPreprocessor()
        self.speech_encoder = SpeechEncoder()
        self.language_model = LanguageModel()
        self.decoder = BeamSearchDecoder()

        # Real-time processing components
        self.audio_buffer = queue.Queue()
        self.result_buffer = queue.Queue()
        self.is_listening = False
        self.listening_thread = None

    def preprocess_audio(self, audio_data: np.ndarray) -> torch.Tensor:
        """Preprocess audio for speech recognition"""
        return self.audio_processor(audio_data)

    def encode_speech(self, features: torch.Tensor) -> torch.Tensor:
        """Encode speech features using neural network"""
        return self.speech_encoder(features)

    def decode_to_text(self, encoded_features: torch.Tensor) -> str:
        """Decode encoded features to text using language model"""
        return self.decoder(encoded_features, self.language_model)

    def recognize_speech(self, audio_data: np.ndarray) -> str:
        """Complete speech recognition pipeline"""
        # Preprocess
        features = self.preprocess_audio(audio_data)

        # Encode
        encoded = self.encode_speech(features)

        # Decode
        text = self.decode_to_text(encoded)

        return text

class AudioPreprocessor:
    """Preprocess audio data for speech recognition"""

    def __init__(self):
        # Common preprocessing parameters
        self.sample_rate = 16000  # Standard for ASR
        self.window_size = 400  # 25ms window at 16kHz
        self.hop_length = 160   # 10ms hop at 16kHz
        self.n_fft = 512
        self.n_mels = 80

        # Create mel filterbank
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            n_stft=self.n_fft // 2 + 1
        )

    def __call__(self, audio_data: np.ndarray) -> torch.Tensor:
        """Apply preprocessing pipeline"""
        # Convert to tensor if needed
        if not isinstance(audio_data, torch.Tensor):
            audio_data = torch.from_numpy(audio_data).float()

        # Resample if necessary
        if audio_data.shape[-1] != self.sample_rate:
            audio_data = torchaudio.functional.resample(audio_data, audio_data.shape[-1], self.sample_rate)

        # Compute spectrogram
        spectrogram = torch.stft(
            audio_data,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_size,
            return_complex=True
        )
        spectrogram = torch.abs(spectrogram)

        # Convert to mel scale
        mel_spec = self.mel_scale(spectrogram)

        # Apply log transform
        log_mel_spec = torch.log(mel_spec + 1e-9)

        # Apply normalization
        mean = log_mel_spec.mean(dim=-1, keepdim=True)
        std = log_mel_spec.std(dim=-1, keepdim=True)
        normalized_spec = (log_mel_spec - mean) / (std + 1e-9)

        return normalized_spec

class SpeechEncoder(nn.Module):
    """Encode speech features using neural network"""

    def __init__(self, input_dim=80, hidden_dim=512, num_layers=3):
        super().__init__()

        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode speech features"""
        # Input: (batch, n_mels, time)
        # Transpose to: (batch, time, n_mels)
        features = features.transpose(1, 2)

        # Apply conv layers
        conv_out = self.conv_layers(features.transpose(1, 2))
        conv_out = conv_out.transpose(1, 2)  # Back to (batch, time, features)

        # Apply LSTM
        lstm_out, _ = self.lstm(conv_out)

        # Apply output projection
        encoded = self.output_projection(lstm_out)

        return encoded

class LanguageModel(nn.Module):
    """Language model for speech recognition"""

    def __init__(self, vocab_size=1000, embedding_dim=256, hidden_dim=512, num_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through language model"""
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        output = self.output_projection(lstm_out)
        return output

class BeamSearchDecoder:
    """Beam search decoder for speech recognition"""

    def __init__(self, beam_width=5, vocab_size=1000):
        self.beam_width = beam_width
        self.vocab_size = vocab_size

    def __call__(self, encoded_features: torch.Tensor, language_model: LanguageModel) -> str:
        """Decode encoded features to text using beam search"""
        # This is a simplified version - real implementation would be more complex
        batch_size, seq_len, hidden_dim = encoded_features.shape

        # Initialize beam with empty sequences
        beams = [(0.0, [])]  # (score, sequence)

        for t in range(seq_len):
            new_beams = []

            for score, sequence in beams:
                # Get probabilities for next token
                if len(sequence) == 0:
                    # For first token, use average of all time steps
                    avg_features = encoded_features.mean(dim=1)
                else:
                    # Use features at current time step
                    avg_features = encoded_features[:, t, :]

                # This is a simplified approach - real ASR uses attention mechanisms
                logits = torch.randn(self.vocab_size)  # Placeholder
                probs = torch.softmax(logits, dim=-1)

                # Get top candidates
                top_probs, top_indices = torch.topk(probs, self.beam_width)

                for prob, idx in zip(top_probs, top_indices):
                    new_score = score + torch.log(prob)
                    new_sequence = sequence + [idx.item()]
                    new_beams.append((new_score, new_sequence))

            # Keep top beams
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:self.beam_width]

        # Return best sequence
        best_sequence = beams[0][1]

        # Convert to text (simplified - real implementation would use tokenizer)
        return " ".join([str(token) for token in best_sequence])

# Example implementation using pre-trained models
class PretrainedSpeechRecognizer:
    """Speech recognition using pre-trained models"""

    def __init__(self, model_name="facebook/wav2vec2-large-960h"):
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

    def recognize(self, audio_data: np.ndarray, sampling_rate: int = 16000) -> str:
        """Recognize speech using pre-trained model"""
        # Process audio
        input_values = self.processor(
            audio_data,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_values

        # Recognize
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription
```

## Real-time Speech Recognition for Robotics

Real-time speech recognition is crucial for natural human-robot interaction:

```python
import pyaudio
import webrtcvad
import collections
import wave
import threading
from dataclasses import dataclass

@dataclass
class SpeechRecognitionConfig:
    """Configuration for speech recognition system"""
    sample_rate: int = 16000
    frame_duration_ms: int = 30  # Supports 10, 20, 30 (ms)
    vad_aggressiveness: int = 3  # 0-3, more aggressive from noice
    chunk_size: int = 1024
    model_path: str = "default_model"
    sensitivity: float = 0.5

class RealTimeSpeechRecognizer:
    """Real-time speech recognition for robotics"""

    def __init__(self, config: SpeechRecognitionConfig):
        self.config = config
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)

        # Audio stream parameters
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = config.sample_rate
        self.chunk = config.chunk_size

        # Voice activity detection
        self.frame_duration_ms = config.frame_duration_ms
        self.frame_size = int(self.rate * self.frame_duration_ms / 1000) * 2  # 2 bytes per sample

        # Audio processing
        self.audio_buffer = collections.deque(maxlen=30)  # Store last 30 frames
        self.speech_segments = []

        # Recognition model
        self.speech_recognizer = PretrainedSpeechRecognizer()

        # Threading
        self.is_recording = False
        self.recording_thread = None
        self.result_queue = queue.Queue()

    def start_listening(self):
        """Start real-time speech recognition"""
        if self.is_recording:
            return

        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._recording_loop)
        self.recording_thread.start()

    def stop_listening(self):
        """Stop real-time speech recognition"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()

    def _recording_loop(self):
        """Main recording loop"""
        audio = pyaudio.PyAudio()

        stream = audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        while self.is_recording:
            try:
                # Read audio data
                data = stream.read(self.chunk, exception_on_overflow=False)

                # Process audio frame
                is_speech = self._is_speech_frame(data)

                if is_speech:
                    # Add to speech buffer
                    self.audio_buffer.extend(data)

                    # Check if we have enough speech to recognize
                    if len(self.audio_buffer) > self.frame_size * 10:  # 300ms minimum
                        audio_segment = list(self.audio_buffer)

                        # Convert to numpy array
                        audio_array = np.frombuffer(b''.join(audio_segment), dtype=np.int16)
                        audio_float = audio_array.astype(np.float32) / 32768.0

                        # Recognize speech
                        result = self.speech_recognizer.recognize(audio_float, self.rate)

                        # Add to result queue
                        self.result_queue.put({
                            'text': result,
                            'timestamp': time.time(),
                            'confidence': self._estimate_confidence(result)
                        })

                        # Clear buffer after recognition
                        self.audio_buffer.clear()
                else:
                    # Clear buffer if no speech detected for a while
                    if len(self.audio_buffer) > 0:
                        self.audio_buffer.clear()

            except Exception as e:
                print(f"Error in recording loop: {e}")
                time.sleep(0.1)

        stream.stop_stream()
        stream.close()
        audio.terminate()

    def _is_speech_frame(self, frame: bytes) -> bool:
        """Check if audio frame contains speech"""
        try:
            # WebRTC VAD requires frames of specific sizes
            # 10ms, 20ms, or 30ms at 16kHz
            return self.vad.is_speech(frame, self.rate)
        except:
            # If frame size is wrong, assume it's speech
            return True

    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence of recognition result"""
        # This is a simplified confidence estimation
        # In practice, this would come from the ASR model
        if len(text.strip()) == 0:
            return 0.0
        return 0.8  # Default confidence

    def get_results(self) -> List[Dict]:
        """Get recognition results from queue"""
        results = []
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results

# Wake Word Detection
class WakeWordDetector:
    """Detect wake words to activate speech recognition"""

    def __init__(self, wake_words: List[str] = None):
        if wake_words is None:
            wake_words = ["robot", "hey robot", "hello robot"]
        self.wake_words = [word.lower() for word in wake_words]

        # Simple keyword matching - in practice, this would use more sophisticated models
        self.keyword_model = None  # Placeholder for keyword spotting model

    def detect_wake_word(self, text: str) -> bool:
        """Check if text contains wake word"""
        text_lower = text.lower()
        for wake_word in self.wake_words:
            if wake_word in text_lower:
                return True
        return False

    def get_wake_word_position(self, text: str) -> Optional[int]:
        """Get position of wake word in text"""
        text_lower = text.lower()
        for wake_word in self.wake_words:
            pos = text_lower.find(wake_word)
            if pos != -1:
                return pos
        return None

# Example integration with robot controller
class RobotSpeechInterface:
    """Complete speech interface for robotics"""

    def __init__(self, robot_controller, speech_config: SpeechRecognitionConfig):
        self.robot_controller = robot_controller
        self.speech_recognizer = RealTimeSpeechRecognizer(speech_config)
        self.wake_word_detector = WakeWordDetector()
        self.command_parser = CommandParser()

        # State management
        self.is_listening_for_command = False
        self.command_timeout = 10.0  # seconds

    def start_interaction(self):
        """Start speech-based interaction with robot"""
        print("Starting speech interaction...")
        self.speech_recognizer.start_listening()

        while True:
            try:
                # Get speech recognition results
                results = self.speech_recognizer.get_results()

                for result in results:
                    text = result['text']
                    confidence = result['confidence']

                    print(f"Recognized: '{text}' (confidence: {confidence:.2f})")

                    # Check for wake word if not already listening for command
                    if not self.is_listening_for_command:
                        if self.wake_word_detector.detect_wake_word(text):
                            print("Wake word detected! Listening for command...")
                            self.is_listening_for_command = True
                            self.command_start_time = time.time()
                    else:
                        # Process command
                        command = self.command_parser.parse(text)
                        if command:
                            print(f"Executing command: {command}")
                            self.execute_command(command)
                            self.is_listening_for_command = False
                        else:
                            # Check for timeout
                            if time.time() - self.command_start_time > self.command_timeout:
                                print("Command timeout - returning to wake word mode")
                                self.is_listening_for_command = False

                time.sleep(0.1)  # Small delay to prevent busy waiting

            except KeyboardInterrupt:
                print("Stopping speech interaction...")
                break

    def execute_command(self, command: Dict):
        """Execute parsed command on robot"""
        command_type = command.get('type')

        if command_type == 'move':
            direction = command.get('direction')
            distance = command.get('distance', 1.0)
            self.robot_controller.move(direction, distance)
        elif command_type == 'turn':
            angle = command.get('angle')
            self.robot_controller.turn(angle)
        elif command_type == 'grasp':
            object_name = command.get('object')
            self.robot_controller.grasp_object(object_name)
        elif command_type == 'speak':
            text = command.get('text')
            self.robot_controller.speak(text)
        else:
            print(f"Unknown command type: {command_type}")

    def stop_interaction(self):
        """Stop speech-based interaction"""
        self.speech_recognizer.stop_listening()

# Command parsing for natural language
class CommandParser:
    """Parse natural language commands into robot actions"""

    def __init__(self):
        # Define command patterns
        self.command_patterns = {
            'move': [
                r'go (?P<direction>forward|backward|left|right) (?P<distance>\d+(?:\.\d+)?) meters?',
                r'move (?P<direction>forward|backward|left|right) (?P<distance>\d+(?:\.\d+)?) meters?',
                r'go (?P<distance>\d+(?:\.\d+)?) meters? (?P<direction>forward|backward|left|right)'
            ],
            'turn': [
                r'turn (?P<angle>\d+(?:\.\d+)?) degrees?',
                r'rotate (?P<angle>\d+(?:\.\d+)?) degrees?',
                r'pivot (?P<angle>\d+(?:\.\d+)?) degrees?'
            ],
            'grasp': [
                r'pick up the (?P<object>\w+)',
                r'grasp the (?P<object>\w+)',
                r'get the (?P<object>\w+)'
            ],
            'speak': [
                r'say "(?P<text>[^"]+)"',
                r'tell me "(?P<text>[^"]+)"',
                r'speak "(?P<text>[^"]+)"'
            ]
        }

    def parse(self, text: str) -> Optional[Dict]:
        """Parse natural language text into command"""
        import re

        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    params = match.groupdict()
                    params['type'] = command_type

                    # Convert numeric values
                    if 'distance' in params:
                        params['distance'] = float(params['distance'])
                    if 'angle' in params:
                        params['angle'] = float(params['angle'])

                    return params

        return None  # No command found
```

## Advanced Speech Recognition Techniques

For robotics applications, advanced techniques are needed to handle challenging conditions:

```python
import librosa
import scipy.signal
from scipy import ndimage

class AdvancedSpeechProcessor:
    """Advanced speech processing for robotics"""

    def __init__(self):
        self.noise_estimator = NoiseEstimator()
        self.beamformer = Beamformer()
        self.dereverberation = Dereverberation()

    def preprocess_for_robotics(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Advanced preprocessing for robotic speech recognition"""
        # Step 1: Noise reduction
        audio_clean = self.noise_estimator.reduce_noise(audio, sample_rate)

        # Step 2: Beamforming (if multiple microphones available)
        # This would combine signals from multiple microphones
        # audio_bf = self.beamformer.process(audio_clean, sample_rate)

        # Step 3: Dereverberation
        audio_dereverb = self.dereverberation.process(audio_clean, sample_rate)

        # Step 4: Speech enhancement
        audio_enhanced = self.enhance_speech(audio_dereverb, sample_rate)

        return audio_enhanced

    def enhance_speech(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Enhance speech signal for better recognition"""
        # Apply spectral subtraction for noise reduction
        enhanced = self._spectral_subtraction(audio, sample_rate)

        # Apply Wiener filtering
        enhanced = self._wiener_filter(enhanced, sample_rate)

        return enhanced

    def _spectral_subtraction(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply spectral subtraction for noise reduction"""
        # This is a simplified implementation
        # In practice, more sophisticated methods would be used

        # Compute STFT
        stft = librosa.stft(audio)

        # Estimate noise spectrum (from beginning of signal)
        noise_frames = stft[:, :50]  # First 50 frames as noise estimate
        noise_power = np.mean(np.abs(noise_frames) ** 2, axis=1, keepdims=True)

        # Apply spectral subtraction
        signal_power = np.abs(stft) ** 2
        enhanced_power = np.maximum(signal_power - noise_power, 0.1 * signal_power)

        # Reconstruct signal
        enhanced_stft = stft * np.sqrt(enhanced_power) / (np.abs(stft) + 1e-8)
        enhanced = librosa.istft(enhanced_stft)

        return enhanced

    def _wiener_filter(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply Wiener filtering for noise reduction"""
        # Simplified Wiener filter implementation
        # In practice, more sophisticated methods would be used

        # Estimate signal and noise power spectra
        window_size = 2048
        hop_size = 512

        # Compute short-time spectra
        f, t, Zxx = scipy.signal.stft(audio, fs=sample_rate, nperseg=window_size, noverlap=window_size-hop_size)

        # Estimate power spectra
        power_spectrum = np.abs(Zxx) ** 2

        # Simple noise estimation (average of low-energy frames)
        frame_energy = np.mean(power_spectrum, axis=0)
        noise_threshold = np.percentile(frame_energy, 20)  # Bottom 20% as noise
        noise_mask = frame_energy < noise_threshold

        if np.any(noise_mask):
            noise_power = np.mean(power_spectrum[:, noise_mask], axis=1, keepdims=True)
        else:
            noise_power = np.mean(power_spectrum, axis=1, keepdims=True) * 0.1  # Conservative estimate

        # Signal power estimation
        signal_power = np.maximum(power_spectrum - noise_power, 0.1 * power_spectrum)

        # Wiener filter gain
        wiener_gain = signal_power / (signal_power + noise_power)

        # Apply filter
        filtered_spectrum = Zxx * wiener_gain

        # Reconstruct signal
        _, enhanced_audio = scipy.signal.istft(filtered_spectrum, fs=sample_rate, nperseg=window_size, noverlap=window_size-hop_size)

        return enhanced_audio

class NoiseEstimator:
    """Estimate and reduce background noise"""

    def __init__(self):
        self.noise_buffer = np.zeros(16000)  # 1 second at 16kHz
        self.buffer_idx = 0
        self.is_initialized = False

    def reduce_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Reduce noise in audio signal"""
        if not self.is_initialized:
            self._initialize_noise_buffer(audio)
            return audio  # Return original for first call

        # Estimate noise spectrum
        noise_spectrum = self._estimate_noise_spectrum()

        # Apply noise reduction
        enhanced_audio = self._apply_noise_reduction(audio, noise_spectrum)

        # Update noise buffer
        self._update_noise_buffer(audio)

        return enhanced_audio

    def _initialize_noise_buffer(self, audio: np.ndarray):
        """Initialize noise buffer with initial audio"""
        buffer_size = len(self.noise_buffer)
        if len(audio) >= buffer_size:
            self.noise_buffer = audio[:buffer_size].copy()
        else:
            self.noise_buffer[:len(audio)] = audio
        self.is_initialized = True

    def _estimate_noise_spectrum(self) -> np.ndarray:
        """Estimate noise spectrum from buffer"""
        # Compute FFT of noise buffer
        noise_fft = np.fft.rfft(self.noise_buffer)
        noise_power = np.abs(noise_fft) ** 2
        return noise_power

    def _apply_noise_reduction(self, audio: np.ndarray, noise_spectrum: np.ndarray) -> np.ndarray:
        """Apply noise reduction to audio"""
        # Pad audio if necessary
        if len(audio) < len(self.noise_buffer):
            padded_audio = np.zeros_like(self.noise_buffer)
            padded_audio[:len(audio)] = audio
        else:
            padded_audio = audio

        # Compute FFT
        audio_fft = np.fft.rfft(padded_audio)
        audio_power = np.abs(audio_fft) ** 2

        # Apply spectral subtraction
        enhanced_power = np.maximum(audio_power - noise_spectrum, 0.1 * audio_power)
        enhanced_magnitude = np.sqrt(enhanced_power)

        # Preserve phase
        enhanced_fft = enhanced_magnitude * np.exp(1j * np.angle(audio_fft))

        # Inverse FFT
        enhanced_audio = np.fft.irfft(enhanced_fft)

        # Return only original length
        return enhanced_audio[:len(audio)]

    def _update_noise_buffer(self, audio: np.ndarray):
        """Update noise buffer with new audio"""
        # Simple approach: replace oldest portion with new audio
        new_len = min(len(audio), len(self.noise_buffer))
        self.buffer_idx = (self.buffer_idx + new_len) % len(self.noise_buffer)

        # Circular buffer update
        if new_len > 0:
            start_idx = (self.buffer_idx - new_len) % len(self.noise_buffer)
            if start_idx + new_len <= len(self.noise_buffer):
                self.noise_buffer[start_idx:start_idx + new_len] = audio[:new_len]
            else:
                # Wrap around
                first_part = len(self.noise_buffer) - start_idx
                self.noise_buffer[start_idx:] = audio[:first_part]
                self.noise_buffer[:new_len - first_part] = audio[first_part:new_len]

# Multi-microphone beamforming for robotics
class Beamformer:
    """Beamformer for multi-microphone speech enhancement"""

    def __init__(self, num_mics=2, mic_distance=0.1):
        self.num_mics = num_mics
        self.mic_distance = mic_distance  # in meters
        self.delay_and_sum_weights = self._compute_delay_and_sum_weights()

    def process(self, multi_channel_audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply beamforming to multi-channel audio"""
        if multi_channel_audio.shape[0] != self.num_mics:
            raise ValueError(f"Expected {self.num_mics} channels, got {multi_channel_audio.shape[0]}")

        # Apply delay-and-sum beamforming
        beamformed = self._delay_and_sum(multi_channel_audio, sample_rate)

        return beamformed

    def _compute_delay_and_sum_weights(self):
        """Compute weights for delay-and-sum beamforming"""
        # For simplicity, assume front-direction beamforming
        # In practice, this would be more sophisticated
        return np.ones(self.num_mics) / self.num_mics

    def _delay_and_sum(self, multi_channel_audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply delay-and-sum beamforming"""
        # For this example, we'll just average the channels
        # In practice, delays would be applied based on source direction
        beamformed = np.mean(multi_channel_audio, axis=0)
        return beamformed

class Dereverberation:
    """Dereverberation for speech enhancement"""

    def __init__(self):
        self.wiener_filter_order = 100

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply dereverberation to audio"""
        # Estimate room impulse response
       rir_estimate = self._estimate_rir(audio, sample_rate)

        # Apply Wiener filtering to remove reverberation
        dereverb_audio = self._wiener_dereverb(audio, rir_estimate)

        return dereverb_audio

    def _estimate_rir(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Estimate room impulse response"""
        # Simplified RIR estimation
        # In practice, more sophisticated methods would be used
        rir_length = min(4096, len(audio) // 4)  # Estimate RIR length
        rir = np.zeros(rir_length)
        rir[0] = 1.0  # Direct path
        for i in range(1, len(rir)):
            rir[i] = 0.9 * rir[i-1] * 0.999  # Exponential decay
        return rir

    def _wiener_dereverb(self, audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
        """Apply Wiener filtering for dereverberation"""
        # Pad RIR to same length as audio
        padded_rir = np.zeros(len(audio))
        padded_rir[:len(rir)] = rir

        # Compute FFTs
        audio_fft = np.fft.fft(audio)
        rir_fft = np.fft.fft(padded_rir)

        # Compute Wiener filter
        rir_power = np.abs(rir_fft) ** 2
        # Simple regularization
        wiener_filter = np.conj(rir_fft) / (rir_power + 0.01)

        # Apply filter
        filtered_fft = audio_fft * wiener_filter

        # Inverse FFT
        dereverb_audio = np.real(np.fft.ifft(filtered_fft))

        return dereverb_audio
```

## Integration with Robotic Systems

Integrating speech recognition with robotic systems requires careful consideration of timing, safety, and reliability:

```python
class RoboticSpeechIntegration:
    """Integration of speech recognition with robotic systems"""

    def __init__(self, robot_controller, speech_config: SpeechRecognitionConfig):
        self.robot_controller = robot_controller
        self.speech_recognizer = RealTimeSpeechRecognizer(speech_config)
        self.safety_monitor = SafetyMonitor()
        self.context_manager = ContextManager()

        # Command queue for thread-safe operation
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()

        # State tracking
        self.current_state = "idle"
        self.last_command_time = time.time()

    def process_speech_command(self, text: str, confidence: float) -> Dict:
        """Process speech command with safety checks"""
        # Check confidence threshold
        if confidence < 0.6:  # Minimum confidence
            return {
                'status': 'rejected',
                'reason': 'Low confidence',
                'confidence': confidence
            }

        # Parse command
        command = self.parse_command(text)
        if not command:
            return {
                'status': 'rejected',
                'reason': 'Unrecognized command',
                'text': text
            }

        # Check safety
        if not self.safety_monitor.is_command_safe(command):
            return {
                'status': 'rejected',
                'reason': 'Safety violation',
                'command': command
            }

        # Update context
        self.context_manager.update_context(text, command)

        # Execute command
        try:
            result = self.execute_command(command)
            return {
                'status': 'success',
                'command': command,
                'result': result,
                'confidence': confidence
            }
        except Exception as e:
            return {
                'status': 'error',
                'reason': str(e),
                'command': command
            }

    def parse_command(self, text: str) -> Optional[Dict]:
        """Parse command with context awareness"""
        # Use context to disambiguate commands
        context = self.context_manager.get_context()

        # Parse using command parser
        parser = CommandParser()
        command = parser.parse(text)

        if command:
            # Apply context to refine command
            command = self._apply_context(command, context)

        return command

    def _apply_context(self, command: Dict, context: Dict) -> Dict:
        """Apply context to refine command interpretation"""
        # Example: if robot was looking at an object, "pick it up" refers to that object
        if command.get('type') == 'grasp' and command.get('object') == 'it':
            last_object = context.get('last_seen_object')
            if last_object:
                command['object'] = last_object

        return command

    def execute_command(self, command: Dict) -> Any:
        """Execute command with robot controller"""
        command_type = command.get('type')

        if command_type == 'move':
            return self.robot_controller.move(
                command.get('direction'),
                command.get('distance', 1.0)
            )
        elif command_type == 'turn':
            return self.robot_controller.turn(command.get('angle'))
        elif command_type == 'grasp':
            return self.robot_controller.grasp_object(command.get('object'))
        elif command_type == 'speak':
            return self.robot_controller.speak(command.get('text'))
        elif command_type == 'look_at':
            return self.robot_controller.look_at(command.get('target'))
        else:
            raise ValueError(f"Unknown command type: {command_type}")

    def get_robot_response(self, command_result: Dict) -> str:
        """Generate natural language response to command"""
        command_type = command_result.get('command', {}).get('type')

        if command_result['status'] == 'success':
            if command_type == 'move':
                return f"Moving {command_result['command'].get('direction')}."
            elif command_type == 'turn':
                return f"Turning {command_result['command'].get('angle')} degrees."
            elif command_type == 'grasp':
                return f"Attempting to grasp the {command_result['command'].get('object')}."
            elif command_type == 'speak':
                return "I have spoken as requested."
            else:
                return "Command executed successfully."
        else:
            return f"Sorry, I couldn't execute that command: {command_result.get('reason', 'Unknown error')}."

class ContextManager:
    """Manage context for speech understanding"""

    def __init__(self):
        self.context = {
            'last_seen_objects': [],
            'last_command': None,
            'conversation_history': [],
            'robot_state': {},
            'environment': {}
        }

    def update_context(self, text: str, command: Dict):
        """Update context based on new input"""
        self.context['last_command'] = {
            'text': text,
            'command': command,
            'timestamp': time.time()
        }

        self.context['conversation_history'].append({
            'text': text,
            'command': command,
            'timestamp': time.time()
        })

        # Keep only recent history
        if len(self.context['conversation_history']) > 10:
            self.context['conversation_history'] = self.context['conversation_history'][-10:]

    def get_context(self) -> Dict:
        """Get current context"""
        return self.context

    def get_last_seen_object(self) -> Optional[str]:
        """Get last seen object"""
        if self.context['last_seen_objects']:
            return self.context['last_seen_objects'][-1]
        return None

class SafetyMonitor:
    """Monitor safety for speech commands"""

    def __init__(self):
        self.safety_rules = [
            self._check_movement_safety,
            self._check_manipulation_safety,
            self._check_navigation_safety
        ]

    def is_command_safe(self, command: Dict) -> bool:
        """Check if command is safe to execute"""
        for rule in self.safety_rules:
            if not rule(command):
                return False
        return True

    def _check_movement_safety(self, command: Dict) -> bool:
        """Check if movement command is safe"""
        if command.get('type') == 'move':
            distance = command.get('distance', 1.0)
            # Check if movement distance is reasonable
            if distance > 5.0:  # More than 5 meters
                return False
        return True

    def _check_manipulation_safety(self, command: Dict) -> bool:
        """Check if manipulation command is safe"""
        if command.get('type') == 'grasp':
            obj = command.get('object', '').lower()
            # Check for dangerous objects
            dangerous_objects = ['knife', 'blade', 'sharp', 'hot', 'fire']
            if any(dangerous in obj for dangerous in dangerous_objects):
                return False
        return True

    def _check_navigation_safety(self, command: Dict) -> bool:
        """Check if navigation command is safe"""
        if command.get('type') in ['move', 'turn']:
            # Check robot's current state and environment
            # This would interface with robot's perception system
            pass
        return True

# Example usage of the complete system
def example_robotic_speech_system():
    """Example of complete robotic speech recognition system"""

    # Mock robot controller (in practice, this would interface with actual robot)
    class MockRobotController:
        def move(self, direction, distance):
            print(f"Moving {direction} {distance} meters")
            return "Movement completed"

        def turn(self, angle):
            print(f"Turning {angle} degrees")
            return "Turn completed"

        def grasp_object(self, obj_name):
            print(f"Attempting to grasp {obj_name}")
            return "Grasp attempt completed"

        def speak(self, text):
            print(f"Speaking: {text}")
            return "Speech completed"

        def look_at(self, target):
            print(f"Looking at {target}")
            return "Looking completed"

    # Create speech configuration
    config = SpeechRecognitionConfig()

    # Create the integration system
    robot_controller = MockRobotController()
    speech_system = RoboticSpeechIntegration(robot_controller, config)

    # Simulate processing a speech command
    sample_text = "Move forward 2 meters"
    result = speech_system.process_speech_command(sample_text, confidence=0.85)

    print(f"Command result: {result}")

    # Generate response
    response = speech_system.get_robot_response(result)
    print(f"Robot response: {response}")

    return speech_system

if __name__ == "__main__":
    system = example_robotic_speech_system()
    print("Robotic speech system example completed")
```

## Privacy and Security Considerations

When implementing speech recognition in robotics, privacy and security are critical:

```python
import hashlib
import hmac
import secrets
from cryptography.fernet import Fernet
from typing import Optional

class SecureSpeechProcessor:
    """Secure processing of speech data in robotics"""

    def __init__(self):
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.privacy_policy = PrivacyPolicy()

    def process_audio_securely(self, audio_data: bytes) -> Optional[bytes]:
        """Process audio with privacy protection"""
        # Check privacy policy
        if not self.privacy_policy.allows_processing():
            return None

        # Encrypt audio data
        encrypted_audio = self.cipher.encrypt(audio_data)

        # Process encrypted data (this would interface with ASR system)
        processed_result = self._process_encrypted_data(encrypted_audio)

        # Decrypt result if needed
        if processed_result:
            return self.cipher.decrypt(processed_result)

        return None

    def _process_encrypted_data(self, encrypted_data: bytes) -> Optional[bytes]:
        """Process encrypted audio data"""
        # This would interface with secure ASR processing
        # In practice, this might involve secure enclaves or homomorphic encryption
        return encrypted_data  # Placeholder

class PrivacyPolicy:
    """Privacy policy enforcement for speech data"""

    def __init__(self):
        self.data_retention_hours = 24  # Hours to retain data
        self.allow_cloud_processing = False  # Whether to allow cloud processing
        self.log_data_usage = True  # Whether to log data usage

    def allows_processing(self) -> bool:
        """Check if processing is allowed under privacy policy"""
        return True  # Simplified for example

    def should_encrypt_data(self) -> bool:
        """Check if data should be encrypted"""
        return True

    def allows_data_sharing(self) -> bool:
        """Check if data sharing is allowed"""
        return False

# On-device processing for privacy
class OnDeviceSpeechRecognizer:
    """On-device speech recognition for privacy"""

    def __init__(self):
        # Load lightweight model for on-device processing
        self.model = self._load_lightweight_model()

    def _load_lightweight_model(self):
        """Load lightweight model suitable for on-device processing"""
        # This would load a quantized or pruned model
        # suitable for edge devices
        return None  # Placeholder

    def recognize(self, audio_data: np.ndarray) -> str:
        """Perform speech recognition on device"""
        # Process audio locally without sending to cloud
        # This ensures privacy by keeping data on the device
        return "Recognized text"  # Placeholder
```

## Performance Optimization

Optimizing speech recognition for robotics requires balancing accuracy with computational efficiency:

```python
class OptimizedSpeechRecognition:
    """Optimized speech recognition for resource-constrained robots"""

    def __init__(self, optimization_level="balanced"):
        self.optimization_level = optimization_level
        self.model_cache = {}
        self._setup_optimizations()

    def _setup_optimizations(self):
        """Setup optimizations based on level"""
        if self.optimization_level == "speed":
            self._setup_speed_optimizations()
        elif self.optimization_level == "accuracy":
            self._setup_accuracy_optimizations()
        else:  # balanced
            self._setup_balanced_optimizations()

    def _setup_speed_optimizations(self):
        """Setup optimizations for speed"""
        self.use_quantized_model = True
        self.processing_rate = 8000  # Lower sampling rate
        self.feature_dim = 40  # Reduced feature dimension
        self.beam_width = 3  # Smaller beam width

    def _setup_accuracy_optimizations(self):
        """Setup optimizations for accuracy"""
        self.use_quantized_model = False
        self.processing_rate = 16000  # Full sampling rate
        self.feature_dim = 80  # Full feature dimension
        self.beam_width = 10  # Larger beam width

    def _setup_balanced_optimizations(self):
        """Setup balanced optimizations"""
        self.use_quantized_model = True
        self.processing_rate = 16000
        self.feature_dim = 80
        self.beam_width = 5

    def recognize_optimized(self, audio_data: np.ndarray) -> str:
        """Optimized speech recognition"""
        # Preprocess with optimized parameters
        features = self._preprocess_optimized(audio_data)

        # Recognize with optimized model
        result = self._recognize_with_model(features)

        return result

    def _preprocess_optimized(self, audio_data: np.ndarray) -> np.ndarray:
        """Optimized preprocessing"""
        # Resample if needed
        if self.processing_rate != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=16000, target_sr=self.processing_rate)

        # Extract optimized features
        # This would extract features based on self.feature_dim
        return audio_data  # Placeholder

    def _recognize_with_model(self, features: np.ndarray) -> str:
        """Recognize with optimized model"""
        # This would use the appropriate model based on optimization level
        return "Recognized text"  # Placeholder

# Model compression for robotics
class CompressedSpeechModel:
    """Compressed speech recognition model for robotics"""

    def __init__(self):
        self.compression_ratio = 4  # 4x compression
        self.quantization_bits = 8  # 8-bit quantization

    def compress_model(self, original_model):
        """Compress the speech recognition model"""
        # Apply quantization
        compressed_model = self._quantize_model(original_model)

        # Apply pruning
        compressed_model = self._prune_model(compressed_model)

        # Apply knowledge distillation if applicable
        compressed_model = self._distill_model(compressed_model)

        return compressed_model

    def _quantize_model(self, model):
        """Apply quantization to reduce model size"""
        # This would apply 8-bit quantization to the model
        return model  # Placeholder

    def _prune_model(self, model):
        """Apply pruning to remove unnecessary connections"""
        # This would remove weights below a threshold
        return model  # Placeholder

    def _distill_model(self, model):
        """Apply knowledge distillation"""
        # This would create a smaller student model
        return model  # Placeholder
```

## Chapter Summary

In this chapter, we explored speech recognition integration for robotics, covering:

- The fundamentals of automatic speech recognition for robotic applications
- Real-time speech processing systems for natural interaction
- Advanced techniques for handling noise and reverberation in robotic environments
- Integration with robotic systems including safety and context management
- Privacy and security considerations for speech data
- Performance optimization strategies for resource-constrained robots

Speech recognition enables natural human-robot interaction, allowing robots to understand verbal commands and respond appropriately. The integration requires careful consideration of real-time processing, acoustic challenges, safety, and privacy requirements.

## Next Steps

In the next chapter, we'll explore motion planning from language, examining how natural language commands can be translated into specific robotic motion plans and trajectories.

## Exercises

1. **Implementation Challenge**: Implement a real-time speech recognition system for a robotic platform, including voice activity detection and wake word recognition.

2. **System Design**: Design a speech recognition architecture for a mobile robot that operates in noisy environments, considering both performance and privacy requirements.

3. **Optimization Task**: Optimize a speech recognition model for deployment on a resource-constrained robot, balancing accuracy with computational efficiency.

4. **Integration Challenge**: Integrate speech recognition with a robot's navigation system, allowing the robot to understand and execute spoken navigation commands.

5. **Privacy Analysis**: Analyze the privacy implications of speech recognition in robotics and implement appropriate data protection measures.
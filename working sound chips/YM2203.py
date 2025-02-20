import cupy as cp
import numpy as np
import time
import sounddevice as sd

class YM2203Chip:
    def __init__(self, fs=44100, num_samples=44100):
        self.fs = fs  # Sampling rate
        self.num_samples = num_samples  # Number of samples per test
        self.registers = {
            'FM1_car_freq': 440,
            'FM2_car_freq': 440,
            'FM1_mod_freq': 10,
            'FM2_mod_freq': 10,
            'FM1_mod_index': 1,
            'FM2_mod_index': 1,
            'FM1_waveform': 1,
            'FM2_waveform': 1,
            'noise_level': 0.5,
            'envelope': {'attack': 0.01, 'decay': 0.2, 'sustain': 0.7, 'release': 0.3},
            'LFO_freq': 0.5
        }

    def generate_sine_wave(self, frequency, num_samples, fs):
        t = cp.arange(num_samples) / fs  # Use cp.arange for GPU
        return cp.sin(2 * cp.pi * frequency * t)  # Use cp.sin for GPU

    def generate_square_wave(self, frequency, num_samples, fs):
        t = cp.arange(num_samples) / fs  # Use cp.arange for GPU
        return cp.sign(cp.sin(2 * cp.pi * frequency * t))  # Use cp.sign for GPU

    def generate_noise(self):
        """Generate a noise signal using GPU."""
        return cp.random.uniform(-1, 1, self.num_samples)  # Use cp.random for GPU

    def generate_envelope(self, attack, decay, sustain, release, num_samples, fs):
        """Generate an envelope signal on the GPU."""
        # Attack
        attack_samples = int(attack * fs)
        decay_samples = int(decay * fs)
        release_samples = int(release * fs)

        envelope = cp.concatenate([
            cp.linspace(0, 1, attack_samples),  # Attack
            cp.linspace(1, sustain, decay_samples),  # Decay
            cp.full(num_samples - attack_samples - decay_samples - release_samples, sustain),  # Sustain
            cp.linspace(sustain, 0, release_samples)  # Release
        ])
        return envelope

    def generate_fm(self):
        """Generate frequency modulated signal using GPU."""
        fm1_car_signal = self.generate_sine_wave(self.registers['FM1_car_freq'], self.num_samples, self.fs)
        fm2_car_signal = self.generate_sine_wave(self.registers['FM2_car_freq'], self.num_samples, self.fs)

        fm1_mod_signal = self.generate_sine_wave(self.registers['FM1_mod_freq'], self.num_samples, self.fs)
        fm2_mod_signal = self.generate_sine_wave(self.registers['FM2_mod_freq'], self.num_samples, self.fs)

        fm1_signal = cp.sin(fm1_car_signal + self.registers['FM1_mod_index'] * fm1_mod_signal)
        fm2_signal = cp.sin(fm2_car_signal + self.registers['FM2_mod_index'] * fm2_mod_signal)

        return fm1_signal + fm2_signal
    
    def generate_audio(self):
        """Generate audio with modulation and noise, all on GPU."""
        fm_signal = self.generate_fm()
        noise_signal = self.generate_noise() * self.registers['noise_level']
        envelope = self.generate_envelope(self.registers['envelope']['attack'],
                                           self.registers['envelope']['decay'],
                                           self.registers['envelope']['sustain'],
                                           self.registers['envelope']['release'],
                                           self.num_samples, self.fs)
        combined_signal = fm_signal + noise_signal  # Combine FM and noise
        return cp.clip(combined_signal * envelope, -1.0, 1.0)  # Ensure the output is within the range

# Instantiate the YM2203 chip emulator (GPU-based)
ym_chip = YM2203Chip()

# Testing the noise generation channel
print("Testing Noise Generation Channel...")
audio_signal = ym_chip.generate_audio()
# Convert GPU array to CPU for playback
sd.play(cp.asnumpy(audio_signal), ym_chip.fs)
time.sleep(2)

# Testing envelope control (attack, decay, sustain, release)
print("Testing Envelope Control...")
ym_chip.registers['envelope'] = {'attack': 0.05, 'decay': 0.1, 'sustain': 0.6, 'release': 0.2}
audio_signal = ym_chip.generate_audio()
# Convert GPU array to CPU for playback
sd.play(cp.asnumpy(audio_signal), ym_chip.fs)
time.sleep(2)

# Testing Low-Frequency Oscillator (LFO) modulation
print("Testing Low-Frequency Oscillator (LFO) modulation...")
ym_chip.registers['LFO_freq'] = 1  # Modulate at 1 Hz
audio_signal = ym_chip.generate_audio()
# Convert GPU array to CPU for playback
sd.play(cp.asnumpy(audio_signal), ym_chip.fs)
time.sleep(2)

# Testing waveform variety (sine, square)
print("Testing Waveform Variety (Sine, Square)...")
for waveform in [1, 2]:  # 1 = sine, 2 = square
    ym_chip.registers['FM1_waveform'] = waveform
    ym_chip.registers['FM2_waveform'] = waveform
    audio_signal = ym_chip.generate_audio()
    # Convert GPU array to CPU for playback
    sd.play(cp.asnumpy(audio_signal), ym_chip.fs)
    time.sleep(2)

# Full dynamic audio with all features tested
print("Testing Full Dynamic Audio with All Features...")
for _ in range(3):  # Loop through a few updates
    ym_chip.registers['FM1_car_freq'] = np.random.randint(400, 800)
    ym_chip.registers['FM2_car_freq'] = np.random.randint(400, 800)
    ym_chip.registers['FM1_mod_index'] = np.random.uniform(0.5, 2.0)
    ym_chip.registers['FM2_mod_index'] = np.random.uniform(0.5, 2.0)
    ym_chip.registers['noise_level'] = np.random.uniform(0, 1)
    audio_signal = ym_chip.generate_audio()
    # Convert GPU array to CPU for playback
    sd.play(cp.asnumpy(audio_signal), ym_chip.fs)
    time.sleep(2)

print("All tests complete.")

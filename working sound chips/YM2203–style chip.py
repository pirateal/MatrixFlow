import cupy as cp
import numpy as np
import time
import sounddevice as sd

class YM2203Chip:
    def __init__(self, fs=44100, num_samples=44100):
        self.fs = fs
        self.num_samples = num_samples

        # --- FM Channels (3 channels) ---
        # Each FM channel has carrier/modulator settings, an ADSR envelope,
        # and optional LFO modulation (frequency and depth).
        self.fm_channels = [
            {
                'carrier_freq': 440,
                'modulator_freq': 110,
                'mod_index': 2.0,
                'carrier_waveform': 1,  # 1 = sine, 2 = square
                'modulator_waveform': 1,
                'envelope': {'attack': 0.01, 'decay': 0.2, 'sustain': 0.8, 'release': 0.3},
                'LFO_freq': 0,      # Default: no LFO
                'LFO_depth': 0.0
            },
            {
                'carrier_freq': 660,
                'modulator_freq': 165,
                'mod_index': 2.5,
                'carrier_waveform': 1,
                'modulator_waveform': 1,
                'envelope': {'attack': 0.02, 'decay': 0.15, 'sustain': 0.7, 'release': 0.25},
                'LFO_freq': 0,
                'LFO_depth': 0.0
            },
            {
                'carrier_freq': 550,
                'modulator_freq': 137,
                'mod_index': 1.8,
                'carrier_waveform': 1,
                'modulator_waveform': 1,
                'envelope': {'attack': 0.015, 'decay': 0.18, 'sustain': 0.75, 'release': 0.28},
                'LFO_freq': 0,
                'LFO_depth': 0.0
            }
        ]
        
        # --- PSG Channels (3 channels) ---
        # These channels simulate the AY-3-8910–style square waves.
        self.psg_channels = [
            {
                'tone_freq': 220,
                'waveform': 2,  # 2 = square wave
                'envelope': {'attack': 0.005, 'decay': 0.1, 'sustain': 0.9, 'release': 0.2}
            },
            {
                'tone_freq': 330,
                'waveform': 2,
                'envelope': {'attack': 0.005, 'decay': 0.1, 'sustain': 0.9, 'release': 0.2}
            },
            {
                'tone_freq': 440,
                'waveform': 2,
                'envelope': {'attack': 0.005, 'decay': 0.1, 'sustain': 0.9, 'release': 0.2}
            }
        ]
        
        # --- Noise Channel ---
        self.noise_level = 0.3  # Noise amplitude scaling
        
        # --- Master Envelope (applied to overall mix) ---
        self.master_envelope = {'attack': 0.01, 'decay': 0.2, 'sustain': 1.0, 'release': 0.3}

    # -- Waveform Generators using GPU Matrix Operations --
    def generate_sine_wave(self, frequency):
        t = cp.arange(self.num_samples) / self.fs
        return cp.sin(2 * cp.pi * frequency * t)
    
    def generate_square_wave(self, frequency):
        t = cp.arange(self.num_samples) / self.fs
        return cp.sign(cp.sin(2 * cp.pi * frequency * t))
    
    def generate_noise(self):
        return cp.random.uniform(-1, 1, self.num_samples)
    
    def generate_envelope(self, attack, decay, sustain, release):
        attack_samples = int(attack * self.fs)
        decay_samples = int(decay * self.fs)
        release_samples = int(release * self.fs)
        sustain_samples = self.num_samples - (attack_samples + decay_samples + release_samples)
        if sustain_samples < 0:
            sustain_samples = 0

        attack_env = cp.linspace(0, 1, attack_samples) if attack_samples > 0 else cp.array([])
        decay_env = cp.linspace(1, sustain, decay_samples) if decay_samples > 0 else cp.array([])
        sustain_env = cp.full(sustain_samples, sustain) if sustain_samples > 0 else cp.array([])
        release_env = cp.linspace(sustain, 0, release_samples) if release_samples > 0 else cp.array([])

        envelope = cp.concatenate([attack_env, decay_env, sustain_env, release_env])
        # Pad or trim to ensure the envelope is exactly num_samples long.
        if envelope.shape[0] < self.num_samples:
            pad = cp.full(self.num_samples - envelope.shape[0], sustain)
            envelope = cp.concatenate([envelope, pad])
        elif envelope.shape[0] > self.num_samples:
            envelope = envelope[:self.num_samples]
        return envelope

    # -- FM Synthesis Channel --
    def generate_fm_channel(self, channel):
        t = cp.arange(self.num_samples) / self.fs
        # Generate the modulator waveform (sine or square)
        if channel['modulator_waveform'] == 1:
            mod_signal = cp.sin(2 * cp.pi * channel['modulator_freq'] * t)
        else:
            mod_signal = cp.sign(cp.sin(2 * cp.pi * channel['modulator_freq'] * t))
        
        # Apply optional LFO modulation if enabled
        lfo_freq = channel.get('LFO_freq', 0)
        if lfo_freq > 0:
            lfo = cp.sin(2 * cp.pi * lfo_freq * t)
            mod_signal = mod_signal + channel.get('LFO_depth', 0.0) * lfo

        # Calculate the instantaneous phase for the carrier.
        instantaneous_phase = 2 * cp.pi * channel['carrier_freq'] * t + channel['mod_index'] * mod_signal
        
        # Generate the carrier waveform (sine or square)
        if channel['carrier_waveform'] == 1:
            carrier_signal = cp.sin(instantaneous_phase)
        else:
            carrier_signal = cp.sign(cp.sin(instantaneous_phase))
        
        # Apply the channel envelope.
        env_params = channel.get('envelope', {'attack':0.01, 'decay':0.1, 'sustain':0.8, 'release':0.3})
        envelope = self.generate_envelope(env_params['attack'],
                                          env_params['decay'],
                                          env_params['sustain'],
                                          env_params['release'])
        return carrier_signal * envelope

    # -- PSG (Square Wave) Channel --
    def generate_psg_channel(self, channel):
        t = cp.arange(self.num_samples) / self.fs
        tone_signal = cp.sign(cp.sin(2 * cp.pi * channel['tone_freq'] * t))
        env_params = channel.get('envelope', {'attack':0.005, 'decay':0.1, 'sustain':0.9, 'release':0.2})
        envelope = self.generate_envelope(env_params['attack'],
                                          env_params['decay'],
                                          env_params['sustain'],
                                          env_params['release'])
        return tone_signal * envelope

    # -- Combine All Channels into a Final Audio Signal --
    def generate_audio(self):
        # Sum FM channels if available
        if self.fm_channels:
            fm_signals = [self.generate_fm_channel(ch) for ch in self.fm_channels]
            fm_sum = cp.sum(cp.stack(fm_signals), axis=0)
        else:
            fm_sum = cp.zeros(self.num_samples)
        
        # Sum PSG channels if available
        if self.psg_channels:
            psg_signals = [self.generate_psg_channel(ch) for ch in self.psg_channels]
            psg_sum = cp.sum(cp.stack(psg_signals), axis=0)
        else:
            psg_sum = cp.zeros(self.num_samples)
        
        # Noise channel
        noise_signal = self.generate_noise() * self.noise_level
        
        # Combined mix of FM, PSG, and Noise
        combined_signal = fm_sum + psg_sum + noise_signal
        
        # Apply the master envelope
        master_env = self.generate_envelope(self.master_envelope['attack'],
                                            self.master_envelope['decay'],
                                            self.master_envelope['sustain'],
                                            self.master_envelope['release'])
        final_signal = cp.clip(combined_signal * master_env, -1.0, 1.0)
        return final_signal

# ----------------------- Full Sound Test Routine -----------------------
if __name__ == "__main__":
    chip = YM2203Chip(fs=44100, num_samples=44100)

    def play_signal(signal, duration=3, description=""):
        print(description)
        audio = cp.asnumpy(signal)
        sd.play(audio, chip.fs)
        time.sleep(duration)
        sd.stop()
        time.sleep(0.5)

    print("Starting full sound test...\n")

    # Test 1: FM channels only.
    print("Test 1: FM channels only")
    # Temporarily mute PSG channels and noise.
    original_psg = chip.psg_channels.copy()
    original_noise = chip.noise_level
    chip.psg_channels = []
    chip.noise_level = 0.0
    fm_audio = cp.sum(cp.stack([chip.generate_fm_channel(ch) for ch in chip.fm_channels]), axis=0)
    play_signal(fm_audio, duration=3, description="Playing FM channels only.")
    
    # Restore PSG and noise.
    chip.psg_channels = original_psg
    chip.noise_level = original_noise

    # Test 2: PSG channels only.
    print("Test 2: PSG channels only")
    original_fm = chip.fm_channels.copy()
    chip.fm_channels = []
    chip.noise_level = 0.0
    psg_audio = cp.sum(cp.stack([chip.generate_psg_channel(ch) for ch in chip.psg_channels]), axis=0)
    play_signal(psg_audio, duration=3, description="Playing PSG channels only.")
    
    # Restore FM channels and noise.
    chip.fm_channels = original_fm
    chip.noise_level = original_noise

    # Test 3: Noise channel only.
    print("Test 3: Noise channel only")
    chip.fm_channels = []
    chip.psg_channels = []
    noise_audio = chip.generate_noise() * chip.noise_level
    play_signal(noise_audio, duration=3, description="Playing noise channel only.")
    
    # Restore FM and PSG channels.
    chip.fm_channels = original_fm
    chip.psg_channels = original_psg

    # Test 4: Combined full mix (FM + PSG + Noise).
    print("Test 4: Combined full mix")
    full_audio = chip.generate_audio()
    play_signal(full_audio, duration=4, description="Playing combined full mix.")

    # Test 5: FM channels with LFO modulation.
    print("Test 5: FM channels with LFO modulation")
    # Enable LFO on the first FM channel.
    chip.fm_channels[0]['LFO_freq'] = 5     # 5 Hz LFO
    chip.fm_channels[0]['LFO_depth'] = 0.5    # Moderate depth
    lfo_audio = cp.sum(cp.stack([chip.generate_fm_channel(ch) for ch in chip.fm_channels]), axis=0)
    play_signal(lfo_audio, duration=4, description="Playing FM channels with LFO modulation on channel 1.")
    
    # Reset LFO parameters.
    chip.fm_channels[0]['LFO_freq'] = 0
    chip.fm_channels[0]['LFO_depth'] = 0.0

    print("Full sound test complete.")

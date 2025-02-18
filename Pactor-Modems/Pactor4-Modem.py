import numpy as np
import pygame
import time
import random
import string
from enum import Enum
import struct
import sys

# --- Packet Type Definition ---
class PacketType(Enum):
    DATA = 1
    ACK = 2
    CONTROL = 3
    CRC_ERROR = 4
    RETRY = 5

# --- PACTOR Configuration ---
FREQUENCY_CONFIGS = {
    'PACTOR_I': {
        'carrier': 1500,
        'shift': 200,
        'name': 'PACTOR-I',
        'speed_level': 1,
        'crc_type': 'CRC-16',
        'compression': False,
        'memory_arq': False
    },
    'PACTOR_II': {
        'carrier': 1700,
        'shift': 450,
        'name': 'PACTOR-II',
        'speed_level': 2,
        'crc_type': 'CRC-16-CCITT',
        'compression': True,
        'memory_arq': True
    },
    'PACTOR_III': {
        'carrier': 2000,
        'shift': 600,
        'name': 'PACTOR-III',
        'speed_level': 3,
        'crc_type': 'CRC-32',
        'compression': True,
        'memory_arq': True
    },
    'PACTOR_IV': {
        'carrier': 2200,
        'shift': 800,
        'name': 'PACTOR-IV',
        'speed_level': 4,
        'crc_type': 'CRC-32',
        'compression': True,
        'memory_arq': True
    }
}

SPEED_LEVELS = {
    1: 200,    # PACTOR-I
    2: 400,    # PACTOR-II
    3: 800,    # PACTOR-III
    4: 1600    # PACTOR-IV
}

SAMPLE_RATE = 48000
TEST_DURATION = 1  # seconds

# --- PACTOR Modem Class ---
class PACTORModem:
    def __init__(self, mode='PACTOR_I'):
        self.mode = mode
        self.config = FREQUENCY_CONFIGS[mode]
        self.carrier_freq = self.config['carrier']
        self.shift = self.config['shift']
        self.mark_freq = self.carrier_freq + self.shift / 2
        self.space_freq = self.carrier_freq - self.shift / 2
        self.speed = SPEED_LEVELS[self.config['speed_level']]
        self.sample_rate = SAMPLE_RATE
        self.compression_enabled = self.config['compression']
        self.memory_arq = self.config['memory_arq']
        self.packet_counter = 0
        self.retry_count = 0
        self.max_retries = 3

        pygame.init()
        pygame.mixer.init(frequency=self.sample_rate, channels=2)
        print(f"Initialized {mode} modem at {self.speed} bps")

    def calculate_crc(self, data):
        """
        Calculate a dummy CRC value by converting the payload to a numeric array.
        This avoids issues with byte-string arithmetic.
        """
        arr = np.frombuffer(data, dtype=np.uint8)
        if self.config['crc_type'] == 'CRC-32':
            return np.int32(arr.sum() % (2**32))
        return np.int16(arr.sum() % (2**16))

    def compress_data(self, data):
        """Simple run-length encoding compression."""
        if not self.compression_enabled:
            return data, 1.0

        compressed = bytearray()
        count = 1
        current = data[0]
        for byte in data[1:]:
            if byte == current and count < 255:
                count += 1
            else:
                compressed.extend([count, current])
                count = 1
                current = byte
        compressed.extend([count, current])
        compression_ratio = len(data) / len(compressed)
        return bytes(compressed), compression_ratio

    def generate_test_packet(self, packet_type=PacketType.DATA):
        """Generate a test PACTOR packet with header and payload."""
        header = bytearray()
        header.extend(struct.pack('B', packet_type.value))
        header.extend(struct.pack('H', self.packet_counter))

        payload_size = self.speed // 8
        payload = ''.join(random.choices(string.ascii_letters + string.digits, k=payload_size)).encode()

        if self.compression_enabled:
            payload, ratio = self.compress_data(payload)
            header.extend(struct.pack('f', ratio))

        crc = self.calculate_crc(payload)
        header.extend(struct.pack('I', crc))
        return header + payload

    def add_doppler_effect(self, signal, velocity=10):
        """Simulate Doppler shift based on relative velocity (m/s)."""
        t = np.linspace(0, len(signal) / self.sample_rate, len(signal))
        doppler_factor = 1 + velocity / 343.0  # speed of sound ~343 m/s
        return signal * np.cos(2 * np.pi * self.carrier_freq * doppler_factor * t)

    def add_multipath(self, signal, delay_ms=2):
        """Simulate multipath propagation."""
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        delayed = np.pad(signal, (delay_samples, 0))[:-delay_samples]
        return signal + 0.3 * delayed

    def add_channel_effects(self, signal, snr_db=20):
        """Apply noise, multipath, Doppler, and fading to the signal."""
        snr = 10 ** (snr_db / 10)
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / snr
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

        signal = self.add_multipath(signal)
        signal = self.add_doppler_effect(signal)

        t = np.linspace(0, len(signal) / self.sample_rate, len(signal))
        slow_fading = 1 + 0.2 * np.sin(2 * np.pi * 0.1 * t)
        fast_fading = 1 + 0.1 * np.sin(2 * np.pi * 5 * t)
        fading = slow_fading * fast_fading

        return (signal * fading + noise) / 2

    def detect_packet_loss(self, signal, threshold=0.3):
        """Detect potential packet loss based on signal quality."""
        signal_power = np.mean(np.abs(signal) ** 2)
        return signal_power < threshold

    def memory_arq_correction(self, signal):
        """Simulate Memory ARQ correction for modes with memory ARQ enabled."""
        if not self.memory_arq:
            return signal
        if self.detect_packet_loss(signal):
            self.retry_count += 1
            if self.retry_count <= self.max_retries:
                print(f"Packet loss detected – Retry {self.retry_count}/{self.max_retries}")
                return self.transmit_with_arq()
        self.retry_count = 0
        return signal

    def transmit_with_arq(self):
        """Implement a basic ARQ protocol."""
        packet = self.generate_test_packet(PacketType.RETRY)
        modulated = self.modulate_fsk(packet)
        return self.add_channel_effects(modulated)

    def modulate_fsk(self, data):
        """Standard FSK modulation with enhanced synchronization."""
        samples_per_bit = int(self.sample_rate / self.speed)
        t = np.linspace(0, 1 / self.speed, samples_per_bit)
        modulated = np.array([])

        # Enhanced preamble with a fixed sync pattern
        sync_pattern = [1, 0, 1, 0, 1, 1, 0, 0]
        for bit in sync_pattern:
            freq = self.mark_freq if bit else self.space_freq
            modulated = np.append(modulated, np.sin(2 * np.pi * freq * t))

        phase = 0
        for byte in data:
            for i in range(8):
                bit = (byte >> (7 - i)) & 1
                freq = self.mark_freq if bit else self.space_freq
                signal = np.sin(2 * np.pi * freq * t + phase)
                phase = np.remainder(phase + 2 * np.pi * freq * t[-1], 2 * np.pi)
                modulated = np.append(modulated, signal)
        return modulated

    def matrix_flow_modulate_fsk(self, data):
        """
        NEW: MatrixFlow-based FSK modulation using dummy matrix operations.
        Replace the identity transformations with your FP8/GPU-based routines.
        """
        samples_per_bit = int(self.sample_rate / self.speed)
        t = np.linspace(0, 1 / self.speed, samples_per_bit)
        modulated = np.array([])

        # Process sync pattern using matrix operations
        sync_pattern = [1, 0, 1, 0, 1, 1, 0, 0]
        for bit in sync_pattern:
            freq = self.mark_freq if bit else self.space_freq
            block = np.sin(2 * np.pi * freq * t)
            # Dummy matrix transformation (identity matrix as placeholder)
            transformation_matrix = np.eye(len(block))
            block = np.dot(transformation_matrix, block)
            modulated = np.append(modulated, block)

        phase = 0
        for byte in data:
            for i in range(8):
                bit = (byte >> (7 - i)) & 1
                freq = self.mark_freq if bit else self.space_freq
                block = np.sin(2 * np.pi * freq * t + phase)
                # Apply dummy matrix processing here
                transformation_matrix = np.eye(len(block))
                block = np.dot(transformation_matrix, block)
                phase = np.remainder(phase + 2 * np.pi * freq * t[-1], 2 * np.pi)
                modulated = np.append(modulated, block)
        return modulated

    def transmit(self, snr_db=20):
        """Perform a standard test transmission using traditional modulation."""
        print(f"\nTransmitting in {self.mode} mode")
        print(f"Carrier: {self.carrier_freq}Hz, Shift: {self.shift}Hz")
        print(f"Features: Compression={self.compression_enabled}, Memory ARQ={self.memory_arq}")

        start_time = time.time()
        packet = self.generate_test_packet()
        modulated = self.modulate_fsk(packet)
        signal_with_effects = self.add_channel_effects(modulated, snr_db)

        if self.memory_arq:
            signal_with_effects = self.memory_arq_correction(signal_with_effects)

        normalized = signal_with_effects / np.max(np.abs(signal_with_effects))
        stereo = np.column_stack((normalized, normalized))
        scaled = np.int16(stereo * 32767)

        sound = pygame.sndarray.make_sound(scaled)
        sound.play()
        pygame.time.wait(TEST_DURATION * 1000)
        end_time = time.time()
        self.packet_counter += 1

        return {
            'mode': self.mode,
            'data_size': len(packet),
            'duration': end_time - start_time,
            'speed': self.speed,
            'snr': snr_db,
            'compression': self.compression_enabled,
            'memory_arq': self.memory_arq,
            'retries': self.retry_count
        }

    def backdoor_test(self, snr_db=20):
        """
        BACKDOOR TEST: Hidden mode that uses the MatrixFlow (matrix logic) technique.
        This method uses the new matrix_flow_modulate_fsk function.
        """
        print("\n[BACKDOOR TEST] Activating hidden MatrixFlow-based modulation test!")
        print(f"[BACKDOOR TEST] Mode: {self.mode} - Using Matrix Logic Technique")
        packet = self.generate_test_packet()
        modulated = self.matrix_flow_modulate_fsk(packet)
        signal_with_effects = self.add_channel_effects(modulated, snr_db)

        if self.memory_arq:
            signal_with_effects = self.memory_arq_correction(signal_with_effects)

        normalized = signal_with_effects / np.max(np.abs(signal_with_effects))
        stereo = np.column_stack((normalized, normalized))
        scaled = np.int16(stereo * 32767)

        sound = pygame.sndarray.make_sound(scaled)
        sound.play()
        pygame.time.wait(TEST_DURATION * 1000)
        print("[BACKDOOR TEST] Transmission complete.\n")
        return {
            'mode': self.mode,
            'data_size': len(packet),
            'speed': self.speed,
            'snr': snr_db,
            'matrix_flow': True
        }

# --- Test Sequences ---
def run_test_sequence():
    """Run the standard test sequence."""
    results = []
    snr_levels = [20, 15, 10, 5]
    print("Starting Enhanced PACTOR Modem Test Sequence")
    print("=" * 50)
    for mode in FREQUENCY_CONFIGS.keys():
        print(f"\nTesting {mode}")
        modem = PACTORModem(mode)
        for snr in snr_levels:
            print(f"\nTesting with SNR: {snr} dB")
            result = modem.transmit(snr)
            results.append(result)
            effective_speed = (result['data_size'] * 8 / result['duration']) if result.get('duration') else 0
            print(f"Transmitted {result['data_size']} bytes at effective speed: {effective_speed:.1f} bps")
            time.sleep(0.5)
    return results

def run_backdoor_tests():
    """Run the hidden MatrixFlow backdoor test sequence."""
    results = []
    snr_levels = [20, 15, 10, 5]
    print("Starting BACKDOOR MatrixFlow-based PACTOR Test Sequence")
    print("=" * 60)
    for mode in FREQUENCY_CONFIGS.keys():
        print(f"\n[BACKDOOR] Testing {mode}")
        modem = PACTORModem(mode)
        for snr in snr_levels:
            print(f"\n[BACKDOOR] Testing with SNR: {snr} dB")
            result = modem.backdoor_test(snr)
            results.append(result)
            time.sleep(0.5)
    return results

def display_detailed_results(results):
    """Display comprehensive test results."""
    print("\nDetailed Test Results")
    print("=" * 60)
    for mode in FREQUENCY_CONFIGS.keys():
        mode_results = [r for r in results if r['mode'] == mode]
        if mode_results:
            print(f"\n{mode} Performance Summary:")
            avg_speed = np.mean([r['data_size'] * 8 / r.get('duration', 1) for r in mode_results])
            print(f"Average Speed: {avg_speed:.1f} bps")
            if mode_results[0].get('matrix_flow', False):
                print("Test Mode: MatrixFlow Backdoor Test")

def main():
    pygame.init()
    pygame.mixer.init(frequency=SAMPLE_RATE, channels=2)
    print("Enhanced PACTOR Modem Simulator with MatrixFlow Integration")
    
    # If a command-line argument "backdoor" is provided, run the hidden test.
    if len(sys.argv) > 1 and sys.argv[1].lower() == "backdoor":
        results = run_backdoor_tests()
    else:
        results = run_test_sequence()
        display_detailed_results(results)
    pygame.quit()

if __name__ == "__main__":
    main()

# dsp_core.py
import cupy as cp
from matrix_mac import matrix_mac
from fft_unit import FFTUnit
from filter_unit import FIRFilter
from modulation_unit import am_modulate, fm_modulate
from adaptive_ai import AdaptiveFilter

class DSPCore:
    def __init__(self, fs=44100):
        self.fs = fs
        self.fft_unit = FFTUnit(fs)
        self.adaptive_filter = AdaptiveFilter(filter_length=10, learning_rate=0.001)
    
    def process_signal(self, signal, filter_coeffs):
        """
        Process an input signal by applying an FIR filter followed by FFT analysis.
        signal: cp.array, 1D input signal.
        filter_coeffs: cp.array, FIR filter coefficients.
        Returns: cp.array, processed signal.
        """
        # Apply FIR filtering
        filtered_signal = FIRFilter(signal, filter_coeffs)
        # Compute FFT of the filtered signal
        freq_domain = self.fft_unit.fft_transform(filtered_signal)
        # Perform inverse FFT to return to time domain
        processed_signal = self.fft_unit.ifft_transform(freq_domain)
        return processed_signal

    def modulate_signal(self, signal, carrier_freq, mode="AM", modulation_index=0.5):
        """
        Modulate the signal using AM or FM modulation.
        """
        if mode.upper() == "AM":
            return am_modulate(signal, carrier_freq, self.fs, modulation_index)
        elif mode.upper() == "FM":
            return fm_modulate(signal, carrier_freq, self.fs, modulation_index)
        else:
            raise ValueError("Unsupported modulation mode.")

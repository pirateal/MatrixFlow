# examples/test_modulation.py
import cupy as cp
import sounddevice as sd
from src.modulation_unit import am_modulate, fm_modulate

def main():
    fs = 44100
    t = cp.linspace(0, 1, fs)
    # Modulating signal: 5 Hz sine wave
    mod_signal = cp.sin(2 * cp.pi * 5 * t)
    # Apply AM modulation with a 1000 Hz carrier
    am_signal = am_modulate(mod_signal, 1000, fs, modulation_index=0.7)
    # Apply FM modulation with a 1000 Hz carrier
    fm_signal = fm_modulate(mod_signal, 1000, fs, modulation_index=50.0)
    
    print("Playing AM modulated signal...")
    sd.play(cp.asnumpy(am_signal), fs)
    sd.wait()
    
    print("Playing FM modulated signal...")
    sd.play(cp.asnumpy(fm_signal), fs)
    sd.wait()

if __name__ == "__main__":
    main()

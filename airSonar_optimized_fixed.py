# coding=utf-8

"""
airSonar_optimized_fixed.py - Fixed Optimized Version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Real-time air sonar system for Windows platform with maximum range of 50m.

Copyright © 2025
"""

import sys, time, csv, logging, queue, math, os, traceback
from pathlib import Path
from threading import Lock, Event, Thread, Timer
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.signal import chirp, iirfilter, filtfilt, firwin, fftconvolve, correlate
import pyaudio
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# GPU Support
try:
    import cupy as cp
    import cupyx.scipy.signal as cpx_signal
    GPU_AVAILABLE = True
    print("GPU (CuPy) Support Enabled")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    print("GPU (CuPy) Support Not Available, using CPU fallback")

# Set font support for better display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ========= Logging Setup ========= #
CSV_PATH = Path("distances.csv")
LOG_PATH = Path("sonar.log")

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Sonar")

# Enable faulthandler for debugging
try:
    import faulthandler
    faulthandler.enable()
except Exception as e:
    logger.warning(f"faulthandler setup failed: {e}")

# ========= Configuration ========= #
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Config:
    FS: int = 48000
    BASE_TEMP: float = 28.0
    R_MIN: float = 0.5
    R_MAX: float = 15.0
    CYCLE_MARGIN: float = 0.02
    CHANNELS: int = 1
    FORMAT: int = pyaudio.paInt16
    BANDS: list = field(default_factory=lambda: [(9500,11500),(13500,15500),(17500,19500)])
    PLOT_UPDATE_INTERVAL: int = 1
    SPECTRUM_UPDATE_INTERVAL: int = 1
    MAX_HIST_POINTS: int = 300
    GUI_UPDATE_RATE: int = 30
    PLOT_DECIMATION: int = 1
    HEARTBEAT_INTERVAL: float = 0.1
    SNR_NOISE_MS: float = 0.005
    SPECTRUM_CACHE_SEC: float = 0.5
    
    @property
    def CHIRP_LEN(self):
        return 2*self.R_MIN/(343.0*math.sqrt(1+(self.BASE_TEMP-20)/273.15))
    
    @property
    def LISTEN_LEN(self):
        return 2*self.R_MAX/(343.0*math.sqrt(1+(self.BASE_TEMP-20)/273.15))+0.003

cfg = Config()

# ========= Core Functions ========= #
def calculate_sound_speed(temperature_c):
    """Calculate sound speed (m/s) based on temperature"""
    return 331.3 + 0.606 * temperature_c

def generate_chirps(fs=None, duration=None):
    fs = fs if fs is not None else cfg.FS
    duration = duration if duration is not None else cfg.CHIRP_LEN
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    chirps = [chirp(t, f0=l, f1=h, t1=duration, method='linear').astype(np.float32)
              for l, h in cfg.BANDS]
    mix = np.sum(chirps, axis=0)
    mix *= 0.85*(2**15-1)/np.max(np.abs(mix))
    return mix.astype(np.int16), chirps

def design_filters(fs=None):
    fs = fs if fs is not None else cfg.FS
    filters = []
    for low, high in cfg.BANDS:
        try:
            ba = iirfilter(6, [low/(0.5*fs), high/(0.5*fs)], btype='band', output='ba')
            if ba is None:
                raise ValueError(f"iirfilter failed for band {low}-{high} Hz")
            b, a = ba[0], ba[1]
            taps = firwin(61, [low, high], fs=fs, pass_zero=False, window='hamming')
            filters.append((b, a, taps))
        except Exception as e:
            logger.error(f"Failed to design filter for band {low}-{high} Hz: {e}")
            b, a = [1], [1]
            taps = np.array([1])
            filters.append((b, a, taps))
    return filters

def bandpass(sig, filt):
    """Bandpass filter with GPU acceleration"""
    if GPU_AVAILABLE:
        return gpu_bandpass(sig, filt)
    
    b, a, taps = filt
    try:
        y = filtfilt(b, a, sig)
        return fftconvolve(y, taps, mode='same')
    except Exception as e:
        logger.warning(f"Filtering failed, returning original signal: {e}")
        return sig

def first_strong_peak(corr, fs, min_delay_samples=None):
    """Adaptive threshold peak detection"""
    if min_delay_samples is None:
        min_delay_samples = int(fs * cfg.CHIRP_LEN * 1.2)
    half = corr.size//2
    pos = corr[half:]
    if pos.size <= min_delay_samples:
        return None, 0.0
    pos[:min_delay_samples] = 0
    noise_samples = int(cfg.SNR_NOISE_MS * fs)
    noise_floor = np.median(np.abs(pos[min_delay_samples:min_delay_samples+noise_samples]))**2
    peak_idx = gpu_argmax(pos)
    peak_power = pos[peak_idx] ** 2
    if noise_floor > 0:
        snr_db = 10 * np.log10(peak_power / noise_floor)
    else:
        snr_db = 0.0
    if snr_db < 6.0:
        return None, 0.0
    return peak_idx, snr_db

def calculate_band_confidence(snr, amplitude, band_idx):
    """Calculate frequency band confidence"""
    snr_weight = min(snr / 10.0, 1.0)
    amp_weight = min(amplitude / 0.1, 1.0)
    freq_weights = [0.8, 1.0, 0.9]
    freq_weight = freq_weights[band_idx]
    confidence = (snr_weight * 0.5 + amp_weight * 0.3 + freq_weight * 0.2)
    return min(confidence, 1.0)

def normalize_confidences(confidences):
    """Normalize confidences to ensure sum equals 100%"""
    confidences = np.array(confidences)
    total = np.sum(confidences)
    if total > 0:
        normalized = (confidences / total) * 100.0
    else:
        normalized = np.full_like(confidences, 100.0 / len(confidences))
    return normalized

class ScalarKalman:
    def __init__(self, q=0.005, r=0.1):
        self.x = None
        self.p = 1.0
        self.q = q
        self.r = r

    def update(self, z):
        if self.x is None:
            self.x = z
            return z
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k*(z - self.x)
        self.p *= (1 - k)
        return self.x

def mag2db(x):
    return 20*np.log10(np.maximum(np.abs(x), 1e-12))

# ========= GPU Functions ========= #
def gpu_correlate(a, b):
    """GPU accelerated correlation"""
    if GPU_AVAILABLE and cp is not None:
        try:
            a_gpu = cp.asarray(a)
            b_gpu = cp.asarray(b)
            result = cpx_signal.fftconvolve(a_gpu, b_gpu[::-1], mode='full')
            return cp.asnumpy(result)
        except Exception as e:
            logger.warning(f"GPU correlation failed: {e}")
            return correlate(a, b, 'full')
    else:
        return correlate(a, b, 'full')

def gpu_bandpass(signal, filt):
    """GPU accelerated bandpass filter"""
    if GPU_AVAILABLE and cp is not None:
        try:
            b, a, taps = filt
            signal_gpu = cp.asarray(signal)
            taps_gpu = cp.asarray(taps)
            filtered = cp.convolve(signal_gpu, taps_gpu, mode='same')
            return cp.asnumpy(filtered)
        except Exception as e:
            logger.warning(f"GPU bandpass failed: {e}, falling back to CPU")
            b, a, taps = filt
            try:
                y = filtfilt(b, a, signal)
                return fftconvolve(y, taps, mode='same')
            except Exception as e2:
                logger.error(f"CPU bandpass also failed: {e2}")
                return signal
    else:
        b, a, taps = filt
        try:
            y = filtfilt(b, a, signal)
            return fftconvolve(y, taps, mode='same')
        except Exception as e:
            logger.error(f"CPU bandpass failed: {e}")
            return signal

def gpu_argmax(signal):
    """GPU accelerated argmax"""
    if GPU_AVAILABLE and cp is not None:
        try:
            signal_gpu = cp.asarray(signal)
            return int(cp.argmax(signal_gpu))
        except:
            return np.argmax(signal)
    else:
        return np.argmax(signal)

# ========= Audio I/O ========= #
class AudioIO:
    """Audio input/output with isolation"""
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.out = self.p.open(format=cfg.FORMAT, channels=cfg.CHANNELS,
                               rate=cfg.FS, output=True, frames_per_buffer=1024)
        self.inp = self.p.open(format=cfg.FORMAT, channels=cfg.CHANNELS,
                               rate=cfg.FS, input=True,
                               frames_per_buffer=int(cfg.FS*cfg.LISTEN_LEN))
        self.silence_buffer = np.zeros(int(cfg.FS * 0.02), dtype=np.int16)
    
    def play(self, pcm16):
        """Play audio with isolation"""
        try:
            self.out.write(self.silence_buffer.tobytes())
            time.sleep(0.02)
            self.out.write(pcm16.tobytes())
            self.out.write(self.silence_buffer.tobytes())
            time.sleep(0.02)
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
    
    def record(self):
        """Record audio with isolation"""
        try:
            time.sleep(0.03)
            for _ in range(5):
                try:
                    self.inp.read(1024, exception_on_overflow=False)
                except Exception:
                    break
            raw = self.inp.read(int(cfg.FS*cfg.LISTEN_LEN), exception_on_overflow=False)
            sig = np.frombuffer(raw, np.int16).astype(np.float32)/2**15
            return sig
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
            return np.zeros(int(cfg.FS*cfg.LISTEN_LEN), dtype=np.float32)
    
    def close(self):
        try:
            self.out.stop_stream()
            self.out.close()
            self.inp.stop_stream() 
            self.inp.close()
            self.p.terminate()
        except Exception as e:
            logger.error(f"Audio close error: {e}")

# ========= Worker Thread ========= #
class SonarWorker(QtCore.QThread):
    """Sonar worker thread"""
    distanceSig = QtCore.pyqtSignal(float, list, float)
    waveSig = QtCore.pyqtSignal(dict)
    errorSig = QtCore.pyqtSignal(str)
    heartbeatSig = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.tx_pcm, self.chirps = generate_chirps()
        self.filters = design_filters()
        self.kf = ScalarKalman()
        self.audio = AudioIO()
        self.running = False
        self.temperature = 20.0
        self.update_counter = 0
        
        if not CSV_PATH.exists():
            with CSV_PATH.open("w", newline='') as f:
                csv.writer(f).writerow(["timestamp", "distance", "confidence", "band_snrs"])

    def run(self):
        self.running = True
        logger.info("SonarWorker started")
        
        try:
            while self.running:
                t0 = time.perf_counter()
                
                # Transmit
                self.audio.play(self.tx_pcm)
                
                # Receive
                rx = self.audio.record()
                
                # Process bands
                results = []
                for i, (chirp, filt) in enumerate(zip(self.chirps, self.filters)):
                    distance, confidence, snr = self._process_band(rx, chirp, filt, i)
                    if distance is not None:
                        results.append((distance, confidence, snr))
                
                # Multi-band fusion
                if results:
                    distances, confidences, snrs = zip(*results)
                    
                    # Normalize confidences
                    confidences_norm = normalize_confidences(confidences)
                    weights = confidences_norm / 100.0 + 1e-9
                    
                    # Weighted average
                    weighted_dist = np.average(distances, weights=weights)
                    avg_confidence = np.mean(confidences_norm)
                    
                    # Kalman filter
                    dist_kf = self.kf.update(weighted_dist)
                    
                    # Emit signals
                    self.distanceSig.emit(dist_kf, list(snrs), avg_confidence)
                    
                    # Log to CSV
                    with CSV_PATH.open("a", newline='') as f:
                        csv.writer(f).writerow([time.time(), dist_kf, avg_confidence, list(snrs)])
                
                # Send wave data for plotting
                self.update_counter += 1
                if self.update_counter % cfg.PLOT_UPDATE_INTERVAL == 0:
                    self.waveSig.emit({
                        'rx': rx,
                        'update_spectrum': True
                    })
                
                # Heartbeat
                self.heartbeatSig.emit()
                
                # Timing
                elapsed = time.perf_counter() - t0
                sleep_time = max(0, (cfg.CHIRP_LEN + cfg.LISTEN_LEN + cfg.CYCLE_MARGIN) - elapsed)
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.exception(f"Worker error: {e}")
            self.errorSig.emit(str(e))
        finally:
            self.audio.close()
            logger.info("SonarWorker stopped")

    def _process_band(self, rx, chirp_sig, filt, band_idx):
        """Process single frequency band"""
        try:
            # Filter
            band_sig = bandpass(rx, filt)
            
            # Correlate
            corr = gpu_correlate(band_sig, chirp_sig)
            
            # Peak detection
            min_delay_samples = int(cfg.FS * cfg.CHIRP_LEN * 1.2)
            peak_idx, snr = first_strong_peak(corr, cfg.FS, min_delay_samples)
            
            if peak_idx is None:
                return None, 0.0, 0.0
            
            # Calculate distance
            delay = peak_idx - (len(chirp_sig) - 1)
            distance = delay / cfg.FS * calculate_sound_speed(self.temperature) / 2
            
            # Calculate confidence
            amplitude = np.max(np.abs(corr))
            confidence = calculate_band_confidence(snr, amplitude, band_idx)
            
            return distance, confidence, snr
            
        except Exception as e:
            logger.exception(f"Band processing error: {e}")
            return None, 0.0, 0.0

    def stop(self):
        self.running = False

# ========= GUI ========= #
class MplCanvas(FigureCanvas):
    """Matplotlib canvas with optimized plotting"""
    def __init__(self, title, parent=None, width=8, height=5.5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        fig.patch.set_facecolor('white')
        self.ax = fig.add_subplot(111)
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        super().__init__(fig)
        self.setMinimumSize(int(width*dpi//2), int(height*dpi//2))

class MainWindow(QtWidgets.QMainWindow):
    """Main GUI window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Air Sonar System (Optimized)")
        self.setupUI()
        self.setupData()

    def setupUI(self):
        """Setup user interface"""
        # Controls
        self.label = QtWidgets.QLabel("Distance: -- m")
        self.label.setStyleSheet("""
            font-size: 28pt; color: #2c3e50; font-weight: bold;
            background: rgba(255,255,255,200); border: 2px solid #3498db;
            border-radius: 15px; padding: 15px;
        """)
        
        self.confidence_label = QtWidgets.QLabel("Confidence: --")
        self.confidence_label.setStyleSheet("""
            font-size: 16pt; color: #27ae60; font-weight: bold;
            background: rgba(255,255,255,150); border: 1px solid #27ae60;
            border-radius: 8px; padding: 8px;
        """)
        
        self.temp_label = QtWidgets.QLabel("Temperature (°C):")
        self.temp_spinbox = QtWidgets.QSpinBox()
        self.temp_spinbox.setRange(-40, 60)
        self.temp_spinbox.setValue(20)
        self.temp_spinbox.setSuffix(" °C")
        self.temp_spinbox.valueChanged.connect(self.on_temp_changed)
        
        self.btnStart = QtWidgets.QPushButton("Start")
        self.btnStop = QtWidgets.QPushButton("Stop")
        self.btnStop.setEnabled(False)
        
        # Charts
        self.txSpec = MplCanvas("Transmit Signal Spectrum", width=6, height=4)
        self.rxSpec = MplCanvas("Received Signal Spectrum", width=6, height=4)
        self.hist = MplCanvas("Distance History", width=12, height=4)
        
        # Layout
        control_frame = QtWidgets.QFrame()
        control_layout = QtWidgets.QHBoxLayout(control_frame)
        control_layout.addWidget(self.btnStart)
        control_layout.addWidget(self.btnStop)
        control_layout.addStretch()
        
        temp_layout = QtWidgets.QHBoxLayout()
        temp_layout.addWidget(self.temp_label)
        temp_layout.addWidget(self.temp_spinbox)
        control_layout.addLayout(temp_layout)
        control_layout.addStretch()
        control_layout.addWidget(self.confidence_label)
        control_layout.addWidget(self.label)
        
        charts_layout = QtWidgets.QGridLayout()
        charts_layout.addWidget(self.txSpec, 0, 0)
        charts_layout.addWidget(self.rxSpec, 0, 1)
        
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(control_frame)
        main_layout.addLayout(charts_layout)
        main_layout.addWidget(self.hist)
        
        central = QtWidgets.QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)
        
        # Signals
        self.btnStart.clicked.connect(self.start)
        self.btnStop.clicked.connect(self.stop)

    def setupData(self):
        """Setup data storage"""
        self.dist_hist = []
        self.time_hist = []
        self.confidence_hist = []
        self.start_time = None
        self.worker = None
        self.last_update_time = 0
        self.min_update_interval = 1.0 / cfg.GUI_UPDATE_RATE
        self.plot_cache = {}

    def on_temp_changed(self, value):
        """Temperature change handler"""
        if self.worker and hasattr(self.worker, 'temperature'):
            self.worker.temperature = float(value)

    @QtCore.pyqtSlot(float, list, float)
    def _on_dist(self, d, snrs, confidence):
        """Distance update handler"""
        current_time = time.time()
        
        if current_time - self.last_update_time < self.min_update_interval:
            return
        self.last_update_time = current_time
        
        if self.start_time is None:
            self.start_time = current_time
        
        elapsed_time = current_time - self.start_time
        
        # Update display
        self.label.setText(f"Distance: {d:6.2f} m")
        self.confidence_label.setText(f"Confidence: {confidence:.1f}% | SNR: {snrs}")
        
        # Add to history
        self.dist_hist.append(d)
        self.time_hist.append(elapsed_time)
        self.confidence_hist.append(confidence)
        
        # Limit history length
        if len(self.dist_hist) > cfg.MAX_HIST_POINTS:
            self.dist_hist = self.dist_hist[-cfg.MAX_HIST_POINTS:]
            self.time_hist = self.time_hist[-cfg.MAX_HIST_POINTS:]
            self.confidence_hist = self.confidence_hist[-cfg.MAX_HIST_POINTS:]
        
        # Plot history
        self.hist.ax.clear()
        self.hist.ax.grid(True, alpha=0.4)
        self.hist.ax.set_title("Distance History", fontsize=14, fontweight='bold')
        
        if len(self.time_hist) > 1:
            colors = ['#e74c3c' if c < 30 else '#f39c12' if c < 70 else '#27ae60' 
                     for c in self.confidence_hist]
            
            self.hist.ax.plot(self.time_hist, self.dist_hist, '-', 
                             color='#3498db', linewidth=2, alpha=0.8)
            self.hist.ax.scatter(self.time_hist, self.dist_hist, 
                               c=colors, s=30, alpha=0.8, edgecolors='white', linewidth=1)
        
        self.hist.ax.set_xlabel("Time (s)", fontsize=12)
        self.hist.ax.set_ylabel("Distance (m)", fontsize=12)
        self.hist.ax.set_xlim(max(0, elapsed_time-60), elapsed_time+2)
        
        if self.dist_hist:
            y_range = max(self.dist_hist) - min(self.dist_hist)
            if y_range > 0:
                self.hist.ax.set_ylim(min(self.dist_hist)-y_range*0.15, 
                                     max(self.dist_hist)+y_range*0.15)
        
        self.hist.draw()

    @QtCore.pyqtSlot(dict)
    def _on_wave(self, data):
        """Waveform data handler"""
        try:
            rx = data['rx']
            
            # TX Spectrum
            if self.worker and hasattr(self.worker, 'tx_pcm'):
                tx_mixed = self.worker.tx_pcm / 32768.0
                f_tx = np.fft.rfftfreq(len(tx_mixed), 1/cfg.FS)
                spec_tx = mag2db(np.fft.rfft(tx_mixed))
                
                self.txSpec.ax.clear()
                self.txSpec.ax.plot(f_tx/1000, spec_tx, color='#27ae60', linewidth=2)
                self.txSpec.ax.set_xlabel("Frequency (kHz)")
                self.txSpec.ax.set_ylabel("Magnitude (dB)")
                self.txSpec.ax.grid(True, alpha=0.3)
                self.txSpec.draw()
            
            # RX Spectrum
            f_rx = np.fft.rfftfreq(rx.size, 1/cfg.FS)
            spec_rx = mag2db(np.fft.rfft(rx))
            
            self.rxSpec.ax.clear()
            self.rxSpec.ax.plot(f_rx/1000, spec_rx, color='#c0392b', linewidth=2)
            self.rxSpec.ax.set_xlabel("Frequency (kHz)")
            self.rxSpec.ax.set_ylabel("Magnitude (dB)")
            self.rxSpec.ax.grid(True, alpha=0.3)
            self.rxSpec.draw()
            
        except Exception as e:
            logger.error(f"Error in _on_wave: {e}")

    def start(self):
        """Start measurement"""
        self.worker = SonarWorker()
        self.worker.temperature = float(self.temp_spinbox.value())
        self.worker.distanceSig.connect(self._on_dist)
        self.worker.waveSig.connect(self._on_wave)
        self.worker.start()
        
        self.btnStart.setEnabled(False)
        self.btnStop.setEnabled(True)
        self.start_time = time.time()
        self.dist_hist.clear()
        self.time_hist.clear()
        self.confidence_hist.clear()

    def stop(self):
        """Stop measurement"""
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.start_time = None

    def closeEvent(self, e):
        """Window close handler"""
        self.stop()
        e.accept()

# ========= Main ========= #
def main():
    logger.info("=== Air-Sonar optimized start ===")
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error, exit")

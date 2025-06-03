# coding=utf-8

"""
realtime_sonar.py - Optimized Version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Real-time air sonar system for Windows platform with maximum range of 50m.

Optimization Features
--------------------
1. Performance optimization: reduced plotting frequency, optimized data processing, cached calculations
2. Raw spectrum display: shows unfiltered raw received signal spectrum
3. Multi-frequency fusion: SNR calculation and confidence-weighted distance estimation
4. Adaptive updates: adjusts plotting frequency based on change magnitude

Copyright © 2025
"""

import sys, time, csv, logging, queue, math, os, traceback, faulthandler, signal
from pathlib import Path
from threading import Lock, Event, Thread, Timer
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import math
import numpy as np
from scipy.signal import chirp, iirfilter, filtfilt, firwin, fftconvolve, correlate
import pyaudio
from PyQt5 import QtCore, QtWidgets, QtOpenGL
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# GPU Support
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU (CuPy) Support Enabled")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    print("GPU (CuPy) Support Not Available, using CPU fallback")

# OpenGL support for matplotlib
try:
    import OpenGL.GL as gl
    plt.rcParams['backend'] = 'Qt5Agg'
    OPENGL_AVAILABLE = True
    print("OpenGL Support Enabled")
except ImportError:
    OPENGL_AVAILABLE = False
    print("OpenGL Support Not Available, using software rendering")

# Set font support for better display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ========= Sonar Parameters ========= #
BASE_TEMP = 28.0  # Base temperature (°C)
R_MIN, R_MAX = 0.5, 15.0        # m
c_air = 343.0 * math.sqrt(1 + (BASE_TEMP - 20) / 273.15)  # Simple temperature correction
CHIRP_LEN = 2 * R_MIN / c_air   # s
LISTEN_LEN = 2 * R_MAX / c_air + 0.003   # 3 ms margin
CYCLE = CHIRP_LEN + LISTEN_LEN + 0.02    # Add 20 ms buffer
FS = 48000
#CHIRP_LEN = 0.05  # Reduced chirp duration: from 0.1s to 0.05s
#LISTEN_LEN = 0.15  # Correspondingly reduced listen duration
#CYCLE = 0.5  # Increased transmission interval: from 0.3s to 0.5s
#BASE_TEMP = 28.0  # Base temperature (°C)
SPEED_SOUND_20C = 343.0  # Sound speed at 20°C (m/s)
CHANNELS = 1
FORMAT = pyaudio.paInt16
CSV_PATH = Path("distances.csv")
LOG_PATH = Path("sonar.log")
BANDS = [(9500,11500),(13500,15500),(17500,19500)]  # Different frequency bands

# Performance optimization parameters - significantly improved refresh rate and reduced stuttering
PLOT_UPDATE_INTERVAL = 1  # Update charts every measurement
SPECTRUM_UPDATE_INTERVAL = 1  # Update spectrum every time for better fluidity
MAX_HIST_POINTS = 300  # Increased history points
GUI_UPDATE_RATE = 30  # GUI update frequency (Hz) - improved to 60FPS
PLOT_DECIMATION = 1  # Plot data decimation factor, 1 means no decimation

# ========= Logging & Monitoring ========= #
# Register faulthandler for debugging thread deadlocks and crashes
faulthandler.enable()
try:
    # Check if register function exists in faulthandler
    if hasattr(faulthandler, 'register'):
        # Windows doesn't have SIGUSR1, use SIGABRT instead
        if sys.platform != "win32" and hasattr(signal, 'SIGUSR1'):
            faulthandler.register(signal.SIGUSR1)
        elif hasattr(signal, 'SIGABRT'):
            faulthandler.register(signal.SIGABRT)  # Use SIGABRT on Windows
    else:
        print("faulthandler.register is not available on this system")
except Exception as e:
    print(f"Could not register faulthandler: {e}")

# Set console encoding to UTF-8
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

# ========= Thread Safety Settings ========= #
LOCK_TIMEOUT = 2.0        # Lock timeout in seconds
QUEUE_TIMEOUT = 1.0       # Queue operations timeout in seconds
HEARTBEAT_INTERVAL = 0.1  # Heartbeat check interval in seconds
HEARTBEAT_TIMEOUT = 0.5   # Heartbeat timeout in seconds
MAX_RESTART_ATTEMPTS = 3  # Maximum number of worker restart attempts

# ------------------------------------------------------------------ #
def calculate_sound_speed(temperature_c):
    """Calculate sound speed (m/s) based on temperature
    Formula: v = 331.3 + 0.606 * T (T in Celsius)
    """
    return 331.3 + 0.606 * temperature_c

# ------------------------------------------------------------------ #
def generate_chirps(fs=FS, duration=CHIRP_LEN):
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    chirps = [chirp(t, f0=l, f1=h, t1=duration, method='linear').astype(np.float32)
              for l, h in BANDS]
    mix = np.sum(chirps, axis=0)
    mix *= 0.85*(2**15-1)/np.max(np.abs(mix))
    return mix.astype(np.int16), chirps

def design_filters(fs=FS):
    filters = []
    for low, high in BANDS:
        try:
            ba = iirfilter(6, [low/(0.5*fs), high/(0.5*fs)], btype='band', output='ba')  # Reduced filter order
            if ba is None:
                raise ValueError(f"iirfilter failed for band {low}-{high} Hz")
            b, a = ba[0], ba[1]
            taps = firwin(61, [low, high], fs=fs, pass_zero=False, window='hamming')  # Reduced FIR length
            filters.append((b, a, taps))
        except Exception as e:            
            logger.error(f"Failed to design filter for band {low}-{high} Hz: {e}")
            # Use default filter parameters
            b, a = [1], [1]
            taps = np.array([1])
            filters.append((b, a, taps))
    return filters

def bandpass(sig, filt):
    """Bandpass filter (FIR + IIR combination) - uses GPU acceleration if available"""
    if GPU_AVAILABLE:
        return gpu_bandpass(sig, filt)
    
    b, a, taps = filt
    try:
        # First use IIR for precise frequency control, then FIR for noise removal
        y = filtfilt(b, a, sig)
        return fftconvolve(y, taps, mode='same')
    except:
        logger.warning("Filtering failed, returning original signal")
        return sig

# === Adaptive threshold peak detection === #
def first_strong_peak(corr, fs, min_delay_samples=500):
    half = corr.size//2
    pos = corr[half:]
    if pos.size <= min_delay_samples:
        return None, 0.0
    pos[:min_delay_samples] = 0       # Remove direct wave
    
    # Improved SNR calculation - uses GPU acceleration
    # Use first 20% as noise baseline
    noise_samples = int(len(pos) * 0.2)
    noise_floor = gpu_mean(pos[:noise_samples] ** 2)  # Noise power
    
    # Find peak - uses GPU acceleration
    peak_idx = gpu_argmax(pos)
    peak_power = pos[peak_idx] ** 2
    
    # Calculate SNR (dB)
    if noise_floor > 0:
        snr_db = 10 * np.log10(peak_power / noise_floor)
    else:
        snr_db = 0.0
    
    # Check if it's a valid peak
    if snr_db < 6.0:  # At least 6dB SNR
        return None, 0.0
        
    return peak_idx, snr_db

# === SNR Calculation and Confidence Assessment === #
def calculate_band_confidence(snr, amplitude, band_idx):
    """Calculate frequency band confidence
    Args:
        snr: Signal-to-noise ratio
        amplitude: Peak amplitude
        band_idx: Band index (0, 1, 2)
    Returns:
        confidence: Raw confidence (0-1), will be normalized to sum 100% later
    """
    # SNR weight (higher SNR means higher confidence)
    snr_weight = min(snr / 10.0, 1.0)  # Normalized to 0-1
    
    # Amplitude weight (higher amplitude means higher confidence)
    amp_weight = min(amplitude / 0.1, 1.0)  # Normalized to 0-1
    
    # Frequency band weight (mid-frequency band is usually more stable)
    freq_weights = [0.8, 1.0, 0.9]  # Low, mid, high frequency
    freq_weight = freq_weights[band_idx]
    
    # Combined confidence
    confidence = (snr_weight * 0.5 + amp_weight * 0.3 + freq_weight * 0.2)
    return min(confidence, 1.0)

def normalize_confidences(confidences):
    """Normalize confidences to ensure sum equals 100%
    Args:
        confidences: List of confidence values for each band
    Returns:
        normalized_confidences: Normalized confidences summing to 100%
    """
    confidences = np.array(confidences)
    total = np.sum(confidences)
    
    if total > 0:
        # Normalize to sum 100%
        normalized = (confidences / total) * 100.0
    else:
        # If all are 0, distribute equally
        normalized = np.full_like(confidences, 100.0 / len(confidences))
    
    return normalized

# === Simple 1-D Kalman Filter === #
class ScalarKalman:
    def __init__(self, q=0.005, r=0.1):  # Adjust parameters to improve response speed
        self.x = None  # state
        self.p = 1.0   # covariance
        self.q = q     # process var
        self.r = r     # meas var

    def update(self, z):
        if self.x is None:
            self.x = z
            return z
        # predict
        self.p += self.q
        # gain
        k = self.p / (self.p + self.r)
        # update
        self.x += k*(z - self.x)
        self.p *= (1 - k)
        return self.x

# === FFT (dB) === #
def mag2db(x):
    return 20*np.log10(np.maximum(np.abs(x), 1e-12))

# === GPU加速函数 === #
def gpu_fft(signal):
    """GPU加速的FFT计算"""
    if GPU_AVAILABLE and cp is not None:
        try:
            signal_gpu = cp.asarray(signal)
            fft_result = cp.fft.rfft(signal_gpu)
            return cp.asnumpy(fft_result)
        except:
            return np.asarray(np.fft.rfft(signal))
    else:
        return np.asarray(np.fft.rfft(signal))

def gpu_correlate(a, b):
    """GPU加速的相关计算"""
    if GPU_AVAILABLE and cp is not None:
        try:
            a_gpu = cp.asarray(a)
            b_gpu = cp.asarray(b)
            result = cp.correlate(a_gpu, b_gpu, mode='full')
            return cp.asnumpy(result)
        except:
            return correlate(a, b, 'full')
    else:
        return correlate(a, b, 'full')

def gpu_bandpass(signal, filt):
    """GPU加速的带通滤波"""
    if GPU_AVAILABLE and cp is not None:
        try:
            b, a, taps = filt
            signal_gpu = cp.asarray(signal)
            
            # 使用FIR滤波（GPU效率更高）
            taps_gpu = cp.asarray(taps)
            filtered = cp.convolve(signal_gpu, taps_gpu, mode='same')
            return cp.asnumpy(filtered)
        except:
            return bandpass(signal, filt)
    else:
        return bandpass(signal, filt)

def gpu_spectrum(signal):
    """GPU加速的频谱计算"""
    if GPU_AVAILABLE and cp is not None:
        try:
            signal_gpu = cp.asarray(signal)
            fft_result = cp.fft.rfft(signal_gpu)
            magnitude = cp.abs(fft_result)
            return cp.asnumpy(magnitude)
        except:
            return np.abs(np.fft.rfft(signal))
    else:
        return np.abs(np.fft.rfft(signal))

def gpu_mag2db(x):
    """GPU加速的幅度转dB"""
    if GPU_AVAILABLE and cp is not None:
        try:
            x_gpu = cp.asarray(x)
            # 避免log(0)
            x_clipped = cp.maximum(x_gpu, 1e-12)
            db_result = 20 * cp.log10(x_clipped)
            return cp.asnumpy(db_result)
        except:
            return mag2db(x)
    else:
        return mag2db(x)

def gpu_argmax(signal):
    """GPU加速的最大值索引查找"""
    if GPU_AVAILABLE and cp is not None:
        try:
            signal_gpu = cp.asarray(signal)
            return int(cp.argmax(signal_gpu))
        except:
            return np.argmax(signal)
    else:
        return np.argmax(signal)

def gpu_mean(signal):
    """GPU加速的均值计算"""
    if GPU_AVAILABLE and cp is not None:
        try:
            signal_gpu = cp.asarray(signal)
            return float(cp.mean(signal_gpu))
        except:
            return np.mean(signal)
    else:
        return np.mean(signal)

def gpu_sqrt(x):
    """GPU加速的平方根"""
    if GPU_AVAILABLE and cp is not None:
        try:
            x_gpu = cp.asarray(x)
            return float(cp.sqrt(x_gpu))
        except:
            return np.sqrt(x)
    else:
        return np.sqrt(x)

# ------------------------------------------------------------------ #
class AudioIO:
    """Audio input/output class - implements audio isolation to avoid speaker signal interference with microphone"""
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.out = self.p.open(format=FORMAT, channels=CHANNELS,
                               rate=FS, output=True, frames_per_buffer=1024)
        self.inp = self.p.open(format=FORMAT, channels=CHANNELS,
                               rate=FS, input=True,
                               frames_per_buffer=int(FS*LISTEN_LEN))
        self.silence_buffer = np.zeros(int(FS * 0.01), dtype=np.int16)  # 10ms silence buffer
    
    def play(self, pcm16):  
        """Play audio while ensuring microphone doesn't record during playback"""
        try:
            # Add brief silence before playback to ensure system stability
            self.out.write(self.silence_buffer.tobytes())
            time.sleep(0.005)  # 5ms wait time to ensure system response
            
            # Play main signal
            self.out.write(pcm16.tobytes())
            
            # Add silence buffer after playback to avoid reverberation
            self.out.write(self.silence_buffer.tobytes())
            time.sleep(0.005)  # Additional 5ms wait time
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
    
    def record(self):
        """Record audio, ensuring recording starts only after playback is complete"""
        try:
            # Additional wait time to ensure playback is completely finished
            time.sleep(0.01)  # 10ms wait for playback system to stabilize
            
            # Clear input buffer to remove any remaining playback signals
            try:
                # Try to read and discard old data in buffer
                for _ in range(5):  # Limit attempts to avoid infinite loop
                    try:
                        self.inp.read(1024, exception_on_overflow=False)
                    except:
                        break  # Buffer is empty, this is normal
            except:
                pass  # Exception when buffer is empty is normal
            
            # Start actual recording
            raw = self.inp.read(int(FS*LISTEN_LEN), exception_on_overflow=False)
            sig = np.frombuffer(raw, np.int16).astype(np.float32)/2**15
            return sig
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
            # Return empty signal if recording fails
            return np.zeros(int(FS*LISTEN_LEN), dtype=np.float32)
    
    def close(self):
        try:
            self.out.stop_stream()
            self.out.close()
            self.inp.stop_stream() 
            self.inp.close()
            self.p.terminate()
        except Exception as e:
            logger.error(f"Audio close error: {e}")

# ------------------------------------------------------------------ #
class SonarWorker(QtCore.QThread):
    distanceSig = QtCore.pyqtSignal(float, list, float)  # Distance + SNR info + confidence
    waveSig = QtCore.pyqtSignal(dict)
    errorSig = QtCore.pyqtSignal(str)  # Signal for error reporting
    heartbeatSig = QtCore.pyqtSignal()  # Signal for heartbeat monitoring

    def __init__(self):
        super().__init__()
        self.tx_pcm, self.chirps = generate_chirps()
        self.filters = design_filters()
        self.kf = ScalarKalman()
        self.audio = None
        
        # Replace boolean flag with Event for thread-safe signaling
        self.stop_event = Event()
        self.paused_event = Event()
        
        self.temperature = 20.0  # Default temperature 20°C
        self.update_counter = 0  # Update counter
        self.executor = None
        
        # Heartbeat monitoring
        self.heartbeat_counter = 0
        self.last_heartbeat_time = time.time()
        self.heartbeat_timer = None
        
        # Thread-safe queues
        self.result_queue = queue.Queue(maxsize=10)  # Buffer for results
          # Pre-compute FFT frequency axis (performance optimization)
        self.tx_freq = np.fft.rfftfreq(len(self.tx_pcm), 1/FS)
        
        if not CSV_PATH.exists():
            with CSV_PATH.open("w", newline='') as f:
                csv.writer(f).writerow(["timestamp", "distance", "confidence", "band_snrs"])
    
    def _send_heartbeat(self):
        """Send heartbeat signal and reschedule next heartbeat"""
        if self.stop_event.is_set():
            return
        
        self.heartbeat_counter += 1
        self.heartbeatSig.emit()
        
        # Reschedule next heartbeat
        if not self.stop_event.is_set():
            self.heartbeat_timer = Timer(HEARTBEAT_INTERVAL, self._send_heartbeat)
            self.heartbeat_timer.daemon = True
            self.heartbeat_timer.start()
    
    def run(self):
        # Use event instead of boolean flag
        self.stop_event.clear()
        self.paused_event.clear()
        
        try:
            # Initialize audio after thread starts
            self.audio = AudioIO()
            
            num_cores = min(4, os.cpu_count() or 1)  # Limit to 4 cores to avoid overhead, default to 1 if cpu_count returns None
            self.executor = ThreadPoolExecutor(max_workers=num_cores)
            logger.info(f"SonarWorker started with full GPU acceleration, utilizing {num_cores} CPU cores.")
            
            # Start heartbeat monitoring
            self._send_heartbeat()
            
            while not self.stop_event.is_set():
                if self.paused_event.is_set():
                    # If paused, wait on the event with timeout to allow for stopping
                    self.paused_event.wait(timeout=0.1)
                    continue
                
                t0 = time.perf_counter()
                try:
                    # -------- Transmit ----------
                    self.audio.play(self.tx_pcm)
                    
                    # -------- Delay to ensure audio isolation ----------
                    # Use wait with timeout instead of sleep for safer interruption
                    if self.stop_event.wait(timeout=CHIRP_LEN + 0.02):  # Returns True if event is set
                        break
                    
                    # -------- Receive ----------
                    rx = self.audio.record()
                    
                    if len(rx) == 0:  # Check if recording failed
                        logger.warning("Empty recording received, skipping this cycle")
                        continue

                    # -------- Parallel filtering + correlation (using GPU acceleration) ----------
                    if not self.stop_event.is_set() and self.executor:
                        # Use timeout for futures to avoid hanging
                        futs = [self.executor.submit(self._process_band_gpu, rx, chirp, filt, i)
                                for i, (chirp, filt) in enumerate(zip(self.chirps, self.filters))]
                        
                        # Collect results with timeout
                        results = []
                        for fut in futs:
                            if self.stop_event.is_set():
                                break
                            try:
                                result = fut.result(timeout=LOCK_TIMEOUT)
                                results.append(result)
                            except TimeoutError:
                                logger.warning("Processing band timed out, skipping")
                        
                        if self.stop_event.is_set():
                            break
                        
                        # -------- Multi-frequency fusion processing (new confidence algorithm) ----------
                        valid_results = [(dist, conf, snr) for dist, conf, snr in results if dist is not None]
                        
                        if valid_results:
                            distances, raw_confidences, snrs = zip(*valid_results)
                            
                            # Normalize confidence to ensure sum equals 100%
                            normalized_confidences = normalize_confidences(raw_confidences)
                            
                            # Use normalized confidence as weights for weighted average
                            weights = normalized_confidences + 1e-9  # Avoid division by zero
                            weighted_dist = np.average(distances, weights=weights)
                            
                            # Kalman filtering
                            dist_kf = self.kf.update(weighted_dist)
                            
                            # Send signal (confidence already normalized to percentage)
                            if not self.stop_event.is_set():
                                try:
                                    # Use non-blocking put with timeout to the result queue
                                    if self.result_queue.full():
                                        # Remove oldest item if queue is full
                                        try:
                                            self.result_queue.get_nowait()
                                        except queue.Empty:
                                            pass
                                    self.result_queue.put((dist_kf, list(normalized_confidences), np.mean(normalized_confidences)), 
                                                          timeout=QUEUE_TIMEOUT)
                                    # Still emit signal for immediate UI updates
                                    self.distanceSig.emit(dist_kf, list(normalized_confidences), np.mean(normalized_confidences))
                                except queue.Full:
                                    logger.warning("Result queue full, dropping measurement")
                            
                            # Log to CSV
                            try:
                                with CSV_PATH.open("a", newline='') as f:
                                    csv.writer(f).writerow([time.time(), dist_kf, np.mean(normalized_confidences), normalized_confidences.tolist()])
                            except Exception as e:
                                logger.error(f"CSV writing error: {e}")
                        
                        # -------- Reduce plotting frequency to improve performance ----------
                        self.update_counter += 1
                        if not self.stop_event.is_set() and self.update_counter % PLOT_UPDATE_INTERVAL == 0:
                            # Send signal to GUI (reduced frequency)
                            band_signals = []
                            correlations = []
                            
                            # Calculate band-filtered signals and cross-correlation (using GPU acceleration)
                            for i, (chirp, filt) in enumerate(zip(self.chirps, self.filters)):
                                band_y = bandpass(rx, filt)  # Built-in GPU acceleration
                                band_signals.append(band_y)
                                # Use GPU-accelerated correlation calculation
                                corr = gpu_correlate(band_y, chirp)
                                correlations.append(corr)
                            
                            if not self.stop_event.is_set():
                                self.waveSig.emit({
                                    'rx': rx,
                                    'band_signals': band_signals,
                                    'correlations': correlations,
                                    'update_spectrum': (self.update_counter % SPECTRUM_UPDATE_INTERVAL == 0)
                                })

                except Exception as e:
                    if not self.stop_event.is_set():  # Only log if we're still supposed to be running
                        logger.exception(f"Worker loop error: {e}")
                        self.errorSig.emit(f"Error in sonar processing: {str(e)}")
                
                # -------- Timing control ----------
                if not self.stop_event.is_set():
                    elapsed = time.perf_counter() - t0
                    sleep_time = max(0, CYCLE - elapsed)
                    if sleep_time > 0:
                        # Use wait with timeout instead of sleep for safer interruption
                        self.stop_event.wait(timeout=sleep_time)
                        
        except Exception as e:
            logger.exception(f"Fatal error in SonarWorker: {e}")
            self.errorSig.emit(f"Fatal error: {str(e)}")
        finally:
            # Cleanup
            if self.heartbeat_timer:
                self.heartbeat_timer.cancel()
                self.heartbeat_timer = None
                
            if self.executor:
                self.executor.shutdown(wait=False)
                self.executor = None
                
            if self.audio:
                self.audio.close()
                self.audio = None
                
            logger.info("SonarWorker stopped")
    def _process_band_gpu(self, rx, chirp_sig, filt, band_idx):
        """Process single frequency band (GPU accelerated version)"""
        # Use GPU-accelerated filtering
        y = bandpass(rx, filt)  # Built-in GPU acceleration check
        
        # Use GPU-accelerated correlation calculation
        corr = gpu_correlate(y, chirp_sig)
        delay, snr = first_strong_peak(corr, FS)
        
        if delay is None:
            return None, 0.0, 0.0
            
        # Calculate distance
        sound_speed = calculate_sound_speed(self.temperature)
        distance = (delay/FS)*sound_speed/2
          # Calculate confidence
        peak_amplitude = np.max(np.abs(corr))
        confidence = calculate_band_confidence(snr, peak_amplitude, band_idx)
        
        return distance, confidence, snr
        
    def _process_band(self, rx, chirp_sig, filt, band_idx):
        """Process single frequency band (compatibility version)"""
        return self._process_band_gpu(rx, chirp_sig, filt, band_idx)
        
    def stop(self):
        """Stop worker thread safely"""
        logger.info("Stopping SonarWorker...")
        self.stop_event.set()
        
    def pause(self):
        """Pause worker without stopping"""
        logger.info("Pausing SonarWorker...")
        self.paused_event.set()
        
    def resume(self):
        """Resume paused worker"""
        logger.info("Resuming SonarWorker...")
        self.paused_event.clear()

# ------------------------------------------------------------------ #
class MplCanvas(FigureCanvas):
    def __init__(self, title, parent=None, width=8, height=5.5, dpi=100):
        # 性能优化：使用较低的DPI和简化的图形设置
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        # 设置性能优化参数
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1.0)
        
        self.ax = fig.add_subplot(111)
        self.ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        
        # 性能优化：设置绘图参数
        self.ax.set_rasterized(True)  # 光栅化以提高性能
        
        super().__init__(fig)
          # 性能优化：启用缓存和减少重绘
        self.setMinimumSize(int(width*dpi//2), int(height*dpi//2))  # 设置最小尺寸防止过度缩放
        
    def clear_and_plot(self, *args, **kwargs):
        """优化的清空和绘图方法"""
        self.ax.clear()
        return self.ax

class MainWindow(QtWidgets.QMainWindow):    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Air Sonar System (Optimized)")
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f0f8ff, stop:1 #e6f3ff);
            }
            QLabel {
                background: transparent;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #45a049);
                border: none;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5CBF60, stop:1 #55b059);
            }
            QPushButton:disabled {
                background: #cccccc;
                color: #666666;
            }
            QSpinBox {
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 5px;
                font-size: 12px;
                background: white;
            }
        """)
        
        # -------- Controls --------
        self.label = QtWidgets.QLabel("Distance: -- m")
        self.label.setStyleSheet("""
            font-size: 28pt; 
            color: #2c3e50; 
            font-weight: bold;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(255,255,255,200), stop:1 rgba(240,248,255,200));
            border: 2px solid #3498db;
            border-radius: 15px;
            padding: 15px;
        """)
        
        # Confidence display
        self.confidence_label = QtWidgets.QLabel("Confidence: --")
        self.confidence_label.setStyleSheet("""
            font-size: 16pt; 
            color: #27ae60; 
            font-weight: bold;
            background: rgba(255,255,255,150);
            border: 1px solid #27ae60;
            border-radius: 8px;
            padding: 8px;
        """)
        
        # Temperature control
        self.temp_label = QtWidgets.QLabel("Temperature (°C):")
        self.temp_label.setStyleSheet("font-size: 14pt; color: #34495e; font-weight: bold;")
        self.temp_spinbox = QtWidgets.QSpinBox()
        self.temp_spinbox.setRange(-40, 60)
        self.temp_spinbox.setValue(20)
        self.temp_spinbox.setSuffix(" °C")
        self.temp_spinbox.valueChanged.connect(self.on_temp_changed)
        
        self.btnStart = QtWidgets.QPushButton("Start")
        self.btnStop = QtWidgets.QPushButton("Stop")
        self.btnStop.setEnabled(False)
        self.btnStop.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f44336, stop:1 #d32f2f);
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f66356, stop:1 #e34f4f);
            }
        """)
        
        # Waveform & spectrum - modified for new layout, adjusted size to reduce stuttering
        self.txSpec = MplCanvas("Transmit Signal Spectrum Analysis", width=6, height=4)
        self.rxSpec = MplCanvas("Received Signal Raw Spectrum", width=6, height=4)
        
        # Three cross-correlation plots - reduced size for better performance
        self.corrPlots = [
            MplCanvas(f"Band {i+1} Cross-correlation ({BANDS[i][0]}-{BANDS[i][1]}Hz)", width=5, height=3) 
            for i in range(3)
        ]
        
        # Three band spectrum plots (not time domain signals) - reduced size for better performance
        self.bandSpecPlots = [
            MplCanvas(f"Band {i+1} Filtered Spectrum ({BANDS[i][0]}-{BANDS[i][1]}Hz)", width=5, height=3) 
            for i in range(3)
        ]
        self.hist = MplCanvas("Distance History Curve", width=16, height=3.5)
        #self.hist = MplCanvas("距离历史曲线", width=16, height=3.5)
        
        # 为图表添加边框样式
        chart_style = """
            border: 2px solid #34495e;
            border-radius: 10px;
            background: white;
            margin: 3px;
        """
        self.txSpec.setStyleSheet(chart_style)
        self.rxSpec.setStyleSheet(chart_style)
        for plot in self.corrPlots:
            plot.setStyleSheet(chart_style)
        for plot in self.bandSpecPlots:
            plot.setStyleSheet(chart_style)
        self.hist.setStyleSheet(chart_style)
        
        # -------- 布局 --------
        # 顶部控制区
        control_frame = QtWidgets.QFrame()
        control_frame.setStyleSheet("""
            QFrame {
                background: rgba(255,255,255,100);
                border: 1px solid #bdc3c7;
                border-radius: 10px;
                margin: 5px;
            }
        """)
        control_layout = QtWidgets.QHBoxLayout(control_frame)
        control_layout.addWidget(self.btnStart)
        control_layout.addWidget(self.btnStop)
        control_layout.addStretch()
        
        # 温度控制区
        temp_layout = QtWidgets.QHBoxLayout()
        temp_layout.addWidget(self.temp_label)
        temp_layout.addWidget(self.temp_spinbox)
        temp_layout.addStretch()
        control_layout.addLayout(temp_layout)
        control_layout.addStretch()
        control_layout.addWidget(self.confidence_label)
        control_layout.addWidget(self.label)          # 图表网格 - 修改为新的布局：第一行2个，下面3列
        g = QtWidgets.QGridLayout()
        g.setSpacing(8)  # 减少间距以节省空间
        # 第一行：发射频谱、接收频谱
        g.addWidget(self.txSpec, 0, 0)
        g.addWidget(self.rxSpec, 0, 1)
        # 第二行：三个频段的自相关图
        g.addWidget(self.corrPlots[0], 1, 0)
        g.addWidget(self.corrPlots[1], 1, 1)
        g.addWidget(self.corrPlots[2], 1, 2)
        # 第三行：三个频段的滤波频谱（在对应自相关下方）
        g.addWidget(self.bandSpecPlots[0], 2, 0)
        g.addWidget(self.bandSpecPlots[1], 2, 1)
        g.addWidget(self.bandSpecPlots[2], 2, 2)
        
        # 主布局
        vbox = QtWidgets.QVBoxLayout()
        vbox.setSpacing(15)
        vbox.addWidget(control_frame)
        vbox.addLayout(g)
        vbox.addWidget(self.hist)
        
        central = QtWidgets.QWidget()
        central.setLayout(vbox)
        self.setCentralWidget(central)
        
        # -------- Signals --------
        self.btnStart.clicked.connect(self.start)
        self.btnStop.clicked.connect(self.stop)
        self.dist_hist = []
        self.time_hist = []
        self.confidence_hist = []
        self.start_time = None
        self.worker = None
        # Performance optimization: reduce redraw and add plot caching
        self.last_update_time = 0
        self.min_update_interval = 1.0 / GUI_UPDATE_RATE  # Update interval based on GUI_UPDATE_RATE
        self.plot_cache = {}  # Plot cache
        self.spectrum_cache_timeout = 0.2  # Spectrum cache timeout (seconds)
    
    def on_temp_changed(self, value):
        """Update worker's temperature value when temperature changes"""
        if self.worker:
            self.worker.temperature = float(value)    # --- worker signal slots ---    @QtCore.pyqtSlot(float, list, float)
    def _on_dist(self, d, snrs, confidence):
        current_time = time.time()
        
        # Performance optimization: limit update frequency
        if current_time - self.last_update_time < self.min_update_interval:
            return
        
        self.last_update_time = current_time
        
        if self.start_time is None:
            self.start_time = current_time
        
        elapsed_time = current_time - self.start_time
        # Update display
        self.label.setText(f"Distance: {d:6.2f} m")
        self.confidence_label.setText(f"Confidence: {confidence:.1f}% | SNR: {snrs}")
        
        # Add to history data        self.dist_hist.append(d)
        self.time_hist.append(elapsed_time)
        self.confidence_hist.append(confidence)
        
        # Limit history data length (performance optimization)
        if len(self.dist_hist) > MAX_HIST_POINTS:
            self.dist_hist = self.dist_hist[-MAX_HIST_POINTS:]
            self.time_hist = self.time_hist[-MAX_HIST_POINTS:]
            self.confidence_hist = self.confidence_hist[-MAX_HIST_POINTS:]
        
        # Draw distance history curve - improved visuals
        self.hist.ax.clear()
        self.hist.ax.grid(True, alpha=0.4, linewidth=0.8)
        self.hist.ax.set_title("Distance History (Confidence Weighted)", fontsize=16, fontweight='bold', pad=15)
        
        if len(self.time_hist) > 1:
            # Set colors based on confidence
            colors = ['#e74c3c' if c < 30 else '#f39c12' if c < 70 else '#27ae60' 
                     for c in self.confidence_hist]
            
            # Draw main curve
            self.hist.ax.plot(self.time_hist, self.dist_hist, '-', 
                             color='#3498db', linewidth=2, alpha=0.8)
            
            # Draw scatter points based on confidence
            self.hist.ax.scatter(self.time_hist, self.dist_hist, 
                               c=colors, s=30, alpha=0.8, edgecolors='white', linewidth=1)
            
            # Add fill area
            self.hist.ax.fill_between(self.time_hist, self.dist_hist, alpha=0.15, color='#3498db')
            
            # Draw scatter points based on confidence (redundant in original code, kept for compatibility)
            self.hist.ax.scatter(self.time_hist, self.dist_hist, 
                               c=colors, s=30, alpha=0.8, edgecolors='white', linewidth=1)
              # Add fill area (redundant in original code, kept for compatibility)
            self.hist.ax.fill_between(self.time_hist, self.dist_hist, alpha=0.15, color='#3498db')
        
        self.hist.ax.set_xlabel("Time (s)", fontsize=14, fontweight='bold')
        self.hist.ax.set_ylabel("Distance (m)", fontsize=14, fontweight='bold')
        self.hist.ax.tick_params(labelsize=12)
        self.hist.ax.set_xlim(max(0, elapsed_time-60), elapsed_time+2)  # Show last 60 seconds
        
        if self.dist_hist:
            y_range = max(self.dist_hist) - min(self.dist_hist)
            if y_range > 0:
                self.hist.ax.set_ylim(min(self.dist_hist)-y_range*0.15, 
                                     max(self.dist_hist)+y_range*0.15)
        
        # Enhance coordinate axes
        self.hist.ax.spines['top'].set_visible(False)
        self.hist.ax.spines['right'].set_visible(False)
        self.hist.ax.spines['left'].set_linewidth(2)
        self.hist.ax.spines['bottom'].set_linewidth(2)
        
        self.hist.draw()
    @QtCore.pyqtSlot(dict)
    def _on_wave(self, data):
        try:
            current_time = time.time()
            rx = data['rx']
            band_signals = data.get('band_signals', [])
            correlations = data.get('correlations', [])
            update_spectrum = data.get('update_spectrum', True)
              # Define colors
            colors = ['#e74c3c', '#f39c12', '#9b59b6']
            
            # Performance optimization: data subsampling (take one point every PLOT_DECIMATION points)
            if len(rx) > 1000 and PLOT_DECIMATION > 1:
                rx_decimated = rx[::PLOT_DECIMATION]
            else:
                rx_decimated = rx
              # Transmit signal spectrum (using cache and subsampling)
            if update_spectrum and hasattr(self, 'worker') and self.worker:
                cache_key = 'tx_spectrum'
                if (cache_key not in self.plot_cache or 
                    current_time - self.plot_cache[cache_key]['timestamp'] > self.spectrum_cache_timeout):
                    
                    tx_mixed = self.worker.tx_pcm / 32768.0  # Normalize
                    # Data subsampling
                    if len(tx_mixed) > 1000 and PLOT_DECIMATION > 1:
                        tx_mixed = tx_mixed[::PLOT_DECIMATION]
                    
                    f_tx = np.fft.rfftfreq(len(tx_mixed), PLOT_DECIMATION/FS)
                    spec_tx = mag2db(np.fft.rfft(tx_mixed))
                    
                    # Cache results
                    self.plot_cache[cache_key] = {
                        'timestamp': current_time,
                        'f_tx': f_tx,
                        'spec_tx': spec_tx
                    }
                
                # Use cached data
                cached_data = self.plot_cache[cache_key]
                
                self.txSpec.ax.clear()
                self.txSpec.ax.plot(cached_data['f_tx']/1000, cached_data['spec_tx'],                                  color='#27ae60', linewidth=2.0, alpha=0.9)  # Reduce line width
                self.txSpec.ax.grid(True, alpha=0.3, linewidth=0.5)  # Reduce grid line width
                self.txSpec.ax.set_xlabel("Frequency (kHz)", fontsize=10, fontweight='bold')  # Reduce font size
                self.txSpec.ax.set_ylabel("Amplitude (dB)", fontsize=10, fontweight='bold')
                self.txSpec.ax.set_title("Transmit Signal Spectrum", fontsize=12, fontweight='bold', pad=10)
                self.txSpec.ax.tick_params(labelsize=9)                # Highlight frequency band regions
                for low, high in BANDS:
                    self.txSpec.ax.axvspan(low/1000, high/1000, alpha=0.12, color='#27ae60')
                # Enhance coordinate axes
                self.txSpec.ax.spines['top'].set_visible(False)
                self.txSpec.ax.spines['right'].set_visible(False)
                self.txSpec.ax.spines['left'].set_linewidth(1.0)  # Reduce border line width                self.txSpec.ax.spines['bottom'].set_linewidth(1.0)
                self.txSpec.draw()                
                # ---- Rx original spectrum ----
                f_rx = np.fft.rfftfreq(rx_decimated.size, PLOT_DECIMATION/FS)
                spec_rx = mag2db(np.fft.rfft(rx_decimated))
                self.rxSpec.ax.clear()
                self.rxSpec.ax.plot(f_rx/1000, spec_rx, color='#c0392b', linewidth=2.0, alpha=0.8)
                self.rxSpec.ax.grid(True, alpha=0.3, linewidth=0.5)  # Reduce grid line width
                self.rxSpec.ax.set_xlabel("Frequency (kHz)", fontsize=10, fontweight='bold')
                self.rxSpec.ax.set_ylabel("Amplitude (dB)", fontsize=10, fontweight='bold')
                self.rxSpec.ax.set_title("Received Signal Raw Spectrum", fontsize=12, fontweight='bold', pad=10)
                self.rxSpec.ax.tick_params(labelsize=9)                # Highlight frequency band regions
                for low, high in BANDS:
                    self.rxSpec.ax.axvspan(low/1000, high/1000, alpha=0.12, color='#c0392b')
                # Enhance coordinate axes
                self.rxSpec.ax.spines['top'].set_visible(False)
                self.rxSpec.ax.spines['right'].set_visible(False)
                self.rxSpec.ax.spines['left'].set_linewidth(1.0)
                self.rxSpec.ax.spines['bottom'].set_linewidth(1.0)
                self.rxSpec.draw()              # ---- 三个独立的自相关函数图 ----            if correlations:
                colors = ['#e74c3c', '#f39c12', '#9b59b6']
                for i, corr in enumerate(correlations[:3]):  # Maximum 3 bands
                    if len(corr) > 0:
                        # Subsample data for better performance
                        if len(corr) > 2000 and PLOT_DECIMATION > 1:
                            corr_decimated = corr[::PLOT_DECIMATION]
                        else:
                            corr_decimated = corr
                        
                        # Time axis (in milliseconds)
                        len_chirp_samples = int(FS * CHIRP_LEN)
                        lag_indices = np.arange(len(corr_decimated))
                        time_axis_samples = lag_indices * PLOT_DECIMATION - (len_chirp_samples - 1)
                        time_axis = time_axis_samples / FS * 1000  # 转换为毫秒
                        
                        self.corrPlots[i].ax.clear()
                        self.corrPlots[i].ax.plot(time_axis, corr_decimated, 
                                                color=colors[i % len(colors)], 
                                                linewidth=1.5, alpha=0.9)  # 减小线宽
                        self.corrPlots[i].ax.grid(True, alpha=0.3, linewidth=0.5)  # 减小网格线宽                        self.corrPlots[i].ax.set_xlabel("Time Delay (ms)", fontsize=9, fontweight='bold')
                        self.corrPlots[i].ax.set_ylabel("Correlation Coefficient", fontsize=9, fontweight='bold')
                        self.corrPlots[i].ax.set_title(f"Band {i+1} Correlation ({BANDS[i][0]}-{BANDS[i][1]}Hz)", 
                                                     fontsize=10, fontweight='bold', pad=8)
                        self.corrPlots[i].ax.tick_params(labelsize=8)
                        # 美化坐标轴
                        self.corrPlots[i].ax.spines['top'].set_visible(False)
                        self.corrPlots[i].ax.spines['right'].set_visible(False)
                        self.corrPlots[i].ax.spines['left'].set_linewidth(1.0)
                        self.corrPlots[i].ax.spines['bottom'].set_linewidth(1.0)
                        self.corrPlots[i].draw()
            # ---- Frequency spectrum for each filtered band signal ----
            if band_signals:
                for i, band_signal in enumerate(band_signals[:3]):  # Maximum 3 bands
                    if len(band_signal) > 0:                        # Calculate spectrum
                        f_band = np.fft.rfftfreq(len(band_signal), 1/FS)
                        spec_band = mag2db(np.fft.rfft(band_signal))
                        self.bandSpecPlots[i].ax.clear()
                        self.bandSpecPlots[i].ax.plot(f_band, spec_band, 
                                                    color=colors[i % len(colors)], 
                                                    linewidth=2.0, alpha=0.9)
                        self.bandSpecPlots[i].ax.grid(True, alpha=0.4, linewidth=0.8)
                        self.bandSpecPlots[i].ax.set_xlabel("Frequency (Hz)", fontsize=10, fontweight='bold')
                        self.bandSpecPlots[i].ax.set_ylabel("Amplitude (dB)", fontsize=10, fontweight='bold')
                        self.bandSpecPlots[i].ax.set_title(f"Band {i+1} Filtered Spectrum ({BANDS[i][0]}-{BANDS[i][1]}Hz)", 
                                                         fontsize=12, fontweight='bold', pad=10)
                        self.bandSpecPlots[i].ax.tick_params(labelsize=9)                        # Highlight corresponding frequency band
                        low, high = BANDS[i]
                        self.bandSpecPlots[i].ax.axvspan(low, high, alpha=0.15, color=colors[i % len(colors)])
                        # Enhance coordinate axes
                        self.bandSpecPlots[i].ax.spines['top'].set_visible(False)
                        self.bandSpecPlots[i].ax.spines['right'].set_visible(False)
                        self.bandSpecPlots[i].ax.spines['left'].set_linewidth(1.5)
                        self.bandSpecPlots[i].ax.spines['bottom'].set_linewidth(1.5)
                        self.bandSpecPlots[i].draw()
        except Exception as e:
            logger.error(f"Error in _on_wave: {e}")
            traceback.print_exc()

    # --- 控制 ---    def start(self):
        self.worker = SonarWorker()
        self.worker.temperature = float(self.temp_spinbox.value())
        self.worker.distanceSig.connect(self._on_dist)
        self.worker.waveSig.connect(self._on_wave)
        self.worker.errorSig.connect(self.showError)  # Connect error signal
        self.worker.heartbeatSig.connect(self.onHeartbeat)  # Connect heartbeat
        
        # Start monitoring for missed heartbeats
        self.last_heartbeat_time = time.time()
        
        # Start worker thread
        self.worker.start()
        self.btnStart.setEnabled(False)
        self.btnStop.setEnabled(True)
        self.start_time = time.time()
        self.dist_hist.clear()
        self.time_hist.clear()
        self.confidence_hist.clear()

    def stop(self):
        """Stop the worker thread safely"""
        if hasattr(self, 'worker') and self.worker:
            # Using the new Event-based stop mechanism
            try:
                self.worker.stop()
                # Give worker thread time to exit gracefully
                if not self.worker.wait(2000):  # 2 seconds timeout
                    logger.warning("Worker thread didn't exit cleanly, forcing...")
            except Exception as e:
                logger.error(f"Error stopping worker thread: {e}")
            self.worker = None
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.start_time = None
    
    def onHeartbeat(self):
        """Receive heartbeat from worker thread"""
        self.last_heartbeat_time = time.time()
    
    def showError(self, error_msg):
        """Display error message from worker thread"""
        logger.error(f"Worker thread error: {error_msg}")
        # Could add GUI popup here if needed

    def closeEvent(self, e):
        """Handle window close event"""
        self.stop()
        e.accept()

# ------------------------------------------------------------------ #
def main():
    logger.info("=== Air-Sonar optimized start ===")
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1400, 900)  # 增大窗口尺寸以适应更大的图表
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error, exit")

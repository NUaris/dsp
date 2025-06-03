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

import sys, time, csv, logging, queue, math, os, traceback, faulthandler, signal, io, warnings
from pathlib import Path
from threading import Lock, Event, Thread, Timer
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
from scipy.signal import chirp, iirfilter, filtfilt, firwin, fftconvolve, correlate
import pyaudio
from PyQt5 import QtCore, QtWidgets, QtOpenGL
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# GPU Support
try:
    import warnings
    # Suppress CuPy experimental warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="cupyx")
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
@dataclass(frozen=True)
class Config:
    FS: int = 48000
    BASE_TEMP: float = 28.0
    R_MIN: float = 0.5
    R_MAX: float = 15.0
    CYCLE_MARGIN: float = 0.02
    CHANNELS: int = 1
    FORMAT: int = pyaudio.paInt16
    BANDS: tuple = ((9500,11500),(13500,15500),(17500,19500))
    PLOT_UPDATE_INTERVAL: int = 1
    SPECTRUM_UPDATE_INTERVAL: int = 1
    MAX_HIST_POINTS: int = 300
    GUI_UPDATE_RATE: int = 50
    PLOT_DECIMATION: int = 1
    LOCK_TIMEOUT: float = 2.0
    QUEUE_TIMEOUT: float = 1.0
    HEARTBEAT_INTERVAL: float = 0.1
    HEARTBEAT_TIMEOUT: float = 0.5
    MAX_RESTART_ATTEMPTS: int = 3
    SILENCE_MS: float = 0.02
    PRE_RECORD_SLEEP: float = 0.03
    SNR_NOISE_MS: float = 0.005
    SPECTRUM_CACHE_SEC: float = 0.5
    CSV_PATH: Path = Path("distances.csv")
    LOG_PATH: Path = Path("sonar.log")

    @property
    def c_air(self):
        return 343.0 * math.sqrt(1 + (self.BASE_TEMP - 20) / 273.15)
    
    @property
    def CHIRP_LEN(self):
        return 2*self.R_MIN/self.c_air
    
    @property
    def LISTEN_LEN(self):
        return 2*self.R_MAX/self.c_air+0.003
    
    @property
    def CYCLE(self):
        return self.CHIRP_LEN + self.LISTEN_LEN + self.CYCLE_MARGIN
    
    @property
    def SPEED_SOUND_20C(self):
        return 343.0

cfg = Config()

# ========= Logging & Monitoring ========= #
# Register faulthandler for debugging thread deadlocks and crashes
faulthandler.enable()

# FIX: logger未定义，需在faulthandler和全局异常处理前定义logger
import logging
import sys
from pathlib import Path
LOG_PATH = Path("sonar.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Sonar")

# FIX: 移除faulthandler.register调用，仅保留faulthandler.enable()，彻底兼容所有平台
try:
    import faulthandler
    faulthandler.enable()
except Exception as e:
    logger.warning(f"faulthandler setup failed: {e}")

# Set console encoding to UTF-8
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ---------- section ----------
# 所有FS、BANDS等参数统一用cfg.FS、cfg.BANDS等

def calculate_sound_speed(temperature_c):
    """Calculate sound speed (m/s) based on temperature
    Formula: v = 331.3 + 0.606 * T (T in Celsius)
    """
    return 331.3 + 0.606 * temperature_c

# ------------------------------------------------------------------ #
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
    """Bandpass filter (FIR + IIR combination) - uses GPU acceleration if available"""
    if GPU_AVAILABLE:
        return gpu_bandpass(sig, filt)
    
    b, a, taps = filt
    try:
        # First use IIR for precise frequency control, then FIR for noise removal
        y = filtfilt(b, a, sig)
        return fftconvolve(y, taps, mode='same')
    except Exception as e:
        logger.warning(f"Filtering failed, returning original signal: {e}")
        return sig

# === Adaptive threshold peak detection === #
def first_strong_peak(corr, fs, min_delay_samples=None):
    # PERF: SNR噪声窗改为静音后首5ms
    if min_delay_samples is None:
        min_delay_samples = int(fs * cfg.CHIRP_LEN * 1.2)
    half = corr.size//2
    pos = corr[half:]
    if pos.size <= min_delay_samples:
        return None, 0.0
    pos[:min_delay_samples] = 0
    noise_samples = int(cfg.SNR_NOISE_MS * fs)
    noise_floor = np.median(np.abs(pos[min_delay_samples:min_delay_samples+noise_samples]))**2  # PERF: MAD近似
    peak_idx = gpu_argmax(pos)
    peak_power = pos[peak_idx] ** 2
    if noise_floor > 0:
        snr_db = 10 * np.log10(peak_power / noise_floor)
    else:
        snr_db = 0.0
    if snr_db < 6.0:
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

# ========== GPU加速函数区 ========== #
def gpu_correlate(a, b):
    """GPU加速的相关计算，优先cupyx.scipy.signal.fftconvolve，CuPy不可用时回退CPU"""
    if GPU_AVAILABLE and cp is not None:
        try:
            import cupyx.scipy.signal as cpx_signal
            a_gpu = cp.asarray(a)
            b_gpu = cp.asarray(b)
            result = cpx_signal.fftconvolve(a_gpu, b_gpu[::-1], mode='full')
            return cp.asnumpy(result)
        except Exception as e:
            logger.exception(e)
            return correlate(a, b, 'full')
    else:
        return correlate(a, b, 'full')

def gpu_bandpass(signal, filt):
    """GPU加速带通滤波，递归死循环修复，出错只回退一次CPU"""
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
        self.out = self.p.open(format=cfg.FORMAT, channels=cfg.CHANNELS,
                               rate=cfg.FS, output=True, frames_per_buffer=1024)
        self.inp = self.p.open(format=cfg.FORMAT, channels=cfg.CHANNELS,
                               rate=cfg.FS, input=True,
                               frames_per_buffer=int(cfg.FS*cfg.LISTEN_LEN))
        self.silence_buffer = np.zeros(int(cfg.FS * 0.02), dtype=np.int16)  # FIX: 20ms silence buffer
    
    def play(self, pcm16):
        """Play audio while ensuring microphone doesn't record during playback"""
        try:
            self.out.write(self.silence_buffer.tobytes())
            time.sleep(0.02)  # FIX: 20ms wait
            
            # Play main signal
            self.out.write(pcm16.tobytes())
            
            # Add silence buffer after playback to avoid reverberation
            self.out.write(self.silence_buffer.tobytes())
            time.sleep(0.02)  # FIX: 20ms wait
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
    
    def record(self):
        """Record audio, ensuring recording starts only after playback is complete"""
        try:
            time.sleep(0.03)  # FIX: 30ms wait before record
            
            # Clear input buffer to remove any remaining playback signals
            for _ in range(5):
                try:
                    self.inp.read(1024, exception_on_overflow=False)
                except Exception:
                    break
              # Start actual recording
            raw = self.inp.read(int(cfg.FS*cfg.LISTEN_LEN), exception_on_overflow=False)
            sig = np.frombuffer(raw, np.int16).astype(np.float32)/2**15
            return sig
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
            # Return empty signal if recording fails
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

# ------------------------------------------------------------------ #
class SonarWorker(QtCore.QThread):
    """Sonar后台线程，负责音频采集、DSP、数据融合与信号发射。"""
    distanceSig = QtCore.pyqtSignal(float, list, float)
    waveSig = QtCore.pyqtSignal(dict)
    errorSig = QtCore.pyqtSignal(str)
    heartbeatSig = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.tx_pcm, self.chirps = generate_chirps(cfg.FS, cfg.CHIRP_LEN)
        self.filters = design_filters(cfg.FS)
        self.kf = ScalarKalman()
        self.audio = None
        self.stop_event = Event()
        self.paused_event = Event()
        self.temperature = 20.0
        self.update_counter = 0
        self.executor = None
        self.heartbeat_timer = QtCore.QTimer()
        self.heartbeat_timer.timeout.connect(self._send_heartbeat)
        self.heartbeat_timer.setInterval(int(cfg.HEARTBEAT_INTERVAL * 1000))
        self.result_queue = queue.Queue(maxsize=10)
        self.tx_freq = np.fft.rfftfreq(len(self.tx_pcm), 1/cfg.FS)
        if not cfg.CSV_PATH.exists():
            with cfg.CSV_PATH.open("w", newline='') as f:
                csv.writer(f).writerow(["timestamp", "distance", "confidence", "band_snrs"])
        # 优化：FIR taps一次性上传到GPU
        if GPU_AVAILABLE and cp is not None:
            self.taps_gpu = [cp.asarray(t) for _, _, t in self.filters]

    def _send_heartbeat(self):
        if self.stop_event.is_set():        return
        self.heartbeatSig.emit()

    def stop(self):
        self.stop_event.set()
        self.heartbeat_timer.stop()

    def _process_band_gpu(self, rx, chirp_sig, filt, band_idx):
        """Process single frequency band with GPU acceleration"""
        try:
            # GPU加速滤波
            band_sig = bandpass(rx, filt)
            # GPU加速相关
            corr = gpu_correlate(band_sig, chirp_sig)
            # 相关峰值检测
            min_delay_samples = int(cfg.FS * cfg.CHIRP_LEN * 1.2)
            peak_idx, snr = first_strong_peak(corr, cfg.FS, min_delay_samples)
            if peak_idx is None:
                return None, 0.0, 0.0
            # 距离估算
            delay = peak_idx - (len(chirp_sig) - 1)
            distance = delay / cfg.FS * calculate_sound_speed(self.temperature) / 2
            amplitude = np.max(np.abs(corr))
            confidence = calculate_band_confidence(snr, amplitude, band_idx)
            return distance, confidence, snr
        except Exception as e:
            logger.exception(f"_process_band_gpu error: {e}")
            return None, 0.0, 0.0

    def run(self):
        """Main worker thread loop"""
        try:
            self.audio = AudioIO()
            logger.info("SonarWorker started")
            if GPU_AVAILABLE and cp is not None:
                try:
                    props = cp.cuda.runtime.getDeviceProperties(0)
                    gpu_name = props['name'].decode() if hasattr(props['name'], 'decode') else props['name']
                except Exception as e:
                    gpu_name = f"Unknown ({e})"
                logger.info(f"cupy version={cp.__version__}, device={gpu_name}")
                with cp.cuda.Stream.null:
                    cp.empty((1,))  # 触发初始化只需一次
            while not self.stop_event.is_set():
                t0 = time.perf_counter()
                self.audio.play(self.tx_pcm)
                rx = self.audio.record()
                band_spectra = [None]*len(cfg.BANDS)
                correlations = [None]*len(cfg.BANDS)
                
                # 计算每个频段的滤波信号、频谱和相关
                def process_band_with_output(args):
                    i, chirp, filt = args
                    # 滤波
                    band_sig = bandpass(rx, filt)
                    # 频谱
                    band_spec = mag2db(np.fft.rfft(band_sig))
                    # 相关
                    corr = gpu_correlate(band_sig, chirp)
                    # 距离计算
                    distance, confidence, snr = self._process_band_gpu(rx, chirp, filt, i)
                    return i, band_spec, corr, (distance, confidence, snr)
                
                # 多线程并行三路band处理
                with ThreadPoolExecutor(max_workers=len(cfg.BANDS)) as pool:
                    futs = [pool.submit(process_band_with_output, (i, chirp, filt))
                            for i, (chirp, filt) in enumerate(zip(self.chirps, self.filters))]
                    results = []
                    for fut in futs:
                        try:
                            i, band_spec, corr, result = fut.result(timeout=cfg.LOCK_TIMEOUT)
                            band_spectra[i] = band_spec
                            correlations[i] = corr
                            if result[0] is not None:  # distance is not None
                                results.append(result)
                        except Exception as e:
                            logger.exception(e)
                
                # 距离融合
                if results:
                    distances, confidences, snrs = zip(*results)
                    confidences_norm = normalize_confidences(confidences)
                    weights = confidences_norm / 100.0 + 1e-9
                    weighted_dist = np.average(distances, weights=weights)
                    avg_confidence = np.mean(confidences_norm)
                    dist_kf = self.kf.update(weighted_dist)
                    self.distanceSig.emit(dist_kf, list(snrs), avg_confidence)
                    with cfg.CSV_PATH.open("a", newline='') as f:
                        csv.writer(f).writerow([time.time(), dist_kf, avg_confidence, list(snrs)])
                
                self.update_counter += 1
                if self.update_counter % cfg.PLOT_UPDATE_INTERVAL == 0:
                    self.waveSig.emit({
                        'rx': rx,
                        'band_spectra': band_spectra,
                        'correlations': correlations,
                        'rx_id': self.update_counter,
                        'update_spectrum': True
                    })
                elapsed = time.perf_counter() - t0
                sleep_time = max(0, (cfg.CHIRP_LEN + cfg.LISTEN_LEN + cfg.CYCLE_MARGIN) - elapsed)
                time.sleep(sleep_time)
        except Exception as e:
            logger.exception(f"Worker error: {e}")
            self.errorSig.emit(str(e))
        finally:
            if self.audio:
                self.audio.close()
            self.heartbeat_timer.stop()
            logger.info("SonarWorker stopped")

# ------------------------------------------------------------------ #
class MplCanvas(FigureCanvas):
    """Matplotlib画布，支持高性能绘图和blitting。"""
    def __init__(self, title, parent=None, width=8, height=5.5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1.0)
        self.ax = fig.add_subplot(111)
        self.ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        self.ax.set_rasterized(True)
        super().__init__(fig)
        self.setMinimumSize(int(width*dpi//2), int(height*dpi//2))
        self._main_line = None
        self._blit_bg = None
    def plot_line(self, x, y, color, **kwargs):
        # 支持blitting的高性能绘图
        if self._main_line is None:
            self._main_line, = self.ax.plot(x, y, color=color, **kwargs)
            self._blit_bg = self.copy_from_bbox(self.ax.bbox)
        else:
            self._main_line.set_data(x, y)
            if self._blit_bg is not None:
                self.restore_region(self._blit_bg)
                self.ax.draw_artist(self._main_line)
                self.blit(self.ax.bbox)
            else:
                self.draw()
    def clear_and_plot(self, *args, **kwargs):
        self.ax.clear()
        self._main_line = None
        self._blit_bg = None
        return self.ax

# ------------------------------------------------------------------ #
class MainWindow(QtWidgets.QMainWindow):
    """主窗口，负责GUI布局、信号连接和高性能绘图。"""
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
            MplCanvas(f"Band {i+1} Cross-correlation ({cfg.BANDS[i][0]}-{cfg.BANDS[i][1]}Hz)", width=5, height=3) 
            for i in range(3)
        ]
        
        # Three band spectrum plots (not time domain signals) - reduced size for better performance
        self.bandSpecPlots = [
            MplCanvas(f"Band {i+1} Filtered Spectrum ({cfg.BANDS[i][0]}-{cfg.BANDS[i][1]}Hz)", width=5, height=3) 
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
        self.worker = None        # Performance optimization: reduce redraw and add plot caching
        self.last_update_time = 0
        self.min_update_interval = 1.0 / cfg.GUI_UPDATE_RATE  # Update interval based on GUI_UPDATE_RATE
        self.plot_cache = {}  # Plot cache
        self.spectrum_cache_timeout = 0.2  # Spectrum cache timeout (seconds)
    
    def on_temp_changed(self, value):
        """Update worker's temperature value when temperature changes"""
        if hasattr(self, 'worker') and self.worker and hasattr(self.worker, 'temperature'):
            try:
                self.worker.temperature = float(value)
            except Exception as e:
                logger.error(f"Failed to set worker temperature: {e}")
    # --- worker signal slots ---    @QtCore.pyqtSlot(float, list, float)
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
        snrs_str = ', '.join(f"{s:.1f}" for s in snrs)
        self.confidence_label.setText(f"Confidence: {confidence:.1f}% | SNR: [{snrs_str}]")
          # Add to history data
        self.dist_hist.append(d)
        self.time_hist.append(elapsed_time)
        self.confidence_hist.append(confidence)
        
        # Limit history data length (performance optimization)
        if len(self.dist_hist) > cfg.MAX_HIST_POINTS:
            self.dist_hist = self.dist_hist[-cfg.MAX_HIST_POINTS:]
            self.time_hist = self.time_hist[-cfg.MAX_HIST_POINTS:]
            self.confidence_hist = self.confidence_hist[-cfg.MAX_HIST_POINTS:]
        
        # Draw distance history curve - improved visuals
        if len(self.time_hist) > 1:
            # Set colors based on confidence
            colors = ['#e74c3c' if c < 30 else '#f39c12' if c < 70 else '#27ae60' 
                     for c in self.confidence_hist]
            self.hist.ax.clear()
            self.hist.ax.grid(True, alpha=0.4, linewidth=0.8)
            self.hist.ax.set_title("Distance History (Confidence Weighted)", fontsize=16, fontweight='bold', pad=15)
            self.hist.ax.plot(self.time_hist, self.dist_hist, '-', 
                             color='#3498db', linewidth=2, alpha=0.8)
            self.hist.ax.scatter(self.time_hist, self.dist_hist, 
                               c=colors, s=30, alpha=0.8, edgecolors='white', linewidth=1)
            self.hist.ax.fill_between(self.time_hist, self.dist_hist, alpha=0.15, color='#3498db')
            self.hist.ax.set_xlabel("Time (s)", fontsize=14, fontweight='bold')
            self.hist.ax.set_ylabel("Distance (m)", fontsize=14, fontweight='bold')
            self.hist.ax.tick_params(labelsize=12)
            self.hist.ax.set_xlim(max(0, elapsed_time-60), elapsed_time+2)
            if self.dist_hist:
                y_range = max(self.dist_hist) - min(self.dist_hist)
                if y_range > 0:
                    self.hist.ax.set_ylim(min(self.dist_hist)-y_range*0.15, 
                                         max(self.dist_hist)+y_range*0.15)
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
            band_spectra = data.get('band_spectra', [])
            correlations = data.get('correlations', [])
            update_spectrum = data.get('update_spectrum', True)
            rx_id = data.get('rx_id', None)
            colors = ['#e74c3c', '#f39c12', '#9b59b6']
            cache_key = f'tx_spectrum_{rx_id}'
            if (cache_key not in self.plot_cache or 
                current_time - self.plot_cache[cache_key]['timestamp'] > cfg.SPECTRUM_CACHE_SEC):
                tx_mixed = self.worker.tx_pcm / 32768.0 if (self.worker is not None and hasattr(self.worker, 'tx_pcm')) else np.zeros(1024)
                f_tx = np.fft.rfftfreq(len(tx_mixed), 1/cfg.FS)
                spec_tx = mag2db(np.fft.rfft(tx_mixed))
                self.plot_cache[cache_key] = {
                    'timestamp': current_time,
                    'f_tx': f_tx,
                    'spec_tx': spec_tx
                }
            cached_data = self.plot_cache[cache_key]
            self.txSpec.plot_line(cached_data['f_tx']/1000, cached_data['spec_tx'], color='#27ae60', linewidth=2.0, alpha=0.9)
            self.txSpec.ax.set_xlabel("Frequency (kHz)")
            self.txSpec.ax.set_ylabel("Magnitude (dB)")
            self.txSpec.ax.grid(True, alpha=0.3)
            for band in cfg.BANDS:
                self.txSpec.ax.axvspan(band[0]/1000, band[1]/1000, color='#b3e5fc', alpha=0.25, lw=0)
            self.txSpec.draw()
            # ---- Rx original spectrum ----
            f_rx = np.fft.rfftfreq(rx.size, 1/cfg.FS)
            spec_rx = mag2db(np.fft.rfft(rx))
            self.rxSpec.plot_line(f_rx/1000, spec_rx, color='#c0392b', linewidth=2.0, alpha=0.8)
            self.rxSpec.ax.set_xlabel("Frequency (kHz)")
            self.rxSpec.ax.set_ylabel("Magnitude (dB)")
            self.rxSpec.ax.grid(True, alpha=0.3)
            
            # ---- Band spectrum and correlation plots (use worker precomputed data) ----
            for i in range(3):
                # 绘制band_spectra
                if i < len(band_spectra) and band_spectra[i] is not None and hasattr(band_spectra[i], '__len__') and len(band_spectra[i]) > 0:
                    f_band = np.fft.rfftfreq(len(band_spectra[i]), 1/cfg.FS)
                    self.bandSpecPlots[i].plot_line(f_band/1000, band_spectra[i], color=colors[i], linewidth=2.0, alpha=0.8)
                    self.bandSpecPlots[i].ax.set_xlabel("Frequency (kHz)")
                    self.bandSpecPlots[i].ax.set_ylabel("Magnitude (dB)")
                    self.bandSpecPlots[i].ax.grid(True, alpha=0.3)
                    band = cfg.BANDS[i]
                    self.bandSpecPlots[i].ax.axvspan(band[0]/1000, band[1]/1000, alpha=0.18, color='#ffe0b2')
                    self.bandSpecPlots[i].draw()
                
                # 绘制correlations
                if i < len(correlations) and correlations[i] is not None and hasattr(correlations[i], '__len__') and len(correlations[i]) > 0:
                    corr = correlations[i]
                    corr_time = np.arange(len(corr)) / cfg.FS * 1000  # ms
                    self.corrPlots[i].plot_line(corr_time, corr, color=colors[i], linewidth=1.5, alpha=0.9)
                    self.corrPlots[i].ax.set_xlabel("Time (ms)")
                    self.corrPlots[i].ax.set_ylabel("Correlation Amplitude")
                    self.corrPlots[i].ax.grid(True, alpha=0.3)
                    self.corrPlots[i].draw()
        except Exception as e:
            logger.error(f"Error in _on_wave: {e}")
            traceback.print_exc()

    def closeEvent(self, e):
        """Handle window close event"""
        self.stop()
        e.accept()
    
    def start(self):
        if self.worker is not None:
            return
        self.worker = SonarWorker()
        self.worker.distanceSig.connect(self._on_dist)
        self.worker.waveSig.connect(self._on_wave)
        self.worker.errorSig.connect(self._on_error)
        self.worker.heartbeatSig.connect(self._on_heartbeat)
        self.worker.start()
        self.btnStart.setEnabled(False)
        self.btnStop.setEnabled(True)

    def stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)

    def _on_error(self, msg):
        logger.error(f"Worker error: {msg}")
        QtWidgets.QMessageBox.critical(self, "Error", str(msg))
        self.stop()

    def _on_heartbeat(self):
        pass

# ------------------------------------------------------------------ #
def main():
    logger.info("=== Air-Sonar optimized start ===")
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    # 检查Qt主屏幕对象，防止NoneType异常
    screen = app.primaryScreen()
    if screen is not None:
        geometry = screen.availableGeometry()
        width = min(1200, geometry.width() - 100)
        height = min(800, geometry.height() - 100)
        win.resize(width, height)
    else:
        win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error, exit")

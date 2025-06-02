# coding=utf-8

"""
realtime_sonar.py - 优化版
~~~~~~~~~~~~~~~~~~~~~~~~~
Windows 平台实时空气声纳系统（最大量程 50 m）优化实现。

优化特点
---------
1. 性能优化：减少绘图频率、优化数据处理、缓存计算结果
2. 原始频谱显示：显示未滤波的原始接收信号频谱
3. 多频融合：SNR计算和置信度加权的距离估计
4. 自适应更新：根据变化幅度调整绘图频率

Copyright © 2025
"""

import sys, time, csv, logging, queue, math, os, traceback
from pathlib import Path
from threading import Lock
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import chirp, iirfilter, filtfilt, firwin, fftconvolve, correlate
import pyaudio
from PyQt5 import QtCore, QtWidgets, QtOpenGL
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# GPU支持
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU (CuPy) 支持已启用")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    print("GPU不可用，使用CPU计算")

# OpenGL支持matplotlib
try:
    import OpenGL.GL as gl
    plt.rcParams['backend'] = 'Qt5Agg'
    OPENGL_AVAILABLE = True
    print("OpenGL 渲染已启用")
except ImportError:
    OPENGL_AVAILABLE = False
    print("OpenGL不可用，使用默认渲染")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========= 声纳参数 ========= #
FS = 48400
CHIRP_LEN = 0.05  # 缩短发射声波时长：从0.1s减少到0.05s
LISTEN_LEN = 0.15  # 相应缩短接收时长
CYCLE = 0.5  # 增大发送间隔：从0.3s增加到0.5s
BASE_TEMP = 28.0  # 基准温度 (°C)
SPEED_SOUND_20C = 343.0  # 20°C时的声速 (m/s)
CHANNELS = 1
FORMAT = pyaudio.paInt16
CSV_PATH = Path("distances.csv")
LOG_PATH = Path("sonar.log")
BANDS = [(21000, 23000), (21000, 23000), (21000, 23000)]  # 修改为不同频段

# 性能优化参数 - 大幅提高刷新频率和减少卡顿
PLOT_UPDATE_INTERVAL = 1  # 每次测量都更新图表
SPECTRUM_UPDATE_INTERVAL = 1  # 每次都更新频谱，提高流畅度
MAX_HIST_POINTS = 300  # 增加历史点数
GUI_UPDATE_RATE = 60  # GUI更新频率 (Hz) - 提高到60FPS
PLOT_DECIMATION = 1  # 绘图数据抽取因子，1表示不抽取

# ========= 日志 ========= #
# 设置控制台编码为UTF-8
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

# ------------------------------------------------------------------ #
def calculate_sound_speed(temperature_c):
    """根据温度计算声速 (m/s)
    公式: v = 331.3 + 0.606 * T (T为摄氏度)
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
            ba = iirfilter(6, [low/(0.5*fs), high/(0.5*fs)], btype='band', output='ba')  # 降低滤波器阶数
            if ba is None:
                raise ValueError(f"iirfilter failed for band {low}-{high} Hz")
            b, a = ba[0], ba[1]
            taps = firwin(61, [low, high], fs=fs, pass_zero=False, window='hamming')  # 减少FIR长度
            filters.append((b, a, taps))
        except Exception as e:
            logger.error(f"Failed to design filter for band {low}-{high} Hz: {e}")
            # 使用默认滤波器参数
            b, a = [1], [1]
            taps = np.array([1])
            filters.append((b, a, taps))
    return filters

def bandpass(sig, filt):
    b, a, taps = filt
    y = filtfilt(b, a, sig)
    return fftconvolve(y, taps, mode='same')

# === 自适应阈值峰检测 === #
def first_strong_peak(corr, fs, min_delay_samples=500):
    half = corr.size//2
    pos = corr[half:]
    if pos.size <= min_delay_samples:
        return None, 0.0
    pos[:min_delay_samples] = 0       # 去直达
    
    # 改进的SNR计算
    # 使用前20%作为噪声基准
    noise_samples = int(len(pos) * 0.2)
    noise_floor = np.mean(pos[:noise_samples] ** 2)  # 噪声功率
    
    # 寻找峰值
    peak_idx = np.argmax(pos)
    peak_power = pos[peak_idx] ** 2
    
    # 计算SNR (dB)
    if noise_floor > 0:
        snr_db = 10 * np.log10(peak_power / noise_floor)
    else:
        snr_db = 0.0
    
    # 检查是否为有效峰值
    if snr_db < 6.0:  # 至少6dB的SNR
        return None, 0.0
        
    return peak_idx, snr_db

# === SNR计算和置信度评估 === #
def calculate_band_confidence(snr, amplitude, band_idx):
    """计算频段置信度
    Args:
        snr: 信噪比
        amplitude: 峰值幅度
        band_idx: 频段索引 (0, 1, 2)
    Returns:
        confidence: 置信度 (0-1)
    """
    # SNR权重 (SNR越高置信度越高)
    snr_weight = min(snr / 10.0, 1.0)  # 归一化到0-1
    
    # 幅度权重 (幅度越大置信度越高)
    amp_weight = min(amplitude / 0.1, 1.0)  # 归一化到0-1
    
    # 频段权重 (中频段通常更稳定)
    freq_weights = [0.8, 1.0, 0.9]  # 低频、中频、高频
    freq_weight = freq_weights[band_idx]
    
    # 综合置信度
    confidence = (snr_weight * 0.5 + amp_weight * 0.3 + freq_weight * 0.2)
    return min(confidence, 1.0)

# === 简单 1-D Kalman === #
class ScalarKalman:
    def __init__(self, q=0.005, r=0.1):  # 调整参数以提高响应速度
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

# ------------------------------------------------------------------ #
class AudioIO:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.out = self.p.open(format=FORMAT, channels=CHANNELS,
                               rate=FS, output=True, frames_per_buffer=1024)
        self.inp = self.p.open(format=FORMAT, channels=CHANNELS,
                               rate=FS, input=True,
                               frames_per_buffer=int(FS*LISTEN_LEN))
        self.lock = Lock()
    def play(self, pcm16):  # int16
        with self.lock:
            self.out.write(pcm16.tobytes())
    def record(self):
        with self.lock:
            raw = self.inp.read(int(FS*LISTEN_LEN),
                                exception_on_overflow=False)
        sig = np.frombuffer(raw, np.int16).astype(np.float32)/2**15
        return sig
    def close(self):
        self.out.stop_stream(); self.out.close()
        self.inp.stop_stream(); self.inp.close()
        self.p.terminate()

# ------------------------------------------------------------------ #
class SonarWorker(QtCore.QThread):
    distanceSig = QtCore.pyqtSignal(float, list, float)  # 距离 + SNR信息 + 置信度
    waveSig = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.tx_pcm, self.chirps = generate_chirps()
        self.filters = design_filters()
        self.kf = ScalarKalman()
        self.audio = AudioIO()
        self.running = False
        self.temperature = 20.0  # 默认温度20°C
        # 移除了 self.rx_buffer
        self.update_counter = 0  # 更新计数器
        
        # 预计算FFT频率轴（性能优化）
        self.tx_freq = np.fft.rfftfreq(len(self.tx_pcm), 1/FS)
        
        if not CSV_PATH.exists():
            with CSV_PATH.open("w", newline='') as f:
                csv.writer(f).writerow(["timestamp", "distance", "confidence", "band_snrs"])

    def run(self):
        self.running = True
        num_cores = os.cpu_count() # 获取CPU核心数
        executor = ThreadPoolExecutor(max_workers=num_cores) # 使用所有核心
        logger.info(f"SonarWorker started (optimized), utilizing {num_cores} CPU cores.")
        
        while self.running:
            t0 = time.perf_counter()
            try:
                # -------- 发射 ----------
                self.audio.play(self.tx_pcm)
                # -------- 接收 ----------
                rx = self.audio.record()
                # 直接处理信号，移除了 self.rx_buffer.extend(rx)

                # -------- 并行滤波+相关 ----------
                futs = [executor.submit(self._process_band, rx, chirp, filt, i)
                        for i, (chirp, filt) in enumerate(zip(self.chirps, self.filters))]
                results = [f.result() for f in futs]
                  # -------- 多频融合处理 ----------
                valid_results = [(dist, conf, snr) for dist, conf, snr in results if dist is not None]
                
                if valid_results:
                    distances, confidences, snrs = zip(*valid_results)
                    
                    # SNR归一化处理
                    snr_array = np.array(snrs)
                    if np.max(snr_array) > 0:
                        normalized_snrs = snr_array / np.max(snr_array) * 100  # 归一化到100%
                    else:
                        normalized_snrs = np.zeros_like(snr_array)
                    
                    # 使用归一化的SNR作为权重进行加权平均
                    weights = normalized_snrs + 1e-9  # 避免除零
                    weighted_dist = np.average(distances, weights=weights)
                    avg_confidence = np.mean(normalized_snrs)
                    
                    # Kalman滤波
                    dist_kf = self.kf.update(weighted_dist)
                    
                    # 发送信号
                    self.distanceSig.emit(dist_kf, list(normalized_snrs), avg_confidence)
                    
                    # 记录到CSV
                    with CSV_PATH.open("a", newline='') as f:
                        csv.writer(f).writerow([time.time(), dist_kf, avg_confidence, normalized_snrs.tolist()])                # -------- 减少绘图频率以提高性能 ----------
                self.update_counter += 1
                if self.update_counter % PLOT_UPDATE_INTERVAL == 0:
                    # 发信号给 GUI (降低频率)
                    band_signals = []
                    correlations = []
                    
                    # 计算各频段滤波后的信号和自相关
                    for i, (chirp, filt) in enumerate(zip(self.chirps, self.filters)):
                        band_y = bandpass(rx, filt)
                        band_signals.append(band_y)
                        # 修正：使用各频段自己的chirp信号进行自相关，而不是混合信号
                        corr = gpu_correlate(band_y, chirp)
                        correlations.append(corr)
                    
                    self.waveSig.emit({
                        'rx': rx,
                        'band_signals': band_signals,
                        'correlations': correlations,
                        'update_spectrum': (self.update_counter % SPECTRUM_UPDATE_INTERVAL == 0)
                    })

            except Exception:
                logger.exception("worker loop error")
            # -------- 节拍 ----------
            time.sleep(max(0, CYCLE - (time.perf_counter()-t0)))
            
        executor.shutdown(wait=False)
        self.audio.close()
        logger.info("SonarWorker stopped")

    def _process_band(self, rx, chirp_sig, filt, band_idx):
        """处理单个频段"""
        y = bandpass(rx, filt)
        corr = correlate(y, chirp_sig, 'full')
        delay, snr = first_strong_peak(corr, FS)
        
        if delay is None:
            return None, 0.0, 0.0
            
        # 计算距离
        sound_speed = calculate_sound_speed(self.temperature)
        distance = (delay/FS)*sound_speed/2
        
        # 计算置信度
        peak_amplitude = np.max(np.abs(corr))
        confidence = calculate_band_confidence(snr, peak_amplitude, band_idx)
        
        return distance, confidence, snr

    def stop(self):
        self.running = False

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
        self.setWindowTitle("实时空气声纳系统 (优化版)")
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
        
        # -------- 控件 --------
        self.label = QtWidgets.QLabel("距离: -- m")
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
        
        # 置信度显示
        self.confidence_label = QtWidgets.QLabel("置信度: --")
        self.confidence_label.setStyleSheet("""
            font-size: 16pt; 
            color: #27ae60; 
            font-weight: bold;
            background: rgba(255,255,255,150);
            border: 1px solid #27ae60;
            border-radius: 8px;
            padding: 8px;
        """)
        
        # 温度控制
        self.temp_label = QtWidgets.QLabel("环境温度 (°C):")
        self.temp_label.setStyleSheet("font-size: 14pt; color: #34495e; font-weight: bold;")
        self.temp_spinbox = QtWidgets.QSpinBox()
        self.temp_spinbox.setRange(-40, 60)
        self.temp_spinbox.setValue(20)
        self.temp_spinbox.setSuffix(" °C")
        self.temp_spinbox.valueChanged.connect(self.on_temp_changed)
        
        self.btnStart = QtWidgets.QPushButton("开始测量")
        self.btnStop = QtWidgets.QPushButton("停止测量")
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
        """)          # 波形 & 频谱 - 修改为新的布局，调整尺寸以减少卡顿
        self.txSpec = MplCanvas("发射信号频谱分析", width=6, height=4)
        self.rxSpec = MplCanvas("接收信号原始频谱", width=6, height=4)
        
        # 三个自相关图 - 减小尺寸提高性能
        self.corrPlots = [
            MplCanvas(f"频段{i+1}自相关 ({BANDS[i][0]}-{BANDS[i][1]}Hz)", width=5, height=3) 
            for i in range(3)
        ]
        
        # 三个频段的频谱图（不是时域信号） - 减小尺寸提高性能
        self.bandSpecPlots = [
            MplCanvas(f"频段{i+1}滤波频谱 ({BANDS[i][0]}-{BANDS[i][1]}Hz)", width=5, height=3) 
            for i in range(3)
        ]  
        self.hist = MplCanvas("距离历史曲线", width=16, height=3.5)
        
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
        
        # -------- 信号 --------
        self.btnStart.clicked.connect(self.start)
        self.btnStop.clicked.connect(self.stop)
        self.dist_hist = []
        self.time_hist = []
        self.confidence_hist = []
        self.start_time = None
        self.worker = None        # 性能优化：减少重绘并添加绘图缓存
        self.last_update_time = 0
        self.min_update_interval = 1.0 / GUI_UPDATE_RATE  # 基于GUI_UPDATE_RATE计算更新间隔
        self.plot_cache = {}  # 绘图缓存
        self.spectrum_cache_timeout = 0.2  # 频谱缓存超时时间(秒)

    def on_temp_changed(self, value):
        """温度改变时更新worker的温度值"""
        if self.worker:
            self.worker.temperature = float(value)    # --- worker 信号槽 ---
    @QtCore.pyqtSlot(float, list, float)
    def _on_dist(self, d, snrs, confidence):
        current_time = time.time()
        
        # 性能优化：限制更新频率
        if current_time - self.last_update_time < self.min_update_interval:
            return
        self.last_update_time = current_time
        
        if self.start_time is None:
            self.start_time = current_time
        
        elapsed_time = current_time - self.start_time
        
        # 更新显示
        self.label.setText(f"距离: {d:6.2f} m")
        self.confidence_label.setText(f"置信度: {confidence:.1f}% | SNR: {snrs}")
        
        # 添加到历史数据
        self.dist_hist.append(d)
        self.time_hist.append(elapsed_time)
        self.confidence_hist.append(confidence)
        
        # 限制历史数据长度（性能优化）
        if len(self.dist_hist) > MAX_HIST_POINTS:
            self.dist_hist = self.dist_hist[-MAX_HIST_POINTS:]
            self.time_hist = self.time_hist[-MAX_HIST_POINTS:]
            self.confidence_hist = self.confidence_hist[-MAX_HIST_POINTS:]
        
        # 绘制距离历史曲线 - 改善视觉效果
        self.hist.ax.clear()
        self.hist.ax.grid(True, alpha=0.4, linewidth=0.8)
        self.hist.ax.set_title("距离历史曲线 (置信度加权)", fontsize=16, fontweight='bold', pad=15)
        
        if len(self.time_hist) > 1:
            # 根据置信度设置颜色
            colors = ['#e74c3c' if c < 30 else '#f39c12' if c < 70 else '#27ae60' 
                     for c in self.confidence_hist]
            
            # 绘制主曲线
            self.hist.ax.plot(self.time_hist, self.dist_hist, '-', 
                             color='#3498db', linewidth=2, alpha=0.8)
            
            # 根据置信度绘制散点
            self.hist.ax.scatter(self.time_hist, self.dist_hist, 
                               c=colors, s=30, alpha=0.8, edgecolors='white', linewidth=1)
            
            # 添加填充区域
            self.hist.ax.fill_between(self.time_hist, self.dist_hist, alpha=0.15, color='#3498db')
            
            # 根据置信度绘制散点
            self.hist.ax.scatter(self.time_hist, self.dist_hist, 
                               c=colors, s=30, alpha=0.8, edgecolors='white', linewidth=1)
            
            # 添加填充区域
            self.hist.ax.fill_between(self.time_hist, self.dist_hist, alpha=0.15, color='#3498db')
        
        self.hist.ax.set_xlabel("时间 (s)", fontsize=14, fontweight='bold')
        self.hist.ax.set_ylabel("距离 (m)", fontsize=14, fontweight='bold')
        self.hist.ax.tick_params(labelsize=12)
        self.hist.ax.set_xlim(max(0, elapsed_time-60), elapsed_time+2)  # 显示最近60秒
        
        if self.dist_hist:
            y_range = max(self.dist_hist) - min(self.dist_hist)
            if y_range > 0:
                self.hist.ax.set_ylim(min(self.dist_hist)-y_range*0.15, 
                                     max(self.dist_hist)+y_range*0.15)
        
        # 美化坐标轴
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
            
            # 定义颜色
            colors = ['#e74c3c', '#f39c12', '#9b59b6']
            
            # 性能优化：数据抽取（每PLOT_DECIMATION个点取一个）
            if len(rx) > 1000 and PLOT_DECIMATION > 1:
                rx_decimated = rx[::PLOT_DECIMATION]
            else:
                rx_decimated = rx
            
            # 发射信号频谱（使用缓存和抽取）
            if update_spectrum and hasattr(self, 'worker') and self.worker:
                cache_key = 'tx_spectrum'
                if (cache_key not in self.plot_cache or 
                    current_time - self.plot_cache[cache_key]['timestamp'] > self.spectrum_cache_timeout):
                    
                    tx_mixed = self.worker.tx_pcm / 32768.0  # 归一化
                    # 数据抽取
                    if len(tx_mixed) > 1000 and PLOT_DECIMATION > 1:
                        tx_mixed = tx_mixed[::PLOT_DECIMATION]
                    
                    f_tx = np.fft.rfftfreq(len(tx_mixed), PLOT_DECIMATION/FS)
                    spec_tx = mag2db(np.fft.rfft(tx_mixed))
                    
                    # 缓存结果
                    self.plot_cache[cache_key] = {
                        'timestamp': current_time,
                        'f_tx': f_tx,
                        'spec_tx': spec_tx
                    }
                
                # 使用缓存数据
                cached_data = self.plot_cache[cache_key]
                
                self.txSpec.ax.clear()
                self.txSpec.ax.plot(cached_data['f_tx']/1000, cached_data['spec_tx'], 
                                  color='#27ae60', linewidth=2.0, alpha=0.9)  # 减小线宽
                self.txSpec.ax.grid(True, alpha=0.3, linewidth=0.5)  # 减小网格线宽
                self.txSpec.ax.set_xlabel("频率 (kHz)", fontsize=10, fontweight='bold')  # 减小字体
                self.txSpec.ax.set_ylabel("幅度 (dB)", fontsize=10, fontweight='bold')
                self.txSpec.ax.set_title("发射信号频谱", fontsize=12, fontweight='bold', pad=10)
                self.txSpec.ax.tick_params(labelsize=9)
                # 高亮频带区域
                for low, high in BANDS:
                    self.txSpec.ax.axvspan(low/1000, high/1000, alpha=0.12, color='#27ae60')
                # 美化坐标轴
                self.txSpec.ax.spines['top'].set_visible(False)
                self.txSpec.ax.spines['right'].set_visible(False)
                self.txSpec.ax.spines['left'].set_linewidth(1.0)  # 减小边框线宽
                self.txSpec.ax.spines['bottom'].set_linewidth(1.0)
                self.txSpec.draw()                
                # ---- Rx 原始频谱 ----
                f_rx = np.fft.rfftfreq(rx_decimated.size, PLOT_DECIMATION/FS)
                spec_rx = mag2db(np.fft.rfft(rx_decimated))
                self.rxSpec.ax.clear()
                self.rxSpec.ax.plot(f_rx/1000, spec_rx, color='#c0392b', linewidth=2.0, alpha=0.8)
                self.rxSpec.ax.grid(True, alpha=0.3, linewidth=0.5)  # 减小网格线宽
                self.rxSpec.ax.set_xlabel("频率 (kHz)", fontsize=10, fontweight='bold')
                self.rxSpec.ax.set_ylabel("幅度 (dB)", fontsize=10, fontweight='bold')
                self.rxSpec.ax.set_title("接收信号原始频谱", fontsize=12, fontweight='bold', pad=10)
                self.rxSpec.ax.tick_params(labelsize=9)
                # 高亮频带区域
                for low, high in BANDS:
                    self.rxSpec.ax.axvspan(low/1000, high/1000, alpha=0.12, color='#c0392b')
                # 美化坐标轴
                self.rxSpec.ax.spines['top'].set_visible(False)
                self.rxSpec.ax.spines['right'].set_visible(False)
                self.rxSpec.ax.spines['left'].set_linewidth(1.0)
                self.rxSpec.ax.spines['bottom'].set_linewidth(1.0)
                self.rxSpec.draw()              # ---- 三个独立的自相关函数图 ----
            if correlations:
                colors = ['#e74c3c', '#f39c12', '#9b59b6']
                for i, corr in enumerate(correlations[:3]):  # 最多3个频段
                    if len(corr) > 0:
                        # 数据抽取以提高性能
                        if len(corr) > 2000 and PLOT_DECIMATION > 1:
                            corr_decimated = corr[::PLOT_DECIMATION]
                        else:
                            corr_decimated = corr
                        
                        # 时间轴（以毫秒为单位）
                        len_chirp_samples = int(FS * CHIRP_LEN)
                        lag_indices = np.arange(len(corr_decimated))
                        time_axis_samples = lag_indices * PLOT_DECIMATION - (len_chirp_samples - 1)
                        time_axis = time_axis_samples / FS * 1000  # 转换为毫秒
                        
                        self.corrPlots[i].ax.clear()
                        self.corrPlots[i].ax.plot(time_axis, corr_decimated, 
                                                color=colors[i % len(colors)], 
                                                linewidth=1.5, alpha=0.9)  # 减小线宽
                        self.corrPlots[i].ax.grid(True, alpha=0.3, linewidth=0.5)  # 减小网格线宽
                        self.corrPlots[i].ax.set_xlabel("时间延迟 (ms)", fontsize=9, fontweight='bold')
                        self.corrPlots[i].ax.set_ylabel("相关系数", fontsize=9, fontweight='bold')
                        self.corrPlots[i].ax.set_title(f"频段{i+1}自相关 ({BANDS[i][0]}-{BANDS[i][1]}Hz)", 
                                                     fontsize=10, fontweight='bold', pad=8)
                        self.corrPlots[i].ax.tick_params(labelsize=8)
                        # 美化坐标轴
                        self.corrPlots[i].ax.spines['top'].set_visible(False)
                        self.corrPlots[i].ax.spines['right'].set_visible(False)
                        self.corrPlots[i].ax.spines['left'].set_linewidth(1.0)
                        self.corrPlots[i].ax.spines['bottom'].set_linewidth(1.0)
                        self.corrPlots[i].draw()
            
            # ---- 各频段滤波信号的频谱图 ----
            if band_signals:
                for i, band_signal in enumerate(band_signals[:3]):  # 最多3个频段
                    if len(band_signal) > 0:
                        # 计算频谱
                        f_band = np.fft.rfftfreq(len(band_signal), 1/FS)
                        spec_band = mag2db(np.fft.rfft(band_signal))
                        
                        self.bandSpecPlots[i].ax.clear()
                        self.bandSpecPlots[i].ax.plot(f_band, spec_band, 
                                                    color=colors[i % len(colors)], 
                                                    linewidth=2.0, alpha=0.9)
                        self.bandSpecPlots[i].ax.grid(True, alpha=0.4, linewidth=0.8)
                        self.bandSpecPlots[i].ax.set_xlabel("频率 (Hz)", fontsize=10, fontweight='bold')
                        self.bandSpecPlots[i].ax.set_ylabel("幅度 (dB)", fontsize=10, fontweight='bold')
                        self.bandSpecPlots[i].ax.set_title(f"频段{i+1}滤波频谱 ({BANDS[i][0]}-{BANDS[i][1]}Hz)", 
                                                         fontsize=12, fontweight='bold', pad=10)
                        self.bandSpecPlots[i].ax.tick_params(labelsize=9)
                        # 高亮对应频带区域
                        low, high = BANDS[i]
                        self.bandSpecPlots[i].ax.axvspan(low, high, alpha=0.15, color=colors[i % len(colors)])
                        # 美化坐标轴
                        self.bandSpecPlots[i].ax.spines['top'].set_visible(False)
                        self.bandSpecPlots[i].ax.spines['right'].set_visible(False)
                        self.bandSpecPlots[i].ax.spines['left'].set_linewidth(1.5)
                        self.bandSpecPlots[i].ax.spines['bottom'].set_linewidth(1.5)
                        self.bandSpecPlots[i].draw()
        except Exception as e:
            logger.error(f"Error in _on_wave: {e}")
            traceback.print_exc()

    # --- 控制 ---
    def start(self):
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
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.start_time = None

    def closeEvent(self, e):
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

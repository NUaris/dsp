# coding=utf-8

"""
realtime_sonar.py
~~~~~~~~~~~~~~~~~
Windows 平台实时空气声纳系统（最大量程 50 m）示例实现。
功能特点
---------
1. **三路线性 FM Chirp**（3-6 kHz / 8-11 kHz / 13-16 kHz）合成发射；
2. **0.1 s 发射 + 0.2 s 监听**，周期 0.3 s；
3. IIR(Butterworth) + 窗 FIR 带通滤波器分离三个频段；
4. 自相关取首个强峰提取回波延时并求三频平均距离；
5. PyQt5 GUI + Matplotlib 实时显示：
   • 当前距离数值 • 距离历史曲线 • 发射/回波波形与频谱图
6. 多线程（QThread）架构，GUI 不卡顿；
7. `logging` 同时输出到控制台和 sonar.log；
8. 测距结果追加写入 distances.csv（timestamp, distance）。

Copyright © 2025
"""


"""
realtime_sonar.py  (增强版 2025-06-02)
-------------------------------------
新增:
  * FFT 频谱图 (dB)   * 自适应阈值   * 1-D Kalman 滤波
  * ThreadPool 并行滤波+相关        * 回波缓存 (deque)
"""

import sys, time, csv, logging, queue, math
from pathlib import Path
from threading import Lock
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import chirp, iirfilter, filtfilt, firwin, fftconvolve, correlate
import pyaudio
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========= 声纳参数 ========= #
FS = 44100
CHIRP_LEN = 0.1
LISTEN_LEN = 0.2
CYCLE = 0.3
BASE_TEMP = 20.0  # 基准温度 (°C)
SPEED_SOUND_20C = 343.0  # 20°C时的声速 (m/s)
CHANNELS = 1
FORMAT = pyaudio.paInt16
CSV_PATH = Path("distances.csv")
LOG_PATH = Path("sonar.log")
BANDS = [(3000, 6000), (8000, 11000), (13000, 16000)]

# ========= 日志 ========= #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)]
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
        ba = iirfilter(8, [low/(0.5*fs), high/(0.5*fs)], btype='band', output='ba')
        if ba is None:
            raise ValueError(f"iirfilter failed for band {low}-{high} Hz")
        b, a = ba[0], ba[1]
        taps = firwin(101, [low, high], fs=fs, pass_zero=False, window='hamming')
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
        return None
    pos[:min_delay_samples] = 0       # 去直达
    # 噪声底估计 (中位数+MAD)
    med = np.median(pos)
    mad = np.median(np.abs(pos - med)) + 1e-9
    noise_thr = med + 6*mad          # 6-σ
    idxs = np.where(pos > noise_thr)[0]
    if idxs.size == 0:
        return None
    return int(idxs[0])

# === 简单 1-D Kalman === #
class ScalarKalman:
    def __init__(self, q=0.01, r=0.2):
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
    distanceSig = QtCore.pyqtSignal(float)
    waveSig = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.tx_pcm, self.chirps = generate_chirps()
        self.filters = design_filters()
        self.kf = ScalarKalman()
        self.audio = AudioIO()
        self.running = False
        self.temperature = 20.0  # 默认温度20°C
        self.rx_buffer = deque(maxlen=int(FS*LISTEN_LEN))  # 回波缓存
        if not CSV_PATH.exists():
            with CSV_PATH.open("w", newline='') as f:
                csv.writer(f).writerow(["timestamp", "distance"])

    def run(self):
        self.running = True
        executor = ThreadPoolExecutor(max_workers=3)
        logger.info("SonarWorker started")
        while self.running:
            t0 = time.perf_counter()
            try:
                # -------- 发射 ----------
                self.audio.play(self.tx_pcm)
                # -------- 接收 ----------
                rx = self.audio.record()
                self.rx_buffer.extend(rx)   # 加入缓存

                # -------- 并行滤波+相关 ----------
                futs = [executor.submit(self._process_band, rx, chirp, filt)
                        for chirp, filt in zip(self.chirps, self.filters)]
                dists = [f.result() for f in futs]
                dists_valid = [d for d in dists if d is not None]
                if dists_valid:
                    dist_raw = float(np.mean(dists_valid))
                    dist_kf = self.kf.update(dist_raw)
                    self.distanceSig.emit(dist_kf)
                    with CSV_PATH.open("a", newline='') as f:
                        csv.writer(f).writerow([time.time(), dist_kf])

                # -------- 发信号给 GUI ----------
                self.waveSig.emit({'tx': self.tx_pcm/32768.0, 'rx': rx})

            except Exception:
                logger.exception("worker loop error")
            # -------- 节拍 ----------
            time.sleep(max(0, CYCLE - (time.perf_counter()-t0)))
        executor.shutdown(wait=False)
        self.audio.close()
        logger.info("SonarWorker stopped")

    def _process_band(self, rx, chirp_sig, filt):
        y = bandpass(rx, filt)
        corr = correlate(y, chirp_sig, 'full')
        delay = first_strong_peak(corr, FS)
        if delay is None:
            return None
        sound_speed = calculate_sound_speed(self.temperature)
        return (delay/FS)*sound_speed/2

    def stop(self):
        self.running = False

# ------------------------------------------------------------------ #
class MplCanvas(FigureCanvas):
    def __init__(self, title, parent=None, width=8, height=5.5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.ax = fig.add_subplot(111)
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        super().__init__(fig)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎯 实时空气声纳系统 (增强版)")
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
        
        # 温度控制
        self.temp_label = QtWidgets.QLabel("环境温度 (°C):")
        self.temp_label.setStyleSheet("font-size: 14pt; color: #34495e; font-weight: bold;")
        self.temp_spinbox = QtWidgets.QSpinBox()
        self.temp_spinbox.setRange(-40, 60)
        self.temp_spinbox.setValue(20)
        self.temp_spinbox.setSuffix(" °C")
        self.temp_spinbox.valueChanged.connect(self.on_temp_changed)
        
        self.btnStart = QtWidgets.QPushButton("▶ 开始测量")
        self.btnStop = QtWidgets.QPushButton("⏹ 停止测量")
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
        """)        # 波形 & 频谱 - 增大尺寸并改善视觉效果
        self.txTime = MplCanvas("发射信号时域波形", width=8, height=5.5)
        self.txSpec = MplCanvas("发射信号频谱分析", width=8, height=5.5)
        self.rxTime = MplCanvas("接收信号时域波形", width=8, height=5.5)
        self.rxSpec = MplCanvas("接收信号频谱分析", width=8, height=5.5)
        self.hist = MplCanvas("距离历史曲线", width=16, height=4)
        
        # 为图表添加边框样式
        chart_style = """
            border: 2px solid #34495e;
            border-radius: 10px;
            background: white;
            margin: 3px;
        """
        self.txTime.setStyleSheet(chart_style)
        self.txSpec.setStyleSheet(chart_style)
        self.rxTime.setStyleSheet(chart_style)
        self.rxSpec.setStyleSheet(chart_style)
        self.hist.setStyleSheet(chart_style)# -------- 布局 --------
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
        control_layout.addWidget(self.label)
          # 图表网格 - 2x2布局，增大间距
        g = QtWidgets.QGridLayout()
        g.setSpacing(12)  # 增加间距让图表更清楚
        g.addWidget(self.txTime, 0, 0)
        g.addWidget(self.txSpec, 0, 1)
        g.addWidget(self.rxTime, 1, 0)
        g.addWidget(self.rxSpec, 1, 1)
        
        # 主布局
        vbox = QtWidgets.QVBoxLayout()
        vbox.setSpacing(15)  # 增加垂直间距
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
        self.start_time = None
        self.worker = None

    def on_temp_changed(self, value):
        """温度改变时更新worker的温度值"""
        if self.worker:
            self.worker.temperature = float(value)

    # --- worker 信号槽 ---    @QtCore.pyqtSlot(float)
    def _on_dist(self, d):
        if self.start_time is None:
            self.start_time = time.time()
        
        current_time = time.time() - self.start_time
        
        self.label.setText(f"距离: {d:6.2f} m")
        self.dist_hist.append(d)
        self.time_hist.append(current_time)
        
        # 限制历史数据长度
        max_points = 200
        if len(self.dist_hist) > max_points:
            self.dist_hist = self.dist_hist[-max_points:]
            self.time_hist = self.time_hist[-max_points:]
        
        # 绘制距离历史曲线 - 改善视觉效果
        self.hist.ax.clear()
        self.hist.ax.grid(True, alpha=0.4, linewidth=0.8)
        self.hist.ax.set_title("距离历史曲线", fontsize=16, fontweight='bold', pad=15)
        
        if len(self.time_hist) > 1:
            self.hist.ax.plot(self.time_hist, self.dist_hist, '-o', 
                             color='#3498db', linewidth=3, markersize=4, 
                             alpha=0.9, markerfacecolor='#2980b9', markeredgecolor='white', markeredgewidth=1)
            
            # 添加填充区域
            self.hist.ax.fill_between(self.time_hist, self.dist_hist, alpha=0.2, color='#3498db')
        
        self.hist.ax.set_xlabel("时间 (s)", fontsize=14, fontweight='bold')
        self.hist.ax.set_ylabel("距离 (m)", fontsize=14, fontweight='bold')
        self.hist.ax.tick_params(labelsize=12)
        self.hist.ax.set_xlim(max(0, current_time-60), current_time+2)  # 显示最近60秒
        
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
        tx, rx = data['tx'], data['rx']
        
        # 时间轴
        tx_time = np.linspace(0, CHIRP_LEN, len(tx))
        rx_time = np.linspace(0, LISTEN_LEN, len(rx))
        
        # ---- Tx 时域 - 增强版 ----        self.txTime.ax.clear()
        self.txTime.ax.plot(tx_time, tx, color='#2ecc71', linewidth=2.5, alpha=0.9)
        self.txTime.ax.grid(True, alpha=0.4, linewidth=0.8)
        self.txTime.ax.set_xlabel("时间 (s)", fontsize=12, fontweight='bold')
        self.txTime.ax.set_ylabel("幅度", fontsize=12, fontweight='bold')
        self.txTime.ax.set_title("发射信号时域", fontsize=14, fontweight='bold', pad=15)
        self.txTime.ax.tick_params(labelsize=11)
        # 美化坐标轴
        self.txTime.ax.spines['top'].set_visible(False)
        self.txTime.ax.spines['right'].set_visible(False)
        self.txTime.ax.spines['left'].set_linewidth(1.5)
        self.txTime.ax.spines['bottom'].set_linewidth(1.5)
        self.txTime.draw()
        
        # ---- Tx 频谱 - 增强版 ----
        f = np.fft.rfftfreq(tx.size, 1/FS)
        spec = mag2db(np.fft.rfft(tx))
        self.txSpec.ax.clear()
        self.txSpec.ax.plot(f/1000, spec, color='#27ae60', linewidth=2.5, alpha=0.9)
        self.txSpec.ax.grid(True, alpha=0.4, linewidth=0.8)
        self.txSpec.ax.set_xlabel("频率 (kHz)", fontsize=12, fontweight='bold')
        self.txSpec.ax.set_ylabel("幅度 (dB)", fontsize=12, fontweight='bold')
        self.txSpec.ax.set_title("发射信号频谱", fontsize=14, fontweight='bold', pad=15)
        self.txSpec.ax.tick_params(labelsize=11)
        # 高亮频带区域
        for low, high in BANDS:
            self.txSpec.ax.axvspan(low/1000, high/1000, alpha=0.15, color='#27ae60')
        # 美化坐标轴
        self.txSpec.ax.spines['top'].set_visible(False)
        self.txSpec.ax.spines['right'].set_visible(False)
        self.txSpec.ax.spines['left'].set_linewidth(1.5)
        self.txSpec.ax.spines['bottom'].set_linewidth(1.5)
        self.txSpec.draw()
        
        # ---- Rx 时域 - 增强版 ----
        self.rxTime.ax.clear()
        self.rxTime.ax.plot(rx_time, rx, color='#e74c3c', linewidth=2.0, alpha=0.8)
        self.rxTime.ax.grid(True, alpha=0.4, linewidth=0.8)
        self.rxTime.ax.set_xlabel("时间 (s)", fontsize=12, fontweight='bold')
        self.rxTime.ax.set_ylabel("幅度", fontsize=12, fontweight='bold')
        self.rxTime.ax.set_title("接收信号时域", fontsize=14, fontweight='bold', pad=15)
        self.rxTime.ax.tick_params(labelsize=11)
        # 美化坐标轴
        self.rxTime.ax.spines['top'].set_visible(False)
        self.rxTime.ax.spines['right'].set_visible(False)
        self.rxTime.ax.spines['left'].set_linewidth(1.5)
        self.rxTime.ax.spines['bottom'].set_linewidth(1.5)
        self.rxTime.draw()
        
        # ---- Rx 频谱 - 增强版 ----
        f2 = np.fft.rfftfreq(rx.size, 1/FS)
        spec2 = mag2db(np.fft.rfft(rx))
        self.rxSpec.ax.clear()
        self.rxSpec.ax.plot(f2/1000, spec2, color='#c0392b', linewidth=2.0, alpha=0.8)
        self.rxSpec.ax.grid(True, alpha=0.4, linewidth=0.8)
        self.rxSpec.ax.set_xlabel("频率 (kHz)", fontsize=12, fontweight='bold')
        self.rxSpec.ax.set_ylabel("幅度 (dB)", fontsize=12, fontweight='bold')
        self.rxSpec.ax.set_title("🎵 接收信号频谱", fontsize=14, fontweight='bold', pad=15)
        self.rxSpec.ax.tick_params(labelsize=11)
        # 高亮频带区域
        for low, high in BANDS:
            self.rxSpec.ax.axvspan(low/1000, high/1000, alpha=0.15, color='#c0392b')
        # 美化坐标轴
        self.rxSpec.ax.spines['top'].set_visible(False)
        self.rxSpec.ax.spines['right'].set_visible(False) 
        self.rxSpec.ax.spines['left'].set_linewidth(1.5)
        self.rxSpec.ax.spines['bottom'].set_linewidth(1.5)
        self.rxSpec.draw()

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

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.start_time = None

    def closeEvent(self, e):
        self.stop(); e.accept()

# ------------------------------------------------------------------ #
def main():
    logger.info("=== Air-Sonar enhanced start ===")
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(); win.resize(1400,900); win.show()  # 增大窗口尺寸以适应更大的图表
    sys.exit(app.exec_())

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error, exit")

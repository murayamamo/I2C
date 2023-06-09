import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

# 101個の.wavファイルのパス
file_paths = ["noise_0.wav","noise_1.wav", "noise_2.wav", ..., "noise_100.wav"]

# .wavファイルを読み込む
signals = []
for file_path in file_paths:
    with wave.open(file_path, "rb") as wf:
        signal = wf.readframes(wf.getnframes())
        signals.append(signal)

# numpy配列に変換する
signals = [np.frombuffer(s, dtype=np.int16) for s in signals]

# 全ての信号のサイズを合わせる
min_size = min([len(s) for s in signals])
signals = [s[:min_size] for s in signals]

# 同期加算して復元する
recovered_signal = np.zeros(min_size)
for s in signals:
    recovered_signal += s
recovered_signal = (recovered_signal / len(signals)).astype(np.int16)

# SN比を計算する
original_signal = signals[0]
noise_signal = recovered_signal - original_signal
snr = 10 * np.log10(np.sum(original_signal ** 2) / np.sum(noise_signal ** 2))

# 同期加算した信号の数とSN比をプロットする
n = len(signals)
x = np.arange(1, n+1)
snr_list = []
for i in range(n):
    r = correlate(signals[i], original_signal, mode='same')
    snr_i = 10 * np.log10(np.sum(original_signal ** 2) / np.sum((signals[i] - r) ** 2))
    snr_list.append(snr_i)
    
fig, ax1 = plt.subplots()
ax1.plot(x, snr_list, label="SN比")
ax1.set_xlabel("同期加算した信号の数")
ax1.set_ylabel("SN比 (dB)")
ax1.tick_params(axis='y')
ax2 = ax1.twinx()
ax2.plot(x, [len(s) for s in signals], color="red", label="信号の長さ")
ax2.set_ylabel("同期加算後の信号の長さ")
ax2.tick_params(axis='y')
plt.title("同期加算による原信号の復元精度")
fig.legend()
plt.show()


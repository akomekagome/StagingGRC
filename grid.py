import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

rewards = np.load('grid_smal_uzumaki_dankai_2.npy', allow_pickle=True)
rewards2 = np.load('grid_smal_uzumaki_normal_2.npy', allow_pickle=True)
# rewards = np.load('grid_large_dankai_1.npy', allow_pickle=True)
# rewards2 = np.load('grid_large_normal_1.npy', allow_pickle=True)
length = 1000

# 結果のプロット
plt.plot(np.arange(length), rewards2, label="GRC")
plt.plot(np.arange(length), rewards, label="段階別大局基準値によるGRC")
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()
plt.savefig("result.jpg")
plt.show()
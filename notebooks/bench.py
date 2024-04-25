import japanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("bench2.csv")

x = {
    "capsize": 3.0,
    "fmt": "o-",
    "markersize": 2,
}

plt.errorbar(df["N_e"], df["kyuu"], df["kyuu_bar"], label="旧実装", **x)
plt.errorbar(df["N_e"], df["cpu"], df["cpu_bar"], label="CPU", **x)
plt.errorbar(df["N_e"], df["gpu"], df["gpu_bar"], label="GPU", **x)

# plt.ylim(0, 5000)

plt.minorticks_on()
plt.legend()
plt.title("Random 4-dim tensor")
plt.ylabel("Runtime (ms)")
plt.xlabel("# X elements")
plt.savefig("bench_full.png")

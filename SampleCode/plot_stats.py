import sys, re, matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("usage: python plot_summary.py  <stats-file>")
    sys.exit(1)

txt = open(sys.argv[1]).read()
def grab(key):          # small helper
    m = re.search(rf"{key}\s+([-+]?\d+(\.\d+)?)", txt, re.I)
    return float(m.group(1)) if m else None

vals   = [grab("Best"), grab("Mean"), grab("Worst")]
labels = ["Best", "Mean", "Worst"]

plt.figure(figsize=(5,3))
plt.plot(labels, vals, marker="o", linewidth=2, color="royalblue")
plt.ylabel("Route length")
plt.title("Performance Evaluation Simulated Annealing")
for x,y in zip(labels, vals):
    plt.text(x, y, f"{y:.1f}", ha="center", va="bottom")
plt.tight_layout()
plt.show()
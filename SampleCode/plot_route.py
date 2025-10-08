import sys, re, numpy as np, matplotlib.pyplot as plt
from itertools import cycle

if len(sys.argv) != 3:
    print("usage: python pretty_plot_route.py  <instance.evrp>  <tour.txt>")
    sys.exit(1)

evrp_file, tour_file = sys.argv[1], sys.argv[2]

with open(evrp_file) as f:
    raw = f.read()

def block(after, until):
    m = re.search(rf"{after}(.*?){until}", raw, re.S)
    return [] if m is None else m.group(1).strip().splitlines()

coords = {}
for line in block("NODE_COORD_SECTION", "DEMAND_SECTION"):
    i, x, y = map(float, line.split())
    coords[int(i)-1] = (x, y)

depot_xy = coords[0]
cust_xy = {i:xy for i,xy in coords.items() if 0 < i < 23}
station_xy = {i:xy for i,xy in coords.items() if i >= 23}

tour = np.loadtxt(tour_file, dtype=int).tolist()

# Parse routes
routes, seg = [], []
for v in tour:
    seg.append(v)
    if v == 0 and len(seg) > 1:        
        routes.append(seg)
        seg = [0]                     
if seg != [0]:
    routes.append(seg)

# Create figure with clean white background
plt.figure(figsize=(10, 8))
plt.gca().set_facecolor('white')

# Extract problem name properly
import os
problem_name = os.path.splitext(os.path.basename(evrp_file))[0]

# Set title with clean formatting
if problem_name:
    plt.title(f"Problem {problem_name}", fontsize=16, fontweight='normal', pad=20)
else:
    plt.title("Problem", fontsize=16, fontweight='normal', pad=20)

# Plot routes with dotted lines (more similar to reference image)
for r in routes:
    xs = [coords[i][0] for i in r]
    ys = [coords[i][1] for i in r]
    
    # Use dotted lines that match the reference image better
    plt.plot(xs, ys, color='#333333', linewidth=1.0,
              linestyle=':', alpha=0.8, zorder=1)

# Plot nodes with styling from reference image
# Define labels
customer_label = "Customer Node"
station_label = "Charging Station Node" 
depot_label = "Depot Node"

# Customer nodes - solid green circles
if cust_xy:
    cx, cy = zip(*cust_xy.values())
    plt.scatter(cx, cy, s=60, color='#2E8B57',  # Sea green color
                edgecolors='none', 
                label=customer_label, zorder=3, alpha=1.0)

# Charging stations - solid blue triangles
if station_xy:
    sx, sy = zip(*station_xy.values())
    plt.scatter(sx, sy, s=80, marker='^', color='#4169E1',  # Royal blue
                edgecolors='none', 
                label=station_label, zorder=4, alpha=1.0)

# Depot - solid red square
plt.scatter(*depot_xy, s=100, marker='s', color='#DC143C',  # Crimson red
            edgecolors='none', 
            label=depot_label, zorder=5)

# Clean plot styling exactly like reference image
plt.gca().set_aspect('equal')

# Remove top and right spines for cleaner look
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')

# Style the ticks and grid
plt.tick_params(axis='both', which='major', labelsize=10, 
                colors='#666666', length=4, width=1)
plt.grid(True, alpha=0.3, color='#CCCCCC', linewidth=0.5)

# Legend with clean styling matching reference
legend = plt.legend(loc='upper right', fontsize=11, frameon=True,
                   fancybox=False, shadow=False, framealpha=1.0,
                   edgecolor='#CCCCCC', facecolor='white')
legend.get_frame().set_linewidth(1)

# Set margins for clean appearance
plt.margins(0.02)

# Final layout adjustment
plt.tight_layout()
plt.show()
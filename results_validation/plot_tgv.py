import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

base = r"D:\FluidX3D-MultiBlock"
outdir = os.path.join(base, "results_validation")

df_uf = pd.read_csv(os.path.join(base, "energy_uniform_fine.csv"))
df_mb = pd.read_csv(os.path.join(base, "energy_multiblock.csv"))

# Normalize
E0_uf = df_uf['energy'].iloc[0]
E0_mb = df_mb['energy_total'].iloc[0]

fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

# Uniform fine: steps are in fine-grid units, convert to coarse-equivalent
ax.plot(df_uf['step']/2.0, df_uf['energy']/E0_uf, 'b-o', markersize=3, linewidth=1.5, label='Uniform fine 128^3')
ax.plot(df_mb['step'], df_mb['energy_total']/E0_mb, 'g-^', markersize=4, linewidth=1.5, label='Multi-block (coarse 64^3 + fine 64^3)')
ax.plot(df_mb['step'], df_mb['energy_coarse']/E0_mb, 'r--', markersize=3, linewidth=1, alpha=0.7, label='Multi-block coarse only')
ax.plot(df_mb['step'], df_mb['energy_fine']/E0_mb, 'g--', markersize=3, linewidth=1, alpha=0.7, label='Multi-block fine only')

ax.set_xlabel('Coarse timestep', fontsize=12)
ax.set_ylabel('E / E0', fontsize=12)
ax.set_title('Taylor-Green Vortex Decay: Energy vs Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
ax.set_ylim(1e-3, 1.5)

fig.savefig(os.path.join(outdir, "tgv_energy_decay.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(outdir, "lowres", "tgv_energy_decay.png"), dpi=100, bbox_inches='tight')
plt.close(fig)

print("=== TGV Summary ===")
print(f"Uniform fine E/E0 final: {df_uf['energy'].iloc[-1]/E0_uf:.4f}")
print(f"Multi-block E/E0 final: {df_mb['energy_total'].iloc[-1]/E0_mb:.4f}")
print(f"Plots saved to {outdir}")

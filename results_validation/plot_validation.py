import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
base = r"D:\FluidX3D-MultiBlock"
df_uniform = pd.read_csv(os.path.join(base, "profile_uniform_fine.csv"))
df_coarse = pd.read_csv(os.path.join(base, "profile_multiblock_coarse.csv"))
df_fine = pd.read_csv(os.path.join(base, "profile_multiblock_fine.csv"))

# Channel parameters (coarse units)
H = 13.0  # wall-to-wall distance (y=0 and y=13 are walls, fluid from y=1 to y=12)
nu_c = 0.2
drho = 0.01
Lx_c = 40.0  # coarse channel length

# Analytical Poiseuille: u(y) = (dp/dx) * y * (H-y) / (2*rho*nu)
# dp/dx = drho * cs^2 / Lx, cs^2 = 1/3
# For LBM on fine grid: nu_f = 2*nu_c, Lx_f = 2*Lx_c, drho same
# In coarse units: dp/dx_c = drho * (1/3) / Lx_c
dpdx = drho * (1.0/3.0) / Lx_c
y_anal = np.linspace(0, H, 500)
# Walls at y=0 and y=H, parabolic profile: u = dpdx/(2*nu) * y*(H-y)
u_anal = dpdx / (2.0 * nu_c) * y_anal * (H - y_anal)

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, constrained_layout=True)

# --- Top: Velocity profiles ---
ax1.plot(y_anal, u_anal, 'k-', linewidth=2, label='Analytical Poiseuille', zorder=5)
ax1.plot(df_uniform['y_physical'], df_uniform['ux'], 'b-o', markersize=3, linewidth=1.5, label='Uniform fine grid (reference)', alpha=0.8)
ax1.plot(df_coarse['y_physical'], df_coarse['ux'], 'r-s', markersize=5, linewidth=1.5, label='Multi-block: coarse region', alpha=0.8)
ax1.plot(df_fine['y_physical'], df_fine['ux'], 'g-^', markersize=3, linewidth=1.5, label='Multi-block: fine region', alpha=0.8)

# Mark the fine zone boundaries
zone_cy0, zone_cy1 = 4.0, 10.0
ax1.axvline(x=zone_cy0, color='gray', linestyle='--', alpha=0.5, label='Fine zone boundary')
ax1.axvline(x=zone_cy1, color='gray', linestyle='--', alpha=0.5)
ax1.axvspan(zone_cy0, zone_cy1, alpha=0.05, color='green')

ax1.set_xlabel('y (coarse lattice units)', fontsize=12)
ax1.set_ylabel('$u_x$ (lattice units)', fontsize=12)
ax1.set_title('Poiseuille Flow Validation: Uniform Fine vs Multi-Block', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, 14)

# --- Bottom: Error between multi-block fine and uniform fine ---
# Interpolate uniform fine onto fine grid y-positions for comparison
y_fine = df_fine['y_physical'].values
ux_fine = df_fine['ux'].values
ux_uniform_interp = np.interp(y_fine, df_uniform['y_physical'].values, df_uniform['ux'].values)

error = ux_fine - ux_uniform_interp
# Normalize by max velocity for relative error
ux_max = np.max(np.abs(ux_uniform_interp))
rel_error_pct = 100.0 * error / ux_max if ux_max > 0 else error

ax2.plot(y_fine, rel_error_pct, 'g-^', markersize=3, linewidth=1.5, label='Fine region - Uniform fine')
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=zone_cy0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=zone_cy1, color='gray', linestyle='--', alpha=0.5)
ax2.axvspan(zone_cy0, zone_cy1, alpha=0.05, color='green')

ax2.set_xlabel('y (coarse lattice units)', fontsize=12)
ax2.set_ylabel('Relative error (%)', fontsize=12)
ax2.set_title('Error: Multi-block fine vs Uniform fine', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-1, 22)

# Print summary statistics
print(f"\n=== Validation Summary ===")
print(f"Uniform fine grid: {len(df_uniform)} points, max ux = {df_uniform['ux'].max():.6f}")
print(f"Multi-block coarse: {len(df_coarse)} points, max ux = {df_coarse['ux'].max():.6f}")
print(f"Multi-block fine:   {len(df_fine)} points, max ux = {df_fine['ux'].max():.6f}")
print(f"Analytical max ux:  {np.max(u_anal):.6f}")
print(f"\nError (fine region vs uniform fine):")
print(f"  Max absolute error: {np.max(np.abs(error)):.6e}")
print(f"  RMS error:          {np.sqrt(np.mean(error**2)):.6e}")
print(f"  Max relative error: {np.max(np.abs(rel_error_pct)):.2f}%")
print(f"  RMS relative error: {np.sqrt(np.mean(rel_error_pct**2)):.2f}%")

# Save high-res
outdir = os.path.join(base, "results_validation")
fig.savefig(os.path.join(outdir, "poiseuille_validation.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(outdir, "poiseuille_validation.pdf"), bbox_inches='tight')

# Save low-res
fig.savefig(os.path.join(outdir, "lowres", "poiseuille_validation.png"), dpi=100, bbox_inches='tight')

print(f"\nPlots saved to {outdir}")
plt.close(fig)

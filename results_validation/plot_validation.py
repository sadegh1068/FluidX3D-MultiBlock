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
H = 63.0  # wall-to-wall distance (y=0 and y=63 are walls, fluid y=1..62)
nu_c = 0.1
fx_c = 1.0e-5  # body force (coarse lattice units)
Ny_c = 64

# Analytical Poiseuille: u(y) = fx/(2*nu) * y * (H-y), with walls at y=0 and y=H
y_anal = np.linspace(0, H, 500)
u_anal = fx_c / (2.0 * nu_c) * y_anal * (H - y_anal)

# Fine zone boundaries (coarse units)
zone_cy0, zone_cy1 = 16.0, 48.0

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11),
    gridspec_kw={'height_ratios': [3, 1, 1]}, constrained_layout=True)

# --- Top: Velocity profiles ---
ax1.plot(y_anal, u_anal, 'k-', linewidth=2, label='Analytical Poiseuille', zorder=5)
ax1.plot(df_uniform['y_physical'], df_uniform['ux'], 'b-o', markersize=2, linewidth=1.5,
         label='Uniform fine grid (reference)', alpha=0.8)
ax1.plot(df_coarse['y_physical'], df_coarse['ux'], 'r-s', markersize=5, linewidth=1.5,
         label='Multi-block: coarse region', alpha=0.8)
ax1.plot(df_fine['y_physical'], df_fine['ux'], 'g-^', markersize=3, linewidth=1.5,
         label='Multi-block: fine region', alpha=0.8)

ax1.axvline(x=zone_cy0, color='gray', linestyle='--', alpha=0.5, label='Fine zone boundary')
ax1.axvline(x=zone_cy1, color='gray', linestyle='--', alpha=0.5)
ax1.axvspan(zone_cy0, zone_cy1, alpha=0.05, color='green')

ax1.set_xlabel('y (coarse lattice units)', fontsize=12)
ax1.set_ylabel('$u_x$ (lattice units)', fontsize=12)
ax1.set_title('Poiseuille Flow Validation (Body Force): Uniform Fine vs Multi-Block', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, 64)

# --- Middle: Error multi-block fine vs uniform fine ---
y_fine = df_fine['y_physical'].values
ux_fine = df_fine['ux'].values
ux_uniform_interp = np.interp(y_fine, df_uniform['y_physical'].values, df_uniform['ux'].values)
error_mb = ux_fine - ux_uniform_interp
ux_max = np.max(np.abs(df_uniform['ux'].values))
rel_error_mb = 100.0 * error_mb / ux_max if ux_max > 0 else error_mb

ax2.plot(y_fine, rel_error_mb, 'g-^', markersize=3, linewidth=1.5, label='Multi-block fine - Uniform fine')
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=zone_cy0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=zone_cy1, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('y (coarse lattice units)', fontsize=12)
ax2.set_ylabel('Relative error (%)', fontsize=12)
ax2.set_title('Error: Multi-block fine vs Uniform fine', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-1, 64)

# --- Bottom: Error vs analytical ---
# Uniform fine vs analytical
y_uf = df_uniform['y_physical'].values
ux_uf = df_uniform['ux'].values
u_anal_at_uf = fx_c / (2.0 * nu_c) * y_uf * (H - y_uf)
u_anal_max = np.max(u_anal)
err_uf_anal = 100.0 * (ux_uf - u_anal_at_uf) / u_anal_max

# Multi-block fine vs analytical
u_anal_at_mf = fx_c / (2.0 * nu_c) * y_fine * (H - y_fine)
err_mf_anal = 100.0 * (ux_fine - u_anal_at_mf) / u_anal_max

ax3.plot(y_uf, err_uf_anal, 'b-o', markersize=2, linewidth=1.5, label='Uniform fine - Analytical')
ax3.plot(y_fine, err_mf_anal, 'g-^', markersize=3, linewidth=1.5, label='Multi-block fine - Analytical')
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=zone_cy0, color='gray', linestyle='--', alpha=0.5)
ax3.axvline(x=zone_cy1, color='gray', linestyle='--', alpha=0.5)
ax3.set_xlabel('y (coarse lattice units)', fontsize=12)
ax3.set_ylabel('Relative error (%)', fontsize=12)
ax3.set_title('Error vs Analytical Poiseuille Solution', fontsize=11)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-1, 64)

# Print summary
print(f"\n=== Validation Summary ===")
print(f"Analytical max ux:  {np.max(u_anal):.6f}")
print(f"Uniform fine grid:  {len(df_uniform)} points, max ux = {df_uniform['ux'].max():.6f}")
print(f"Multi-block coarse: {len(df_coarse)} points, max ux = {df_coarse['ux'].max():.6f}")
print(f"Multi-block fine:   {len(df_fine)} points, max ux = {df_fine['ux'].max():.6f}")

print(f"\nMulti-block fine vs Uniform fine:")
print(f"  Max relative error: {np.max(np.abs(rel_error_mb)):.2f}%")
print(f"  RMS relative error: {np.sqrt(np.mean(rel_error_mb**2)):.2f}%")

mask_interior = (y_uf > 1) & (y_uf < H-1)
print(f"\nUniform fine vs Analytical (interior only):")
print(f"  Max relative error: {np.max(np.abs(err_uf_anal[mask_interior])):.2f}%")
print(f"  RMS relative error: {np.sqrt(np.mean(err_uf_anal[mask_interior]**2)):.2f}%")

mask_mf_interior = (y_fine > zone_cy0+1) & (y_fine < zone_cy1-1)
if np.any(mask_mf_interior):
    print(f"\nMulti-block fine vs Analytical (fine zone interior):")
    print(f"  Max relative error: {np.max(np.abs(err_mf_anal[mask_mf_interior])):.2f}%")
    print(f"  RMS relative error: {np.sqrt(np.mean(err_mf_anal[mask_mf_interior]**2)):.2f}%")

# Save
outdir = os.path.join(base, "results_validation")
fig.savefig(os.path.join(outdir, "poiseuille_validation.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(outdir, "poiseuille_validation.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(outdir, "lowres", "poiseuille_validation.png"), dpi=100, bbox_inches='tight')

print(f"\nPlots saved to {outdir}")
plt.close(fig)

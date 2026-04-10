import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

base = r"D:\FluidX3D-MultiBlock"
outdir = os.path.join(base, "results_validation")
os.makedirs(os.path.join(outdir, "lowres"), exist_ok=True)

# === FIGURE 1: Mass Conservation ===
fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

try:
    df_mass_uf = pd.read_csv(os.path.join(base, "mass_uniform_fine.csv"))
    ax1a.plot(df_mass_uf['step'], df_mass_uf['mass_drift_pct'], 'b-o', markersize=4, label='Uniform fine')
except: pass

try:
    df_mass_mb = pd.read_csv(os.path.join(base, "mass_multiblock.csv"))
    ax1a.plot(df_mass_mb['step'], df_mass_mb['mass_drift_pct'], 'g-^', markersize=4, label='Multi-block (total)')
    ax1b.plot(df_mass_mb['step'], df_mass_mb['mass_coarse'], 'r-s', markersize=4, label='Coarse mass')
    ax1b.plot(df_mass_mb['step'], df_mass_mb['mass_fine'], 'g-^', markersize=4, label='Fine mass')
except: pass

ax1a.set_xlabel('Coarse timestep')
ax1a.set_ylabel('Mass drift (%)')
ax1a.set_title('Mass Conservation Test', fontsize=14, fontweight='bold')
ax1a.legend()
ax1a.grid(True, alpha=0.3)
ax1a.axhline(y=0, color='k', linewidth=0.5)

ax1b.set_xlabel('Coarse timestep')
ax1b.set_ylabel('Total mass')
ax1b.set_title('Multi-block: Coarse vs Fine Grid Mass')
ax1b.legend()
ax1b.grid(True, alpha=0.3)

fig1.savefig(os.path.join(outdir, "mass_conservation.png"), dpi=300, bbox_inches='tight')
fig1.savefig(os.path.join(outdir, "lowres", "mass_conservation.png"), dpi=100, bbox_inches='tight')
plt.close(fig1)

# === FIGURE 2: Velocity Profiles ===
df_uniform = pd.read_csv(os.path.join(base, "profile_uniform_fine.csv"))
df_coarse = pd.read_csv(os.path.join(base, "profile_multiblock_coarse.csv"))
df_fine = pd.read_csv(os.path.join(base, "profile_multiblock_fine.csv"))

H = 63.0
nu_c = 0.1
fx_c = 1.0e-5
zone_cy0, zone_cy1 = 16.0, 48.0

y_anal = np.linspace(0, H, 500)
u_anal = fx_c / (2.0 * nu_c) * y_anal * (H - y_anal)

fig2, (ax2a, ax2b, ax2c) = plt.subplots(3, 1, figsize=(10, 11),
    gridspec_kw={'height_ratios': [3, 1, 1]}, constrained_layout=True)

ax2a.plot(y_anal, u_anal, 'k-', linewidth=2, label='Analytical', zorder=5)
ax2a.plot(df_uniform['y_physical'], df_uniform['ux'], 'b-o', markersize=2, linewidth=1.5, label='Uniform fine', alpha=0.8)
ax2a.plot(df_coarse['y_physical'], df_coarse['ux'], 'r-s', markersize=4, linewidth=1.5, label='Multi-block coarse', alpha=0.8)
ax2a.plot(df_fine['y_physical'], df_fine['ux'], 'g-^', markersize=2, linewidth=1.5, label='Multi-block fine', alpha=0.8)
ax2a.axvline(x=zone_cy0, color='gray', linestyle='--', alpha=0.5, label='Fine zone boundary')
ax2a.axvline(x=zone_cy1, color='gray', linestyle='--', alpha=0.5)
ax2a.axvspan(zone_cy0, zone_cy1, alpha=0.05, color='green')
ax2a.set_xlabel('y (coarse units)'); ax2a.set_ylabel('$u_x$')
ax2a.set_title('3D Poiseuille Flow (Body Force): Uniform Fine vs Multi-Block', fontsize=14, fontweight='bold')
ax2a.legend(fontsize=9); ax2a.grid(True, alpha=0.3); ax2a.set_xlim(-1, 64)

# Error: multi-block fine vs uniform fine
y_fine = df_fine['y_physical'].values
ux_fine = df_fine['ux'].values
ux_uf_interp = np.interp(y_fine, df_uniform['y_physical'].values, df_uniform['ux'].values)
ux_max = np.max(np.abs(df_uniform['ux'].values))
if ux_max > 0:
    rel_err_mb = 100.0 * (ux_fine - ux_uf_interp) / ux_max
else:
    rel_err_mb = np.zeros_like(ux_fine)

ax2b.plot(y_fine, rel_err_mb, 'g-^', markersize=2, linewidth=1.5, label='MB fine - Uniform fine')
ax2b.axhline(y=0, color='k', linewidth=0.5)
ax2b.axvline(x=zone_cy0, color='gray', linestyle='--', alpha=0.5)
ax2b.axvline(x=zone_cy1, color='gray', linestyle='--', alpha=0.5)
ax2b.set_xlabel('y (coarse units)'); ax2b.set_ylabel('Relative error (%)')
ax2b.set_title('Error: Multi-block fine vs Uniform fine'); ax2b.legend(); ax2b.grid(True, alpha=0.3); ax2b.set_xlim(-1, 64)

# Error vs analytical
y_uf = df_uniform['y_physical'].values; ux_uf = df_uniform['ux'].values
u_anal_max = np.max(u_anal)
u_anal_at_uf = fx_c / (2.0 * nu_c) * y_uf * (H - y_uf)
u_anal_at_mf = fx_c / (2.0 * nu_c) * y_fine * (H - y_fine)
err_uf = 100.0 * (ux_uf - u_anal_at_uf) / u_anal_max if u_anal_max > 0 else np.zeros_like(ux_uf)
err_mf = 100.0 * (ux_fine - u_anal_at_mf) / u_anal_max if u_anal_max > 0 else np.zeros_like(ux_fine)

ax2c.plot(y_uf, err_uf, 'b-o', markersize=2, linewidth=1.5, label='Uniform fine - Analytical')
ax2c.plot(y_fine, err_mf, 'g-^', markersize=2, linewidth=1.5, label='MB fine - Analytical')
ax2c.axhline(y=0, color='k', linewidth=0.5)
ax2c.axvline(x=zone_cy0, color='gray', linestyle='--', alpha=0.5)
ax2c.axvline(x=zone_cy1, color='gray', linestyle='--', alpha=0.5)
ax2c.set_xlabel('y (coarse units)'); ax2c.set_ylabel('Relative error (%)')
ax2c.set_title('Error vs Analytical Poiseuille'); ax2c.legend(); ax2c.grid(True, alpha=0.3); ax2c.set_xlim(-1, 64)

fig2.savefig(os.path.join(outdir, "poiseuille_validation.png"), dpi=300, bbox_inches='tight')
fig2.savefig(os.path.join(outdir, "lowres", "poiseuille_validation.png"), dpi=100, bbox_inches='tight')
plt.close(fig2)

# === Summary ===
print(f"\n=== Validation Summary ===")
print(f"Analytical max ux: {np.max(u_anal):.6f}")
print(f"Uniform fine: {len(df_uniform)} pts, max ux = {df_uniform['ux'].max():.6f}")
print(f"MB coarse: {len(df_coarse)} pts, max ux = {df_coarse['ux'].max():.6f}")
print(f"MB fine: {len(df_fine)} pts, max ux = {df_fine['ux'].max():.6f}")

mask_int = (y_fine > zone_cy0+1) & (y_fine < zone_cy1-1)
if np.any(mask_int):
    print(f"\nMB fine vs Analytical (zone interior):")
    print(f"  Max relative error: {np.max(np.abs(err_mf[mask_int])):.2f}%")
    print(f"  RMS relative error: {np.sqrt(np.mean(err_mf[mask_int]**2)):.2f}%")

try:
    df_mass_mb = pd.read_csv(os.path.join(base, "mass_multiblock.csv"))
    print(f"\nMass conservation (multi-block):")
    print(f"  Initial: {df_mass_mb['mass_total'].iloc[0]:.2f}")
    print(f"  Final:   {df_mass_mb['mass_total'].iloc[-1]:.2f}")
    print(f"  Drift:   {df_mass_mb['mass_drift_pct'].iloc[-1]:.4f}%")
except: pass

print(f"\nPlots saved to {outdir}")

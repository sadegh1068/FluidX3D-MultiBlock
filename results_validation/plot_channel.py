import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

base = r"D:\FluidX3D-MultiBlock"
outdir = os.path.join(base, "results_validation")

df_c = pd.read_csv(os.path.join(base, "channel_profile_coarse.csv"))
df_f = pd.read_csv(os.path.join(base, "channel_profile_fine.csv"))

# Log-law reference: u+ = (1/kappa) * ln(y+) + B, kappa=0.41, B=5.2
# Viscous sublayer: u+ = y+
yp_ref = np.logspace(-0.5, 2.5, 200)
up_viscous = yp_ref  # u+ = y+ (viscous sublayer)
up_loglaw = (1.0/0.41) * np.log(yp_ref) + 5.2  # log law

# Fine zone boundary in y+: zone.cy0=2, zone.cy1=16 coarse units
# y+ = y * u_tau / nu = y * 0.05 / 0.016 = y * 3.125
# Re_tau=30: nu = 0.05*32/30 = 0.0533, y+ = y * 0.05/0.0533
nu_val = 0.05 * 32.0 / 30.0
u_tau = 0.05
zone_yp0 = 8.0 * u_tau / nu_val
zone_yp1 = 24.0 * u_tau / nu_val

fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

# Reference curves
ax.plot(yp_ref, up_viscous, 'k--', linewidth=1, label='u+ = y+ (viscous sublayer)')
ax.plot(yp_ref, up_loglaw, 'k-', linewidth=1.5, label='Log law: u+ = 2.44 ln(y+) + 5.2')

# Coarse grid (only bottom half, y < Ny/2)
mask_c = (df_c['y_plus'] > 0) & (df_c['y'] < 32) & (df_c['u_plus'] > 0)
ax.plot(df_c['y_plus'][mask_c], df_c['u_plus'][mask_c], 'r-s', markersize=5, linewidth=1.5,
        label='Multi-block coarse', alpha=0.8)

# Fine grid
mask_f = (df_f['y_plus'] > 0) & (df_f['u_plus'] > 0)
ax.plot(df_f['y_plus'][mask_f], df_f['u_plus'][mask_f], 'g-^', markersize=3, linewidth=1.5,
        label='Multi-block fine (near wall)', alpha=0.8)

# Mark fine zone boundaries
ax.axvline(x=zone_yp0, color='gray', linestyle='--', alpha=0.5, label='Fine zone boundary')
ax.axvline(x=zone_yp1, color='gray', linestyle='--', alpha=0.5)

ax.set_xscale('log')
ax.set_xlabel('y+', fontsize=14)
ax.set_ylabel('u+', fontsize=14)
ax.set_title('Turbulent Channel Flow Re_tau=100: Mean Velocity Profile', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(0.5, 200)
ax.set_ylim(0, 20)

fig.savefig(os.path.join(outdir, "channel_u_plus.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(outdir, "lowres", "channel_u_plus.png"), dpi=100, bbox_inches='tight')
plt.close(fig)

print("=== Channel Flow Summary ===")
print(f"Coarse: {len(df_c)} points, max u+ = {df_c['u_plus'].max():.2f}")
print(f"Fine: {len(df_f)} points, max u+ = {df_f['u_plus'].max():.2f}")
print(f"Fine zone: y+ = {zone_yp0:.1f} to {zone_yp1:.1f}")
print(f"Plots saved to {outdir}")

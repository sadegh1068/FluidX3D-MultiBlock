# Static Multi-Block Refinement for FluidX3D — Implementation Guide

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Mathematical Framework](#2-mathematical-framework)
3. [Architecture Design](#3-architecture-design)
4. [Memory Budget Analysis (8GB VRAM)](#4-memory-budget-analysis)
5. [Data Structures](#5-data-structures)
6. [Kernel Modifications](#6-kernel-modifications)
7. [Sub-Cycling Algorithm](#7-sub-cycling-algorithm)
8. [Coarse-Fine Coupling](#8-coarse-fine-coupling)
9. [Refinement Zone Layout for Jets](#9-refinement-zone-layout-for-jets)
10. [Step-by-Step Implementation Plan](#10-step-by-step-implementation-plan)
11. [Known Pitfalls & Solutions](#11-known-pitfalls--solutions)
12. [References](#12-references)

---

## 1. Problem Statement

**Goal:** Distribute mesh resolution non-uniformly within the same 8GB VRAM budget to get
better resolution near the jet exit (where physics matters) and coarse resolution in the far-field
(where it's wasted).

**Current situation (uniform grid):**
- 8GB VRAM with D3Q19 FP16S: ~55 bytes/cell → ~145M cells max (~525³)
- For a rectangular jet at aspect ratio 2:1, the domain needs to be ~40h long, ~20h wide, ~20h tall
- If h (slot height) = 20 cells → domain = 800×400×400 = 128M cells (fits, but 20 cells/h is too coarse for shear layer LES)
- If h = 40 cells → domain = 1600×800×800 = 1024M cells (doesn't fit — needs 56 GB VRAM)
- The shear layer needs h ≈ 40-60 cells, but far-field (r/h > 5) only needs h ≈ 10-20 cells

**With 2-level static refinement (factor 2):**
- Fine zone near jet: h = 40 cells, covering ~10h × 4h × 4h = 400×160×160 = 10.2M fine cells
- Coarse zone everywhere else: h_eff = 20 cells, covering ~40h × 20h × 20h = 800×400×400 = 128M coarse cells
- Coarse cells that overlap with fine zone are removed: ~200×80×80 = 1.3M removed
- Total: (128 - 1.3 + 10.2)M = 136.9M cells × 55 bytes = 7.5 GB — fits in 8GB!
- **Result: 40 cells/h resolution where it matters, 20 cells/h in far-field, same VRAM**

---

## 2. Mathematical Framework

### 2.1 Scaling Relations (Convective Scaling, Refinement Ratio n=2)

All quantities on the fine grid (subscript f) and coarse grid (subscript c):

```
dx_f = dx_c / 2          (fine cells are half the size)
dt_f = dt_c / 2          (fine timestep is half — CFL condition)
e_i = dx/dt = same       (lattice velocities identical on both grids)
cs = 1/sqrt(3) = same    (speed of sound identical)
u_LBM = same              (Mach number identical on both grids)
```

### 2.2 Relaxation Time Rescaling

Physical viscosity must be identical on both grids:

```
nu = cs² * (tau - 0.5) * dt
```

Setting nu_c = nu_f:

```
cs² * (tau_c - 0.5) * dt_c = cs² * (tau_f - 0.5) * dt_f
(tau_c - 0.5) * dt_c = (tau_f - 0.5) * dt_f
```

Since dt_c = 2 * dt_f:

```
(tau_c - 0.5) * 2 = (tau_f - 0.5)
tau_f = 2 * (tau_c - 0.5) + 0.5
tau_f = 2 * tau_c - 0.5
```

**General formula for refinement ratio n:**
```
tau_f = n * (tau_c - 0.5) + 0.5
tau_c = (tau_f - 0.5) / n + 0.5
```

**Examples:**
| tau_c | tau_f (n=2) | Comment |
|-------|-------------|---------|
| 0.55  | 0.60        | Both barely stable — high Re |
| 0.60  | 0.70        | Safe operating range |
| 0.80  | 1.10        | Comfortable |
| 1.00  | 1.50        | Low Re |

**CRITICAL:** Both tau_c and tau_f MUST be > 0.5. Since tau_f > tau_c always (for n>1),
if tau_c > 0.5, tau_f is automatically > 0.5. The constraint is tau_c > 0.5.

### 2.3 Reynolds Number on Each Grid

```
Re = U * L / nu
nu = cs² * (tau - 0.5) * dt
```

On the fine grid, L is measured in fine-grid cells (L_f = 2 * L_c), and dt_f = dt_c/2:
```
nu_f = (1/3) * (tau_f - 0.5) * dt_f = (1/3) * (2*tau_c - 1) * (dt_c/2) = (1/3) * (tau_c - 0.5) * dt_c = nu_c
```
Re is identical on both grids. Correct.

### 2.4 Smagorinsky SGS Model Across Levels

With Smagorinsky LES: `nu_eff = nu + nu_SGS`, where `nu_SGS = (Cs * dx)² * |S|`

On the fine grid: `nu_SGS_f = (Cs * dx_f)² * |S_f| = (Cs * dx_c/2)² * |S_f|`

Since |S| is better resolved on the fine grid, nu_SGS_f < nu_SGS_c. This is GOOD — less
SGS dissipation where you have more resolution. This is exactly what you want for the jet.

The effective tau on each grid becomes:
```
tau_eff_c = 0.5 + 3 * (nu + nu_SGS_c) / dt_c
tau_eff_f = 0.5 + 3 * (nu + nu_SGS_f) / dt_f
```

At the coarse-fine interface, the rescaling must use the LOCAL tau values (including SGS).

### 2.5 Distribution Function Rescaling (Dupuis-Chopard)

At the coarse-fine interface, distribution functions must be rescaled to account for the
different relaxation times. The non-equilibrium part of f scales with tau:

**Coarse → Fine (filling fine ghost cells):**
```
f_i^fine = f_i^eq(rho, u) + (2*tau_f - 1) / (2*tau_c - 1) * [f_i^coarse - f_i^eq(rho, u)]
```

**Fine → Coarse (restricting to coarse cells):**
```
f_i^coarse = f_i^eq(rho, u) + (2*tau_c - 1) / (2*tau_f - 1) * [f_i^fine_avg - f_i^eq(rho, u)]
```

Where f_i^fine_avg is the volume-averaged DDF from the 2³=8 fine cells that make up one
coarse cell.

With convective scaling and n=2:
```
(2*tau_f - 1) / (2*tau_c - 1) = (2*(2*tau_c - 0.5) - 1) / (2*tau_c - 1) = (4*tau_c - 2) / (2*tau_c - 1) = 2
```

**So the rescaling factor is exactly 2 (coarse→fine) and 1/2 (fine→coarse)** when using
standard convective scaling. This simplifies implementation enormously.

**WITH Smagorinsky:** The tau values include SGS viscosity, so the rescaling factor varies
per cell. Use local tau_eff values.

### 2.6 Equilibrium Distribution Function (for reference)

```
f_i^eq(rho, u) = w_i * rho * [1 + (e_i · u)/cs² + (e_i · u)²/(2*cs⁴) - u·u/(2*cs²)]
```

For D3Q19, w_i values:
- w_0 = 1/3 (rest)
- w_{1-6} = 1/18 (face neighbors)
- w_{7-18} = 1/36 (edge neighbors)

---

## 3. Architecture Design

### 3.1 Key Design Decision: Domains as Blocks

**Use FluidX3D's existing multi-domain system.** Each refinement zone becomes one or more
LBM_Domain objects. The existing halo exchange, OpenCL compilation, and memory management
are reused.

```
┌─────────────────────────────────────────────────────────┐
│                   COARSE DOMAIN                          │
│  (Dx × Dy × Dz GPU domains, dx_c, dt_c, tau_c)        │
│                                                          │
│       ┌─────────────────────────┐                        │
│       │     FINE DOMAIN         │                        │
│       │  (dx_f=dx_c/2, dt_f,   │                        │
│       │   tau_f, sub-cycled)    │                        │
│       │                         │                        │
│       │   JET EXIT HERE =====>  │                        │
│       │                         │                        │
│       └─────────────────────────┘                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Two Approaches

**Approach A: Separate LBM objects (simplest)**
- One `LBM` object for the coarse grid
- One `LBM` object for the fine grid
- A new `MultiBlockManager` class orchestrates sub-cycling and coupling
- Coarse grid has a "hole" where the fine grid lives (cells flagged as TYPE_REFINED — skip collision/streaming)
- Coupling via CPU-side buffer exchange (like existing multi-GPU communication)

**Approach B: Extended domain decomposition (more integrated)**
- Modify `LBM_Domain` to support per-domain resolution level
- Fine domains have 2× more cells per physical length
- Modify communication layer to handle resolution mismatch at domain boundaries
- Requires changes deep in the codebase

**Recommendation: Approach A.** It's minimally invasive to the existing code. The coupling
happens outside the LBM kernels, in a new orchestration layer.

### 3.3 Architecture Diagram (Approach A)

```
class MultiBlockLBM {
    LBM* coarse;           // Coarse grid — full domain
    LBM* fine;             // Fine grid — sub-domain, 2× resolution
    
    // Overlap region (in coarse-grid coordinates)
    uint fine_x0, fine_y0, fine_z0;  // Where fine grid starts in coarse coords
    uint fine_Lx, fine_Ly, fine_Lz;  // Size in coarse cells
    
    // Coupling buffers (host memory)
    float* coarse_to_fine_buffer;  // DDFs from coarse ghost layer → fine ghost cells
    float* fine_to_coarse_buffer;  // Averaged DDFs from fine interface → coarse cells
    
    // Rescaling parameters
    float tau_c, tau_f;
    float rescale_c2f;  // = (2*tau_f - 1) / (2*tau_c - 1)
    float rescale_f2c;  // = (2*tau_c - 1) / (2*tau_f - 1) = 1/rescale_c2f
    
    void do_time_step();  // Orchestrates sub-cycling
};
```

---

## 4. Memory Budget Analysis (8GB VRAM, D3Q19 FP16S)

### 4.1 Bytes per Cell

| Component | FP16S | FP32 |
|-----------|-------|------|
| fi (19 DDFs) | 38 B | 76 B |
| rho (density) | 4 B | 4 B |
| u (velocity 3D) | 12 B | 12 B |
| flags | 1 B | 1 B |
| **Total (basic)** | **55 B** | **93 B** |
| + FORCE_FIELD (F) | +12 B | +12 B |
| + transfer buffers | ~5% overhead | ~5% overhead |
| **Total (with forces)** | **~70 B** | **~110 B** |

### 4.2 Cell Budget

With 8 GB VRAM, ~7.5 GB usable (OS + driver overhead):

| Precision | Bytes/cell | Max cells | Equivalent uniform cube |
|-----------|-----------|-----------|------------------------|
| FP16S basic | 55 B | 136M | 515³ |
| FP16S + forces | 70 B | 107M | 475³ |
| FP32 basic | 93 B | 80M | 431³ |
| FP32 + forces | 110 B | 68M | 408³ |

### 4.3 Example: Rectangular Jet (AR=2:1, h=slot height)

**Uniform grid (current approach):**
- Domain: 40h × 20h × 20h
- h = 20 cells: 800 × 400 × 400 = 128M cells (55 B → 7.0 GB) — fits but h=20 is too coarse
- h = 30 cells: 1200 × 600 × 600 = 432M cells — needs 23.8 GB — DOESN'T FIT

**2-level static refinement:**
- Coarse grid (h_c = 20 cells): 40h × 20h × 20h = 800 × 400 × 400 = 128M cells
- Fine zone (h_f = 40 cells): covering 0 < x < 15h, -3h < y < 3h, -3h < z < 3h
  - Fine cells: (15×40) × (6×40) × (6×40) = 600 × 240 × 240 = 34.6M fine cells
  - Coarse cells removed (overlap): (15×20) × (6×20) × (6×20) = 300 × 120 × 120 = 4.3M removed
- Total: (128 - 4.3 + 34.6)M = 158.3M cells
- Memory: 158.3M × 55 B = 8.7 GB — slightly over budget

**Adjusted fine zone (more conservative):**
- Fine zone: 0 < x < 12h, -2.5h < y < 2.5h, -2.5h < z < 2.5h
  - Fine cells: 480 × 200 × 200 = 19.2M fine cells
  - Coarse removed: 240 × 100 × 100 = 2.4M
- Total: (128 - 2.4 + 19.2)M = 144.8M cells
- Memory: 144.8M × 55 B = 8.0 GB — FITS!
- **Result: 40 cells/h in shear layer (x < 12h), 20 cells/h in far-field**

**3-level refinement (more aggressive):**
- Level 0 (coarsest): h = 10 cells, full domain 400 × 200 × 200 = 16M cells
- Level 1 (medium): h = 20 cells, zone 0-20h × ±5h × ±5h → 400 × 200 × 200 = 16M cells
  minus overlap with coarse: 200 × 100 × 100 = 2M → 14M net
- Level 2 (finest): h = 40 cells, zone 0-10h × ±2.5h × ±2.5h → 400 × 200 × 200 = 16M cells
  minus overlap with medium: 200 × 100 × 100 = 2M → 14M net
- Total: 16 + 14 + 14 = 44M cells
- Memory: 44M × 55 B = 2.4 GB — fits with room to spare!
- **Result: 40 cells/h near nozzle, 20 cells/h in mid-field, 10 cells/h in far-field**
- **Could use h = 60 at finest level → 60 cells/h near nozzle in same budget!**

---

## 5. Data Structures

### 5.1 New MultiBlock Manager (New File: multiblock.hpp / multiblock.cpp)

```cpp
struct RefinementLevel {
    LBM* lbm;              // FluidX3D LBM object for this level
    uint level;             // 0 = coarsest, 1 = finer, etc.
    float tau;              // Relaxation time at this level
    float dx;               // Grid spacing at this level
    float dt;               // Time step at this level
    
    // Position of this level's origin in GLOBAL (coarsest) coordinates
    float3 origin_global;   // Physical position of (0,0,0) corner
    
    // Overlap region in THIS level's local coordinates
    // (where the NEXT finer level covers — skip these cells)
    uint3 skip_start;       // Start of "hole" in local coords
    uint3 skip_end;         // End of "hole" in local coords
};

struct CouplingInterface {
    uint coarse_level;      // Index into levels[]
    uint fine_level;        // Index into levels[]
    
    // Ghost layer thickness (2 cells on each side of the interface)
    // These are the cells in the fine grid that receive interpolated data from coarse
    uint ghost_layers;      // = 2
    
    // Buffers for coupling (allocated on host + device)
    // Size: interface_area × 19 DDFs × sizeof(float)
    Memory<float> c2f_buffer;  // Coarse-to-fine transfer
    Memory<float> f2c_buffer;  // Fine-to-coarse transfer
};

class MultiBlockLBM {
    vector<RefinementLevel> levels;
    vector<CouplingInterface> interfaces;
    
    void initialize(/* setup parameters */);
    void do_time_step();    // Main loop with sub-cycling
    void interpolate_c2f(CouplingInterface& iface);
    void average_f2c(CouplingInterface& iface);
};
```

### 5.2 New Cell Type Flag

In `defines.hpp`, repurpose one bit (or use existing TYPE_Y):

```cpp
#define TYPE_SKIP (0b10000000u)  // Cell is covered by finer level — skip computation
```

Cells in the coarse grid that overlap with the fine grid are flagged TYPE_SKIP.
The stream_collide kernel already skips non-fluid cells — just add this check.

### 5.3 Coupling Buffers

For the coarse-fine interface, we need to transfer DDFs for ghost cells.

Interface area = perimeter of fine zone in coarse cells × ghost_layers.
For a fine zone of 240×100×100 coarse cells:
- Interface area ≈ 2 × (240×100 + 240×100 + 100×100) = 2 × (24000 + 24000 + 10000) = 116,000 cells
- × ghost_layers(2) = 232,000 cells
- × 19 DDFs × 4 bytes = 17.6 MB per coupling direction

This is negligible compared to the main grid memory.

---

## 6. Kernel Modifications

### 6.1 Coarse Grid: Skip Refined Region

**Minimal change to stream_collide kernel:**

```opencl
// In stream_collide kernel, after loading flags:
if(flags == TYPE_SKIP) return;  // Cell covered by finer level — do nothing
```

This is a one-line addition. Cells flagged TYPE_SKIP don't collide, don't stream, don't
contribute to anything. They're placeholders that maintain the coarse grid's contiguous
memory layout (important for coalesced access on non-skipped cells).

**Alternative (more efficient):** Don't flag cells. Instead, use the existing domain
decomposition to split the coarse grid into domains that surround the fine zone. But this
is more complex to set up.

### 6.2 Fine Grid: Standard FluidX3D Kernels

The fine grid runs completely standard FluidX3D kernels. No modifications needed.
It's just a regular LBM simulation with smaller dx and dt.

The only special treatment is at its boundaries: ghost cells on the fine grid's edges
that face the coarse grid receive interpolated DDFs from the coarse grid.

### 6.3 New Kernel: Coarse-to-Fine Interpolation

This kernel runs on the fine grid's ghost cells (the 2-cell-thick layer at the fine-coarse
boundary). It reads coarse grid DDFs and writes interpolated+rescaled values to fine ghost cells.

```opencl
kernel void interpolate_coarse_to_fine(
    global fpxx* fi_fine,         // Fine grid DDFs (write target)
    global const float* c2f_buf,  // Coarse DDFs at interface (pre-extracted to buffer)
    const float tau_c,
    const float tau_f,
    const uint fine_Nx, const uint fine_Ny, const uint fine_Nz
) {
    const uint n_fine = get_global_id(0);  // Fine ghost cell index
    
    // Map fine cell to coarse cell coordinates
    // Fine cell (xf, yf, zf) corresponds to coarse cell (xf/2, yf/2, zf/2)
    // with sub-cell offset (xf%2, yf%2, zf%2) ∈ {0,1}³
    
    uint3 xyz_fine = coordinates(n_fine);  // fine grid coordinates
    
    // Coarse parent cell (integer division)
    uint3 xyz_coarse = xyz_fine / 2;
    
    // Sub-cell position within coarse cell: {-0.25, +0.25} per axis
    float3 sub = (float3)(
        (xyz_fine.x % 2 == 0) ? -0.25f : 0.25f,
        (xyz_fine.y % 2 == 0) ? -0.25f : 0.25f,
        (xyz_fine.z % 2 == 0) ? -0.25f : 0.25f
    );
    
    // ---- TRILINEAR INTERPOLATION ----
    // Interpolate rho and u from 8 surrounding coarse cells
    // (xyz_coarse and its 6 face-neighbors that bracket the sub-cell position)
    
    float rho_interp = 0.0f;
    float3 u_interp = (float3)(0.0f);
    
    // 8 corners of the interpolation cube
    for(int dz=0; dz<=1; dz++) {
        for(int dy=0; dy<=1; dy++) {
            for(int dx=0; dx<=1; dx++) {
                float wx = (dx == 0) ? (0.5f - sub.x) : (0.5f + sub.x);
                float wy = (dy == 0) ? (0.5f - sub.y) : (0.5f + sub.y);
                float wz = (dz == 0) ? (0.5f - sub.z) : (0.5f + sub.z);
                float w = wx * wy * wz;
                
                // Read coarse cell macroscopic values from buffer
                uint idx_c = coarse_index(xyz_coarse.x + dx - (sub.x < 0 ? 1 : 0),
                                          xyz_coarse.y + dy - (sub.y < 0 ? 1 : 0),
                                          xyz_coarse.z + dz - (sub.z < 0 ? 1 : 0));
                rho_interp += w * rho_coarse[idx_c];
                u_interp   += w * u_coarse[idx_c];
            }
        }
    }
    
    // ---- RESCALING ----
    // For each DDF direction i:
    float rescale = (2.0f * tau_f - 1.0f) / (2.0f * tau_c - 1.0f);
    
    for(uint i = 0; i < 19; i++) {
        // Compute equilibrium at interpolated macroscopic values
        float feq = compute_feq(i, rho_interp, u_interp);
        
        // Interpolate non-equilibrium part from coarse (same trilinear weights)
        float fneq_coarse_interp = 0.0f;
        for(int dz=0; dz<=1; dz++) {
            for(int dy=0; dy<=1; dy++) {
                for(int dx=0; dx<=1; dx++) {
                    float w = wx * wy * wz;  // same weights as above
                    float f_c = read_coarse_ddf(idx_c, i);  // from buffer
                    float feq_c = compute_feq(i, rho_c, u_c);
                    fneq_coarse_interp += w * (f_c - feq_c);
                }
            }
        }
        
        // Rescaled fine DDF
        float f_fine = feq + rescale * fneq_coarse_interp;
        
        // Write to fine grid ghost cell
        store(fi_fine, index_f(n_fine, i), f_fine);
    }
}
```

**Note on interpolation order:** Lagrava et al. (2012) proved that **cubic interpolation is
needed for full 2nd-order accuracy**. However, AGAL found cubic interpolation to be
**unstable** in practice and uses linear. PowerFLOW uses linear. For a first implementation,
**use trilinear interpolation** — it works and is much simpler. Upgrade to cubic later if needed.

### 6.4 New Kernel: Fine-to-Coarse Averaging (Restriction)

This kernel runs on coarse cells at the fine-coarse interface. It reads averaged fine-grid
DDFs and writes rescaled values to the coarse grid.

```opencl
kernel void average_fine_to_coarse(
    global fpxx* fi_coarse,       // Coarse grid DDFs (write target)
    global const float* f2c_buf,  // Fine DDFs averaged to coarse cell size
    const float tau_c,
    const float tau_f,
    const uint n_interface_cells  // Number of coarse cells at interface
) {
    const uint idx = get_global_id(0);  // Interface coarse cell index
    if(idx >= n_interface_cells) return;
    
    const uint n_coarse = interface_cell_map[idx];  // Map to coarse grid index
    
    float rescale_inv = (2.0f * tau_c - 1.0f) / (2.0f * tau_f - 1.0f);
    
    for(uint i = 0; i < 19; i++) {
        // Read volume-averaged fine DDFs (average of 8 fine children)
        float f_fine_avg = f2c_buf[idx * 19 + i];
        
        // Compute equilibrium from averaged macroscopic quantities
        float rho_avg = /* sum of f_fine_avg over all i */;
        float3 u_avg = /* momentum sum / rho_avg */;
        float feq = compute_feq(i, rho_avg, u_avg);
        
        // Rescale non-equilibrium part
        float f_coarse = feq + rescale_inv * (f_fine_avg - feq);
        
        // Write to coarse grid
        store(fi_coarse, index_f(n_coarse, i), f_coarse);
    }
}
```

### 6.5 Extract/Insert Kernels for Coupling

Reuse FluidX3D's existing `transfer_extract` / `transfer_insert` pattern:

```opencl
// Extract coarse DDFs at interface region → buffer (runs on coarse grid GPU)
kernel void extract_coarse_interface(
    global const fpxx* fi_coarse,
    global float* c2f_buffer,
    global const uint* interface_coarse_indices,  // List of coarse cells at interface
    const uint n_cells
) {
    const uint idx = get_global_id(0);
    if(idx >= n_cells) return;
    const uint n = interface_coarse_indices[idx];
    
    // Also extract rho and u for interpolation
    float fhn[19];
    load_f(n, fhn, fi_coarse, j, t);
    
    for(uint i = 0; i < 19; i++) {
        c2f_buffer[idx * 19 + i] = fhn[i];
    }
}
```

---

## 7. Sub-Cycling Algorithm

### 7.1 Two-Level Sub-Cycling (n=2)

The fine grid takes 2 time steps for every 1 coarse step. The correct ordering
(following Lagrava et al. 2012, "explosion" algorithm):

```
ONE COARSE TIME STEP (t_c → t_c + dt_c):
═══════════════════════════════════════════

Step 1: COARSE collision + streaming
    coarse.do_time_step()
    // Now coarse is at t_c + dt_c

Step 2: Extract coarse data at interface
    extract_coarse_interface()
    // Get coarse DDFs + rho + u at interface cells

Step 3: TEMPORAL INTERPOLATION for fine ghost cells at t_c + dt_f = t_c + dt_c/2
    // The fine grid needs BCs at t_c + dt_f, but coarse only has data at t_c and t_c + dt_c
    // Linear temporal interpolation:
    // f_ghost(t_c + dt_f) = 0.5 * f_coarse(t_c) + 0.5 * f_coarse(t_c + dt_c)
    // This requires storing f_coarse(t_c) from the PREVIOUS coarse step
    temporal_interpolate(f_coarse_old, f_coarse_new, 0.5)
    
Step 4: Interpolate + rescale → fine ghost cells (for fine sub-step 1)
    interpolate_coarse_to_fine()

Step 5: FINE collision + streaming (sub-step 1)
    fine.do_time_step()
    // Now fine is at t_c + dt_f

Step 6: Interpolate + rescale → fine ghost cells (for fine sub-step 2)
    // Use f_coarse(t_c + dt_c) directly (no temporal interpolation needed — coincident time)
    interpolate_coarse_to_fine()

Step 7: FINE collision + streaming (sub-step 2)
    fine.do_time_step()
    // Now fine is at t_c + dt_c (synchronized!)

Step 8: Average fine → coarse at interface
    extract_fine_interface()
    average_and_rescale()
    insert_coarse_interface()
    // Coarse cells at interface now have restricted fine data

Step 9: Store f_coarse_old = f_coarse_new for next temporal interpolation
```

### 7.2 Three-Level Sub-Cycling

For 3 levels (L0 coarsest, L1 medium, L2 finest), the recursion is:

```
Advance L0 (1 step):
    L0.stream_collide()
    
    Advance L1 (2 steps):
        interpolate L0 → L1 (temporal interp at t + dt_1)
        L1.stream_collide()  // sub-step 1
        
        Advance L2 (2 steps):
            interpolate L1 → L2 (temporal interp at t + dt_2)
            L2.stream_collide()  // sub-step 1
            interpolate L1 → L2 (at synchronized time)
            L2.stream_collide()  // sub-step 2
            average L2 → L1
        
        interpolate L0 → L1 (at synchronized time)
        L1.stream_collide()  // sub-step 2
        
        Advance L2 (2 steps):
            interpolate L1 → L2 (temporal interp)
            L2.stream_collide()  // sub-step 3
            interpolate L1 → L2 (at synchronized time)
            L2.stream_collide()  // sub-step 4
            average L2 → L1
        
        average L1 → L0

Total per coarse step: L0=1 step, L1=2 steps, L2=4 steps
```

### 7.3 Simplified Algorithm (No Temporal Interpolation)

**If you want to avoid temporal interpolation** (simpler implementation), use this ordering
from AGAL:

```
ONE COARSE TIME STEP:

1. interpolate_c2f()    // Use current coarse data for fine ghost cells
2. fine.stream_collide()  // Fine sub-step 1
3. average_f2c()        // Restrict fine → coarse
4. coarse.stream_collide()  // Coarse step
5. fine.stream_collide()  // Fine sub-step 2 (using coarse data that's now at t+dt_c)
6. average_f2c()        // Restrict fine → coarse again
```

This avoids storing f_coarse_old and temporal interpolation, at the cost of slightly
reduced temporal accuracy. For LES of turbulent jets, this is probably fine.

---

## 8. Coarse-Fine Coupling — Detailed Algorithm

### 8.1 Interface Cell Classification

At the fine-coarse boundary, classify cells as:

**On the fine grid:**
- **Ghost cells** (2 layers): Receive interpolated data from coarse grid. Run streaming but NOT collision. They exist to provide neighbor data for the inner fine cells' streaming.
- **Interior cells**: Normal LBM cells. No special treatment.

**On the coarse grid:**
- **Interface cells** (1 layer): Receive restricted data from fine grid after each coarse step. Their DDFs are overwritten by the fine-to-coarse average.
- **Skip cells**: Inside the fine zone — flagged TYPE_SKIP, not computed.

```
COARSE GRID:

  normal | normal | INTERFACE | SKIP | SKIP | SKIP | INTERFACE | normal | normal
                        ↕                                  ↕
FINE GRID:        ghost|ghost| interior ... interior |ghost|ghost
```

### 8.2 Spatial Interpolation Details (Trilinear, Coarse → Fine)

Each fine ghost cell at position (x_f, y_f, z_f) maps to coarse coordinates:

```
x_c = x_f / 2.0    (in coarse grid units, can be fractional)
y_c = y_f / 2.0
z_c = z_f / 2.0
```

The 8 coarse cells surrounding this point contribute with trilinear weights:

```
i0 = floor(x_c - 0.5), i1 = i0 + 1
j0 = floor(y_c - 0.5), j1 = j0 + 1
k0 = floor(z_c - 0.5), k1 = k0 + 1

dx = x_c - (i0 + 0.5)   // ∈ [0, 1)
dy = y_c - (j0 + 0.5)
dz = z_c - (k0 + 0.5)

w[0] = (1-dx)*(1-dy)*(1-dz)   // weight for cell (i0,j0,k0)
w[1] = (  dx)*(1-dy)*(1-dz)   // weight for cell (i1,j0,k0)
w[2] = (1-dx)*(  dy)*(1-dz)   // weight for cell (i0,j1,k0)
... etc (8 weights summing to 1.0)
```

### 8.3 Volume Averaging (Fine → Coarse)

Each coarse cell at (i_c, j_c, k_c) contains 2³ = 8 fine cells:

```
Fine cells: (2*i_c + a, 2*j_c + b, 2*k_c + c) for a,b,c ∈ {0, 1}

f_coarse_avg[q] = (1/8) * Σ_{a,b,c} f_fine[q](2*i_c+a, 2*j_c+b, 2*k_c+c)
rho_avg = (1/8) * Σ rho_fine
u_avg = (1/8) * Σ (rho_fine * u_fine) / rho_avg
```

Then apply rescaling:
```
f_coarse[q] = f_eq(q, rho_avg, u_avg) + rescale_f2c * (f_coarse_avg[q] - f_eq(q, rho_avg, u_avg))
```

### 8.4 Esoteric-Pull Compatibility

**IMPORTANT:** FluidX3D uses Esoteric-Pull AA streaming, where the storage location of
DDFs alternates based on even/odd timestep. The coupling kernels must account for this.

When extracting DDFs from the coarse grid, you must use the correct load_f() pattern
(accounting for t%2). When inserting into the fine grid's ghost cells, you must use the
correct store_f() pattern.

**Simplification:** Extract DDFs to a temporary buffer in PHYSICAL (not esoteric) order.
The extract kernel uses load_f() to read the actual physical DDFs regardless of timestep
parity. The insert kernel uses store_f() to write them back in the correct esoteric order.

---

## 9. Refinement Zone Layout for Rectangular Jets

### 9.1 Physics-Informed Zone Placement

For a rectangular jet with slot height h, aspect ratio AR:

```
CRITICAL RULE: Never place a coarse-fine interface inside a shear layer or
region of large velocity gradients. Place interfaces in quiescent regions.
```

**Recommended 2-level layout:**

```
Fine zone (h_fine = 2 * h_coarse):
  x: -2h to 15h    (captures potential core + transition region)
  y: -3h to 3h     (captures shear layer + near-field entrainment)
  z: -3h to 3h     (same in spanwise direction)

Coarse zone (full domain):
  x: -5h to 40h
  y: -12h to 12h
  z: -12h to 12h
```

**Why these dimensions:**
- Potential core ends at x ≈ 4-6h for slot jets
- Shear layer half-width at x = 15h is about y ≈ 2h — safely inside the fine zone
- The coarse-fine interface at y = ±3h is in the irrotational far-field where gradients are small
- The x = 15h interface is past the region of intense mixing but before the self-similar region

### 9.2 Recommended 3-Level Layout (More Efficient)

```
Level 2 (finest, h = 40 cells):
  x: -1h to 8h, y: ±2h, z: ±2h
  → Captures nozzle exit + initial shear layer + potential core

Level 1 (medium, h = 20 cells):
  x: -3h to 20h, y: ±5h, z: ±5h
  → Captures transition region + developing jet

Level 0 (coarsest, h = 10 cells):
  x: -5h to 40h, y: ±12h, z: ±12h
  → Full domain with sponge layers at boundaries
```

### 9.3 Cell Count Estimate (3-Level)

| Level | Zone (cells) | Cells | Memory (FP16S) |
|-------|-------------|-------|----------------|
| L0 (h=10) | 400×240×240 minus L1 overlap | ~21.8M | 1.2 GB |
| L1 (h=20) | 460×200×200 minus L2 overlap | ~17.0M | 0.9 GB |
| L2 (h=40) | 360×160×160 | 9.2M | 0.5 GB |
| **Total** | | **48.0M** | **2.6 GB** |

**This fits in 8 GB with room to spare!** You could increase h to 60 at the finest level
(240×240 = 57.6M fine cells → ~4.8 GB total) and still fit.

Compare: uniform grid at h=40 would need (40h × 20h × 20h) = 1600×800×800 = 1024M cells
= 56 GB. Multi-block gets the same near-field resolution in **2.6 GB**.

---

## 10. Step-by-Step Implementation Plan

### Phase 1: Proof of Concept (2D, ~2-3 weeks)

**Goal:** Get 2-level static refinement working in 2D with a simple test case (lid-driven cavity).

1. **Create `multiblock.hpp` / `multiblock.cpp`**
   - MultiBlockLBM class with two LBM objects (coarse + fine)
   - Coupling buffers allocated on host
   - Sub-cycling loop (simplified, no temporal interpolation)

2. **Add TYPE_SKIP flag to defines.hpp**
   - Add `#define TYPE_SKIP (0x80u)` or similar
   - Modify stream_collide kernel: `if(flags & TYPE_SKIP) return;`

3. **Write extract/insert kernels**
   - Extract coarse interface DDFs → host buffer
   - Insert interpolated DDFs → fine ghost cells
   - Extract fine interface DDFs → host buffer
   - Insert averaged DDFs → coarse interface cells

4. **Write interpolation kernel (bilinear for 2D, trilinear for 3D)**
   - Input: coarse DDFs at interface
   - Output: fine ghost cell DDFs
   - Include Dupuis-Chopard rescaling

5. **Write averaging kernel**
   - Input: fine DDFs at interface
   - Output: coarse cell DDFs
   - Include inverse rescaling

6. **Test: 2D lid-driven cavity at Re=1000**
   - Fine zone in top-left corner (where primary vortex is)
   - Compare velocity profiles with uniform fine-grid reference
   - Check mass conservation

### Phase 2: 3D Implementation (~2-3 weeks)

7. **Extend to 3D**
   - Trilinear interpolation
   - 8-child averaging
   - 3D interface cell detection

8. **Test: 3D flow past sphere at Re=300**
   - Fine zone around sphere
   - Compare Cd with uniform grid

### Phase 3: Jet Application (~2 weeks)

9. **Set up rectangular jet with multi-block**
   - Fine zone near nozzle exit
   - Coarse far-field
   - Sponge layers on coarse grid boundaries

10. **Validate against your existing uniform-grid jet results**
    - Compare velocity decay, spreading rate, turbulence statistics
    - Verify shear layer structure in fine zone

### Phase 4: Optimization (~1-2 weeks)

11. **Overlap computation and communication**
    - While fine grid computes, extract coarse data
    - Use OpenCL event-based synchronization

12. **Add 3-level support**
    - Recursive sub-cycling
    - Intermediate coupling buffers

13. **Performance profiling**
    - Measure coupling overhead
    - Optimize buffer sizes and transfer patterns

### Phase 5: Optional Enhancements

14. **Temporal interpolation** (improves acoustic behavior)
15. **Cubic spatial interpolation** (improves accuracy — Lagrava recommends)
16. **Smagorinsky SGS-aware rescaling** (use local tau_eff)
17. **Mass correction** (monitor and fix any drift)

---

## 11. Known Pitfalls & Solutions

### 11.1 Instability at Interface

**Symptom:** Simulation blows up at the coarse-fine boundary.

**Causes and fixes:**
- Interface inside shear layer → MOVE interface to quiescent region (r > 3h for jets)
- Missing rescaling → ALWAYS apply Dupuis-Chopard rescaling
- Wrong τ relationship → Verify: tau_f = 2*(tau_c - 0.5) + 0.5
- tau_c too close to 0.5 → Increase Mach number or reduce Re per grid level

### 11.2 Mass Loss/Gain

**Symptom:** Density slowly drifts over time.

**Fix:** Monitor total mass each step. Apply correction:
```
delta_mass = mass_initial - mass_current
// Distribute delta_mass uniformly across all cells
rho_correction = delta_mass / N_total_cells
```

### 11.3 Acoustic Reflections at Interface

**Symptom:** Spurious pressure waves at the coarse-fine boundary.

**Fix:** Temporal interpolation (Step 3 in Section 7.1) smooths the time-mismatch
and reduces reflections. Also ensure Mach number is low (<0.1).

### 11.4 Esoteric-Pull Timing Issues

**Symptom:** Alternating-timestep artifacts at the interface.

**Fix:** When extracting/inserting DDFs, always use the physical (not storage) DDF order.
The extract kernel must call load_f() with the correct timestep parity. The insert kernel
must call store_f() with the correct parity for the TARGET grid.

**Alternative:** Use a temporary flat buffer in natural DDF order as an intermediary.

### 11.5 SGS Model Discontinuity

**Symptom:** Turbulence statistics show a jump at the interface.

**Fix:** When computing SGS viscosity near the interface, use the strain rate from the
fine grid (interpolated to the interface) rather than from the coarse grid.

### 11.6 Single-GPU Memory: Two LBM Objects

**Issue:** If both coarse and fine grids are on the SAME GPU (8GB card), both LBM objects
share the same OpenCL context and compete for VRAM.

**Fix:** FluidX3D's multi-domain already handles this — multiple LBM_Domain objects can
share one GPU. Just ensure total memory fits. Use the budget in Section 4.

---

## 12. References

### Essential (read these)
1. **Dupuis & Chopard (2003)** — "Theory and finite-size effects for the lattice Boltzmann
   grid refinement." Phys. Rev. E 67, 066707. [The rescaling formula]
2. **Lagrava et al. (2012)** — "Advances in multi-domain lattice Boltzmann grid refinement."
   J. Comput. Phys. 231, 4808-4822. [The definitive reference — sub-cycling, interpolation order]
3. **Rohde et al. (2006)** — "A generic, mass conservative local grid refinement technique."
   Int. J. Numer. Meth. Fluids 51, 439-468. [Mass-conservative volumetric approach]

### Background
4. **Filippova & Hänel (1998)** — "Grid refinement for lattice-BGK models."
   J. Comput. Phys. 147, 219-228. [Original method — superseded by Dupuis-Chopard]
5. **Jaber (2025)** — "GPU-Native Adaptive Mesh Refinement with Application to LBM."
   Comput. Phys. Commun. 311, 109543. [AGAL — GPU implementation reference]
6. **Schornbaum & Rüde (2016)** — "Massively parallel algorithms for the LBM on nonuniform grids."
   SIAM J. Sci. Comput. 38(2), C96-C126. [waLBerla — scalability reference]

### Application
7. **Casalino & Lele (2014)** — "Lattice-Boltzmann simulation of coaxial jet noise."
   CTR Stanford. [PowerFLOW VRR for jets — practical validation]

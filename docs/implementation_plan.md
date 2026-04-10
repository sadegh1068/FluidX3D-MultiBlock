# Static Multi-Block Mesh Refinement for FluidX3D — Implementation Plan

## Context

The user runs rectangular jet LES simulations on an 8GB GPU. With uniform grids, the VRAM limits resolution to ~20 cells/h (slot height), but the shear layer needs ~40-60 cells/h. The far-field wastes most cells. Static multi-block refinement (2:1 ratio, fixed zones) solves this: 40 cells/h near the nozzle + 20 cells/h in far-field fits in the same 8GB budget (~145M cells vs ~1024M uniform).

**Codebase:** `D:\FluidX3D-MultiBlock` (fresh clone from GitHub)

## Approach: Two LBM Objects + Coupling Layer

Two separate `LBM` instances share one GPU:
- **Coarse LBM**: Full domain at base resolution. Interior of fine zone flagged `TYPE_Y` (skip computation).
- **Fine LBM**: Sub-domain at 2× resolution. Outermost ring flagged `TYPE_E` (equilibrium boundary fed by coarse data).

A new `MultiBlockLBM` class orchestrates sub-cycling and coupling.

### Why TYPE_E for Coupling (Not DDF-Level Rescaling)

FluidX3D's `stream_collide` kernel already has full TYPE_E support:
- Line 1496-1503: TYPE_E cells read `rho/u` from stored arrays (not from DDFs)
- Line 1614: TYPE_E cells set `fhn[i] = feq[i]` (override DDFs with equilibrium)
- Line 1578: TYPE_E cells' `rho/u` are NOT overwritten by the kernel

This means: **writing `rho/u` to TYPE_E cells is sufficient** — the kernel automatically imposes the correct equilibrium DDFs. No DDF-level Dupuis-Chopard rescaling is needed for v1. This is first-order accurate at the interface, acceptable when the interface is placed in quiescent flow (r > 3h from the jet).

### Cell Classification

```
COARSE GRID:
  fluid | fluid | TYPE_E (ring) | TYPE_Y (skip) | ... | TYPE_Y | TYPE_E (ring) | fluid | fluid
                      ↕ coupling                                      ↕ coupling
FINE GRID:      TYPE_E (ghost) | fluid | fluid | ... | fluid | fluid | TYPE_E (ghost)
```

- **Coarse TYPE_Y cells** (zone interior, excluding 1-cell border): Skip computation entirely. Saves GPU time.
- **Coarse TYPE_E ring** (1-cell border of zone): Receives volume-averaged `rho/u` from fine grid.
- **Fine TYPE_E ghost ring** (outermost 1 cell on all 6 faces): Receives trilinearly interpolated `rho/u` from coarse grid.
- **Fine fluid cells**: Normal LBM computation.

### Selective Device Transfer (Not Full Array Roundtrip)

Full array PCIe roundtrips (~640 MB for 20M cells) would dominate runtime. Instead:
- **Coupling index lists** (~100K indices) uploaded to device once at init
- **Small coupling buffers** (~1.6 MB) for gather/scatter
- **New OpenCL kernels** in each LBM_Domain gather/scatter `rho/u` at coupling indices only
- PCIe transfer of ~1.6 MB takes ~0.1 ms (vs ~50 ms for full arrays)

---

## Files to Modify

### 1. `src/defines.hpp` — Enable extensions + add MULTIBLOCK flag

```
Line 16: Comment out BENCHMARK (#define BENCHMARK → //#define BENCHMARK)
Line 20: Uncomment EQUILIBRIUM_BOUNDARIES
Line 24: Uncomment SUBGRID
Add after line 24: #define MULTIBLOCK
```

### 2. `src/kernel.cpp` — Add TYPE_Y skip in stream_collide (~line 1480)

After the existing early-return:
```cpp
if(flagsn_bo==TYPE_S||flagsn_su==TYPE_G) return;
```
Add:
```cpp
if(flagsn&TYPE_Y) return; // MULTIBLOCK: skip coarse cells covered by fine grid
```

This is inside `#ifdef MULTIBLOCK` guard. One line.

### 3. `src/lbm.hpp` — Add friend declaration + coupling support

In the LBM class (before `private:` at line ~210), add:
```cpp
friend class MultiBlockLBM;
```

Add public coupling methods:
```cpp
// Coupling support for multi-block refinement
void setup_coupling(const vector<uint>& cell_indices);  // upload index list to device
void coupling_gather(float* host_buf);                    // device rho/u → small host buffer
void coupling_scatter(const float* host_buf);             // small host buffer → device rho/u
```

### 4. `src/lbm.cpp` — Implement coupling methods

Add `setup_coupling()`, `coupling_gather()`, `coupling_scatter()`:
- `setup_coupling()`: Allocates `Memory<uint>` for indices and `Memory<float>` for coupling buffer on each relevant domain. Compiles two new OpenCL kernels (`coupling_gather_kernel`, `coupling_scatter_kernel`) in each domain's context.
- `coupling_gather()`: Enqueues gather kernel → reads coupling buffer from device → copies to host_buf.
- `coupling_scatter()`: Writes host_buf to coupling buffer on device → enqueues scatter kernel.

New OpenCL kernel code (added to `opencl_c_container()` string):
```opencl
kernel void coupling_gather_kernel(
    global const float* rho, global const float* u,
    global const uint* indices, global float* buf, const uint N_coupling
) {
    const uint i = get_global_id(0);
    if(i >= N_coupling) return;
    const uint n = indices[i];
    buf[i*4+0] = rho[n];
    buf[i*4+1] = u[n];
    buf[i*4+2] = u[def_N+(ulong)n];
    buf[i*4+3] = u[2ul*def_N+(ulong)n];
}

kernel void coupling_scatter_kernel(
    global float* rho, global float* u,
    global const uint* indices, global const float* buf, const uint N_coupling
) {
    const uint i = get_global_id(0);
    if(i >= N_coupling) return;
    const uint n = indices[i];
    rho[n]                = buf[i*4+0];
    u[n]                  = buf[i*4+1];
    u[def_N+(ulong)n]     = buf[i*4+2];
    u[2ul*def_N+(ulong)n] = buf[i*4+3];
}
```

### 5. `src/setup.cpp` — New main_setup() for multi-block jet

Replace BENCHMARK setup with MultiBlockLBM-based jet setup.

### 6. NEW: `src/multiblock.hpp` — MultiBlockLBM class

```cpp
#pragma once
#include "lbm.hpp"

struct RefinementZone {
    uint cx0, cy0, cz0;  // start in coarse coords (must be even)
    uint cx1, cy1, cz1;  // end in coarse coords (must be even)
};

class MultiBlockLBM {
public:
    MultiBlockLBM(
        uint cNx, uint cNy, uint cNz,
        float nu,
        const RefinementZone& zone,
        float fx=0.f, float fy=0.f, float fz=0.f
    );
    ~MultiBlockLBM();

    LBM* coarse() { return lbm_c; }
    LBM* fine()   { return lbm_f; }

    void initialize();
    void run(ulong coarse_steps);

private:
    LBM* lbm_c;  // coarse grid (full domain)
    LBM* lbm_f;  // fine grid (sub-domain, 2x resolution)

    RefinementZone zone;
    float nu_c, nu_f;
    float tau_c, tau_f;

    // Coupling cell lists (host-side, built once at init)
    vector<uint> coarse_interface_indices;  // TYPE_E ring on coarse grid
    vector<uint> fine_ghost_indices;        // TYPE_E ring on fine grid
    vector<uint> coarse_extract_indices;    // 2-cell shell on coarse for interpolation source

    // Host coupling buffers
    vector<float> buf_c2f;  // coarse extract data (rho,ux,uy,uz per cell)
    vector<float> buf_f2c;  // fine extract data

    void build_index_lists();
    void flag_coarse_zone();       // set TYPE_Y interior + TYPE_E ring
    void flag_fine_boundaries();   // set TYPE_E on fine grid outer ring

    void interpolate_c2f();  // trilinear interp: coarse → fine ghost rho/u
    void average_f2c();      // volume average: fine → coarse interface rho/u
};
```

### 7. NEW: `src/multiblock.cpp` — Implementation

Key methods:

**Constructor:**
- Create `lbm_c = new LBM(cNx, cNy, cNz, 1, 1, 1, nu, ...)` for coarse grid
- Fine grid size: `fNx = (cx1-cx0)*2, fNy = (cy1-cy0)*2, fNz = (cz1-cz0)*2`
- Create `lbm_f = new LBM(fNx, fNy, fNz, 1, 1, 1, nu, ...)` — SAME nu (physics match)
- Compute `tau_c` and `tau_f` from nu and dx

**build_index_lists():**
- Walk the zone boundary in coarse coords → build `coarse_interface_indices` (TYPE_E ring)
- Walk the fine grid boundary (x=0, x=fNx-1, y=0, etc.) → build `fine_ghost_indices`
- Walk the 2-cell shell around zone in coarse coords → build `coarse_extract_indices` (for interpolation stencil)

**flag_coarse_zone():**
- `parallel_for` over coarse grid: cells strictly inside zone → `TYPE_Y`
- Cells on zone boundary (1-cell ring) → `TYPE_E`

**flag_fine_boundaries():**
- `parallel_for` over fine grid: outermost 1-cell ring on all 6 faces → `TYPE_E`

**initialize():**
- Call `flag_coarse_zone()` and `flag_fine_boundaries()`
- Call `build_index_lists()`
- Set up coupling on both LBMs: `lbm_c->setup_coupling(coarse_extract_indices)` etc.
- Do initial coupling: set fine ghost rho/u from coarse initial conditions
- Call `lbm_c->initialize()` and `lbm_f->initialize()` via `run(0)` trick or friend access

**run(coarse_steps):**
```
for step = 0 to coarse_steps-1:
    // 1. Coarse step
    lbm_c->do_time_step()

    // 2. C→F coupling
    lbm_c->coupling_gather(buf_c2f.data())   // extract coarse rho/u at interface shell
    interpolate_c2f()                          // trilinear interp → buf for fine ghost cells
    lbm_f->coupling_scatter(buf_f2c_interp)  // write to fine ghost TYPE_E cells

    // 3. Fine sub-step 1
    lbm_f->do_time_step()

    // 4. Fine sub-step 2 (reuse same coupling data — simplified, no temporal interp in v1)
    lbm_f->do_time_step()

    // 5. F→C coupling
    lbm_f->coupling_gather(buf_f2c.data())   // extract fine rho/u at interface
    average_f2c()                              // volume average 8 fine → 1 coarse
    lbm_c->coupling_scatter(buf_averaged)    // write to coarse TYPE_E ring
```

**interpolate_c2f():**
- For each fine ghost cell, compute coarse fractional coordinates
- Trilinear interpolation from 8 surrounding coarse cells
- Write interpolated `rho/u` to output buffer

**average_f2c():**
- For each coarse interface cell, identify 8 fine children
- Average `rho = mean(rho_fine)`, `u = mean(u_fine)`
- Write to output buffer

---

## Sub-Cycling Timing

```
Coarse:  |-------- dt_c --------|-------- dt_c --------|
Fine:    |--- dt_f ---|--- dt_f ---|--- dt_f ---|--- dt_f ---|
         ^            ^            ^            ^
         C→F         (reuse)      C→F         (reuse)
                      ^                         ^
                      F→C                       F→C
```

v1 simplification: No temporal interpolation at mid-step. The C→F coupling uses coarse data from the START of the coarse step for both fine sub-steps. This introduces O(dt) temporal error but is simple and stable.

---

## Physics Parameters

For the jet simulation with slot height h:
```
nu_LBM = u_jet * h_cells / Re
tau_c = 0.5 + 3 * nu_LBM        (coarse grid)
tau_f = 2 * (tau_c - 0.5) + 0.5  (fine grid)
```

Both grids use the SAME `nu` in the LBM constructor — FluidX3D computes tau from nu and dx=1. Since the fine grid also uses dx=1 (it just has more cells), nu is the same but the PHYSICAL viscosity is matched because the fine grid represents half the physical length per cell.

**Wait — critical correction:** In FluidX3D, `nu` in the constructor is in LATTICE units (nu_LBM = (tau-0.5)/3). Both grids use dx_LBM = 1. To get the same PHYSICAL viscosity, we need:
- `nu_f_LBM = 2 * nu_c_LBM` (because dt_f = dt_c/2 and nu_phys = nu_LBM * dx²/dt)
- Actually: `nu_phys = nu_LBM * dx²/dt`. With dx_f = dx_c/2 and dt_f = dt_c/2:
  `nu_phys = nu_c_LBM * dx_c²/dt_c = nu_f_LBM * (dx_c/2)²/(dt_c/2) = nu_f_LBM * dx_c²/(2*dt_c)`
  So: `nu_f_LBM = 2 * nu_c_LBM`

This means we create the fine LBM with `nu_f = 2 * nu_c`.

---

## Build System

Add to makefile `OBJECTS`:
```
multiblock.o: src/multiblock.cpp src/multiblock.hpp
```

---

## Phased Implementation Order

### Phase 1: Infrastructure (files + compilation)
1. Create `multiblock.hpp` and `multiblock.cpp` (skeleton)
2. Add `#define MULTIBLOCK` and enable extensions in `defines.hpp`
3. Add TYPE_Y skip to `kernel.cpp` (1 line)
4. Add `friend class MultiBlockLBM` to `lbm.hpp`
5. Update build system
6. Verify it compiles

### Phase 2: Coupling Kernels
7. Add `coupling_gather_kernel` and `coupling_scatter_kernel` to `kernel.cpp`
8. Implement `setup_coupling()`, `coupling_gather()`, `coupling_scatter()` in `lbm.cpp`
9. Test: create coupling, gather, modify, scatter, verify values

### Phase 3: MultiBlockLBM Core
10. Implement constructor (two LBM objects, zone validation)
11. Implement `build_index_lists()` 
12. Implement `flag_coarse_zone()` and `flag_fine_boundaries()`
13. Implement `initialize()`
14. Test: both grids initialize correctly, TYPE_Y/TYPE_E flags are set

### Phase 4: Coupling Logic
15. Implement `interpolate_c2f()` (trilinear on host)
16. Implement `average_f2c()` (volume average on host)
17. Implement `run()` with sub-cycling
18. Test: run a few steps, check no crashes, check rho/u continuity at interface

### Phase 5: Validation
19. Poiseuille flow through multi-block domain — compare with analytic solution
20. Monitor mass conservation over 10K steps
21. Check velocity continuity at interface

### Phase 6: Jet Application
22. Set up rectangular jet with multi-block zones
23. Compare with existing uniform-grid results

---

## Verification

1. **Compilation test**: `make` succeeds with no errors
2. **Unit test**: Create 2-level grid, verify TYPE_Y/TYPE_E flags are correct
3. **Interpolation test**: Linear velocity field → C2F interpolation should be exact
4. **Mass conservation**: `sum(rho)` stable to < 0.1% over 10K steps
5. **Poiseuille flow**: Parabolic profile matches analytical within 2% across the interface
6. **Performance**: Coupling overhead < 5% of total step time (verify with timing)

---

## Critical Files Summary

| File | Action | Lines Changed |
|------|--------|---------------|
| `src/defines.hpp` | Modify: enable extensions, add MULTIBLOCK | ~5 lines |
| `src/kernel.cpp` | Modify: add TYPE_Y skip + coupling kernels | ~25 lines |
| `src/lbm.hpp` | Modify: add friend + coupling method declarations | ~10 lines |
| `src/lbm.cpp` | Modify: implement coupling methods | ~60 lines |
| `src/setup.cpp` | Modify: new main_setup() | ~50 lines |
| `src/multiblock.hpp` | **NEW** | ~60 lines |
| `src/multiblock.cpp` | **NEW** | ~300 lines |
| `makefile` | Modify: add multiblock.o | ~3 lines |

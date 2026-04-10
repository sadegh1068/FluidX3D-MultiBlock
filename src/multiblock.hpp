#pragma once

#include "lbm.hpp"
#include <vector>
using std::vector;

struct RefinementZone {
	uint cx0, cy0, cz0; // start in coarse coords (must be even)
	uint cx1, cy1, cz1; // end in coarse coords (must be even)
};

class MultiBlockLBM {
public:
	MultiBlockLBM(
		const uint cNx, const uint cNy, const uint cNz,
		const float nu,
		const RefinementZone& zone,
		const float fx=0.0f, const float fy=0.0f, const float fz=0.0f
	);
	~MultiBlockLBM();

	LBM* coarse() { return lbm_c; }
	LBM* fine() { return lbm_f; }

	void initialize();
	void run(const ulong coarse_steps);

	RefinementZone zone;

private:
	LBM* lbm_c = nullptr; // coarse grid (full domain)
	LBM* lbm_f = nullptr; // fine grid (sub-domain, 2x resolution)

	float nu_c, nu_f; // lattice viscosities (nu_f = 2 * nu_c)
	float tau_c, tau_f;

	ulong t_coarse = 0ull;
	ulong t_fine = 0ull;

	// GPU-side coupling buffers and kernels (shared OpenCL context)
	// Index arrays on device: list of cell indices for coupling
	Memory<uint>* dev_fine_ghost_indices = nullptr; // fine ghost cell indices
	Memory<uint>* dev_coarse_iface_indices = nullptr; // coarse interface cell indices

	// Mapping arrays on device for trilinear interpolation
	// For each fine ghost cell: 8 coarse source indices + 8 weights
	Memory<uint>* dev_interp_coarse_indices = nullptr; // [N_ghost * 8] coarse cell indices
	Memory<float>* dev_interp_weights = nullptr; // [N_ghost * 8] trilinear weights

	// Mapping arrays on device for fine→coarse averaging
	// For each coarse interface cell: 8 fine child indices
	Memory<uint>* dev_avg_fine_children = nullptr; // [N_iface * 8] fine child indices

	// Visualization: map TYPE_Y interior coarse cells → fine children
	Memory<uint>* dev_vis_coarse_indices = nullptr; // [N_interior] coarse TYPE_Y cell indices
	Memory<uint>* dev_vis_fine_children = nullptr; // [N_interior * 8] fine child indices

	uint n_fine_ghost = 0u;
	uint n_coarse_iface = 0u;
	uint n_coarse_interior = 0u; // TYPE_Y cells for visualization sync

	// GPU coupling kernels (compiled in shared context)
	Kernel* kernel_c2f = nullptr; // coarse→fine interpolation
	Kernel* kernel_f2c = nullptr; // fine→coarse averaging
	Kernel* kernel_vis_sync = nullptr; // copy fine data into coarse TYPE_Y cells for rendering

	void build_coupling_gpu();
	void flag_coarse_zone();
	void flag_fine_boundaries();

	void gpu_push_c2f(); // run C→F kernel on GPU (no PCIe transfer)
	void gpu_push_f2c(); // run F→C kernel on GPU (no PCIe transfer)
	void gpu_vis_sync(); // sync fine→coarse interior for visualization
};

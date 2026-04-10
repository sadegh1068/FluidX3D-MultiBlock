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

	ulong t_coarse = 0ull; // coarse timestep counter (for Esoteric-Pull parity)
	ulong t_fine = 0ull; // fine timestep counter

	// Coupling cell index lists (host-side, built once at init)
	vector<uint> coarse_interface_indices; // TYPE_E ring on coarse grid (receives F→C data)
	vector<uint> fine_ghost_indices; // TYPE_Y ring on fine grid (receives C→F data)
	vector<uint> coarse_extract_indices; // shell on coarse for interpolation source

	// Host coupling buffers: 4 floats per cell (rho, ux, uy, uz)
	vector<float> buf_c2f_extract; // extracted coarse data at shell
	vector<float> buf_c2f_interp; // interpolated data for fine ghost cells
	vector<float> buf_f2c_extract; // extracted fine data at interface
	vector<float> buf_f2c_avg; // averaged data for coarse interface cells

	// Coordinate mapping: fine ghost cell index → coarse fractional coords for interpolation
	struct FineGhostMapping {
		uint fine_idx; // linear index in fine grid
		float cx, cy, cz; // fractional position in coarse grid
	};
	vector<FineGhostMapping> ghost_mappings;

	// Coordinate mapping: coarse interface cell → 8 fine children for averaging
	struct CoarseInterfaceMapping {
		uint coarse_idx; // linear index in coarse grid
		uint fine_children[8]; // 8 fine cell indices that map to this coarse cell
	};
	vector<CoarseInterfaceMapping> interface_mappings;

	void build_index_lists();
	void flag_coarse_zone(); // set TYPE_Y interior + TYPE_E ring on coarse
	void flag_fine_boundaries(); // set TYPE_Y on fine grid outer ring

	void interpolate_c2f(); // trilinear interp: coarse → fine ghost rho/u
	void average_f2c(); // volume average: fine → coarse interface rho/u
};

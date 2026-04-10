#include "multiblock.hpp"
#include "setup.hpp"

MultiBlockLBM::MultiBlockLBM(
	const uint cNx, const uint cNy, const uint cNz,
	const float nu,
	const RefinementZone& zone,
	const float fx, const float fy, const float fz
) : zone(zone), nu_c(nu), nu_f(2.0f*nu) {
	// Validate zone alignment (must be even coordinates, inside coarse grid with margin)
	if(zone.cx0%2u!=0u || zone.cy0%2u!=0u || zone.cz0%2u!=0u ||
	   zone.cx1%2u!=0u || zone.cy1%2u!=0u || zone.cz1%2u!=0u) {
		print_error("RefinementZone coordinates must be even multiples of 2");
	}
	if(zone.cx0<2u || zone.cy0<2u || zone.cz0<2u ||
	   zone.cx1>cNx-2u || zone.cy1>cNy-2u || zone.cz1>cNz-2u) {
		print_error("RefinementZone must have at least 2-cell margin from coarse grid boundary");
	}

	// Compute tau values
	tau_c = 0.5f + 3.0f * nu_c;
	tau_f = 0.5f + 3.0f * nu_f; // = 2*(tau_c - 0.5) + 0.5

	// Fine grid dimensions: 2x resolution in each direction
	const uint fNx = (zone.cx1 - zone.cx0) * 2u;
	const uint fNy = (zone.cy1 - zone.cy0) * 2u;
	const uint fNz = (zone.cz1 - zone.cz0) * 2u;

	print_info("MultiBlockLBM: coarse grid " + to_string(cNx) + "x" + to_string(cNy) + "x" + to_string(cNz));
	print_info("MultiBlockLBM: fine grid " + to_string(fNx) + "x" + to_string(fNy) + "x" + to_string(fNz));
	print_info("MultiBlockLBM: tau_c=" + to_string(tau_c) + " tau_f=" + to_string(tau_f));

	// Create both LBM objects (single GPU, single domain each)
	lbm_c = new LBM(cNx, cNy, cNz, nu_c, fx, fy, fz);
	lbm_f = new LBM(fNx, fNy, fNz, nu_f, fx, fy, fz);
}

MultiBlockLBM::~MultiBlockLBM() {
	delete lbm_c;
	delete lbm_f;
}

void MultiBlockLBM::build_index_lists() {
	const uint cNx = lbm_c->get_Nx(), cNy = lbm_c->get_Ny(), cNz = lbm_c->get_Nz();
	const uint fNx = lbm_f->get_Nx(), fNy = lbm_f->get_Ny(), fNz = lbm_f->get_Nz();

	// Build coarse_interface_indices: 1-cell ring at zone boundary (TYPE_E cells)
	// Build coarse_extract_indices: 2-cell shell around zone (for trilinear interpolation stencil)
	coarse_interface_indices.clear();
	coarse_extract_indices.clear();
	for(uint z=zone.cz0; z<zone.cz1; z++) {
		for(uint y=zone.cy0; y<zone.cy1; y++) {
			for(uint x=zone.cx0; x<zone.cx1; x++) {
				const bool on_border = (x==zone.cx0 || x==zone.cx1-1u ||
				                        y==zone.cy0 || y==zone.cy1-1u ||
				                        z==zone.cz0 || z==zone.cz1-1u);
				if(on_border) {
					coarse_interface_indices.push_back(x + (y + z*cNy)*cNx);
				}
			}
		}
	}
	// Extract shell: zone boundary ±1 cell (for interpolation source data)
	for(uint z=zone.cz0-1u; z<=zone.cz1; z++) {
		for(uint y=zone.cy0-1u; y<=zone.cy1; y++) {
			for(uint x=zone.cx0-1u; x<=zone.cx1; x++) {
				const bool outside_zone = (x<zone.cx0 || x>=zone.cx1 ||
				                           y<zone.cy0 || y>=zone.cy1 ||
				                           z<zone.cz0 || z>=zone.cz1);
				const bool on_border = !outside_zone && (x==zone.cx0 || x==zone.cx1-1u ||
				                                          y==zone.cy0 || y==zone.cy1-1u ||
				                                          z==zone.cz0 || z==zone.cz1-1u);
				if(outside_zone || on_border) {
					coarse_extract_indices.push_back(x + (y + z*cNy)*cNx);
				}
			}
		}
	}

	// Build fine_ghost_indices: outermost 1-cell ring on all 6 faces of fine grid
	// Build ghost_mappings: for each fine ghost cell, compute coarse fractional position
	fine_ghost_indices.clear();
	ghost_mappings.clear();
	for(uint z=0u; z<fNz; z++) {
		for(uint y=0u; y<fNy; y++) {
			for(uint x=0u; x<fNx; x++) {
				const bool on_boundary = (x==0u || x==fNx-1u ||
				                          y==0u || y==fNy-1u ||
				                          z==0u || z==fNz-1u);
				if(on_boundary) {
					const uint n_fine = x + (y + z*fNy)*fNx;
					fine_ghost_indices.push_back(n_fine);
					// Map fine cell to coarse fractional coordinates
					// Fine cell (x,y,z) in fine grid → coarse coords: (x/2 + cx0, y/2 + cy0, z/2 + cz0)
					// Sub-cell offset: center of fine cell in coarse units
					FineGhostMapping m;
					m.fine_idx = n_fine;
					m.cx = (float)zone.cx0 + ((float)x + 0.5f) * 0.5f;
					m.cy = (float)zone.cy0 + ((float)y + 0.5f) * 0.5f;
					m.cz = (float)zone.cz0 + ((float)z + 0.5f) * 0.5f;
					ghost_mappings.push_back(m);
				}
			}
		}
	}

	// Build interface_mappings: for each coarse interface cell, find 8 fine children
	interface_mappings.clear();
	for(uint i=0u; i<(uint)coarse_interface_indices.size(); i++) {
		const uint n_c = coarse_interface_indices[i];
		const uint cx = n_c % cNx;
		const uint cy = (n_c / cNx) % cNy;
		const uint cz = n_c / (cNx * cNy);

		// Fine children: (2*(cx-cx0) + a, 2*(cy-cy0) + b, 2*(cz-cz0) + c) for a,b,c in {0,1}
		const uint fx0 = 2u * (cx - zone.cx0);
		const uint fy0 = 2u * (cy - zone.cy0);
		const uint fz0 = 2u * (cz - zone.cz0);

		CoarseInterfaceMapping m;
		m.coarse_idx = n_c;
		uint child = 0u;
		for(uint dz=0u; dz<2u; dz++) {
			for(uint dy=0u; dy<2u; dy++) {
				for(uint dx=0u; dx<2u; dx++) {
					m.fine_children[child++] = (fx0+dx) + ((fy0+dy) + (fz0+dz)*fNy)*fNx;
				}
			}
		}
		interface_mappings.push_back(m);
	}

	// Allocate host coupling buffers
	buf_c2f_extract.resize(coarse_extract_indices.size() * 4u);
	buf_c2f_interp.resize(fine_ghost_indices.size() * 4u);
	buf_f2c_extract.resize(fine_ghost_indices.size() * 4u); // reuse size — extract from fine interface region
	buf_f2c_avg.resize(coarse_interface_indices.size() * 4u);

	print_info("MultiBlockLBM: coarse interface cells = " + to_string(coarse_interface_indices.size()));
	print_info("MultiBlockLBM: coarse extract shell = " + to_string(coarse_extract_indices.size()));
	print_info("MultiBlockLBM: fine ghost cells = " + to_string(fine_ghost_indices.size()));
}

void MultiBlockLBM::flag_coarse_zone() {
	const uint cNx = lbm_c->get_Nx(), cNy = lbm_c->get_Ny();
	parallel_for(lbm_c->get_N(), [&](ulong n) {
		uint x, y, z;
		lbm_c->coordinates(n, x, y, z);
		if(x>=zone.cx0 && x<zone.cx1 && y>=zone.cy0 && y<zone.cy1 && z>=zone.cz0 && z<zone.cz1) {
			const bool on_border = (x==zone.cx0 || x==zone.cx1-1u ||
			                        y==zone.cy0 || y==zone.cy1-1u ||
			                        z==zone.cz0 || z==zone.cz1-1u);
			if(on_border) {
				lbm_c->flags[n] = TYPE_E; // interface ring: receives averaged fine data
			} else {
				lbm_c->flags[n] = TYPE_Y; // interior: skip computation
			}
		}
	});
}

void MultiBlockLBM::flag_fine_boundaries() {
	const uint fNx = lbm_f->get_Nx(), fNy = lbm_f->get_Ny(), fNz = lbm_f->get_Nz();
	parallel_for(lbm_f->get_N(), [&](ulong n) {
		uint x, y, z;
		lbm_f->coordinates(n, x, y, z);
		const bool on_boundary = (x==0u || x==fNx-1u ||
		                          y==0u || y==fNy-1u ||
		                          z==0u || z==fNz-1u);
		if(on_boundary) {
			lbm_f->flags[n] = TYPE_E; // ghost ring: TYPE_E reads stored rho/u and sets DDFs=feq
		}
	});
}

void MultiBlockLBM::interpolate_c2f() {
	const uint cNx = lbm_c->get_Nx(), cNy = lbm_c->get_Ny(), cNz = lbm_c->get_Nz();

	// Build a lookup from coarse extract indices → position in buf_c2f_extract
	// For simplicity, use direct coarse grid access (host-side rho/u arrays)
	// This avoids building a hash map — we read directly from lbm_c->rho[n], lbm_c->u.*[n]

	for(uint i=0u; i<(uint)ghost_mappings.size(); i++) {
		const FineGhostMapping& m = ghost_mappings[i];

		// Trilinear interpolation from 8 surrounding coarse cells
		const int i0 = (int)floorf(m.cx - 0.5f), j0 = (int)floorf(m.cy - 0.5f), k0 = (int)floorf(m.cz - 0.5f);
		const float dx = m.cx - ((float)i0 + 0.5f);
		const float dy = m.cy - ((float)j0 + 0.5f);
		const float dz = m.cz - ((float)k0 + 0.5f);

		float rho_interp=0.0f, ux_interp=0.0f, uy_interp=0.0f, uz_interp=0.0f;
		for(uint ddk=0u; ddk<2u; ddk++) {
			for(uint ddj=0u; ddj<2u; ddj++) {
				for(uint ddi=0u; ddi<2u; ddi++) {
					const uint ci = clamp((uint)(i0+(int)ddi), 0u, cNx-1u);
					const uint cj = clamp((uint)(j0+(int)ddj), 0u, cNy-1u);
					const uint ck = clamp((uint)(k0+(int)ddk), 0u, cNz-1u);
					const float wx = ddi==0u ? (1.0f-dx) : dx;
					const float wy = ddj==0u ? (1.0f-dy) : dy;
					const float wz = ddk==0u ? (1.0f-dz) : dz;
					const float w = wx * wy * wz;

					const ulong nc = (ulong)ci + ((ulong)cj + (ulong)ck*(ulong)cNy)*(ulong)cNx;
					rho_interp += w * lbm_c->rho[nc];
					ux_interp  += w * lbm_c->u.x[nc];
					uy_interp  += w * lbm_c->u.y[nc];
					uz_interp  += w * lbm_c->u.z[nc];
				}
			}
		}

		buf_c2f_interp[i*4u+0u] = rho_interp;
		buf_c2f_interp[i*4u+1u] = ux_interp;
		buf_c2f_interp[i*4u+2u] = uy_interp;
		buf_c2f_interp[i*4u+3u] = uz_interp;
	}
}

void MultiBlockLBM::average_f2c() {
	for(uint i=0u; i<(uint)interface_mappings.size(); i++) {
		const CoarseInterfaceMapping& m = interface_mappings[i];
		float rho_avg=0.0f, ux_avg=0.0f, uy_avg=0.0f, uz_avg=0.0f;
		for(uint c=0u; c<8u; c++) {
			const ulong nf = (ulong)m.fine_children[c];
			rho_avg += lbm_f->rho[nf];
			ux_avg  += lbm_f->u.x[nf];
			uy_avg  += lbm_f->u.y[nf];
			uz_avg  += lbm_f->u.z[nf];
		}
		buf_f2c_avg[i*4u+0u] = rho_avg * 0.125f; // /8
		buf_f2c_avg[i*4u+1u] = ux_avg  * 0.125f;
		buf_f2c_avg[i*4u+2u] = uy_avg  * 0.125f;
		buf_f2c_avg[i*4u+3u] = uz_avg  * 0.125f;
	}
}

void MultiBlockLBM::initialize() {
	// Flag coarse zone and fine boundaries (before user setup, so user can override interior cells)
	flag_coarse_zone();
	flag_fine_boundaries();

	// Build coupling index lists and buffers
	build_index_lists();

	// Set initial rho/u for coupling cells from coarse initial conditions
	interpolate_c2f();
	for(uint i=0u; i<(uint)fine_ghost_indices.size(); i++) {
		const ulong nf = (ulong)fine_ghost_indices[i];
		lbm_f->rho[nf]  = buf_c2f_interp[i*4u+0u];
		lbm_f->u.x[nf]  = buf_c2f_interp[i*4u+1u];
		lbm_f->u.y[nf]  = buf_c2f_interp[i*4u+2u];
		lbm_f->u.z[nf]  = buf_c2f_interp[i*4u+3u];
	}

	// Initialize both LBM objects (copies host data to GPU, runs kernel_initialize)
	lbm_c->initialize();
	lbm_f->initialize();

	t_coarse = 0ull;
	t_fine = 0ull;

	print_info("MultiBlockLBM: initialized");
}

void MultiBlockLBM::run(const ulong coarse_steps) {
	if(!lbm_c->initialized) initialize();

	// Helper lambdas for coupling transfers (read-modify-write only coupling cells)
	auto push_c2f = [&]() { // interpolate coarse→fine ghost cells, write rho/u to fine device
		// Read coarse rho/u from device
		for(uint d=0u; d<lbm_c->get_D(); d++) {
			lbm_c->lbm_domain[d]->rho.read_from_device();
			lbm_c->lbm_domain[d]->u.read_from_device();
		}
		interpolate_c2f();
		// Read fine rho/u from device (so we don't overwrite interior cells)
		for(uint d=0u; d<lbm_f->get_D(); d++) {
			lbm_f->lbm_domain[d]->rho.read_from_device();
			lbm_f->lbm_domain[d]->u.read_from_device();
		}
		// Modify ONLY ghost cells on host
		for(uint i=0u; i<(uint)fine_ghost_indices.size(); i++) {
			const ulong nf = (ulong)fine_ghost_indices[i];
			lbm_f->rho[nf]  = buf_c2f_interp[i*4u+0u];
			lbm_f->u.x[nf]  = buf_c2f_interp[i*4u+1u];
			lbm_f->u.y[nf]  = buf_c2f_interp[i*4u+2u];
			lbm_f->u.z[nf]  = buf_c2f_interp[i*4u+3u];
		}
		// Write full arrays back (now only ghost cells differ from device state)
		for(uint d=0u; d<lbm_f->get_D(); d++) {
			lbm_f->lbm_domain[d]->rho.write_to_device();
			lbm_f->lbm_domain[d]->u.write_to_device();
		}
	};

	auto push_f2c = [&]() { // average fine→coarse interface cells, write rho/u to coarse device
		// Read fine rho/u from device
		for(uint d=0u; d<lbm_f->get_D(); d++) {
			lbm_f->lbm_domain[d]->rho.read_from_device();
			lbm_f->lbm_domain[d]->u.read_from_device();
		}
		average_f2c();
		// Read coarse rho/u from device (so we don't overwrite non-interface cells)
		for(uint d=0u; d<lbm_c->get_D(); d++) {
			lbm_c->lbm_domain[d]->rho.read_from_device();
			lbm_c->lbm_domain[d]->u.read_from_device();
		}
		// Modify ONLY interface cells on host
		for(uint i=0u; i<(uint)coarse_interface_indices.size(); i++) {
			const ulong nc = (ulong)coarse_interface_indices[i];
			lbm_c->rho[nc]  = buf_f2c_avg[i*4u+0u];
			lbm_c->u.x[nc]  = buf_f2c_avg[i*4u+1u];
			lbm_c->u.y[nc]  = buf_f2c_avg[i*4u+2u];
			lbm_c->u.z[nc]  = buf_f2c_avg[i*4u+3u];
		}
		// Write full arrays back
		for(uint d=0u; d<lbm_c->get_D(); d++) {
			lbm_c->lbm_domain[d]->rho.write_to_device();
			lbm_c->lbm_domain[d]->u.write_to_device();
		}
	};

	for(ulong step=0ull; step<coarse_steps; step++) {
		// === Step 1: C→F coupling (use current coarse data) ===
		push_c2f();

		// === Step 2: Fine sub-step 1 ===
		lbm_f->do_time_step();
		t_fine++;

		// === Step 3: F→C coupling (restrict fine → coarse interface) ===
		push_f2c();

		// === Step 4: Coarse step (with fresh interface data) ===
		lbm_c->do_time_step();
		t_coarse++;

		// === Step 5: C→F coupling again (use NEW coarse data post-step) ===
		push_c2f();

		// === Step 6: Fine sub-step 2 ===
		lbm_f->do_time_step();
		t_fine++;

		// === Step 7: F→C coupling (final restriction) ===
		push_f2c();
	}

	// Final sync
	for(uint d=0u; d<lbm_c->get_D(); d++) lbm_c->lbm_domain[d]->finish_queue();
	for(uint d=0u; d<lbm_f->get_D(); d++) lbm_f->lbm_domain[d]->finish_queue();
}

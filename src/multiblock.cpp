#include "multiblock.hpp"
#include "setup.hpp"

MultiBlockLBM::MultiBlockLBM(
	const uint cNx, const uint cNy, const uint cNz,
	const float nu,
	const RefinementZone& zone,
	const float fx, const float fy, const float fz
) : zone(zone), nu_c(nu), nu_f(2.0f*nu) {
	// Validate zone alignment
	if(zone.cx0%2u!=0u || zone.cy0%2u!=0u || zone.cz0%2u!=0u ||
	   zone.cx1%2u!=0u || zone.cy1%2u!=0u || zone.cz1%2u!=0u) {
		print_error("RefinementZone coordinates must be even multiples of 2");
	}
	if(zone.cx0<2u || zone.cy0<2u || zone.cz0<2u ||
	   zone.cx1>cNx-2u || zone.cy1>cNy-2u || zone.cz1>cNz-2u) {
		print_error("RefinementZone must have at least 2-cell margin from coarse grid boundary");
	}

	tau_c = 0.5f + 3.0f * nu_c;
	tau_f = 0.5f + 3.0f * nu_f;

	const uint fNx = (zone.cx1 - zone.cx0) * 2u;
	const uint fNy = (zone.cy1 - zone.cy0) * 2u;
	const uint fNz = (zone.cz1 - zone.cz0) * 2u;

	print_info("MultiBlockLBM: coarse grid " + to_string(cNx) + "x" + to_string(cNy) + "x" + to_string(cNz));
	print_info("MultiBlockLBM: fine grid " + to_string(fNx) + "x" + to_string(fNy) + "x" + to_string(fNz));
	print_info("MultiBlockLBM: tau_c=" + to_string(tau_c) + " tau_f=" + to_string(tau_f));

	// Create SHARED OpenCL context — both LBM objects use the same GPU device
	const Device_Info shared_device = select_device_with_most_flops();

	// Create both LBM objects using the shared device context
	lbm_c = new LBM(shared_device, cNx, cNy, cNz, nu_c, fx, fy, fz);
	lbm_f = new LBM(shared_device, fNx, fNy, fNz, nu_f, fx, fy, fz);

	print_info("MultiBlockLBM: shared OpenCL context — GPU-side coupling enabled");
}

MultiBlockLBM::~MultiBlockLBM() {
	delete kernel_c2f;
	delete kernel_f2c;
	delete dev_fine_ghost_indices;
	delete dev_coarse_iface_indices;
	delete dev_interp_coarse_indices;
	delete dev_interp_weights;
	delete dev_avg_fine_children;
	delete lbm_c;
	delete lbm_f;
}

void MultiBlockLBM::flag_coarse_zone() {
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
			lbm_f->flags[n] = TYPE_E; // ghost ring: TYPE_E reads stored rho/u, sets DDFs=feq
		}
	});
}

void MultiBlockLBM::build_coupling_gpu() {
	const uint cNx = lbm_c->get_Nx(), cNy = lbm_c->get_Ny(), cNz = lbm_c->get_Nz();
	const uint fNx = lbm_f->get_Nx(), fNy = lbm_f->get_Ny(), fNz = lbm_f->get_Nz();

	// Build host-side index lists first, then upload to GPU

	// === Fine ghost cells + interpolation mappings ===
	vector<uint> h_fine_ghost;
	vector<uint> h_interp_coarse; // 8 per ghost cell
	vector<float> h_interp_weights; // 8 per ghost cell

	for(uint z=0u; z<fNz; z++) {
		for(uint y=0u; y<fNy; y++) {
			for(uint x=0u; x<fNx; x++) {
				const bool on_boundary = (x==0u || x==fNx-1u || y==0u || y==fNy-1u || z==0u || z==fNz-1u);
				if(!on_boundary) continue;

				const uint n_fine = x + (y + z*fNy)*fNx;
				h_fine_ghost.push_back(n_fine);

				// Map fine cell center to coarse fractional coordinates
				const float fcx = (float)zone.cx0 + ((float)x + 0.5f) * 0.5f;
				const float fcy = (float)zone.cy0 + ((float)y + 0.5f) * 0.5f;
				const float fcz = (float)zone.cz0 + ((float)z + 0.5f) * 0.5f;

				// Trilinear stencil: 8 coarse neighbors
				const int i0 = (int)floorf(fcx - 0.5f);
				const int j0 = (int)floorf(fcy - 0.5f);
				const int k0 = (int)floorf(fcz - 0.5f);
				const float dx = fcx - ((float)i0 + 0.5f);
				const float dy = fcy - ((float)j0 + 0.5f);
				const float dz = fcz - ((float)k0 + 0.5f);

				for(uint dk=0u; dk<2u; dk++) {
					for(uint dj=0u; dj<2u; dj++) {
						for(uint di=0u; di<2u; di++) {
							const uint ci = (uint)max(0, min((int)(cNx-1u), i0+(int)di));
							const uint cj = (uint)max(0, min((int)(cNy-1u), j0+(int)dj));
							const uint ck = (uint)max(0, min((int)(cNz-1u), k0+(int)dk));
							h_interp_coarse.push_back(ci + (cj + ck*cNy)*cNx);

							const float wx = di==0u ? (1.0f-dx) : dx;
							const float wy = dj==0u ? (1.0f-dy) : dy;
							const float wz = dk==0u ? (1.0f-dz) : dz;
							h_interp_weights.push_back(wx * wy * wz);
						}
					}
				}
			}
		}
	}
	n_fine_ghost = (uint)h_fine_ghost.size();

	// === Coarse interface cells + averaging mappings ===
	vector<uint> h_coarse_iface;
	vector<uint> h_avg_children; // 8 per interface cell

	for(uint z=zone.cz0; z<zone.cz1; z++) {
		for(uint y=zone.cy0; y<zone.cy1; y++) {
			for(uint x=zone.cx0; x<zone.cx1; x++) {
				const bool on_border = (x==zone.cx0 || x==zone.cx1-1u ||
				                        y==zone.cy0 || y==zone.cy1-1u ||
				                        z==zone.cz0 || z==zone.cz1-1u);
				if(!on_border) continue;

				h_coarse_iface.push_back(x + (y + z*cNy)*cNx);

				// 8 fine children
				const uint fx0 = 2u * (x - zone.cx0);
				const uint fy0 = 2u * (y - zone.cy0);
				const uint fz0 = 2u * (z - zone.cz0);
				for(uint dk=0u; dk<2u; dk++) {
					for(uint dj=0u; dj<2u; dj++) {
						for(uint di=0u; di<2u; di++) {
							h_avg_children.push_back((fx0+di) + ((fy0+dj) + (fz0+dk)*fNy)*fNx);
						}
					}
				}
			}
		}
	}
	n_coarse_iface = (uint)h_coarse_iface.size();

	print_info("MultiBlockLBM: fine ghost cells = " + to_string(n_fine_ghost) + ", coarse interface cells = " + to_string(n_coarse_iface));

	// === Upload index/weight arrays to GPU ===
	Device& dev = lbm_c->lbm_domain[0]->device; // shared device

	dev_fine_ghost_indices = new Memory<uint>(dev, n_fine_ghost);
	dev_coarse_iface_indices = new Memory<uint>(dev, n_coarse_iface);
	dev_interp_coarse_indices = new Memory<uint>(dev, n_fine_ghost * 8u);
	dev_interp_weights = new Memory<float>(dev, n_fine_ghost * 8u);
	dev_avg_fine_children = new Memory<uint>(dev, n_coarse_iface * 8u);

	for(uint i=0u; i<n_fine_ghost; i++) (*dev_fine_ghost_indices)[i] = h_fine_ghost[i];
	for(uint i=0u; i<n_coarse_iface; i++) (*dev_coarse_iface_indices)[i] = h_coarse_iface[i];
	for(uint i=0u; i<n_fine_ghost*8u; i++) (*dev_interp_coarse_indices)[i] = h_interp_coarse[i];
	for(uint i=0u; i<n_fine_ghost*8u; i++) (*dev_interp_weights)[i] = h_interp_weights[i];
	for(uint i=0u; i<n_coarse_iface*8u; i++) (*dev_avg_fine_children)[i] = h_avg_children[i];

	dev_fine_ghost_indices->enqueue_write_to_device();
	dev_coarse_iface_indices->enqueue_write_to_device();
	dev_interp_coarse_indices->enqueue_write_to_device();
	dev_interp_weights->enqueue_write_to_device();
	dev_avg_fine_children->enqueue_write_to_device();
	dev.finish_queue();

	// === Compile GPU coupling kernels ===
	// These kernels run in the shared context and directly access both grids' rho/u buffers
	const ulong cN = lbm_c->get_N();
	const ulong fN = lbm_f->get_N();

	// C→F kernel: trilinear interpolation from coarse rho/u → fine ghost rho/u
	kernel_c2f = new Kernel(dev, n_fine_ghost, "coupling_c2f",
		lbm_c->lbm_domain[0]->rho, // coarse rho (read)
		lbm_c->lbm_domain[0]->u,   // coarse u (read)
		lbm_f->lbm_domain[0]->rho, // fine rho (write)
		lbm_f->lbm_domain[0]->u,   // fine u (write)
		*dev_fine_ghost_indices,     // fine ghost cell indices
		*dev_interp_coarse_indices,  // 8 coarse source indices per ghost cell
		*dev_interp_weights,         // 8 weights per ghost cell
		cN, fN, n_fine_ghost
	);

	// F→C kernel: volume average from fine rho/u → coarse interface rho/u
	kernel_f2c = new Kernel(dev, n_coarse_iface, "coupling_f2c",
		lbm_f->lbm_domain[0]->rho, // fine rho (read)
		lbm_f->lbm_domain[0]->u,   // fine u (read)
		lbm_c->lbm_domain[0]->rho, // coarse rho (write)
		lbm_c->lbm_domain[0]->u,   // coarse u (write)
		*dev_coarse_iface_indices,   // coarse interface cell indices
		*dev_avg_fine_children,      // 8 fine child indices per interface cell
		cN, fN, n_coarse_iface
	);

	print_info("MultiBlockLBM: GPU coupling kernels compiled");
}

void MultiBlockLBM::gpu_push_c2f() {
	kernel_c2f->enqueue_run();
}

void MultiBlockLBM::gpu_push_f2c() {
	kernel_f2c->enqueue_run();
}

void MultiBlockLBM::initialize() {
	flag_coarse_zone();
	flag_fine_boundaries();

	// Initialize both LBM objects (copies host data to GPU, compiles kernels)
	lbm_c->initialize();
	lbm_f->initialize();

	// Build GPU coupling after LBM initialization (buffers must exist on device)
	build_coupling_gpu();

	// Initial C→F coupling to set fine ghost cells from coarse initial conditions
	gpu_push_c2f();
	lbm_c->lbm_domain[0]->finish_queue();

	t_coarse = 0ull;
	t_fine = 0ull;

	print_info("MultiBlockLBM: initialized (GPU-side coupling, zero PCIe transfers per step)");
}

void MultiBlockLBM::run(const ulong coarse_steps) {
	if(!lbm_c->initialized) initialize();

	for(ulong step=0ull; step<coarse_steps; step++) {
		// === Step 1: C→F coupling (GPU kernel, no PCIe) ===
		gpu_push_c2f();

		// === Step 2: Fine sub-step 1 ===
		lbm_f->do_time_step();
		t_fine++;

		// === Step 3: F→C coupling (GPU kernel, no PCIe) ===
		gpu_push_f2c();

		// === Step 4: Coarse step ===
		lbm_c->do_time_step();
		t_coarse++;

		// === Step 5: C→F coupling again (post-coarse data) ===
		gpu_push_c2f();

		// === Step 6: Fine sub-step 2 ===
		lbm_f->do_time_step();
		t_fine++;

		// === Step 7: F→C coupling (final restriction) ===
		gpu_push_f2c();
	}

	lbm_c->lbm_domain[0]->finish_queue();
	lbm_f->lbm_domain[0]->finish_queue();
}

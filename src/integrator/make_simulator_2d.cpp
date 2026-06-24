#include "integrator/make_simulator_2d.h"
#include "integrator/factory.h"
#include <iostream>
#include <stdexcept>
// CUDA is OPTIONAL: compiled by nvcc (HAVE_CUDA) so backend=gpu can inject the
// GPU Poisson solver, or by the C++ compiler for a CPU-only binary.
#ifdef HAVE_CUDA
#include "solver/cuda_pcg_solver.h"
#include <cuda_runtime.h>
#endif

namespace scene2d {

// Is a usable GPU actually present in THIS build / on THIS machine?
static bool gpu_present() {
#ifdef HAVE_CUDA
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
#else
    return false;
#endif
}

// 2D's GPU is a GPU Poisson solver (CudaPCGSolver) injected into the CPU
// simulator loop — not a GPU-resident simulator like 3D. So it is OPT-IN:
// backend=gpu offloads the pressure solve to the GPU; auto/cpu keep the
// configured CPU solver (so CPU reproductions stay bit-identical).
std::unique_ptr<Simulator> make_simulator(Config& cfg) {
    std::string backend = cfg.sget("backend", "cpu");
    if (backend != "auto" && backend != "gpu" && backend != "cpu")
        throw std::runtime_error("cfdsim: backend must be 'auto' | 'gpu' | 'cpu' (got '" + backend +
                                 "')");
    if (backend == "auto")
        backend = "cpu"; // 2D: the GPU solver is an opt-in, not the default
    if (backend == "gpu" && !gpu_present()) {
        std::cerr << "cfdsim: no usable CUDA device — running on the CPU instead\n";
        backend = "cpu";
    }
    cfg.extra["backend"] = backend; // record the real choice for the run banner

#ifdef HAVE_CUDA
    if (backend == "gpu")
        return SimulatorFactory::create(cfg, std::make_unique<CudaPCGSolver>(true));
#endif
    return SimulatorFactory::create(cfg); // CPU pressure solver
}

} // namespace scene2d

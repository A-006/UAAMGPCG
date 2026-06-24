#include "simulator/make_simulator_3d.h"
#include "simulator/lfm_simulator_3d.h"
#include "simulator/scene_3d.h"
#include "solver/factory_3d.h"
#include <iostream>
#include <stdexcept>
// CUDA is OPTIONAL: this file is compiled by nvcc (HAVE_CUDA) so it can build
// the GPU-resident simulator, or by the C++ compiler for a CPU-only binary.
#ifdef HAVE_CUDA
#include "simulator/cuda_lfm_simulator_3d.h"
#include <cuda_runtime.h>
#endif

namespace scene3d {

// Is a usable GPU actually present in THIS build / on THIS machine?
static bool gpu_present() {
#ifdef HAVE_CUDA
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
#else
    return false;
#endif
}

std::unique_ptr<Simulator3D> make_simulator(Config& cfg) {
    std::string backend = cfg.sget("backend", "auto");
    if (backend != "auto" && backend != "gpu" && backend != "cpu")
        throw std::runtime_error("cfdsim: backend must be 'auto' | 'gpu' | 'cpu' (got '" + backend + "')");
    if (backend == "auto")
        backend = gpu_present() ? "gpu" : "cpu";
    if (backend == "gpu" && !gpu_present()) {
        std::cerr << "cfdsim: no usable CUDA device — running on the CPU instead\n";
        backend = "cpu";
    }
    cfg.extra["backend"] = backend; // record the real choice for the run banner

#ifdef HAVE_CUDA
    if (backend == "gpu") {
        int dev = cfg.iget("gpu", -1); // multi-GPU device index
        if (dev >= 0 && cudaSetDevice(dev) != cudaSuccess)
            throw std::runtime_error("cfdsim: cudaSetDevice(" + std::to_string(dev) + ") failed");
        auto s = std::make_unique<CudaLFMSimulator3D>(cfg);
        setup(s->mutable_grid(), cfg); // initial condition
        s->commit();                   // upload IC + apply wall BCs
        return s;
    }
#endif
    auto s = std::make_unique<LFMSimulator3D>(cfg, Factory3D::create(cfg.solver), make_cpu_bcs(cfg));
    setup(s->mutable_grid(), cfg); // initial condition
    s->commit();                   // apply wall BCs
    return s;
}

} // namespace scene3d

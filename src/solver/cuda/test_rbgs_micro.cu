/**
 * Micro-test: one RBGS sweep on a tiny grid — compare CPU vs GPU cell by cell.
 */
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>

// ── GPU kernels (copied from cuda_uaamg_preconditioner.cu) ──
__device__ inline int dev_idx(int i, int j, int stride) { return i + j * stride; }

__global__ void rbgs_red_kernel(
    double *x, const double *b, const bool *solid,
    int nx, int ny, int stride, double idx2, double idy2, double diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    if ((i + j) & 1) return;
    int id = dev_idx(i, j, stride);
    if (solid[id]) return;
    double pC = x[id];
    double pL = (i>1 && !solid[dev_idx(i-1,j,stride)]) ? x[dev_idx(i-1,j,stride)] : pC;
    double pR = (i<nx && !solid[dev_idx(i+1,j,stride)]) ? x[dev_idx(i+1,j,stride)] : pC;
    double pB = (j>1 && !solid[dev_idx(i,j-1,stride)]) ? x[dev_idx(i,j-1,stride)] : pC;
    double pT = (j<ny && !solid[dev_idx(i,j+1,stride)]) ? x[dev_idx(i,j+1,stride)] : pC;
    double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
    double eff_d = diag;
    if (i==1||solid[dev_idx(i-1,j,stride)]) eff_d -= idx2;
    if (i==nx||solid[dev_idx(i+1,j,stride)]) eff_d -= idx2;
    if (j==1||solid[dev_idx(i,j-1,stride)]) eff_d -= idy2;
    if (j==ny||solid[dev_idx(i,j+1,stride)]) eff_d -= idy2;
    x[id] += (eff_d < 1e-15 ? 0.0 : 1.0/eff_d) * (b[id] - diag * pC + lap);
}

__global__ void rbgs_black_kernel(
    double *x, const double *b, const bool *solid,
    int nx, int ny, int stride, double idx2, double idy2, double diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    if (!((i + j) & 1)) return;
    int id = dev_idx(i, j, stride);
    if (solid[id]) return;
    double pC = x[id];
    double pL = (i>1 && !solid[dev_idx(i-1,j,stride)]) ? x[dev_idx(i-1,j,stride)] : pC;
    double pR = (i<nx && !solid[dev_idx(i+1,j,stride)]) ? x[dev_idx(i+1,j,stride)] : pC;
    double pB = (j>1 && !solid[dev_idx(i,j-1,stride)]) ? x[dev_idx(i,j-1,stride)] : pC;
    double pT = (j<ny && !solid[dev_idx(i,j+1,stride)]) ? x[dev_idx(i,j+1,stride)] : pC;
    double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
    double eff_d = diag;
    if (i==1||solid[dev_idx(i-1,j,stride)]) eff_d -= idx2;
    if (i==nx||solid[dev_idx(i+1,j,stride)]) eff_d -= idx2;
    if (j==1||solid[dev_idx(i,j-1,stride)]) eff_d -= idy2;
    if (j==ny||solid[dev_idx(i,j+1,stride)]) eff_d -= idy2;
    x[id] += (eff_d < 1e-15 ? 0.0 : 1.0/eff_d) * (b[id] - diag * pC + lap);
}

// ── CPU RBGS ──
static void cpu_rbgs_sweep(double *x, const double *b, const bool *solid,
                           int nx, int ny, int stride, double idx2, double idy2, double diag) {
    // Red pass
    for (int i = 1; i <= nx; i++)
        for (int j = 1 + (i % 2); j <= ny; j += 2) {
            int id = i + j * stride;
            if (solid[id]) continue;
            double pC = x[id];
            double pL = (i>1 && !solid[(i-1)+j*stride]) ? x[(i-1)+j*stride] : pC;
            double pR = (i<nx && !solid[(i+1)+j*stride]) ? x[(i+1)+j*stride] : pC;
            double pB = (j>1 && !solid[i+(j-1)*stride]) ? x[i+(j-1)*stride] : pC;
            double pT = (j<ny && !solid[i+(j+1)*stride]) ? x[i+(j+1)*stride] : pC;
            double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
            double eff_d = diag;
            if (i==1||solid[(i-1)+j*stride]) eff_d -= idx2;
            if (i==nx||solid[(i+1)+j*stride]) eff_d -= idx2;
            if (j==1||solid[i+(j-1)*stride]) eff_d -= idy2;
            if (j==ny||solid[i+(j+1)*stride]) eff_d -= idy2;
            x[id] += (eff_d < 1e-15 ? 0.0 : 1.0/eff_d) * (b[id] - diag * pC + lap);
        }
    // Black pass
    for (int i = 1; i <= nx; i++)
        for (int j = 1 + ((i+1) % 2); j <= ny; j += 2) {
            int id = i + j * stride;
            if (solid[id]) continue;
            double pC = x[id];
            double pL = (i>1 && !solid[(i-1)+j*stride]) ? x[(i-1)+j*stride] : pC;
            double pR = (i<nx && !solid[(i+1)+j*stride]) ? x[(i+1)+j*stride] : pC;
            double pB = (j>1 && !solid[i+(j-1)*stride]) ? x[i+(j-1)*stride] : pC;
            double pT = (j<ny && !solid[i+(j+1)*stride]) ? x[i+(j+1)*stride] : pC;
            double lap = (pL+pR)*idx2 + (pB+pT)*idy2;
            double eff_d = diag;
            if (i==1||solid[(i-1)+j*stride]) eff_d -= idx2;
            if (i==nx||solid[(i+1)+j*stride]) eff_d -= idx2;
            if (j==1||solid[i+(j-1)*stride]) eff_d -= idy2;
            if (j==ny||solid[i+(j+1)*stride]) eff_d -= idy2;
            x[id] += (eff_d < 1e-15 ? 0.0 : 1.0/eff_d) * (b[id] - diag * pC + lap);
        }
}

int main() {
    int nx = 4, ny = 4;
    double dx = 1.0, dy = 1.0;  // unit grid
    double idx2 = 1.0/(dx*dx), idy2 = 1.0/(dy*dy), diag = 2.0*(idx2+idy2);
    int stride = nx + 2;  // = 6
    int N = (nx+2)*(ny+2); // = 36

    // Setup: x=0, b=1 on interior, solid=false
    std::vector<double> h_x(N, 0.0), h_b(N, 0.0);
    std::vector<char> h_solid(N, 0);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            h_b[i + j*stride] = 1.0;

    // ── CPU ──
    std::vector<double> cpu_x = h_x;
    cpu_rbgs_sweep(cpu_x.data(), h_b.data(), (bool*)h_solid.data(),
                   nx, ny, stride, idx2, idy2, diag);

    // ── GPU ──
    double *d_x, *d_b; bool *d_solid;
    cudaMalloc(&d_x, N * sizeof(double));
    cudaMalloc(&d_b, N * sizeof(double));
    cudaMalloc(&d_solid, N * sizeof(bool));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_solid, h_solid.data(), N * sizeof(bool), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(1, 1);
    rbgs_red_kernel<<<grid, block>>>(d_x, d_b, d_solid, nx, ny, stride, idx2, idy2, diag);
    cudaDeviceSynchronize();
    rbgs_black_kernel<<<grid, block>>>(d_x, d_b, d_solid, nx, ny, stride, idx2, idy2, diag);
    cudaDeviceSynchronize();

    std::vector<double> gpu_x(N);
    cudaMemcpy(gpu_x.data(), d_x, N * sizeof(double), cudaMemcpyDeviceToHost);

    // ── Compare cell by cell ──
    printf("i j   CPU       GPU       diff\n");
    printf("------------------------------\n");
    double max_diff = 0;
    for (int i = 1; i <= nx; i++) {
        for (int j = 1; j <= ny; j++) {
            int id = i + j * stride;
            double d = std::abs(cpu_x[id] - gpu_x[id]);
            if (d > max_diff) max_diff = d;
            printf("%d %d  %+.6f  %+.6f  %e\n", i, j, cpu_x[id], gpu_x[id], d);
        }
    }
    printf("\nMax diff: %e\n", max_diff);
    printf("%s\n", max_diff < 1e-14 ? "PASS" : "FAIL");

    cudaFree(d_x); cudaFree(d_b); cudaFree(d_solid);
    return max_diff >= 1e-14;
}

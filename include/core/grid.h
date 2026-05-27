#pragma once
#include <vector>
#include <algorithm>

// MAC staggered grid (2D)
//
// NX×NY cells. Physical coordinates:
//   Cell (i,j): [(i-1)dx, i·dx] × [(j-1)dy, j·dy],  i∈[1,NX], j∈[1,NY]
//   u(i,j): x-velocity, on right face,  at (i·dx,  (j-0.5)dy), i∈[0,NX],   j∈[1,NY]
//   v(i,j): y-velocity, on top face,    at ((i-0.5)dx, j·dy),  i∈[1,NX],   j∈[0,NY]
//   p(i,j): pressure,    at cell center,                        i∈[1,NX],   j∈[1,NY]
//
// All arrays include one layer of ghost cells (0 and N+1).
// Storage: 1D flattened, column-major (i stride = 1).

class Grid {
public:
    int nx, ny;
    double dx, dy;
    std::vector<double> u, v, p;
    std::vector<bool>   solid;

    Grid(int nx_, int ny_, double lx, double ly);

    double Lx() const { return nx * dx; }
    double Ly() const { return ny * dy; }
    const std::vector<double>& u_data() const { return u; }
    const std::vector<double>& v_data() const { return v; }

    // Variable Laplacian coefficients for ∇·((1-φ)∇p).
    // When empty, PCG uses uniform coefficients.
    // diag > 0, off_x < 0, off_y < 0.
    std::vector<double> lap_diag, lap_off_x, lap_off_y;
    bool has_variable_lap() const { return !lap_diag.empty(); }
    void init_variable_lap();  // allocate to match p array size

    // u: i∈[0,nx], j∈[0,ny+1]
    int    iu(int i, int j) const;
    double  u_at(int i, int j) const;
    double& u_at(int i, int j);

    // v: i∈[0,nx+1], j∈[0,ny]
    int    iv(int i, int j) const;
    double  v_at(int i, int j) const;
    double& v_at(int i, int j);

    // p/solid: i∈[0,nx+1], j∈[0,ny+1]
    int    ip(int i, int j) const;
    double  p_at(int i, int j) const;
    double& p_at(int i, int j);
    bool    is_solid(int i, int j) const;
    void    set_solid(int i, int j);

    // Divergence at cell (i,j): du/dx + dv/dy
    double divergence(int i, int j) const {
        return (u_at(i,j) - u_at(i-1,j)) / dx
             + (v_at(i,j) - v_at(i,j-1)) / dy;
    }

    // Clamp value to [lo, hi]
    static double clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

/**
 * @file poly_mesh.h
 * @brief Face-based unstructured 2D mesh (OpenFOAM-style owner/neighbour).
 * @author liutao
 * @date 2026-06-23
 *
 * Minimal cell-centred polygonal mesh for the unstructured finite-volume
 * pressure-Poisson solver. Each internal face couples an owner and a neighbour
 * cell; boundary faces have nb < 0. The face area vector Sf points owner->nb
 * (outward on boundary faces) with |Sf| equal to the edge length (2D).
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

namespace ufvm {

struct Vec2 {
    double x = 0, y = 0;
};
inline Vec2 operator-(Vec2 a, Vec2 b) {
    return {a.x - b.x, a.y - b.y};
}
inline Vec2 operator+(Vec2 a, Vec2 b) {
    return {a.x + b.x, a.y + b.y};
}
inline Vec2 operator*(double s, Vec2 a) {
    return {s * a.x, s * a.y};
}
inline double dot(Vec2 a, Vec2 b) {
    return a.x * b.x + a.y * b.y;
}

struct Face {
    int owner, nb; ///< nb < 0 => boundary face
    Vec2 Sf;       ///< area vector (length * normal), oriented owner -> nb
    Vec2 Cf;       ///< face centre
};

struct PolyMesh {
    int n_cells = 0;
    std::vector<Vec2> centroid;
    std::vector<double> vol; ///< cell area (2D)
    std::vector<Face> faces;
};

/// Build a face-based mesh from polygon cells (vertex-index lists) + vertices.
inline PolyMesh build(const std::vector<Vec2>& pts, const std::vector<std::vector<int>>& cells) {
    PolyMesh m;
    m.n_cells = static_cast<int>(cells.size());
    m.centroid.resize(m.n_cells);
    m.vol.resize(m.n_cells);
    for (int c = 0; c < m.n_cells; ++c) {
        const auto& vs = cells[c];
        double A       = 0;
        Vec2 cen{0, 0};
        for (size_t k = 0; k < vs.size(); ++k) {
            Vec2 p0 = pts[vs[k]], p1 = pts[vs[(k + 1) % vs.size()]];
            double cross = p0.x * p1.y - p1.x * p0.y;
            A += cross;
            cen.x += (p0.x + p1.x) * cross;
            cen.y += (p0.y + p1.y) * cross;
        }
        A *= 0.5;
        m.vol[c]      = std::fabs(A);
        m.centroid[c] = {cen.x / (6 * A), cen.y / (6 * A)};
    }
    std::map<std::pair<int, int>, std::vector<std::pair<int, std::pair<int, int>>>> edge;
    for (int c = 0; c < m.n_cells; ++c) {
        const auto& vs = cells[c];
        for (size_t k = 0; k < vs.size(); ++k) {
            int a = vs[k], b = vs[(k + 1) % vs.size()];
            auto key = std::minmax(a, b);
            edge[{key.first, key.second}].push_back({c, {a, b}});
        }
    }
    for (auto& kv : edge) {
        Vec2 A = pts[kv.first.first], B = pts[kv.first.second];
        Vec2 Cf = 0.5 * (A + B);
        Vec2 e  = B - A;
        Vec2 nrm{e.y, -e.x}; // |nrm| = edge length
        Face f;
        f.Cf = Cf;
        if (kv.second.size() == 2) {
            int c0 = kv.second[0].first, c1 = kv.second[1].first;
            f.owner = std::min(c0, c1);
            f.nb    = std::max(c0, c1);
        } else {
            f.owner = kv.second[0].first;
            f.nb    = -1;
        }
        Vec2 ref = (f.nb >= 0 ? m.centroid[f.nb] : Cf) - m.centroid[f.owner];
        if (dot(nrm, ref) < 0)
            nrm = {-nrm.x, -nrm.y};
        f.Sf = nrm;
        m.faces.push_back(f);
    }
    return m;
}

/// Triangulated unit square [0,1]^2: n x n quads, each split into 2 triangles.
inline PolyMesh make_rect_tri(int n) {
    std::vector<Vec2> pts;
    auto id = [&](int i, int j) { return j * (n + 1) + i; };
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i)
            pts.push_back({(double)i / n, (double)j / n});
    std::vector<std::vector<int>> cells;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            int a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
            cells.push_back({a, b, c});
            cells.push_back({a, c, d});
        }
    return build(pts, cells);
}

} // namespace ufvm

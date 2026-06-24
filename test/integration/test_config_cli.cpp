// Unit tests for the configuration / CLI parsing module.
//
// Exercises the PUBLIC CPU-only API:
//   config::parse_cli(argc, argv)         — 2D registry path (returns optional)
//   scene3d::build_config(argc, argv)     — 3D preset path (throws on bad name)
//   Config::dget / iget / sget            — typed `extra` accessors
//
// Both entry points: apply scenario presets first, then re-apply the INI/CLI
// assignments so any user-set field wins.
#include "../test_utils.h"
#include "config/cli.h"
#include "config/config.h"
#include "simulator/scene_3d.h"

#include <string>
#include <vector>

// Turn {"prog", "scenario=karman", ...} into a mutable argv for parse_cli.
// The pointers stay valid for the lifetime of `storage`.
struct Argv {
    std::vector<std::string> storage;
    std::vector<char*>       ptrs;
    Argv(std::vector<std::string> args) : storage(std::move(args)) {
        for (auto& s : storage)
            ptrs.push_back(s.data());
    }
    int    argc() { return static_cast<int>(ptrs.size()); }
    char** argv() { return ptrs.data(); }
};

int main() {
    test_header("config / CLI parsing");

    // ── 2D karman: scenario presets are applied (KarmanScenario::configure) ──
    {
        Argv a({"prog", "scenario=karman", "NX=128"});
        auto cfg = config::parse_cli(a.argc(), a.argv());
        check(cfg.has_value(), "parse_cli(scenario=karman) returns a Config");
        check(cfg->scenario == "karman", "scenario stored as karman");
        check_approx(cfg->Lx, 4.0, 1e-12, "karman preset Lx = 4.0");
        check_approx(cfg->Ly, 1.0, 1e-12, "karman preset Ly = 1.0");
        // NY = max(NX/4, 16) = max(32, 16) = 32
        check(cfg->NY == 32, "karman preset NY = NX/4 = 32");
        // dt = 0.5 * (Lx/NX) / U_inf = 0.5 * (4/128) / 1 = 0.015625
        check_approx(cfg->dt, 0.5 * (4.0 / 128.0) / 1.0, 1e-12, "karman preset CFL-derived dt");
        check(cfg->time_integrator == "chorin", "karman preset forces chorin integrator");
        check(cfg->solver == "pcg", "karman preset forces pcg solver");
        check(cfg->out_dir == "output_karman", "karman preset out_dir");
    }

    // ── CLI overrides apply AFTER presets ──
    {
        // dt=0.001 must beat the karman CFL-derived dt.
        Argv a({"prog", "scenario=karman", "NX=128", "dt=0.001"});
        auto cfg = config::parse_cli(a.argc(), a.argv());
        check(cfg.has_value(), "parse_cli(karman, dt override) returns a Config");
        check_approx(cfg->dt, 0.001, 1e-12, "CLI dt=0.001 overrides scenario-derived dt");
        // solver override beats the preset's pcg.
        Argv b({"prog", "scenario=karman", "NX=128", "solver=cg"});
        auto cfg2 = config::parse_cli(b.argc(), b.argv());
        check(cfg2.has_value() && cfg2->solver == "cg", "CLI solver=cg overrides karman preset pcg");
    }

    // ── default solver + default_solve_iters (smoke sets neither solver) ──
    {
        // smoke::configure does NOT set solver → build_config default "jacobi".
        Argv a({"prog", "scenario=smoke", "NX=64"});
        auto cfg = config::parse_cli(a.argc(), a.argv());
        check(cfg.has_value(), "parse_cli(scenario=smoke) returns a Config");
        check(cfg->solver == "jacobi", "default solver is jacobi when scenario leaves it unset");
        // default_solve_iters("jacobi") == 2000 (stationary relaxation).
        check(cfg->solve_iters == 2000, "jacobi → default_solve_iters = 2000");
        check(cfg->time_integrator == "chorin", "smoke leaves default integrator chorin");
        // Krylov default: explicit pcg → 50 iters.
        Argv b({"prog", "scenario=smoke", "NX=64", "solver=pcg"});
        auto cfg2 = config::parse_cli(b.argc(), b.argv());
        check(cfg2.has_value() && cfg2->solve_iters == 50,
              "pcg → default_solve_iters = 50 (Krylov)");
    }

    // ── unknown keys land in `extra`, readable via typed accessors ──
    {
        Argv a({"prog", "scenario=smoke", "NX=64", "my_double=2.5", "my_int=7", "my_str=hello"});
        auto cfg = config::parse_cli(a.argc(), a.argv());
        check(cfg.has_value(), "parse_cli with unknown keys still succeeds");
        check(cfg->extra.count("my_double") == 1, "unknown key my_double stored in extra");
        check_approx(cfg->dget("my_double", -1.0), 2.5, 1e-12, "dget reads extra double");
        check(cfg->iget("my_int", -1) == 7, "iget reads extra int");
        check(cfg->sget("my_str", "def") == "hello", "sget reads extra string");
        // Defaults when the key is absent.
        check_approx(cfg->dget("absent", 9.0), 9.0, 1e-12, "dget returns default when key absent");
        check(cfg->iget("absent", -3) == -3, "iget returns default when key absent");
        check(cfg->sget("absent", "fallback") == "fallback",
              "sget returns default when key absent");
    }

    // ── unknown scenario: parse_cli reports via std::nullopt (no throw) ──
    {
        Argv a({"prog", "scenario=does_not_exist"});
        auto cfg = config::parse_cli(a.argc(), a.argv());
        check(!cfg.has_value(), "parse_cli(unknown scenario) returns nullopt");
    }

    // ── 3D presets via scene3d::build_config (collision_paper) ──
    {
        Argv a({"prog", "scenario=collision_paper"});
        Config cfg = scene3d::build_config(a.argc(), a.argv());
        check(cfg.dim == 3, "collision_paper preset dim = 3");
        check(cfg.NX == 128, "collision_paper preset NX = 128");
        check(cfg.NY == 256, "collision_paper preset NY = 256");
        check(cfg.NZ == 256, "collision_paper preset NZ = 256");
        check_approx(cfg.dt, 1e-4, 1e-12, "collision_paper preset dt = 1e-4 (variant-D)");
        check_approx(cfg.Re, 0.0, 1e-12, "collision_paper preset Re = 0 (inviscid)");
        check(cfg.lfm_bfecc_clamp, "collision_paper preset enables BFECC clamp");
        check(cfg.lfm_cycle_steps == 5, "collision_paper preset lfm_cycle_steps = 5");
        check(cfg.time_integrator == "lfm", "3D preset uses lfm integrator");
    }

    // ── 3D CLI override after preset ──
    {
        Argv a({"prog", "scenario=collision_paper", "NX=64", "dt=0.01"});
        Config cfg = scene3d::build_config(a.argc(), a.argv());
        check(cfg.NX == 64, "CLI NX=64 overrides collision_paper preset 128");
        check_approx(cfg.dt, 0.01, 1e-12, "CLI dt overrides collision_paper preset");
        // NY/NZ untouched by the override remain at the preset.
        check(cfg.NY == 256, "un-overridden NY stays at preset 256");
    }

    // ── 3D unknown scenario: scene3d::build_config THROWS ──
    {
        bool threw = false;
        Argv a({"prog", "scenario=no_such_3d_scene"});
        try {
            scene3d::build_config(a.argc(), a.argv());
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "scene3d::build_config(unknown scenario) throws");
    }

    return test_summary();
}

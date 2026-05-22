#pragma once
#include <iostream>
#include <cmath>
#include <string>
#include <chrono>

static int g_passed = 0;
static int g_failed = 0;

inline void test_header(const std::string& name) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << name << "\n";
    std::cout << std::string(60, '=') << "\n";
}

inline void check(bool cond, const std::string& desc) {
    if (cond) { g_passed++; std::cout << "  [PASS] " << desc << "\n"; }
    else      { g_failed++; std::cout << "  [FAIL] " << desc << "\n"; }
}

inline void check_approx(double a, double b, double tol, const std::string& desc) {
    bool ok = std::abs(a - b) < tol;
    if (ok) { g_passed++; std::cout << "  [PASS] " << desc << " (" << a << " ~ " << b << ")\n"; }
    else    { g_failed++; std::cout << "  [FAIL] " << desc << " (|" << a << " - " << b << "| = " << std::abs(a-b) << " >= " << tol << ")\n"; }
}

inline int test_summary() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  Passed: " << g_passed << "  Failed: " << g_failed << "\n";
    std::cout << std::string(60, '=') << "\n";
    return g_failed > 0 ? 1 : 0;
}

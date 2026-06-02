#include "config/cli.h"
#include "simulator/factory.h"
#include "simulator/runner.h"

int main(int argc, char* argv[]) {
    auto cfg = config::parse_cli(argc, argv);
    if (!cfg)
        return 1;

    auto sim = SimulatorFactory::create(*cfg);
    sim::run(*sim, *cfg);
    return 0;
}

#include "config/cli.h"
#include "simulator/simulator.h"
#include "solver/factory.h"

int main(int argc, char* argv[]) {
    auto cfg = config::parse_cli(argc, argv);
    if (!cfg)
        return 1;

    ChorinSimulator sim(*cfg, Factory::create(cfg->solver));
    sim.run();
    return 0;
}

#pragma once
#include <map>
#include <string>
#include <vector>
#include <signal.h>

extern volatile sig_atomic_t stop_signal;
void install_signal_handlers();
void pause_tick();
bool port_ready(int port);

struct Command {
    std::string id, executable, cwd;
    std::vector<std::string> args;
    std::map<std::string, std::string> env;
    std::vector<std::string> unset_env;
};
struct Child { int pid; bool running = true; int result = 0; };
class Processes {
public:
    std::vector<Child> children;
    explicit Processes(int grace_ms = 1500) : grace_ms_(grace_ms) {}
    ~Processes();
    size_t start(const Command&, bool quiet_stdout = false);
    void poll();
private:
    int grace_ms_;
};
int run_probe(const Command&, int timeout_ms);

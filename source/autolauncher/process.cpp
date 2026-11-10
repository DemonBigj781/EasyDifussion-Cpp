#include "process.hpp"
#include "platform.hpp"
#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <spawn.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <time.h>
#include <unistd.h>

extern char** environ;
volatile sig_atomic_t stop_signal = 0;
using Clock = std::chrono::steady_clock;
static void stopped(int signal) { stop_signal = signal; }

void install_signal_handlers() {
    struct sigaction sa{};
    sa.sa_handler = stopped;
    sigemptyset(&sa.sa_mask);
    for (int s : {SIGINT, SIGTERM, SIGHUP}) {
        if (sigaction(s, &sa, nullptr)) throw std::runtime_error("Cannot install signal handler");
    }
}
void pause_tick() {
    timespec delay{0, 20000000};
    nanosleep(&delay, nullptr);
}
static void spawn_check(int error) {
    if (error) throw std::runtime_error(std::string("Process setup: ") + std::strerror(error));
}

size_t Processes::start(const Command& command, bool quiet_stdout) {
    if (stop_signal) throw std::runtime_error("Launch interrupted");
    // Allocate bookkeeping before a child exists, so allocation failure cannot
    // leave an untracked process behind.
    children.reserve(children.size() + 1);
    std::vector<std::string> args{command.executable};
    args.insert(args.end(), command.args.begin(), command.args.end());
    std::map<std::string, std::string> env;
    for (char** p = environ; *p; ++p) {
        std::string item(*p);
        auto eq = item.find('=');
        if (eq != std::string::npos) env[item.substr(0, eq)] = item.substr(eq + 1);
    }
    for (const auto& key : command.unset_env) env.erase(key);
    for (const auto& [key, value] : command.env) env[key] = value;
    std::vector<std::string> env_strings;
    for (const auto& [key, value] : env) env_strings.push_back(key + "=" + value);
    std::vector<char*> argv, envp;
    for (auto& arg : args) argv.push_back(arg.data());
    for (auto& value : env_strings) envp.push_back(value.data());
    argv.push_back(nullptr);
    envp.push_back(nullptr);

    posix_spawnattr_t attr;
    spawn_check(posix_spawnattr_init(&attr));
    posix_spawn_file_actions_t actions;
    int error = posix_spawn_file_actions_init(&actions);
    if (error) { posix_spawnattr_destroy(&attr); spawn_check(error); }
    pid_t pid = -1;
    try {
        sigset_t defaults, mask;
        sigemptyset(&mask);
        sigemptyset(&defaults);
        for (int s : {SIGINT, SIGTERM, SIGHUP, SIGPIPE}) sigaddset(&defaults, s);
        spawn_check(posix_spawnattr_setsigmask(&attr, &mask));
        spawn_check(posix_spawnattr_setsigdefault(&attr, &defaults));
        spawn_check(posix_spawnattr_setpgroup(&attr, 0));
        spawn_check(posix_spawnattr_setflags(&attr,
            POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK));
        spawn_check(posix_spawn_file_actions_addchdir_np(&actions, command.cwd.c_str()));
        if (quiet_stdout) {
            spawn_check(posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO, STDOUT_FILENO));
            spawn_check(posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0));
        }
        error = posix_spawn(&pid, command.executable.c_str(), &actions, &attr, argv.data(), envp.data());
    } catch (...) {
        posix_spawn_file_actions_destroy(&actions);
        posix_spawnattr_destroy(&attr);
        throw;
    }
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);
    if (error) throw std::runtime_error("Cannot start " + command.id + ": " + std::strerror(error));
    children.push_back({pid, true, 0});
    return children.size() - 1;
}

void Processes::poll() {
    for (auto& child : children) {
        if (!child.running) continue;
        int status = 0;
        int result = waitpid(child.pid, &status, WNOHANG);
        if (result == child.pid) {
            child.running = false;
            child.result = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
        } else if (result < 0 && errno != EINTR) {
            child.running = false;
            child.result = 125;
        }
    }
}

Processes::~Processes() {
    if (children.empty()) return;
    auto send = [&](int signal) {
        for (const auto& child : children) {
            if (!on_windows() || child.running) kill(on_windows() ? child.pid : -child.pid, signal);
        }
    };
    send(stop_signal ? stop_signal : SIGTERM);
    auto deadline = Clock::now() + std::chrono::milliseconds(grace_ms_);
    while (Clock::now() < deadline) {
        poll();
        bool alive = false;
        for (const auto& child : children) {
            alive |= on_windows() ? child.running : kill(-child.pid, 0) == 0;
        }
        if (!alive) break;
        pause_tick();
    }
    send(SIGKILL);
    deadline = Clock::now() + std::chrono::milliseconds(500);
    while (Clock::now() < deadline) {
        poll();
        bool alive = false;
        for (const auto& child : children) alive |= child.running;
        if (!alive) break;
        pause_tick();
    }
}

int run_probe(const Command& command, int timeout_ms) {
    Processes processes(100);
    processes.start(command, true);
    auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    do {
        processes.poll();
        if (!processes.children[0].running) return processes.children[0].result;
        if (stop_signal) return 128 + stop_signal;
        pause_tick();
    } while (Clock::now() < deadline);
    return 124;
}

bool port_ready(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) { close(fd); return false; }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int result = connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    if (result < 0 && errno == EINPROGRESS) {
        pollfd pfd{fd, POLLOUT, 0};
        if (::poll(&pfd, 1, 20) > 0) {
            int error = 0;
            socklen_t size = sizeof(error);
            result = getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size) == 0 && error == 0 ? 0 : -1;
        }
    }
    close(fd);
    return result == 0;
}

/* SPDX-License-Identifier: MIT */
#ifndef _COSMO_SOURCE
#define _COSMO_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#include "shell.h"
#ifdef __COSMOPOLITAN__
#include <libc/dce.h>
#endif
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utime.h>

#define SHELL_MAX_INPUT (1024u * 1024u)
#define SHELL_MAX_ARGS 256
#define SHELL_MAX_COMMANDS 4096
#define SHELL_MAX_SOURCE_DEPTH 8

enum connector { NEXT_ALWAYS, NEXT_SUCCESS, NEXT_FAILURE };
struct command { int argc; char **argv; enum connector next; };
struct program { size_t count; struct command *commands; };
struct shell {
    char *directory;
    const struct cosmo_shell_services *services;
    bool exiting;
    int exit_status;
    int depth;
};

static int failure(const char *operation, const char *path) {
    fprintf(stderr, "%s: %s: %s\n", operation, path, strerror(errno));
    return 1;
}

static void free_program(struct program *program) {
    for (size_t i = 0; i < program->count; ++i) {
        for (int j = 0; j < program->commands[i].argc; ++j)
            free(program->commands[i].argv[j]);
        free(program->commands[i].argv);
    }
    free(program->commands);
    *program = (struct program){0};
}

static int append_command(struct program *program) {
    if (program->count == SHELL_MAX_COMMANDS) return -1;
    struct command *next = realloc(program->commands,
                                    (program->count + 1) * sizeof(*next));
    if (!next) return -1;
    program->commands = next;
    next[program->count++] = (struct command){0};
    return 0;
}

static int append_word(struct command *command, const char *word, size_t bytes) {
    if (command->argc == SHELL_MAX_ARGS) return -1;
    char *copy = malloc(bytes + 1);
    if (!copy) return -1;
    memcpy(copy, word, bytes); copy[bytes] = 0;
    char **args = realloc(command->argv, (size_t)(command->argc + 2) * sizeof(*args));
    if (!args) { free(copy); return -1; }
    command->argv = args;
    args[command->argc++] = copy;
    args[command->argc] = NULL;
    return 0;
}

/* Validate a whole input before executing it. Quoted operators are literal;
   a bare pipe, redirection or background operator is an explicit syntax error.
   Backslashes before ordinary letters remain literal (Windows paths). */
static int parse(const char *text, struct program *program) {
    size_t size = strlen(text), used = 0;
    if (size > SHELL_MAX_INPUT) goto invalid;
    char *word = malloc(size + 1);
    if (!word || append_command(program)) { free(word); goto exhausted; }
    char quote = 0;
    bool started = false, requires_command = false;
    for (size_t i = 0; i <= size; ++i) {
        const char c = text[i];
        if (quote) {
            if (!c) { free(word); goto invalid; }
            if (c == quote) { quote = 0; continue; }
            if (quote == '"' && c == '\\' &&
                (text[i + 1] == '"' || text[i + 1] == '\\')) {
                word[used++] = text[++i];
            } else word[used++] = c;
            continue;
        }
        if (c == '\'' || c == '"') { quote = c; started = true; continue; }
        if (c == '\\' && text[i + 1] &&
            strchr("\\\"' \t#;&|<>", text[i + 1])) {
            word[used++] = text[++i]; started = true; continue;
        }
        if (c == '#' && !started) {
            while (text[i] && text[i] != '\n') ++i;
            --i;
            continue;
        }
        const bool separator = !c || c == '\n' || c == ';' || c == '&' || c == '|';
        if (c == '<' || c == '>') { free(word); goto invalid; }
        if (separator || c == ' ' || c == '\t' || c == '\r') {
            struct command *command = &program->commands[program->count - 1];
            if (started) {
                if (append_word(command, word, used)) { free(word); goto exhausted; }
                started = false; used = 0; requires_command = false;
            }
            if (!separator) continue;
            if (c == '&' || c == '|') {
                if (text[i + 1] != c || !command->argc) { free(word); goto invalid; }
                command->next = c == '&' ? NEXT_SUCCESS : NEXT_FAILURE;
                ++i; requires_command = true;
                if (append_command(program)) { free(word); goto exhausted; }
            } else if (command->argc && c) {
                if (append_command(program)) { free(word); goto exhausted; }
            } else if (requires_command && c != '\n') {
                free(word); goto invalid;
            }
            if (!c) break;
        } else { word[used++] = c; started = true; }
    }
    free(word);
    if (requires_command) goto invalid;
    if (program->count && !program->commands[program->count - 1].argc)
        --program->count;
    return 0;
invalid:
    fputs("shell: invalid syntax (check quotes, &&/|| operands, and unsupported |, &, < or >)\n", stderr);
    free_program(program);
    return 2;
exhausted:
    fputs("shell: command, argument or allocation limit exceeded\n", stderr);
    free_program(program);
    return 2;
}

static char *join_path(const char *left, const char *right) {
    const size_t a = strlen(left), b = strlen(right);
    if (a > SIZE_MAX - b - 2) return NULL;
    char *value = malloc(a + b + 2);
    if (value) snprintf(value, a + b + 2, "%s/%s", left, right);
    return value;
}

static char *resolve_path(const struct shell *shell, const char *text) {
    if (!*text) { errno = EINVAL; return NULL; }
    char *path = strdup(text);
    if (!path) return NULL;
#ifdef __COSMOPOLITAN__
    if (IsWindows()) {
        for (char *p = path; *p; ++p) if (*p == '\\') *p = '/';
        if (isalpha((unsigned char)path[0]) && path[1] == ':') {
            if (path[2] != '/') { free(path); errno = EINVAL; return NULL; }
            char drive = path[0]; path[0] = '/'; path[1] = drive;
        }
    }
#endif
    if (*path == '/') return path;
    char *absolute = join_path(shell->directory, path);
    free(path);
    return absolute;
}

static int copy_bytes(FILE *source, FILE *destination) {
    unsigned char block[32768];
    for (;;) {
        size_t count = fread(block, 1, sizeof(block), source);
        if (count && fwrite(block, 1, count, destination) != count) return -1;
        if (count != sizeof(block)) return ferror(source) ? -1 : 0;
    }
}

static int compare_names(const void *a, const void *b) {
    return strcmp(*(const char * const *)a, *(const char * const *)b);
}

static int list_path(const char *path) {
    struct stat info;
    if (stat(path, &info)) return failure("ls", path);
    if (!S_ISDIR(info.st_mode)) return puts(path) == EOF ? 1 : 0;
    DIR *directory = opendir(path);
    if (!directory) return failure("ls", path);
    char **names = NULL;
    size_t count = 0;
    int result = 0;
    for (;;) {
        errno = 0;
        struct dirent *entry = readdir(directory);
        if (!entry) { if (errno) result = failure("ls", path); break; }
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        char **next = realloc(names, (count + 1) * sizeof(*names));
        if (!next) { result = failure("ls", path); break; }
        names = next;
        if (!(names[count] = strdup(entry->d_name))) { result = failure("ls", path); break; }
        ++count;
    }
    if (closedir(directory)) result = failure("ls", path);
    if (count) qsort(names, count, sizeof(*names), compare_names);
    for (size_t i = 0; i < count; ++i) {
        if (!result && puts(names[i]) == EOF) result = 1;
        free(names[i]);
    }
    free(names);
    return result;
}

static int make_directory(char *path, bool parents) {
    if (!parents) return mkdir(path, 0777) ? failure("mkdir", path) : 0;
    size_t length = strlen(path);
    for (size_t i = 1; i <= length; ++i) {
        if (path[i] && path[i] != '/') continue;
        const char c = path[i]; path[i] = 0;
        if (mkdir(path, 0777) && errno != EEXIST) return failure("mkdir", path);
        struct stat info;
        if (stat(path, &info)) return failure("mkdir", path);
        if (!S_ISDIR(info.st_mode)) { errno = ENOTDIR; return failure("mkdir", path); }
        path[i] = c;
    }
    return 0;
}

static int copy_file(const char *source, const char *destination) {
    struct stat from, to;
    if (stat(source, &from)) return failure("cp", source);
    if (!S_ISREG(from.st_mode)) { errno = EINVAL; return failure("cp (regular files only)", source); }
    if (!stat(destination, &to) && from.st_dev == to.st_dev && from.st_ino == to.st_ino) {
        errno = EINVAL; return failure("cp (same file)", destination);
    }
    FILE *input = fopen(source, "rb");
    if (!input) return failure("cp", source);
    size_t bytes = strlen(destination) + sizeof(".tmp.XXXXXX");
    char *temporary = malloc(bytes);
    if (!temporary) { fclose(input); return failure("cp", destination); }
    snprintf(temporary, bytes, "%s.tmp.XXXXXX", destination);
    int fd = mkstemp(temporary);
    const bool created = fd >= 0;
    FILE *output = fd < 0 ? NULL : fdopen(fd, "wb");
    int result = 1;
    if (!output) { if (fd >= 0) close(fd); failure("cp", destination); goto done; }
    bool ok = !copy_bytes(input, output);
    if (fflush(output)) ok = false;
    if (fsync(fileno(output))) ok = false;
    if (fclose(output)) ok = false;
    if (!ok) { failure("cp", destination); goto done; }
    if (rename(temporary, destination)) { failure("cp", destination); goto done; }
    result = 0;
done:
    fclose(input);
    if (result && created) unlink(temporary);
    free(temporary);
    return result;
}

static int run_text(struct shell *shell, const char *text);
static int run_file(struct shell *shell, const char *filename) {
    if (shell->depth >= SHELL_MAX_SOURCE_DEPTH) {
        fputs("shell: source nesting limit exceeded\n", stderr); return 2;
    }
    char *path = resolve_path(shell, filename);
    if (!path) return failure("source", filename);
    FILE *file = fopen(path, "rb");
    if (!file) { int result = failure("source", path); free(path); return result; }
    char *text = malloc(SHELL_MAX_INPUT + 2);
    if (!text) { fclose(file); free(path); return failure("source", filename); }
    const size_t size = fread(text, 1, SHELL_MAX_INPUT + 1, file);
    int result = 2;
    if (ferror(file)) failure("source", path);
    else if (size > SHELL_MAX_INPUT || memchr(text, 0, size))
        fputs("shell: scripts must be NUL-free text of at most 1 MiB\n", stderr);
    else {
        text[size] = 0; ++shell->depth;
        result = run_text(shell, text);
        --shell->depth;
    }
    fclose(file); free(path); free(text);
    return result;
}

static void help(void) {
    puts("Easy Diffusion portable shell\n"
         "  shell [-c SCRIPT | --file PATH]   Otherwise read stdin\n"
         "  help, echo [TEXT...], pwd, cd PATH, ls [PATH...]\n"
         "  mkdir [-p] PATH..., touch FILE..., cat FILE...\n"
         "  cp SOURCE DEST, mv SOURCE DEST, rm [-f] FILE..., rmdir DIR...\n"
         "  test -e|-f|-d PATH, true, false, source FILE, exit [STATUS]\n"
         "Syntax: single/double quotes, comments (#), newlines, ;, &&, ||.\n"
         "Scripts stop after an unsuccessful command chain; || can handle it.\n"
         "Arguments are literal: no glob/variable expansion or host processes.\n"
         "cd changes only this shell's directory; HTTP/model paths stay fixed.\n"
         "Application commands:");
}

static int dispatch(struct shell *shell, int argc, char **argv) {
    const char *name = argv[0];
    if (!strcmp(name, "help")) {
        if (argc != 1) goto usage;
        help();
        if (shell->services && shell->services->dispatch)
            shell->services->dispatch(shell->services->context, shell->directory, argc, argv, stdout, stderr);
        return 0;
    }
    if (!strcmp(name, "echo")) {
        for (int i = 1; i < argc; ++i) {
            if ((i > 1 && putchar(' ') == EOF) || fputs(argv[i], stdout) == EOF) return 1;
        }
        return putchar('\n') == EOF ? 1 : 0;
    }
    if (!strcmp(name, "true") || !strcmp(name, "false")) {
        if (argc != 1) goto usage;
        return !strcmp(name, "false");
    }
    if (!strcmp(name, "exit")) {
        if (argc > 2) goto usage;
        unsigned long value = 0;
        if (argc == 2) {
            char *end; errno = 0;
            value = strtoul(argv[1], &end, 10);
            if (errno || !*argv[1] || *end || value > 255 || !isdigit((unsigned char)argv[1][0])) goto usage;
        }
        shell->exiting = true; shell->exit_status = (int)value;
        return (int)value;
    }
    if (!strcmp(name, "pwd")) {
        if (argc != 1) goto usage;
        return puts(shell->directory) == EOF ? 1 : 0;
    }
    if (!strcmp(name, "source")) {
        if (argc != 2) goto usage;
        return run_file(shell, argv[1]);
    }
    if (!strcmp(name, "cd")) {
        if (argc != 2) goto usage;
        char *path = resolve_path(shell, argv[1]);
        if (!path) return failure(name, argv[1]);
        char *actual = realpath(path, NULL);
        struct stat info;
        int result = 0;
        if (!actual || stat(actual, &info)) result = failure(name, path);
        else if (!S_ISDIR(info.st_mode)) { errno = ENOTDIR; result = failure(name, path); }
        else { free(shell->directory); shell->directory = actual; actual = NULL; }
        free(actual); free(path);
        return result;
    }
    if (!strcmp(name, "test")) {
        if (argc != 3 || (strcmp(argv[1], "-e") && strcmp(argv[1], "-f") && strcmp(argv[1], "-d"))) goto usage;
        char *path = resolve_path(shell, argv[2]);
        if (!path) return failure(name, argv[2]);
        struct stat info;
        int result = stat(path, &info) ? 1 :
            !strcmp(argv[1], "-f") ? !S_ISREG(info.st_mode) :
            !strcmp(argv[1], "-d") ? !S_ISDIR(info.st_mode) : 0;
        free(path); return result;
    }
    if (!strcmp(name, "cp") || !strcmp(name, "mv")) {
        if (argc != 3) goto usage;
        char *source = resolve_path(shell, argv[1]), *destination = resolve_path(shell, argv[2]);
        int result = 1;
        if (!source || !destination) { failure(name, "path allocation"); goto moved; }
        struct stat info;
        if (!stat(destination, &info) && S_ISDIR(info.st_mode)) {
            size_t length = strlen(source);
            while (length > 1 && source[length - 1] == '/') source[--length] = 0;
            const char *base = strrchr(source, '/');
            char *inside = join_path(destination, base ? base + 1 : source);
            if (!inside) { failure(name, destination); goto moved; }
            free(destination); destination = inside;
        }
        result = !strcmp(name, "cp") ? copy_file(source, destination) :
            (rename(source, destination) ? failure(name, destination) : 0);
moved:
        free(source); free(destination); return result;
    }
    if (!strcmp(name, "ls") || !strcmp(name, "mkdir") || !strcmp(name, "cat") ||
        !strcmp(name, "touch") || !strcmp(name, "rm") || !strcmp(name, "rmdir")) {
        int first = 1;
        bool parents = false, force = false;
        if (argc > first && !strcmp(name, "mkdir") && !strcmp(argv[first], "-p")) { parents = true; ++first; }
        if (argc > first && !strcmp(name, "rm") && !strcmp(argv[first], "-f")) { force = true; ++first; }
        if (argc > first && !strcmp(argv[first], "--")) ++first;
        if (argc == first) {
            if (!strcmp(name, "ls")) return list_path(shell->directory);
            goto usage;
        }
        for (int i = first; i < argc; ++i) {
            if (argv[i][0] == '-' && first == 1) goto usage;
            char *path = resolve_path(shell, argv[i]);
            if (!path) return failure(name, argv[i]);
            int result = 0;
            if (!strcmp(name, "ls")) result = list_path(path);
            else if (!strcmp(name, "mkdir")) result = make_directory(path, parents);
            else if (!strcmp(name, "rm")) {
                if (unlink(path) && !(force && errno == ENOENT)) result = failure(name, path);
            } else if (!strcmp(name, "rmdir")) {
                if (rmdir(path)) result = failure(name, path);
            } else if (!strcmp(name, "touch")) {
                int fd = open(path, O_CREAT | O_WRONLY, 0666);
                if (fd < 0) result = failure(name, path);
                else { if (close(fd) || utime(path, NULL)) result = failure(name, path); }
            } else {
                FILE *file = fopen(path, "rb");
                if (!file) result = failure(name, path);
                else { if (copy_bytes(file, stdout)) result = failure(name, path); fclose(file); }
            }
            free(path);
            if (result) return result;
        }
        return 0;
    }
    if (shell->services && shell->services->dispatch) {
        int result = shell->services->dispatch(shell->services->context, shell->directory,
                                                argc, argv, stdout, stderr);
        if (result != 127) return result;
    }
    fprintf(stderr, "shell: unknown command: %s (use help)\n", name);
    return 127;
usage:
    fprintf(stderr, "shell: invalid arguments for %s (use help)\n", name);
    return 2;
}

static int run_text(struct shell *shell, const char *text) {
    struct program program = {0};
    int result = parse(text, &program);
    if (result) return result;
    for (size_t i = 0; i < program.count && !shell->exiting; ++i) {
        const enum connector previous = i ? program.commands[i - 1].next : NEXT_ALWAYS;
        if (previous == NEXT_ALWAYS || (previous == NEXT_SUCCESS && !result) ||
            (previous == NEXT_FAILURE && result))
            result = dispatch(shell, program.commands[i].argc, program.commands[i].argv);
        if (program.commands[i].next == NEXT_ALWAYS && result) break;
    }
    free_program(&program);
    if (fflush(stdout)) result = 1;
    return shell->exiting ? shell->exit_status : result;
}

int cosmo_shell_run(int argc, char **argv, const struct cosmo_shell_services *services) {
    struct shell shell = {.services = services};
    if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
        shell.directory = getcwd(NULL, 0);
        if (!shell.directory) return failure("shell", "working directory");
        char *command[] = {"help", NULL};
        int result = dispatch(&shell, 1, command);
        free(shell.directory); return result;
    }
    if (argc != 1 && (argc != 3 || (strcmp(argv[1], "-c") && strcmp(argv[1], "--file")))) {
        fputs("Usage: shell [-c SCRIPT | --file PATH]\n", stderr); return 2;
    }
    shell.directory = getcwd(NULL, 0);
    if (!shell.directory) return failure("shell", "working directory");
    int result = 0;
    if (argc == 3) result = !strcmp(argv[1], "-c") ? run_text(&shell, argv[2]) : run_file(&shell, argv[2]);
    else {
        bool interactive = isatty(fileno(stdin)) && isatty(fileno(stdout));
        if (interactive) puts("Easy Diffusion shell. Type help for commands, exit to close.");
        char *line = malloc(SHELL_MAX_INPUT + 2);
        if (!line) result = failure("shell", "input allocation");
        while (line && !shell.exiting) {
            if (interactive) { fputs("ed> ", stdout); fflush(stdout); }
            size_t length = 0;
            bool invalid = false, received = false;
            int c;
            while ((c = fgetc(stdin)) != EOF) {
                received = true;
                if (!c || length >= SHELL_MAX_INPUT) invalid = true;
                else line[length++] = (char)c;
                if (c == '\n') break;
            }
            if (ferror(stdin)) { result = failure("shell", "stdin"); break; }
            if (!received) break;
            line[length] = 0;
            if (invalid) {
                fputs("shell: input must be NUL-free text of at most 1 MiB per line\n", stderr); result = 2;
            } else result = run_text(&shell, line);
            if (result && !interactive) break;
        }
        free(line);
    }
    free(shell.directory);
    return shell.exiting ? shell.exit_status : result;
}

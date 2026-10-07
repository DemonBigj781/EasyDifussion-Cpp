/* SPDX-License-Identifier: MIT
 * Links the production shell; only the application-service dispatcher is fake.
 * No model, HTTP service, graphics driver or subprocess launcher is involved. */
#define _POSIX_C_SOURCE 200809L
#include "shell.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct probe {
    char *initial_directory;
    char *mutable_input;
    unsigned calls;
};

static void quoted(FILE *output, const char *text) {
    fputc('"', output);
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*p == '"' || *p == '\\') { fputc('\\', output); fputc(*p, output); }
        else if (*p < 32) fprintf(output, "\\u%04x", *p);
        else fputc(*p, output);
    }
    fputc('"', output);
}

static int dispatch(void *opaque, const char *directory, int argc, char **argv,
                    FILE *output, FILE *error) {
    struct probe *probe = opaque;
    if (strcmp(argv[0], "capture") && strcmp(argv[0], "capture-mutate-input") &&
        strcmp(argv[0], "capture-status")) return 127;
    char *actual = getcwd(NULL, 0);
    if (!actual || strcmp(actual, probe->initial_directory)) {
        free(actual); fputs("PROBE process directory changed\n", error); return 70;
    }
    free(actual);
    ++probe->calls;
    fprintf(output, "{\"call\":%u,\"directory\":", probe->calls);
    quoted(output, directory);
    fputs(",\"process_directory\":", output); quoted(output, probe->initial_directory);
    fputs(",\"argv\":[", output);
    for (int i = 0; i < argc; ++i) {
        if (i) fputc(',', output);
        quoted(output, argv[i]);
    }
    fputs("]}\n", output);
    if (!strcmp(argv[0], "capture-mutate-input")) {
        if (!probe->mutable_input) return 70;
        /* Later parsed argv must own its bytes independently of original input. */
        memset(probe->mutable_input, '#', strlen(probe->mutable_input));
    }
    if (!strcmp(argv[0], "capture-status")) {
        if (argc != 2 || strcmp(argv[1], "1")) return 70;
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    struct probe probe = {.initial_directory = getcwd(NULL, 0)};
    if (!probe.initial_directory) return 70;
    if (argc == 3 && !strcmp(argv[1], "-c")) probe.mutable_input = argv[2];
    const struct cosmo_shell_services services = {.context = &probe, .dispatch = dispatch};
    int status = cosmo_shell_run(argc, argv, &services);
    char *after = getcwd(NULL, 0);
    const int unchanged = after && !strcmp(after, probe.initial_directory);
    fprintf(stderr, "SHELL_PROBE calls=%u process_cwd_unchanged=%d\n", probe.calls, unchanged);
    free(after); free(probe.initial_directory);
    return unchanged ? status : 70;
}

/* SPDX-License-Identifier: MIT */
#ifndef COSMO_APPLICATION_SHELL_H
#define COSMO_APPLICATION_SHELL_H
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif

/* The shell has no process launcher and no application state of its own.
   Application commands use the caller's existing service context. Paths for
   those commands resolve against working_directory; cd never calls chdir().
   Return 127 only for an unrecognized command. No argv pointer is retained. */
struct cosmo_shell_services {
    void *context;
    int (*dispatch)(void *context, const char *working_directory,
                    int argc, char **argv, FILE *output, FILE *error);
};

/* argv[0] is shell. Supports -c TEXT, --file PATH, or commands from stdin.
   Scripts stop at an unhandled failed command chain. An interactive terminal
   keeps accepting input. All syntax is parsed internally on every OS. */
int cosmo_shell_run(int argc, char **argv,
                    const struct cosmo_shell_services *services);

#ifdef __cplusplus
}
#endif
#endif

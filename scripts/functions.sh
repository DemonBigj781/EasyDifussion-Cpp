# Utility functions for the Linux-only local fork.

fail() {
    echo >&2
    if [ -n "${1:-}" ]; then
        echo "ERROR: $1" >&2
    else
        echo "ERROR: Easy Diffusion could not start." >&2
    fi
    echo "Check the messages above and the local README for recovery steps." >&2
    if [ -t 0 ]; then
        read -r -p "Press Enter to continue" _
    fi
    exit 1
}

filesize() {
    stat -c "%s" "$1"
}

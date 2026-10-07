#!/usr/bin/env python3
"""Keep Asio's Linux-kernel header probe out of Cosmopolitan builds."""

from pathlib import Path
import sys


def main() -> int:
    header = Path(__file__).resolve().parents[2] / "source/sdkit3-port-source/third_party/asio/detail/config.hpp"
    lines = header.read_text().splitlines()
    marker = "// Linux: epoll, eventfd, timerfd and io_uring."
    try:
        start = lines.index(marker)
    except ValueError:
        print(f"Missing Asio Linux capability block: {header}", file=sys.stderr)
        return 1

    expected = [
        "#if defined(__linux__) && !defined(__COSMOPOLITAN__)",
        "# include <linux/version.h>",
    ]
    actual = [line.strip() for line in lines[start + 1 : start + 3]]
    if actual != expected:
        print(
            "Asio must avoid linux/version.h for Cosmopolitan even though its "
            f"compiler defines __linux__; got {actual!r}",
            file=sys.stderr,
        )
        return 1

    print("ASIO_COSMOPOLITAN_HEADERS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

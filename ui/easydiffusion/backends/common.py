import logging
import re
import subprocess
import threading
import psutil

from easydiffusion.privacy_debug import redact_log_message


_BACKEND_LOG = logging.getLogger("easydiffusion.native_backend")
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_NON_FINITE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z0-9_])|non[- ]?finite",
    re.IGNORECASE,
)


def _backend_log_level(message):
    lowered = message.lower()
    if _NON_FINITE_PATTERN.search(message) or "error" in lowered:
        return logging.ERROR
    if "warning" in lowered or "warn" in lowered:
        return logging.WARNING
    return logging.INFO


def _normalize_backend_output(message):
    """Collapse terminal redraw frames into the last visible line."""

    clean = _ANSI_ESCAPE_PATTERN.sub("", message)
    frames = [frame for frame in re.split(r"[\r\n]+", clean) if frame]
    return frames[-1] if frames else ""


def read_output(pipe, prefix=""):
    while True:
        output = pipe.readline()
        if output:
            decoded = output.decode("utf-8", errors="replace").rstrip("\r\n")
            visible_output = _normalize_backend_output(decoded)
            if not visible_output:
                continue
            safe_output = redact_log_message(visible_output)
            _BACKEND_LOG.log(_backend_log_level(visible_output), "%s%s", prefix, safe_output)
        else:
            break  # Pipe is closed, subprocess has likely exited


def run(cmds: list, cwd=None, env=None, stream_output=True, wait=True, output_prefix=""):
    p = subprocess.Popen(cmds, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if stream_output:
        output_thread = threading.Thread(target=read_output, args=(p.stdout, output_prefix))
        output_thread.start()

    if wait:
        p.wait()

    return p


# https://stackoverflow.com/a/25134985
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

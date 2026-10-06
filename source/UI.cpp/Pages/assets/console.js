(() => {
  const output = document.getElementById("ui-console-output");
  const clear = document.getElementById("ui-console-clear");
  const form = document.getElementById("ui-console-form");
  const input = document.getElementById("ui-console-command");
  const run = document.getElementById("ui-console-run");
  if (!output || !clear || !form || !input || !run) return;

  const entries = [];
  const record = (level, message) => {
    entries.push(`${new Date().toLocaleTimeString()} ${level}: ${String(message)}`);
    if (entries.length > 250) entries.shift();
    output.textContent = entries.join("\n");
    output.scrollTop = output.scrollHeight;
  };
  let cwd = ".";
  const history = [];
  let historyIndex = 0;
  async function get(url) {
    const response = await fetch(url, {cache: "no-store"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
    return result;
  }
  function relativePath(argument) {
    let path = argument.trim();
    if ((path.startsWith('"') && path.endsWith('"')) || (path.startsWith("'") && path.endsWith("'"))) path = path.slice(1, -1);
    if (!path) return cwd;
    if (path.startsWith("/")) throw new Error("Paths start at the application root, not filesystem /. Use cd to return to root.");
    return `${cwd}/${path}`;
  }
  async function files(path, offset = 0) {
    return get(`/cpp-ui/api/files?${new URLSearchParams({path, offset})}`);
  }
  async function command(text) {
    const space = text.search(/\s/);
    const name = (space < 0 ? text : text.slice(0, space)).toLowerCase();
    const argument = space < 0 ? "" : text.slice(space).trim();
    if (name === "help") return "help | status | logs [1-1000] | events | clear | pwd | ls [path] | cd [path] | stat <path>\nPaths are relative to the application root; cd without a path returns there. Symlinks are followed. No shell commands or file editing. Use Up/Down for command history.";
    if (name === "clear") { entries.length = 0; output.textContent = ""; return; }
    if (name === "pwd") return cwd;
    if (name === "status") return JSON.stringify(await get("/ping"), null, 2);
    if (name === "events") {
      const result = await get("/cpp-ui/api/events");
      return result.events.map(event => `${event.timestamp} ${event.level.toUpperCase()} ${event.message}`).join("\n") || "No serving events.";
    }
    if (name === "logs") {
      if (argument && !/^\d+$/.test(argument)) throw new Error("Usage: logs [1-1000]");
      const count = argument ? Number(argument) : 100;
      if (count < 1 || count > 1000) throw new Error("Usage: logs [1-1000]");
      const result = await get(`/cpp-ui/api/logs?lines=${count}`);
      return result.lines.join("\n") || "Log is empty.";
    }
    if (name === "cd") {
      const result = await files(argument ? relativePath(argument) : ".");
      if (result.type !== "directory") throw new Error("Not a directory");
      cwd = result.path;
      return `Directory: ${cwd}`;
    }
    if (name === "stat") {
      if (!argument) throw new Error("Usage: stat <path>");
      const {entries: children, next_offset: next, ...metadata} = await files(relativePath(argument));
      return JSON.stringify(metadata, null, 2);
    }
    if (name === "ls") {
      const path = relativePath(argument);
      let offset = 0, lines = [];
      do {
        const result = await files(path, offset);
        if (result.type !== "directory") return JSON.stringify(result, null, 2);
        lines.push(...result.entries.map(entry => `${entry.type.padEnd(14)} ${entry.name}${entry.symlink ? " -> " + entry.target : ""}`));
        offset = result.next_offset;
      } while (offset !== null && lines.length < 2000);
      if (offset !== null) lines.push("Listing limited to 2000 entries. Navigate into a subfolder for more detail.");
      return lines.join("\n") || "Empty directory.";
    }
    throw new Error(`Unknown command: ${name}. Type help. Shell execution is not supported.`);
  }
  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (run.disabled) return;
    const text = input.value.trim();
    if (!text) return;
    history.push(text);
    if (history.length > 100) history.shift();
    historyIndex = history.length;
    input.value = "";
    run.disabled = true;
    record(">", text);
    try {
      const result = await command(text);
      if (result !== undefined) record("info", result);
    } catch (error) { record("error", error.message || error); }
    finally { run.disabled = false; input.focus(); }
  });
  input.addEventListener("keydown", event => {
    if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
    event.preventDefault();
    historyIndex = Math.max(0, Math.min(history.length, historyIndex + (event.key === "ArrowUp" ? -1 : 1)));
    input.value = history[historyIndex] || "";
  });

  window.addEventListener("error", event => record("error", event.message || "Unhandled browser error"));
  window.addEventListener("unhandledrejection", event => record("error", event.reason?.message || event.reason || "Unhandled promise rejection"));
  clear.addEventListener("click", () => {
    entries.length = 0;
    output.textContent = "";
  });
  record("info", "Diagnostics console ready. Type help for commands. File navigation starts at the application root.");
})();

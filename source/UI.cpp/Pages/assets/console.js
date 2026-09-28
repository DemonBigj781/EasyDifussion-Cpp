(() => {
  const output = document.getElementById("ui-console-output");
  const clear = document.getElementById("ui-console-clear");
  if (!output || !clear) return;

  const entries = [];
  const record = (level, message) => {
    entries.push(`${new Date().toLocaleTimeString()} ${level}: ${String(message)}`);
    if (entries.length > 250) entries.shift();
    output.textContent = entries.join("\n");
    output.scrollTop = output.scrollHeight;
  };

  window.addEventListener("error", event => record("error", event.message || "Unhandled browser error"));
  window.addEventListener("unhandledrejection", event => record("error", event.reason?.message || event.reason || "Unhandled promise rejection"));
  clear.addEventListener("click", () => {
    entries.length = 0;
    output.textContent = "";
  });
  record("info", "C++ UI console connected");
})();

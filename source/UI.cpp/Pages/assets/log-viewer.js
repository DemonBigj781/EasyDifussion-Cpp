(() => {
  const output = document.getElementById("log-viewer-output");
  const status = document.getElementById("log-viewer-status");
  const refresh = document.getElementById("log-viewer-refresh");
  if (!output || !status || !refresh) return;

  async function loadEvents() {
    if (refresh.disabled) return;
    refresh.disabled = true;
    status.textContent = "Loading application and backend logs…";
    try {
      const response = await fetch("/cpp-ui/api/logs", { cache: "no-store" });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
      output.textContent = (result.lines || []).join("\n");
      status.textContent = `Showing ${result.count || 0} recent application log lines${result.truncated ? " (older lines omitted)" : ""}. Refreshes every 5 seconds. Prompt/privacy filtering is preserved.`;
      output.scrollTop = output.scrollHeight;
    } catch (error) {
      status.textContent = `Could not load application logs: ${error.message || error}`;
    } finally {
      refresh.disabled = false;
    }
  }

  refresh.addEventListener("click", loadEvents);
  loadEvents();
  window.setInterval(loadEvents, 5000);
})();

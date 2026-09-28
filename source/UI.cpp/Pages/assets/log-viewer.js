(() => {
  const output = document.getElementById("log-viewer-output");
  const status = document.getElementById("log-viewer-status");
  const refresh = document.getElementById("log-viewer-refresh");
  if (!output || !status || !refresh) return;

  async function loadEvents() {
    refresh.disabled = true;
    status.textContent = "Loading C++ UI serving events…";
    try {
      const response = await fetch("/cpp-ui/api/events", { cache: "no-store" });
      const result = await response.json();
      output.textContent = (result.events || []).map(event =>
        `${event.timestamp} ${event.level.toUpperCase()} ${event.message}`
      ).join("\n");
      status.textContent = `Showing ${result.count || 0} recent C++ UI events.`;
      output.scrollTop = output.scrollHeight;
    } catch (error) {
      status.textContent = `Could not load serving events: ${error.message || error}`;
    } finally {
      refresh.disabled = false;
    }
  }

  refresh.addEventListener("click", loadEvents);
  loadEvents();
  window.setInterval(loadEvents, 5000);
})();

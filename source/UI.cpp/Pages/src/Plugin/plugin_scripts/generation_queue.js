(() => {
    "use strict";
    function createQueue({run, changed = () => {}, limit = 512}) {
        const pending = [];
        let active = null, newestFirst = false;
        const cancelled = () => new Error("Generation cancelled.");
        const state = () => ({active: !!active, pending: pending.length,
            images: [...pending, ...(active ? [active] : [])].reduce((n, job) => n + (job.request.num_outputs || 1), 0)});
        async function drain() {
            if (active || !pending.length) return;
            const job = active = newestFirst ? pending.pop() : pending.shift();
            changed(state());
            try {
                const result = await run(job.request, () => job.cancelled);
                if (job.cancelled) throw cancelled();
                job.resolve(result);
            } catch (error) { job.reject(error); }
            finally { active = null; changed(state()); drain(); }
        }
        return {
            get state() { return state(); },
            get newestFirst() { return newestFirst; },
            set newestFirst(value) { newestFirst = !!value; },
            enqueue(request) {
                if (pending.length >= limit) return Promise.reject(new Error(`Queue limit is ${limit} tasks.`));
                return new Promise((resolve, reject) => {
                    pending.push({request: structuredClone(request), resolve, reject, cancelled: false});
                    changed(state()); drain();
                });
            },
            cancel() {
                if (active) active.cancelled = true;
                for (const job of pending.splice(0)) job.reject(cancelled());
                changed(state());
            },
        };
    }
    if (typeof module !== "undefined" && module.exports) module.exports = {createQueue};
    else window.CppGenerationQueue = {createQueue};
})();

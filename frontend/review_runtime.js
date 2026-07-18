(function attachReviewRuntime(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.ReviewRuntime = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function buildReviewRuntime() {
  function computeInitialWindow(
    fullStart,
    fullEnd,
    now,
    totalHours = 96,
    historyHours = 12
  ) {
    const startMs = new Date(fullStart).getTime();
    const endMs = new Date(fullEnd).getTime();
    const spanMs = Math.max(0, endMs - startMs);
    const desiredMs = totalHours * 3600000;
    if (spanMs <= desiredMs) {
      return {
        start: new Date(startMs).toISOString(),
        end: new Date(endMs).toISOString(),
      };
    }
    const nowMs = new Date(now).getTime();
    const anchor = Number.isFinite(nowMs) ? nowMs : startMs;
    let visibleStart = Math.max(
      startMs,
      Math.min(endMs, anchor) - historyHours * 3600000
    );
    let visibleEnd = visibleStart + desiredMs;
    if (visibleEnd > endMs) {
      visibleEnd = endMs;
      visibleStart = endMs - desiredMs;
    }
    return {
      start: new Date(visibleStart).toISOString(),
      end: new Date(visibleEnd).toISOString(),
    };
  }

  function normalizeIds(ids) {
    return Array.from(new Set((ids || []).filter(Boolean))).sort().slice(0, 4);
  }

  function selectionKey(taskId, ids) {
    return `${taskId}::${normalizeIds(ids).join(",")}`;
  }

  function scheduleKey(taskId, ids, orderId) {
    return `${selectionKey(taskId, ids)}::${orderId || ""}`;
  }

  function rankOrders(orders, query, limit = 50) {
    const needle = String(query || "").trim().toLowerCase();
    const bucket = (item) => {
      const id = String(item.order_id || "").toLowerCase();
      const name = String(item.order_name || "").toLowerCase();
      if (id === needle) return 0;
      if (id.startsWith(needle)) return 1;
      if (id.includes(needle)) return 2;
      if (name.includes(needle)) return 3;
      return 4;
    };
    return (orders || [])
      .filter((item) => bucket(item) < 4)
      .slice()
      .sort(
        (a, b) =>
          bucket(a) - bucket(b) ||
          String(a.order_id).localeCompare(String(b.order_id), "zh-CN", {
            numeric: true,
          })
      )
      .slice(0, Math.max(1, Math.min(Number(limit) || 50, 50)));
  }

  function createOrderComboboxController({
    search,
    select,
    delay = 200,
    limit = 50,
    schedule = (fn, ms) => setTimeout(fn, ms),
    cancelSchedule = (handle) => clearTimeout(handle),
    onState = () => {},
  }) {
    let timer = null;
    let requestController = null;
    let generation = 0;
    let selecting = false;
    let disposed = false;
    let state = { results: [], activeIndex: -1, open: false };

    function getState() {
      return {
        results: state.results.slice(),
        activeIndex: state.activeIndex,
        open: state.open,
      };
    }

    function emit() {
      onState(getState());
    }

    function invalidateSearch() {
      generation += 1;
      if (timer !== null) cancelSchedule(timer);
      timer = null;
      if (requestController) requestController.abort();
      requestController = null;
    }

    function resetResults() {
      state = { results: [], activeIndex: -1, open: false };
    }

    function close() {
      invalidateSearch();
      resetResults();
      emit();
    }

    async function runSearch(query, requestedGeneration) {
      timer = null;
      if (disposed || requestedGeneration !== generation) return;
      const controller = new AbortController();
      requestController = controller;
      let matches;
      try {
        matches = await search(query, controller.signal);
      } catch (error) {
        if (
          disposed ||
          controller.signal.aborted ||
          requestedGeneration !== generation ||
          error?.name === "AbortError"
        ) {
          return;
        }
        matches = [];
      }
      if (
        disposed ||
        controller.signal.aborted ||
        requestedGeneration !== generation ||
        requestController !== controller
      ) {
        return;
      }
      requestController = null;
      const results = Array.isArray(matches)
        ? matches.slice(0, Math.max(1, Math.min(Number(limit) || 50, 50)))
        : [];
      state = {
        results,
        activeIndex: results.length ? 0 : -1,
        open: true,
      };
      emit();
    }

    function input(query) {
      if (disposed) return;
      invalidateSearch();
      resetResults();
      emit();
      const requestedGeneration = generation;
      timer = schedule(
        () => runSearch(query, requestedGeneration),
        delay
      );
    }

    function move(delta) {
      if (!state.open || !state.results.length) return false;
      state = {
        ...state,
        activeIndex: Math.max(
          0,
          Math.min(state.results.length - 1, state.activeIndex + delta)
        ),
      };
      emit();
      return true;
    }

    async function choose(order) {
      if (
        disposed ||
        selecting ||
        !state.open ||
        !order ||
        !state.results.includes(order)
      ) {
        return false;
      }
      selecting = true;
      close();
      try {
        await select(order);
        return true;
      } finally {
        selecting = false;
      }
    }

    function chooseActive() {
      if (!state.open || state.activeIndex < 0) {
        return Promise.resolve(false);
      }
      return choose(state.results[state.activeIndex]);
    }

    function dispose() {
      if (disposed) return;
      invalidateSearch();
      disposed = true;
      resetResults();
    }

    return {
      input,
      close,
      move,
      choose,
      chooseActive,
      dispose,
      getState,
    };
  }

  async function handleOrderComboboxKey(controller, event) {
    if (event.key === "Escape") {
      event.preventDefault();
      controller.close();
      return true;
    }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      controller.move(event.key === "ArrowDown" ? 1 : -1);
      return true;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      await controller.chooseActive();
      return true;
    }
    return false;
  }

  function createClient({ fetchReviewData, fetchOrders }) {
    const dataCache = new Map();
    const orderCache = new Map();
    let dataController = null;
    let orderController = null;
    let dataGeneration = 0;
    let orderGeneration = 0;

    async function loadData(args) {
      const key = `${scheduleKey(
        args.taskId,
        args.ids,
        args.orderId
      )}::${args.includeUtilization ? "u1" : "u0"}`;
      const generation = ++dataGeneration;
      if (dataController) dataController.abort();
      dataController = null;
      if (dataCache.has(key)) {
        return { payload: dataCache.get(key), fromCache: true };
      }
      const controller = new AbortController();
      dataController = controller;
      try {
        const payload = await fetchReviewData(args, controller.signal);
        if (generation !== dataGeneration) return { cancelled: true };
        dataCache.set(key, payload);
        return { payload, fromCache: false };
      } catch (error) {
        if (error?.name === "AbortError" || generation !== dataGeneration) {
          return { cancelled: true };
        }
        throw error;
      } finally {
        if (generation === dataGeneration) dataController = null;
      }
    }

    async function searchOrders(args, externalSignal) {
      const key = `${selectionKey(
        args.taskId,
        args.ids
      )}::${String(args.query || "").trim().toLowerCase()}`;
      const generation = ++orderGeneration;
      if (orderController) orderController.abort();
      orderController = null;
      if (externalSignal?.aborted) return { cancelled: true };
      if (orderCache.has(key)) {
        return { orders: orderCache.get(key), fromCache: true };
      }
      const controller = new AbortController();
      orderController = controller;
      const abortFromExternal = () => controller.abort();
      externalSignal?.addEventListener("abort", abortFromExternal, { once: true });
      try {
        const payload = await fetchOrders(args, controller.signal);
        if (
          controller.signal.aborted ||
          externalSignal?.aborted ||
          generation !== orderGeneration
        ) {
          return { cancelled: true };
        }
        const orders = payload.orders || [];
        orderCache.set(key, orders);
        return { orders, fromCache: false };
      } catch (error) {
        if (
          error?.name === "AbortError" ||
          controller.signal.aborted ||
          externalSignal?.aborted ||
          generation !== orderGeneration
        ) {
          return { cancelled: true };
        }
        throw error;
      } finally {
        externalSignal?.removeEventListener("abort", abortFromExternal);
        if (generation === orderGeneration) orderController = null;
      }
    }

    function reset() {
      dataGeneration += 1;
      orderGeneration += 1;
      if (dataController) dataController.abort();
      if (orderController) orderController.abort();
      dataController = null;
      orderController = null;
      dataCache.clear();
      orderCache.clear();
    }

    return { loadData, searchOrders, reset };
  }

  return {
    computeInitialWindow,
    normalizeIds,
    selectionKey,
    scheduleKey,
    rankOrders,
    createOrderComboboxController,
    handleOrderComboboxKey,
    createClient,
  };
});

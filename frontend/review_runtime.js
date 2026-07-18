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

  function createClient({ fetchReviewData, fetchOrders }) {
    const dataCache = new Map();
    const orderCache = new Map();
    let dataController = null;
    let orderController = null;

    async function loadData(args) {
      const key = `${scheduleKey(
        args.taskId,
        args.ids,
        args.orderId
      )}::${args.includeUtilization ? "u1" : "u0"}`;
      if (dataCache.has(key)) {
        return { payload: dataCache.get(key), fromCache: true };
      }
      if (dataController) dataController.abort();
      dataController = new AbortController();
      try {
        const payload = await fetchReviewData(args, dataController.signal);
        dataCache.set(key, payload);
        return { payload, fromCache: false };
      } catch (error) {
        if (error?.name === "AbortError") return { cancelled: true };
        throw error;
      }
    }

    async function searchOrders(args) {
      const key = `${selectionKey(
        args.taskId,
        args.ids
      )}::${String(args.query || "").trim().toLowerCase()}`;
      if (orderCache.has(key)) {
        return { orders: orderCache.get(key), fromCache: true };
      }
      if (orderController) orderController.abort();
      orderController = new AbortController();
      try {
        const payload = await fetchOrders(args, orderController.signal);
        const orders = payload.orders || [];
        orderCache.set(key, orders);
        return { orders, fromCache: false };
      } catch (error) {
        if (error?.name === "AbortError") return { cancelled: true };
        throw error;
      }
    }

    function reset() {
      if (dataController) dataController.abort();
      if (orderController) orderController.abort();
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
    createClient,
  };
});

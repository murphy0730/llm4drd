(function attachOptimizeProgress(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.OptimizeProgress = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function buildOptimizeProgress() {
  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function optimizeProgress(status) {
    const state = String(status?.status || "").toLowerCase();
    if (["done", "completed", "success"].includes(state)) return 100;
    if (state === "submitting") return 2;

    const realProgress = Number(status?.real_progress);
    if (Number.isFinite(realProgress)) {
      return Math.round(clamp(realProgress, 0, 1) * 100);
    }

    const progressPercent = Number(status?.progress);
    if (Number.isFinite(progressPercent)) {
      return Math.round(clamp(progressPercent, 0, 100));
    }

    // 兼容尚未返回 real_progress 的旧后端，只使用代数这种真实工作量；
    // elapsed_s 仅是墙钟时间，绝不能拿来制造百分比。
    const generations = Number(status?.config?.generations || 0);
    const generation = Number(status?.current_generation || 0);
    if (generations > 0 && generation > 0) {
      return Math.round(8 + clamp(generation / generations, 0, 1) * 57);
    }
    return state === "error" ? 2 : 5;
  }

  function optimizeActivity(status, nowSeconds = Date.now() / 1000, thresholdSeconds = 60) {
    const state = String(status?.status || "").toLowerCase();
    const running = ["started", "queued", "running"].includes(state);
    const lastRealProgressAt = Number(status?.last_real_progress_at);
    const backendAge = Number(status?.seconds_since_real_progress);
    const computedAge = Number.isFinite(lastRealProgressAt)
      ? Math.max(0, nowSeconds - lastRealProgressAt)
      : 0;
    const secondsSinceRealProgress = Math.floor(
      Number.isFinite(backendAge) ? Math.max(backendAge, computedAge) : computedAge
    );
    const threshold = Math.max(
      1,
      Number(status?.stall_threshold_s || thresholdSeconds || 60)
    );
    const stalled = running && (
      status?.stalled === true || (
        Number.isFinite(lastRealProgressAt)
        && secondsSinceRealProgress >= threshold
      )
    );
    return {
      stalled,
      secondsSinceRealProgress,
      lastRealProgressAt: Number.isFinite(lastRealProgressAt)
        ? lastRealProgressAt
        : null,
      message: stalled
        ? `正在计算，已 ${secondsSinceRealProgress} 秒无新的真实进度；进度条已停止，后台可能正在处理单个耗时评估。`
        : "",
    };
  }

  return { optimizeProgress, optimizeActivity };
});

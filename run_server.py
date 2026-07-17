import importlib.util
import pathlib
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

import uvicorn


HOST = "127.0.0.1"
PORT = 8888
TARGET_URL = f"http://{HOST}:{PORT}/"


def _register_package() -> None:
    """把项目根目录注册成 llm4drd_platform 包。

    必须在模块顶层执行：spawn 出的子进程（进程池 worker）会以 __mp_main__ 之名
    重新执行本文件，随后才反序列化任务负载。若注册只在 main() 里做，子进程反
    序列化 llm4drd_platform.* 对象时会 ModuleNotFoundError。
    """
    if "llm4drd_platform" in sys.modules:
        return

    root = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "llm4drd_platform",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create package spec for llm4drd_platform")

    pkg = importlib.util.module_from_spec(spec)
    sys.modules["llm4drd_platform"] = pkg
    spec.loader.exec_module(pkg)


_register_package()


def _port_in_use() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((HOST, PORT)) == 0


def _is_this_app_running() -> bool:
    try:
        with urllib.request.urlopen(TARGET_URL, timeout=1.0) as response:
            return "LLM4DRD" in response.read(8192).decode("utf-8", errors="ignore")
    except Exception:
        return False


def _open_when_ready() -> None:
    for _ in range(80):
        if _is_this_app_running():
            webbrowser.open(TARGET_URL)
            return
        time.sleep(0.1)


def main() -> None:
    if _port_in_use():
        if _is_this_app_running():
            print(f"LLM4DRD 已在运行：{TARGET_URL}")
            webbrowser.open(TARGET_URL)
            return
        raise SystemExit(
            f"无法启动：{HOST}:{PORT} 已被其他程序占用。\n"
            f"请先停止占用 {PORT} 端口的服务，再重新运行 python run_server.py。"
        )

    threading.Thread(target=_open_when_ready, daemon=True).start()
    uvicorn.run("llm4drd_platform.api.server:app", host=HOST, port=PORT)


if __name__ == "__main__":
    main()

import importlib.util
import pathlib
import sys

import uvicorn


def main() -> None:
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

    uvicorn.run("llm4drd_platform.api.server:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()

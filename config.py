"""配置管理 — config.json + 环境变量"""
import os, json, logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)
CONFIG_PATH = os.environ.get("LLM4DRD_CONFIG", os.path.join(os.path.dirname(__file__), "config.json"))

@dataclass
class LLMConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    max_tokens: int = 2048
    timeout: int = 60

@dataclass
class PlatformConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    db_path: str = "llm4drd.db"

def load_config(path=CONFIG_PATH):
    cfg = PlatformConfig()
    if os.path.exists(path):
        try:
            with open(path) as f: raw = json.load(f)
            for k in ("base_url","api_key","model","max_tokens","timeout"):
                if k in raw.get("llm",{}): setattr(cfg.llm, k, raw["llm"][k])
            cfg.db_path = raw.get("database",{}).get("path", cfg.db_path)
        except Exception as e: logger.warning(f"Config load error: {e}")
    cfg.llm.api_key = os.environ.get("LLM_API_KEY", cfg.llm.api_key)
    cfg.llm.base_url = os.environ.get("LLM_BASE_URL", cfg.llm.base_url)
    cfg.llm.model = os.environ.get("LLM_MODEL", cfg.llm.model)
    return cfg

_cfg = None
def get_config():
    global _cfg
    if not _cfg: _cfg = load_config()
    return _cfg
def reload_config():
    global _cfg; _cfg = load_config(); return _cfg

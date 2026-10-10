"""兼容 all_test.py / all_test_vis.py：`from main import get_args_parser`。
直接 `python main.py ...` 时转交 `main--.py` 入口，与 `python main--.py ...` 等价。
"""
import runpy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_main_dash = Path(__file__).with_name("main--.py")

if __name__ == "__main__":
    runpy.run_path(str(_main_dash), run_name="__main__")
else:
    _spec = spec_from_file_location("main_dash", _main_dash)
    _mod = module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    get_args_parser = _mod.get_args_parser
    __all__ = ["get_args_parser"]

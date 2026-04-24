"""配置转换脚本测试。

验证 configs/convert_examples.py 中的格式迁移逻辑。
"""

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "configs" / "convert_examples.py"
_SPEC = importlib.util.spec_from_file_location("convert_examples", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
convert_example = _MODULE.convert_example


def test_convert_example_outputs_new_runtime_schema_fields() -> None:
    converted = convert_example(
        {
            "v": [[0, 0, 0], [1, 0, 0]],
            "e": [[0, 1]],
            "fixedVs": [1, 0],
            "edgeChannel": [0],
            "edgeActive": [1],
            "script": [[1, 0]],
            "numChannels": 1,
            "numActions": 2,
        }
    )

    assert converted["sites"]["s1"]["projection_target"] is False
    assert converted["sites"]["s2"]["projection_target"] is True
    rod_group = converted["rod_groups"][0]
    assert rod_group["rod_type"] == "active"
    assert rod_group["length_limits"] == [1.0, 1.1]
    assert rod_group["force_limits"] == [-20000.0, 20000.0]
    assert "rest_length" not in rod_group
    assert "min_length" not in rod_group

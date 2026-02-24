import copy
from typing import Any, Optional

from func_timeout import func_set_timeout

from agentlego.types import Annotated, Info
from ..base import BaseTool

import os
import io
import traceback
from typing import Any, Optional, Dict
from contextlib import redirect_stdout, redirect_stderr

DESC_EN = '''\
This tool can execute Python code. The code should include a function named 'solution'. The function should return its answer in str format. Avoid printing the answer. The code instance format is as follows:

```python
# import packages
import xxx
def solution():
    # python code to get the final answer
    ...
    return final_answer
```
'''  # noqa: E501


class GenericRuntime:

    def __init__(
        self,
        global_dict: Optional[dict] = None,
        headers: list = [],
    ):
        self._global_vars = copy.copy(global_dict) if global_dict else {}

        for c in headers:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars, self._global_vars)


class PythonInterpreter(BaseTool):
    """A Python executor that can execute Python scripts.

    WARNING: The PythonInterpreter only has minimal protection, don't expose to
    trustless environment.

    Args:
        timeout (int, Optional): Upper bound of waiting time for Python script execution.
            Defaults to ``20``.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = DESC_EN
    answer_expr = 'solution()'

    def __init__(self, timeout: int = 20, toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.timeout = timeout

    def apply(self, command: Annotated[str, Info('Markdown format Python code')]) -> str:

        if '```python' in command:
            command = command.split('```python')[1].split('```')[0]
        elif '```' in command:
            command = command.split('```')[1].split('```')[0]

        res = func_set_timeout(self.timeout)(self._call)(command)
        return str(res)

    def _call(self, command: str) -> Any:
        runtime = GenericRuntime()
        runtime.exec_code(command)
        return runtime.eval_code(self.answer_expr)

class UniversalPythonInterpreter(BaseTool):
    default_desc = DESC_EN

    def __init__(self, timeout: int = 20, toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.timeout = timeout

    # def apply(self, command: Annotated[str, Info("JSON or Markdown Python code")]) -> str:
    #     """
    #     Accepts:
    #       - JSON: {"code": "..."} or {"path": "...", "eval": "..."}
    #       - Markdown code fence: ```python ... ```
    #     Returns a JSON string.
    #     """
    #     import json

    #     payload = None
    #     raw = (command or "").strip()

    #     # If markdown code block, treat whole thing as code
    #     if "```python" in raw:
    #         code = raw.split("```python", 1)[1].split("```", 1)[0]
    #         payload = {"code": code}
    #     elif raw.startswith("```"):
    #         code = raw.split("```", 1)[1].split("```", 1)[0]
    #         payload = {"code": code}
    #     else:
    #         # Try JSON
    #         try:
    #             payload = json.loads(raw)
    #             if not isinstance(payload, dict):
    #                 payload = {"code": raw}
    #         except Exception:
    #             payload = {"code": raw}

    #     res = func_set_timeout(self.timeout)(self._call)(payload)
    #     return json.dumps(res, ensure_ascii=False)

    def apply(self, command: Annotated[str, Info("JSON or Markdown Python code")]) -> str:
        import json
        from func_timeout.exceptions import FunctionTimedOut

        payload = None
        raw = (command or "").strip()

        if "```python" in raw:
            code = raw.split("```python", 1)[1].split("```", 1)[0]
            payload = {"code": code}
        elif raw.startswith("```"):
            code = raw.split("```", 1)[1].split("```", 1)[0]
            payload = {"code": code}
        else:
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    payload = {"code": raw}
            except Exception:
                payload = {"code": raw}

        # optional per-request timeout override
        req_timeout = payload.get("timeout") if isinstance(payload, dict) else None
        try:
            t = int(req_timeout) if req_timeout is not None else int(self.timeout)
        except Exception:
            t = int(self.timeout)

        try:
            res = func_set_timeout(t)(self._call)(payload)
            return json.dumps(res, ensure_ascii=False)
        except FunctionTimedOut:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"Timeout: execution exceeded {t} seconds",
                    "traceback": "",
                    "stdout": "",
                    "stderr": "",
                    "value": None,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                    "stdout": "",
                    "stderr": "",
                    "value": None,
                },
                ensure_ascii=False,
            )


    def _call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import json

        code = payload.get("code")
        path = payload.get("path")
        eval_expr = payload.get("eval")  # optional

        if code is None and path is None:
            return {
                "ok": False,
                "error": "Provide either 'code' or 'path'.",
                "stdout": "",
                "stderr": "",
                "value": None,
            }

        if code is None and path is not None:
            if not os.path.exists(path):
                return {
                    "ok": False,
                    "error": f"File not found: {path}",
                    "stdout": "",
                    "stderr": "",
                    "value": None,
                }
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()

        # runtime = GenericRuntime()
        runtime = GenericRuntime(global_dict={"__name__": "__main__"})

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                runtime.exec_code(code)

                value = None
                if eval_expr:
                    value = runtime.eval_code(eval_expr)

        except Exception as e:
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue(),
                "value": None,
            }

        out = stdout_buf.getvalue()
        err = stderr_buf.getvalue()

        # If eval_expr was provided, return it as string (even if stdout exists)
        if eval_expr:
            return {
                "ok": True,
                "stdout": out,
                "stderr": err,
                "value": "" if value is None else str(value),
            }

        # Otherwise return stdout as the primary output
        return {
            "ok": True,
            "stdout": out,
            "stderr": err,
            "value": None,
        }
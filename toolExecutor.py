import json
import inspect
from typing import Any, Callable, Dict, Optional
from tools import calculator, search

class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(
        self,
        name: str,
        description: str,
        func: Callable[..., str],
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {
            "description": description,
            "func": func,
            "parameters": parameters or self._defaultStringParameterSchema(),
        }
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])

    def getToolSchemas(self) -> list[Dict[str, Any]]:
        """
        获取 OpenAI Chat Completions function calling 所需的工具定义。
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["parameters"],
                },
            }
            for name, info in self.tools.items()
        ]

    def executeToolCall(self, name: str, arguments: str | Dict[str, Any]) -> str:
        """
        执行模型返回的结构化 tool call。
        """
        tool = self.tools.get(name)
        if not tool:
            return f"错误:未找到名为 '{name}' 的工具。"

        try:
            parsed_arguments = self._parseArguments(arguments)
            return str(self._callToolFunction(tool["func"], parsed_arguments))
        except Exception as exc:
            return f"工具 '{name}' 执行失败:{exc}"

    def _callToolFunction(self, func: Callable[..., str], arguments: Dict[str, Any]) -> str:
        signature = inspect.signature(func)
        parameters = signature.parameters

        if all(name in parameters for name in arguments):
            return func(**arguments)

        if len(arguments) == 1:
            return func(next(iter(arguments.values())))

        return func(**arguments)

    def _parseArguments(self, arguments: str | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if not arguments:
            return {}
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"工具参数不是合法 JSON:{exc}") from exc
        if not isinstance(parsed_arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象。")
        return parsed_arguments

    def _defaultStringParameterSchema(self) -> Dict[str, Any]:
        """
        兼容旧的单字符串工具注册方式。
        """
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "传递给工具的输入文本。",
                }
            },
            "required": ["input"],
            "additionalProperties": False,
        }

# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool(
        "Search",
        search_description,
        search,
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要搜索的关键词或问题。",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    calculator_description = "一个安全的计算器工具。适用于四则运算、括号、幂运算，以及 sqrt/log/sin/cos 等常见数学函数计算。"
    toolExecutor.registerTool(
        "Calculator",
        calculator_description,
        calculator,
        {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式。",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    )
    
    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 模拟一次结构化 tool call，这次我们问一个实时性的问题
    print("\n--- 执行 Tool Call: Search({\"query\":\"英伟达最新的GPU型号是什么\"}) ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    observation = toolExecutor.executeToolCall(tool_name, {"query": tool_input})
    print("--- 工具结果 ---")
    print(observation)

    print("\n--- 执行 Tool Call: Calculator({\"expression\":\"(123 + 456) * 789 / 12\"}) ---")
    tool_name = "Calculator"
    tool_input = "(123 + 456) * 789 / 12"

    observation = toolExecutor.executeToolCall(tool_name, {"expression": tool_input})
    print("--- 工具结果 ---")
    print(observation)

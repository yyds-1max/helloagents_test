import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm import HelloAgentsLLM
from toolExecutor import ToolExecutor


SYSTEM_PROMPT = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}
请根据用户问题自主决定是否调用工具。

- 已经获得足够信息时，直接给出最终答案。

不要输出 Action[...]、Finish[...] 或其他文本协议；工具调用必须使用原生 function calling。
""".strip()


class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history: List[Dict[str, Any]] = []

    def run(self, question: str) -> Optional[str]:
        """
        使用原生 function calling 运行智能体。
        """
        system_prompt = SYSTEM_PROMPT.format(tools = self.tool_executor.getAvailableTools())
        # 调试
        print(system_prompt)

        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        tools = self.tool_executor.getToolSchemas()

        for current_step in range(1, self.max_steps + 1):
            print(f"--- 第 {current_step} 步 ---")

            assistant_message = self.llm_client.chat(
                messages=self.history,
                tools=tools,
                temperature=0,
            )
            if not assistant_message:
                print("错误:LLM未能返回有效响应。")
                return None

            self.history.append(assistant_message)
            tool_calls = assistant_message.get("tool_calls", [])

            if not tool_calls:
                final_answer = assistant_message.get("content")
                if final_answer:
                    print(f"🎉 最终答案: {final_answer}")
                    return final_answer
                print("警告:模型未返回最终答案，也未请求工具调用。")
                return None

            for tool_call in tool_calls:
                function_call = tool_call.get("function", {})
                tool_name = function_call.get("name", "")
                arguments = function_call.get("arguments", "{}")

                print(f"🎬 工具调用: {tool_name}({self._formatArguments(arguments)})")
                observation = self.tool_executor.executeToolCall(tool_name, arguments)
                print(f"👀 工具结果: {observation}")

                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_name,
                        "content": observation,
                    }
                )

        print("已达到最大步数，流程终止。")
        return None

    def _formatArguments(self, arguments: str | Dict[str, Any]) -> str:
        if isinstance(arguments, dict):
            return json.dumps(arguments, ensure_ascii=False)
        try:
            return json.dumps(json.loads(arguments), ensure_ascii=False)
        except json.JSONDecodeError:
            return arguments


if __name__ == "__main__":
    from tools import calculator, search

    llm_client = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    tool_executor.registerTool(
        "Search",
        "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。",
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
    tool_executor.registerTool(
        "Calculator",
        "一个安全的计算器工具。适用于四则运算、括号、幂运算，以及 sqrt/log/sin/cos 等常见数学函数计算。",
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

    agent = ReActAgent(llm_client=llm_client, tool_executor=tool_executor)
    demo_question = "计算 (123 + 456) × 789 / 12 = ? 的结果。"
    print(f"\n=== Function Calling Agent 数学计算演示 ===\n问题: {demo_question}\n")
    agent.run(demo_question)

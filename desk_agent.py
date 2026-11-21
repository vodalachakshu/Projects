from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("desk-agent")


@dataclass(frozen=True)
class ToolResult:
    tool: str
    ok: bool
    data: dict[str, Any]


def lookup_customer(customer_id: str) -> dict[str, Any]:
    """Look up a customer record by customer ID."""
    customers = {
        "C100": {"id": "C100", "name": "Asha Rao", "plan": "Pro", "status": "active"},
        "C200": {"id": "C200", "name": "Miguel Santos", "plan": "Starter", "status": "on hold"},
    }
    customer = customers.get(customer_id.upper())
    if not customer:
        raise ValueError(f"Customer '{customer_id}' was not found")
    return customer


def check_service_status(service: str) -> dict[str, Any]:
    """Check the operational status of a named service."""
    normalized = service.strip().lower()
    statuses = {
        "billing": {"status": "operational", "latency_ms": 120},
        "authentication": {"status": "degraded", "latency_ms": 480},
        "support": {"status": "operational", "latency_ms": 95},
    }
    if normalized not in statuses:
        raise ValueError(f"Unknown service '{service}'")
    return {"service": normalized, **statuses[normalized]}


customer_tool = StructuredTool.from_function(
    func=lookup_customer,
    name="lookup_customer",
    description="Look up a customer by ID, for example C100 or C200.",
)
status_tool = StructuredTool.from_function(
    func=check_service_status,
    name="check_service_status",
    description="Check the status of billing, authentication, or support.",
)


class DeskAgent:
    """A minimal LangChain agent that routes queries to predefined tools."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        self.llm = ChatOpenAI(model=model, temperature=0)
        self.tools = {tool.name: tool for tool in (customer_tool, status_tool)}
        self.routed_llm = self.llm.bind_tools(list(self.tools.values()))

    def run(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            return self._failure("Query cannot be empty", "invalid_input")

        logger.info("Received desk-agent query: %s", query)
        try:
            response = self.routed_llm.invoke([HumanMessage(content=query)])
            tool_calls = getattr(response, "tool_calls", [])

            if not tool_calls:
                answer = response.content.strip() if response.content else ""
                if not answer:
                    return self._failure("The model could not answer the query", "no_answer")
                logger.info("Returned direct model answer")
                return {"ok": True, "answer": answer, "tool_calls": []}

            tool_results: list[dict[str, Any]] = []
            for call in tool_calls:
                tool_name = call["name"]
                arguments = call.get("args", {})
                logger.info("Routing query to tool=%s arguments=%s", tool_name, arguments)

                tool = self.tools.get(tool_name)
                if tool is None:
                    tool_results.append(
                        self._failure_result(tool_name, f"Unknown tool '{tool_name}'", "unknown_tool")
                    )
                    continue

                try:
                    data = tool.invoke(arguments)
                    tool_results.append({"tool": tool_name, "ok": True, "data": data})
                except Exception as exc:
                    logger.exception("Tool execution failed: %s", tool_name)
                    tool_results.append(
                        self._failure_result(tool_name, str(exc), "tool_error")
                    )

            failed = [result for result in tool_results if not result["ok"]]
            if failed:
                return self._failure(failed[0]["error"], failed[0]["error_code"])

            combined = "\n".join(json.dumps(result["data"], sort_keys=True) for result in tool_results)
            synthesis = self.llm.invoke(
                [
                    HumanMessage(
                        content=(
                            "Use the following tool results to answer the user's query concisely.\n"
                            f"User query: {query}\nTool results:\n{combined}"
                        )
                    )
                ]
            )
            logger.info("Completed tool-backed answer")
            return {"ok": True, "answer": synthesis.content, "tool_calls": tool_results}

        except Exception as exc:
            logger.exception("Desk-agent execution failed")
            return self._failure("The desk agent could not complete the request", "agent_error")

    @staticmethod
    def _failure_result(tool: str, message: str, code: str) -> dict[str, Any]:
        return {"tool": tool, "ok": False, "error": message, "error_code": code}

    @staticmethod
    def _failure(message: str, code: str) -> dict[str, Any]:
        return {"ok": False, "answer": message, "error_code": code, "tool_calls": []}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the desk agent")
    parser.add_argument("query", nargs="+", help="User request")
    args = parser.parse_args()

    agent = DeskAgent()
    print(json.dumps(agent.run(" ".join(args.query)), indent=2))

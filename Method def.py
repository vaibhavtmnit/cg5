# method_definition_extractor.py
# Method Definition extractor (names only; no spans). Two independent runs + merge.
# - SAME-CLASS method: return direct, unqualified calls in that method body.
# - EXTERNAL method: return a single instruction-child with requires_definition_expansion=True.
#
# You pass in your AzureChatOpenAI instance. No top-k knobs.
#
# Requires:
#   pip install langchain langchain-openai pydantic

from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# ---------------------------------------------------------------------------
# NOTE: I DO NOT define your TypedDicts. I access fields by name:
# request: {
#   "object_name": str,
#   "java_code": str,
#   "java_code_line": int,                 # 1-based
#   "java_code_line_content": str,         # exact code on that line (you provide this)
#   "analytical_chain": str
# }
#
# Child record you already have; I will populate keys:
#   child_name, child_type, code_snippet, code_block,
#   further_expand, found_in,
#   requires_definition_expansion  # <— assume your schema includes/accepts this flag
# If your schema uses a different flag name, change FLAG_FIELD below.
# ---------------------------------------------------------------------------

FLAG_FIELD = "requires_definition_expansion"  # rename if your TypedDict uses a different field


# ─────────────────────────────────────────────────────────────────────────────
# Structured outputs expected back from the LLM
# ─────────────────────────────────────────────────────────────────────────────

class MDItem(BaseModel):
    child_name: str
    code_snippet: str
    code_block: str
    conditioned: bool = False
    guards: List[str] = Field(default_factory=list)
    confidence: float
    comment: Optional[str] = None
    requires_definition_expansion: bool = False

    @field_validator("confidence")
    @classmethod
    def _clip_conf(cls, v: float) -> float:
        return float(max(0.0, min(1.0, v)))


class MDOut(BaseModel):
    children: List[MDItem] = Field(default_factory=list)
    uncertain: List[MDItem] = Field(default_factory=list)
    stop_reason: Optional[str] = None
    mode: Optional[str] = None  # "SAME_CLASS" | "EXTERNAL" (hint from the model)


class LineExplained(BaseModel):
    line: int
    text: str


class ExplainOut(BaseModel):
    lines: List[LineExplained] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Denylist (keep noise out)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DENYLIST = [
    "System.out.println",
    "logger.info",
    "logger.debug",
    "logger.trace",
    "Objects.requireNonNull",
    "Collections.emptyList",
]


# ─────────────────────────────────────────────────────────────────────────────
# Prompts — concise but robust; include anchor line content
# ─────────────────────────────────────────────────────────────────────────────

_RUNA_SYSTEM = """
Task: Given a METHOD NAME as the input object, decide whether it is defined in the SAME CLASS in the provided code,
and then extract method-call children accordingly. Return STRICT JSON. Prefer empty results over guesses.

MODE DECISION:
• SAME_CLASS if there is a method declaration with that exact name in this code (same class/compilation unit).
• EXTERNAL if no such declaration exists here (the name appears only at call sites/imports/other classes).

EXTRACTION (only when SAME_CLASS):
• Return only direct UNQUALIFIED calls inside that method body (helper(), this.helper(), super.helper()).
• Exclude: calls on other receivers (svc.run()), deeper chain hops (x.a().b()), and calls inside lambda/anonymous-class bodies.
• Ignore logging/printing and trivial JDK utilities (denylist provided).

EXTERNAL HANDLING:
• Return ONE instruction-child telling the orchestrator to expand the method definition in its declaring class:
  - child_name = the method name
  - child_type = "Method Definition"
  - code_snippet = the ANCHOR_LINE_CONTENT
  - code_block = the nearest full statement containing the anchor (or the snippet itself)
  - requires_definition_expansion = true
  - further_expand = true
• No other children are needed in EXTERNAL mode.

All children should include: child_name, code_snippet, code_block, confidence [0,1], optional conditioned/guards and a short comment (≤12 words).
""".strip()

_RUNA_EXAMPLES = """
Few-shot examples (generic):

1) SAME_CLASS
class U {
  void boot(){ prep(); worker.run(); this.flush(); }
  void prep(){} void flush(){}
}
Input: object_name="boot"
Output: children = ["prep","flush"] (unqualified only), mode="SAME_CLASS"

2) EXTERNAL (no definition here)
class V {
  void main(){ engine.start(); }
}
Input: object_name="start"
Output: one child:
  - child_name="start", child_type="Method Definition", requires_definition_expansion=true
  - code_snippet should show "engine.start(...)" or the anchor line content
  - mode="EXTERNAL"
""".strip()

_RUNA_USER_TMPL = """OBJECT_NAME (method): {method_name}
ANCHOR_LINE (1-based): {anchor_line}
ANCHOR_LINE_CONTENT: {anchor_line_content}
ANALYTICAL_CHAIN (≤2): {analytical_chain}
DENYLIST: {denylist}

CODE:
{code}

{examples}

Selection rubric:
1) Decide MODE (SAME_CLASS vs EXTERNAL) from declarations in this code and anchor line context.
2) If SAME_CLASS: include only direct unqualified calls inside that method body; exclude receiver calls & lambda internals.
3) If EXTERNAL: return exactly ONE instruction-child with requires_definition_expansion=true as specified.
4) Prefer empty over guessing.

Output JSON schema:
{{"children":[{{"child_name":"...","code_snippet":"...","code_block":"...","conditioned":false,"guards":[],"confidence":0.0,"comment":"...|null","requires_definition_expansion":false}}], "uncertain":[...], "stop_reason":"...|null", "mode":"SAME_CLASS|EXTERNAL"}}
Return ONLY the JSON object.
"""

_RUNB_EXPLAIN_SYSTEM = """
Convert the Java code to concise, factual natural language, one sentence per original line (1-based).
Preserve identifiers and receivers. No speculation. No added/removed lines. Strict JSON.
""".strip()

_RUNB_EXTRACT_SYSTEM = """
Using the per-line explanations, repeat the SAME task:
• Decide SAME_CLASS vs EXTERNAL for the given method name.
• If SAME_CLASS: list direct UNQUALIFIED calls inside that method body.
• If EXTERNAL: produce the single instruction-child with requires_definition_expansion=true, using the ANCHOR_LINE_CONTENT.
Respect the denylist. Prefer empty over guessing. Strict JSON.
""".strip()

_RUNB_USER_TMPL = """OBJECT_NAME (method): {method_name}
ANCHOR_LINE (1-based): {anchor_line}
ANCHOR_LINE_CONTENT: {anchor_line_content}
ANALYTICAL_CHAIN (≤2): {analytical_chain}
DENYLIST: {denylist}

LINES_NL (JSON array of {{line:int, text:str}}):
{explained_json}

Few-shot hints:
- "line 7: call prep(); then worker.run()" with method=boot → SAME_CLASS children=["prep"] (exclude worker.run)
- "line 3: engine.start()" with method=start and no declaration present → EXTERNAL (instruction-child)

Output JSON schema:
{{"children":[{{"child_name":"...","code_snippet":"...","code_block":"...","conditioned":false,"guards":[],"confidence":0.0,"comment":"...|null","requires_definition_expansion":false}}], "uncertain":[...], "stop_reason":"...|null", "mode":"SAME_CLASS|EXTERNAL"}}
Return ONLY the JSON object.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Small LLM helper (structured output + one retry)
# ─────────────────────────────────────────────────────────────────────────────

def _invoke_structured(llm: AzureChatOpenAI, schema, system: str, user: str, retry: bool = True):
    try:
        return llm.with_structured_output(schema).invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
    except Exception:
        if not retry:
            raise
        user2 = user + "\n\nREMINDER: Return ONLY a valid JSON object. Use empty lists when unsure."
        return llm.with_structured_output(schema).invoke(
            [SystemMessage(content=system), HumanMessage(content=user2)]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MDExtractorConfig:
    denylist: List[str] = None

    def get_denylist(self) -> List[str]:
        return list(DEFAULT_DENYLIST if self.denylist is None else self.denylist)


def extract_method_definition_children(
    llm: AzureChatOpenAI,
    *,
    request: Dict[str, Any],
    config: Optional[MDExtractorConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Method Definition extractor.

    Inputs (request TypedDict keys expected):
      - object_name: str                 # method name to analyze
      - java_code: str                   # full source string
      - java_code_line: int              # 1-based anchor line
      - java_code_line_content: str      # exact code on that line
      - analytical_chain: str            # up to two predecessors

    Returns: list[ChildRecord] (your TypedDict), each with:
      - child_name, child_type, code_snippet, code_block, further_expand, found_in
      - requires_definition_expansion (bool)  # field name controlled by FLAG_FIELD
    """
    cfg = config or MDExtractorConfig()
    denylist = cfg.get_denylist()

    method_name = request["object_name"]
    code = request["java_code"]
    anchor = int(request["java_code_line"])
    anchor_content = request.get("java_code_line_content", "")
    chain = request.get("analytical_chain", "")

    # RUN A — Original code
    user_a = _RUNA_USER_TMPL.format(
        method_name=method_name,
        anchor_line=anchor,
        anchor_line_content=anchor_content,
        analytical_chain=chain,
        denylist=denylist,
        code=code,
        examples=_RUNA_EXAMPLES,
    )
    out_a: MDOut = _invoke_structured(llm, MDOut, _RUNA_SYSTEM, user_a)

    # RUN B — NL explanation → extract
    # 1) explain lines
    explain_user = f"CODE:\n{code}"
    explained: ExplainOut = _invoke_structured(llm, ExplainOut, _RUNB_EXPLAIN_SYSTEM, explain_user)
    explained_json = ExplainOut(lines=explained.lines).model_dump_json()

    # 2) extract from NL
    user_b = _RUNB_USER_TMPL.format(
        method_name=method_name,
        anchor_line=anchor,
        anchor_line_content=anchor_content,
        analytical_chain=chain,
        denylist=denylist,
        explained_json=explained_json,
    )
    out_b: MDOut = _invoke_structured(llm, MDOut, _RUNB_EXTRACT_SYSTEM, user_b)

    # MERGE results by child_name and found_in tags
    merged: Dict[str, Dict[str, Any]] = {}

    def _push(items: List[MDItem], source: str):
        for it in items:
            name = it.child_name.strip()
            if not name:
                continue
            existing = merged.get(name)
            rec: Dict[str, Any] = {
                "child_name": name,
                "child_type": "Method Call",  # default for SAME_CLASS children
                "code_snippet": it.code_snippet.strip(),
                "code_block": it.code_block.strip(),
                "further_expand": bool(it.requires_definition_expansion),  # for EXTERNAL, set True
                "found_in": source,
                FLAG_FIELD: bool(it.requires_definition_expansion),
            }
            # If this is the EXTERNAL instruction-child, set type accordingly
            if it.requires_definition_expansion:
                rec["child_type"] = "Method Definition"  # external definition expansion

            if existing is None:
                merged[name] = rec
            else:
                # Mark found in both, keep shortest blocks/snippets
                existing["found_in"] = "both" if existing["found_in"] != source else existing["found_in"]
                if len(rec["code_block"]) < len(existing["code_block"]):
                    existing["code_block"] = rec["code_block"]
                if len(rec["code_snippet"]) < len(existing["code_snippet"]):
                    existing["code_snippet"] = rec["code_snippet"]
                # If either says requires expansion, keep it true and adjust type
                if rec[FLAG_FIELD] or existing[FLAG_FIELD]:
                    existing[FLAG_FIELD] = True
                    existing["further_expand"] = True
                    existing["child_type"] = "Method Definition"

    _push(out_a.children, "original")
    _push(out_b.children, "processed")

    # If BOTH runs concluded EXTERNAL and produced no child (edge case),
    # synthesize a single instruction-child so the orchestrator can act.
    if not merged and ((out_a.mode == "EXTERNAL") or (out_b.mode == "EXTERNAL")):
        key = method_name
        merged[key] = {
            "child_name": method_name,
            "child_type": "Method Definition",
            "code_snippet": anchor_content or "",
            "code_block": anchor_content or "",
            "further_expand": True,
            "found_in": "original" if out_a.mode == "EXTERNAL" else "processed",
            FLAG_FIELD: True,
        }

    # Return sorted by name for stability
    return [merged[k] for k in sorted(merged.keys())]

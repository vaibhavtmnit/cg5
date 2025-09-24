# method_call_extractor.py
# End-to-end Method-Call extractor (names only, no spans).
# - Two independent runs:
#   (A) Original code → method calls
#   (B) Processed lines (NL) → method calls
# - Merge by child_name with found_in: "original" | "processed" | "both"
# - Works with ANY focus name; model infers whether it's a method, variable, or call-result at the anchor line.
#
# Requires:
#   pip install langchain langchain-openai pydantic
#
# You provide the LLM instances (AzureChatOpenAI) from your app and pass them in.

from __future__ import annotations

from typing import List, Optional, Dict, TypedDict
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


# ─────────────────────────────────────────────────────────────────────────────
# TypedDicts — placeholders matching your contract (replace with your own if you already have them)
# ─────────────────────────────────────────────────────────────────────────────

class MethodCallInput(TypedDict):
    object_name: str            # focus name (method name, variable name, or call-result method name)
    java_code: str              # full source string
    java_code_line: int         # 1-based line number anchoring the occurrence to analyze
    analytical_chain: str       # up to 2 predecessors, e.g., "A->B->C" (same-file only)


class ChildRecord(TypedDict):
    child_name: str
    child_type: str                 # you may fill from a separate list; leave "" if not used here
    code_snippet: str               # minimal fragment showing the child occurrence
    code_block: str                 # minimal block showing parent+child+relation (often the full statement)
    further_expand: bool            # you’ll decide eligibility later; default False here
    found_in: str                   # "original" | "processed" | "both"


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models for STRUCTURED LLM OUTPUT (internal only)
# ─────────────────────────────────────────────────────────────────────────────

class MCItem(BaseModel):
    child_name: str
    code_snippet: str
    code_block: str
    conditioned: bool = False              # if edge occurs under guard/loop/try (annotation only)
    guards: List[str] = Field(default_factory=list)
    confidence: float
    comment: Optional[str] = None          # short 1-liner, no chain-of-thought

    @field_validator("confidence")
    @classmethod
    def _clip_conf(cls, v: float) -> float:
        return float(max(0.0, min(1.0, v)))


class MCOut(BaseModel):
    children: List[MCItem] = Field(default_factory=list)
    uncertain: List[MCItem] = Field(default_factory=list)
    stop_reason: Optional[str] = None


class LineExplained(BaseModel):
    line: int
    text: str


class ExplainOut(BaseModel):
    lines: List[LineExplained] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Constants — denylist & few-shot examples (generic, NOT from your earlier code)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DENYLIST = [
    "System.out.println",
    "logger.info",
    "logger.debug",
    "logger.trace",
    "Objects.requireNonNull",
    "Collections.emptyList",
]

_RUNA_SYSTEM = """
Task: Extract method calls directly linked to the input object (one-hop adjacency), using ONLY the original Java code.
Return STRICT JSON. Prefer empty results over guesses.

How to interpret the input object name (infer from the anchor line and context):
• If it's a METHOD (focus=method): return only direct UNQUALIFIED calls inside that method's body
  (e.g., helper(), this.helper(), super.helper()).
• If it's a VARIABLE (focus=object): return only one-hop calls where RECEIVER == that variable name
  (e.g., obj.doIt(...)). Do NOT include the next chained hop.
• If it's a CALL RESULT (focus=call_result, i.e., the named method appears as a callee at the anchor):
  return only the IMMEDIATE next chained call (e.g., x.a().b() → from a return b).

Mandatory exclusions:
• Do NOT return calls on other receivers, deeper chain hops, or calls inside lambda/anonymous-class bodies.
• Ignore logging/printing and trivial JDK utilities (denylist provided).
• If nothing qualifies, return children=[].

For each kept child, include: child_name, code_snippet (exact fragment), code_block (smallest original block showing parent+child+relation),
confidence [0,1], optional conditioned/guards and a short comment (≤12 words).
""".strip()

_RUNA_EXAMPLES = """
Few-shot examples (diverse):

1) Method focus (unqualified only)
class S { void work(){ init(); worker.run(); client.fetch().map(x -> x.id()); this.flush(); } void init(){} void flush(){} }
Input: object_name="work" → children = ["init","flush"]

2) Object focus with chain
void m(){ Processor p=new Processor(); p.stage().commit(); p.reset(); other.tick(); }
Input: object_name="p" → children=["stage","reset"]   # not "commit", not "tick"

3) Aliasing excludes deeper hops
void m(){ R r = user.profile(); r.update(); }
Input: object_name="user" → children=["profile"]       # not "update"

4) Lambda body excluded
void m(){ items.forEach(it -> it.process()); }
Input: object_name="items" → children=["forEach"]      # not "process"

5) Call-result next hop
void m(){ db.connect().query().close(); }
Input: object_name="connect" → children=["query"]
Input: object_name="query"   → children=["close"]

6) Cast allowed; ternary receiver not allowed
void m(){ ((Writer) out).write(s); (cond ? out : alt).flush(); }
Input: object_name="out" → children=["write"]          # exclude flush under ternary receiver
""".strip()

_RUNB_EXPLAIN_SYSTEM = """
Convert the Java code into concise, factual natural language, one sentence per source line.
Keep identifiers (variable names, method names, receivers) intact. No speculation. No added lines.
Return STRICT JSON mapping each line number (1-based) to its sentence.
""".strip()

_RUNB_EXTRACT_SYSTEM = """
Task: Using the natural-language per-line explanations, extract method calls directly linked to the input object.
Apply the SAME one-hop rules as in the original-code run (UNQUALIFIED for method focus; RECEIVER==object for variable focus; next chained hop for call_result focus).
Anchor at the given line; prefer the occurrence nearest to that line within the same enclosing method/initializer.
Return STRICT JSON. Prefer empty over guesses.
""".strip()

_RUNB_EXAMPLES = """
Few-shots in NL form:

A) "5: call init() with no receiver; then worker.run()" → method focus="work" → ["init"]
B) "7: variable p calls stage(); chained .commit()"     → object focus="p"   → ["stage"]
C) "11: db.connect() then chained .query()"             → call_result="connect" → ["query"]
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_run_a_prompts(
    *,
    code: str,
    object_name: str,
    anchor_line: int,
    analytical_chain: str,
    denylist: List[str],
    top_k: int,
) -> Dict[str, str]:
    system = _RUNA_SYSTEM
    user = (
        f"OBJECT_NAME: {object_name}\n"
        f"ANCHOR_LINE (1-based): {anchor_line}\n"
        f"ANALYTICAL_CHAIN (last up to 2): {analytical_chain}\n"
        f"TOP_K: {top_k}\n"
        f"DENYLIST: {denylist}\n\n"
        "CODE:\n"
        f"{code}\n\n"
        f"{_RUNA_EXAMPLES}\n"
        "Selection rubric:\n"
        "1) One-hop adjacency to the focused occurrence\n"
        "2) Proximity to ANCHOR_LINE (same method/initializer)\n"
        "3) Flow impact (consumes/transforms/produces focus)\n"
        "4) Not denylisted; not inside lambda/anonymous class\n\n"
        'Output JSON schema: {"children":[{...}], "uncertain":[{...}], "stop_reason":"...|null"}\n'
        "Return ONLY the JSON object."
    )
    return {"system": system, "user": user}


def _build_explain_prompts(*, code: str) -> Dict[str, str]:
    system = _RUNB_EXPLAIN_SYSTEM
    user = (
        "Return one sentence per ORIGINAL line (1-based), preserving identifiers.\n"
        "Do not merge or split lines. No extra commentary.\n\n"
        "CODE:\n"
        f"{code}"
    )
    return {"system": system, "user": user}


def _build_run_b_prompts(
    *,
    explained_lines_json: str,
    object_name: str,
    anchor_line: int,
    analytical_chain: str,
    denylist: List[str],
    top_k: int,
) -> Dict[str, str]:
    system = _RUNB_EXTRACT_SYSTEM
    user = (
        f"OBJECT_NAME: {object_name}\n"
        f"ANCHOR_LINE (1-based): {anchor_line}\n"
        f"ANALYTICAL_CHAIN (last up to 2): {analytical_chain}\n"
        f"TOP_K: {top_k}\n"
        f"DENYLIST: {denylist}\n\n"
        "LINES_NL (JSON array of {line, text}):\n"
        f"{explained_lines_json}\n\n"
        f"{_RUNB_EXAMPLES}\n"
        'Output JSON schema: {"children":[{...}], "uncertain":[{...}], "stop_reason":"...|null"}\n'
        "Return ONLY the JSON object."
    )
    return {"system": system, "user": user}


# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers (structured output + one retry)
# ─────────────────────────────────────────────────────────────────────────────

def _invoke_structured(llm: AzureChatOpenAI, schema, system: str, user: str, retry: bool = True):
    try:
        return llm.with_structured_output(schema).invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
    except Exception:
        if not retry:
            raise
        # One nudge retry for strict JSON
        user2 = user + "\n\nREMINDER: Return ONLY a valid JSON object. Use empty lists when unsure."
        return llm.with_structured_output(schema).invoke(
            [SystemMessage(content=system), HumanMessage(content=user2)]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API — the extractor you call
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractorConfig:
    top_k: int = 5
    denylist: List[str] = None

    def get_denylist(self) -> List[str]:
        return list(DEFAULT_DENYLIST if self.denylist is None else self.denylist)


def extract_method_calls(
    llm_fast: AzureChatOpenAI,
    *,
    request: MethodCallInput,
    config: Optional[ExtractorConfig] = None,
) -> List[ChildRecord]:
    """
    Two-run method-call extraction with merge-by-name and `found_in` tagging.
    - llm_fast: your AzureChatOpenAI instance (e.g., o3-mini, temperature=0).
    - request: TypedDict with object_name, java_code, java_code_line, analytical_chain.
    - config: optional ExtractorConfig (top_k, denylist).
    Returns: list[ChildRecord]
    """
    cfg = config or ExtractorConfig()
    denylist = cfg.get_denylist()
    top_k = cfg.top_k

    object_name = request["object_name"]
    code = request["java_code"]
    anchor = int(request["java_code_line"])
    chain = request.get("analytical_chain", "")

    # RUN A — Original code
    pa = _build_run_a_prompts(
        code=code,
        object_name=object_name,
        anchor_line=anchor,
        analytical_chain=chain,
        denylist=denylist,
        top_k=top_k,
    )
    out_a: MCOut = _invoke_structured(llm_fast, MCOut, pa["system"], pa["user"])

    # RUN B — Processed code (explain lines ➜ extract)
    pb1 = _build_explain_prompts(code=code)
    explained: ExplainOut = _invoke_structured(llm_fast, ExplainOut, pb1["system"], pb1["user"])

    # serialize explained lines back to JSON string (model already returned JSON via structured output)
    # but we need a plain string to include in the next prompt; pydantic .model_dump_json is fine.
    explained_json = ExplainOut(lines=explained.lines).model_dump_json()

    pb2 = _build_run_b_prompts(
        explained_lines_json=explained_json,
        object_name=object_name,
        anchor_line=anchor,
        analytical_chain=chain,
        denylist=denylist,
        top_k=top_k,
    )
    out_b: MCOut = _invoke_structured(llm_fast, MCOut, pb2["system"], pb2["user"])

    # MERGE — by child_name, set found_in
    merged: Dict[str, ChildRecord] = {}

    def _add_from(source: str, items: List[MCItem]):
        for it in items:
            name = it.child_name.strip()
            if not name:
                continue
            rec = merged.get(name)
            # Build a ChildRecord from this MCItem
            child = ChildRecord(
                child_name=name,
                child_type="",  # you can fill from your external enum later
                code_snippet=it.code_snippet.strip(),
                code_block=it.code_block.strip(),
                further_expand=False,
                found_in=source,
            )
            if rec is None:
                merged[name] = child
            else:
                # Prefer "both"; pick the shorter block that still includes parent+child (we assume both do)
                rec["found_in"] = "both" if rec["found_in"] != source else rec["found_in"]
                # Keep shorter code_block/snippet
                if len(child["code_block"]) < len(rec["code_block"]):
                    rec["code_block"] = child["code_block"]
                if len(child["code_snippet"]) < len(rec["code_snippet"]):
                    rec["code_snippet"] = child["code_snippet"]

    _add_from("original", out_a.children)
    _add_from("processed", out_b.children)

    # Return list sorted by name for stability
    results = [merged[k] for k in sorted(merged.keys())]
    return results

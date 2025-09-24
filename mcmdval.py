Awesome—here are two relationship validators you can drop in right now:

validate_method_call_relations(...) — checks that each proposed child is a valid method-call child of the input object under your one-hop rules.

validate_method_definition_relations(...) — checks that a method-name focus is SAME_CLASS vs EXTERNAL, and that the child correctly signals definition expansion when external.

Both functions:

take your AzureChatOpenAI instance (you pass it in)

accept your existing TypedDict request (I don’t redefine it; I only access keys)

accept a list of your child TypedDicts to validate (again, not redefining your schema)

return a list of verdict dicts with valid, confidence, reason, etc., so you can gate additions to the tree

include few-shot examples and strict, minimal prompts

avoid top-k; prefer empty over guessing

Save as relationship_validators.py.



# relationship_validators.py
# Validator utilities for:
#  1) Method Call relations
#  2) Method Definition relations (SAME_CLASS vs EXTERNAL)
#
# You pass your AzureChatOpenAI instance in; this module never creates clients.
#
# Requires:
#   pip install langchain langchain-openai pydantic

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic output schemas (structured LLM responses)
# ─────────────────────────────────────────────────────────────────────────────

class Verdict(BaseModel):
    child_name: str
    valid: bool
    confidence: float
    reason: str
    # Optional normalizations/hints:
    normalized_child_name: Optional[str] = None
    # Echoes so you can trace what was judged:
    code_snippet_checked: Optional[str] = None
    code_block_checked: Optional[str] = None

    @field_validator("confidence")
    @classmethod
    def _clip_conf(cls, v: float) -> float:
        return float(max(0.0, min(1.0, v)))


class VerdictsOut(BaseModel):
    verdicts: List[Verdict] = Field(default_factory=list)
    summary: Optional[str] = None


class MDVerdict(BaseModel):
    child_name: str
    mode: str                            # "SAME_CLASS" | "EXTERNAL"
    valid: bool
    confidence: float
    reason: str
    requires_definition_expansion_consistent: Optional[bool] = None
    code_snippet_checked: Optional[str] = None
    code_block_checked: Optional[str] = None

    @field_validator("confidence")
    @classmethod
    def _clip_conf_md(cls, v: float) -> float:
        return float(max(0.0, min(1.0, v)))


class MDVerdictsOut(BaseModel):
    verdicts: List[MDVerdict] = Field(default_factory=list)
    summary: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Generic helper for structured calls (one retry)
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
# Method Call — Relationship validator
# Validates one-hop adjacency according to your rules.
# ─────────────────────────────────────────────────────────────────────────────

_MC_SYSTEM = """
You are a strict validator for method-call relationships. Return STRICT JSON only.
Decide, for each candidate child, whether it is a VALID immediate method-call child of the input object at the anchored occurrence.

Focus kinds (infer from anchor line and context):
• METHOD focus → Only direct UNQUALIFIED calls inside that method body (helper(), this.helper(), super.helper()).
• OBJECT focus → Only one-hop calls where RECEIVER == object name (obj.doIt(...)). Do NOT include the next chained hop.
• CALL_RESULT focus → Only the IMMEDIATE next chained call after the named call (x.a().b() → child 'b' for focus 'a').

Mandatory exclusions:
• Calls on other receivers, deeper chain hops, anything inside lambda/anonymous-class bodies.
• Logging/printing and trivial JDK utilities (denylist if provided).
• Cross-file or other methods’ bodies are out of scope.

For each candidate, return: valid (bool), confidence [0..1], reason (short), and optionally a normalized_child_name.
Prefer EMPTY verdicts over guesses when unsure.
""".strip()

_MC_FEWSHOTS = """
Few-shot examples (generic; not tied to the user code):

1) METHOD focus
class S { void work(){ init(); worker.run(); this.flush(); } void init(){} void flush(){} }
Input: focus="work"; candidates=['init','flush','run']
→ valid: init=true, flush=true, run=false (run is on receiver 'worker')

2) OBJECT focus with chain
void m(){ X x=new X(); x.a().b(); x.c(); y.d(); }
Input: focus="x"; candidates=['a','b','c','d']
→ a=true, c=true, b=false (next hop), d=false (different receiver)

3) CALL_RESULT focus
void m(){ db.connect().query().close(); }
Input: focus="connect"; candidates=['query','close']
→ query=true, close=false (two hops away)
""".strip()

def validate_method_call_relations(
    llm: AzureChatOpenAI,
    *,
    request: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    denylist: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Validate a batch of candidate method-call children for the current focus.

    Parameters
    ----------
    llm : AzureChatOpenAI
        Your configured Azure client (e.g., o3-mini, temp=0).
    request : TypedDict-like
        {
          "object_name": str,
          "java_code": str,
          "java_code_line": int,               # 1-based
          "java_code_line_content": str,       # optional but recommended
          "analytical_chain": str
        }
    candidates : list[child TypedDict]
        Each must contain at least: child_name, code_snippet, code_block.
    denylist : list[str] | None
        Optional noise denylist (logger, println, etc.). If None, it's omitted from prompt.

    Returns
    -------
    list[dict]  # verdicts
        Each verdict: {
          "child_name", "valid", "confidence", "reason",
          "normalized_child_name"?, "code_snippet_checked"?, "code_block_checked"?
        }
    """
    object_name = request["object_name"]
    code = request["java_code"]
    anchor_line = int(request["java_code_line"])
    anchor_line_content = request.get("java_code_line_content", "")
    chain = request.get("analytical_chain", "")

    # Build user prompt
    denyline = f"DENYLIST: {denylist}\n" if denylist else ""
    user = (
        f"FOCUS_NAME: {object_name}\n"
        f"ANCHOR_LINE (1-based): {anchor_line}\n"
        f"ANCHOR_LINE_CONTENT: {anchor_line_content}\n"
        f"ANALYTICAL_CHAIN (≤2): {chain}\n"
        f"{denyline}"
        "CANDIDATE_CHILDREN (JSON array of objects):\n"
        f"{candidates}\n\n"
        "CODE:\n"
        f"{code}\n\n"
        f"{_MC_FEWSHOTS}\n"
        "Judging rubric:\n"
        "1) One-hop adjacency relative to the anchored occurrence of the focus.\n"
        "2) For METHOD focus: only unqualified calls in that method body.\n"
        "3) For OBJECT focus: receiver must equal the object name; exclude next hops.\n"
        "4) For CALL_RESULT focus: the immediate next hop only.\n"
        "5) Exclude lambda/anonymous-class internals and denylisted utilities.\n\n"
        'Output JSON schema: {"verdicts":[{"child_name":"...","valid":true|false,"confidence":0.0,"reason":"...","normalized_child_name":"...|null","code_snippet_checked":"...|null","code_block_checked":"...|null"}], "summary":"...|null"}\n'
        "Return ONLY the JSON object."
    )

    out: VerdictsOut = _invoke_structured(llm, VerdictsOut, _MC_SYSTEM, user)
    # Convert Pydantic to plain dicts for your pipeline
    return [v.model_dump() for v in out.verdicts]


# ─────────────────────────────────────────────────────────────────────────────
# Method Definition — Relationship validator
# Confirms SAME_CLASS vs EXTERNAL and expansion flag consistency.
# ─────────────────────────────────────────────────────────────────────────────

_MD_SYSTEM = """
You validate relationships for a METHOD-NAME focus with respect to its definition.
Return STRICT JSON only.

Decide MODE:
• SAME_CLASS if this code contains a method declaration with that exact name in the same class/compilation unit.
• EXTERNAL if no such declaration exists here (the method is referenced but defined elsewhere).

Validation rules:
• SAME_CLASS → children must be direct UNQUALIFIED calls inside that method body (helper(), this.helper(), super.helper()).
• EXTERNAL  → exactly ONE instruction-child is expected with 'requires_definition_expansion=true', pointing the orchestrator to expand at the defining class.
  The child_name is the method name, child_type should be 'Method Definition', and code_snippet/code_block should reflect the anchor line content/call site.

Exclude calls on other receivers, deeper chain hops, lambda/anonymous-class internals, and trivial utilities.

For each candidate child, return:
- mode ("SAME_CLASS" or "EXTERNAL")
- valid (bool), confidence [0..1], reason
- requires_definition_expansion_consistent (bool|null): only meaningful in EXTERNAL mode
Include code_snippet_checked and code_block_checked for traceability.
""".strip()

_MD_FEWSHOTS = """
Few-shot examples:

A) SAME_CLASS
class U {
  void boot(){ prep(); worker.run(); this.flush(); }
  void prep(){} void flush(){}
}
Focus: "boot"
Candidates: ['prep','flush','run']
→ prep: valid, SAME_CLASS; flush: valid; run: invalid (receiver 'worker').

B) EXTERNAL
class V {
  void main(){ engine.start(); } // no 'start' definition here
}
Focus: "start"
Candidate: one instruction-child { child_name:'start', child_type:'Method Definition', requires_definition_expansion:true }
→ valid, EXTERNAL, requires_definition_expansion_consistent:true
""".strip()

def validate_method_definition_relations(
    llm: AzureChatOpenAI,
    *,
    request: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    denylist: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Validate a batch of candidate children for METHOD DEFINITION focus.

    Parameters
    ----------
    llm : AzureChatOpenAI
    request : TypedDict-like
        {
          "object_name": str,                 # method name to analyze
          "java_code": str,
          "java_code_line": int,              # 1-based
          "java_code_line_content": str,      # anchor line content
          "analytical_chain": str
        }
    candidates : list[child TypedDict]
        (From your extractor) Each should include child_name, child_type,
        code_snippet, code_block, further_expand, and ideally a boolean flag like
        requires_definition_expansion (whatever field name you use).
    denylist : list[str] | None
        Optional noise denylist.

    Returns
    -------
    list[dict]  # MD verdicts
        Each item: {
          "child_name", "mode", "valid", "confidence", "reason",
          "requires_definition_expansion_consistent"?, "code_snippet_checked"?, "code_block_checked"?
        }
    """
    method_name = request["object_name"]
    code = request["java_code"]
    anchor_line = int(request["java_code_line"])
    anchor_line_content = request.get("java_code_line_content", "")
    chain = request.get("analytical_chain", "")

    denyline = f"DENYLIST: {denylist}\n" if denylist else ""
    user = (
        f"FOCUS_METHOD_NAME: {method_name}\n"
        f"ANCHOR_LINE (1-based): {anchor_line}\n"
        f"ANCHOR_LINE_CONTENT: {anchor_line_content}\n"
        f"ANALYTICAL_CHAIN (≤2): {chain}\n"
        f"{denyline}"
        "CANDIDATE_CHILDREN (JSON array of objects):\n"
        f"{candidates}\n\n"
        "CODE:\n"
        f"{code}\n\n"
        f"{_MD_FEWSHOTS}\n"
        "Judging rubric:\n"
        "1) Decide MODE from the presence/absence of a method declaration with this name in this code.\n"
        "2) SAME_CLASS: only unqualified calls inside that method body are valid children.\n"
        "3) EXTERNAL: a single instruction-child with requires_definition_expansion=true is expected; others invalid.\n"
        "4) Exclude lambda internals and denylisted utilities.\n\n"
        'Output JSON schema: {"verdicts":[{"child_name":"...","mode":"SAME_CLASS|EXTERNAL","valid":true|false,"confidence":0.0,"reason":"...","requires_definition_expansion_consistent":true|false|null,"code_snippet_checked":"...|null","code_block_checked":"...|null"}], "summary":"...|null"}\n'
        "Return ONLY the JSON object."
    )

    out: MDVerdictsOut = _invoke_structured(llm, MDVerdictsOut, _MD_SYSTEM, user)
    return [v.model_dump() for v in out.verdicts]



from langchain_openai import AzureChatOpenAI
from relationship_validators import (
    validate_method_call_relations,
    validate_method_definition_relations,
)

llm_fast = AzureChatOpenAI(azure_deployment="o3-mini", temperature=0)

request = {
    "object_name": "x",                     # method/variable/call-result (for call validator)
    "java_code": "...full code here...",
    "java_code_line": 17,
    "java_code_line_content": "x.start().finish();",
    "analytical_chain": "A->B->x",
}

# candidates = list of child dicts from your extractor
verdicts_calls = validate_method_call_relations(llm_fast, request=request, candidates=candidates)

# For method-definition validator:
request_md = {
    "object_name": "boot",                  # the method name focus
    "java_code": "...full code here...",
    "java_code_line": 6,
    "java_code_line_content": "svc.boot();",  # if external call-site
    "analytical_chain": "Class->boot",
}
verdicts_md = validate_method_definition_relations(llm_fast, request=request_md, candidates=candidates)


Notes & best practices baked in

Validators are focus-aware and anchor-aware; they apply your adjacency rules identically to extraction.

They return confidence and a short reason so you can gate acceptance (e.g., accept only confidence ≥ 0.7).

Few-shots are generic (not using your earlier code), covering method focus, object focus, chained calls, and EXTERNAL method cases.

If you maintain a requires_definition_expansion flag under a different name, you can still pass it inside candidates; the validator doesn’t require a specific field name, it only checks consistency based on the candidate’s content and the mode it decides.

If you want these validators to also auto-correct child names (e.g., stripping () or normalizing casing) and return a “fixed” candidate list, I can add a helper to merge verdicts back into your children in one go.


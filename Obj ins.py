# object_instantiation_pipeline.py
# End-to-end "Approach A" pipeline:
#   Phase-1: instantiations-only (variables/fields) within focus scope
#   Phase-2: eligible uses of those variables (and optional external-in-scope vars), excluding argument uses
#   Single validator: validates the relationship between Phase-1 and Phase-2 outputs
#
# Requires:
#   pip install langchain langchain-openai

from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional, Tuple, Set
import json

from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

class EC(TypedDict):
    name: str
    code_snippet: str
    code_block: str
    further_expand: bool
    confidence: float
    conditioned: bool
    guards: List[str]
    comment: str            # e.g. "instantiated variable", "source variable for 'g' on same line", "field instantiated on focus", "used as receiver", ...
    variant: int            # 0-based index for same-name occurrences on the SAME line (left-to-right)

class PipelineInput(TypedDict):
    object_name: str
    java_code: str
    java_code_line: int
    java_code_line_content: str
    analytical_chain: str
    include_external_uses: bool  # Phase-2: also show external-in-scope uses

class VerdictTD(TypedDict):
    name: str
    valid: bool
    confidence: float
    reason: str

class PipelineResult(TypedDict):
    instantiations: List[EC]
    uses: List[EC]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DENYLIST = [
    "System.out.println",
    "logger.info",
    "logger.debug",
    "logger.trace",
    "Objects.requireNonNull",
    "Collections.emptyList",
]

def _invoke_json(llm: AzureChatOpenAI, *, system: str, user: str, retry: bool = True) -> Any:
    msgs = [SystemMessage(content=system), HumanMessage(content=user)]
    try:
        return json.loads(llm.invoke(msgs).content)
    except Exception:
        if not retry:
            raise
        user2 = user + "\n\nREMINDER: Return ONLY a valid JSON object. If nothing, return {\"children\": []}."
        return json.loads(llm.invoke([SystemMessage(content=system), HumanMessage(content=user2)]).content)

def _norm_ec_list(items: List[Dict[str, Any]]) -> List[EC]:
    out: List[EC] = []
    for it in items or []:
        ec: EC = {
            "name": str(it.get("name", "")).strip(),
            "code_snippet": str(it.get("code_snippet", "")).strip(),
            "code_block": str(it.get("code_block", "")).strip(),
            "further_expand": bool(it.get("further_expand", False)),
            "confidence": float(it.get("confidence", 0.0)),
            "conditioned": bool(it.get("conditioned", False)),
            "guards": list(it.get("guards", []) or []),
            "comment": str(it.get("comment", "")).strip(),
            "variant": int(it.get("variant", 0)),
        }
        if not ec["name"]:
            continue
        ec["confidence"] = max(0.0, min(1.0, ec["confidence"]))
        if ec["variant"] < 0:
            ec["variant"] = 0
        out.append(ec)
    return out

def _merge_key_with_comment(ec: EC) -> Tuple[str, str, str]:
    # Preserve same-name duplicates on the same line by including code_snippet + comment in key
    return (ec["name"], ec["code_snippet"], ec.get("comment", ""))

def _merge_ec_lists(a: List[EC], b: List[EC]) -> List[EC]:
    """
    Merge two EC lists while preserving duplicates (variants).
    Key: (name, code_snippet, comment)
    Keep shorter code_block, higher confidence; OR flags; union guards; prefer smaller variant on conflict.
    """
    by: Dict[Tuple[str, str, str], EC] = {}
    def push(lst: List[EC]):
        for it in lst:
            key = _merge_key_with_comment(it)
            if key not in by:
                by[key] = it
            else:
                cur = by[key]
                if it["code_block"] and (not cur["code_block"] or len(it["code_block"]) < len(cur["code_block"])):
                    cur["code_block"] = it["code_block"]
                if it["confidence"] > cur["confidence"]:
                    cur["confidence"] = it["confidence"]
                cur["guards"] = list(dict.fromkeys(cur["guards"] + it["guards"]))
                cur["conditioned"] = cur["conditioned"] or it["conditioned"]
                cur["further_expand"] = cur["further_expand"] or it["further_expand"]
                if it["variant"] < cur["variant"]:
                    cur["variant"] = it["variant"]
    push(a); push(b)
    # deterministic ordering
    return sorted(by.values(), key=lambda ec: (ec["code_snippet"], ec["variant"], ec["name"]))


# ─────────────────────────────────────────────────────────────────────────────
# Phase-1 — Instantiations only (no NL pass)
# ─────────────────────────────────────────────────────────────────────────────

_PHASE1_SYSTEM = """
You extract ONLY variable/field creations within the FOCUS SCOPE.

STRICT RULES:
• Children are variables/fields only (never class names).
• INCLUDE:
  1) Direct constructor assignment (declaration or reassignment):
     "Foo f = new Foo(...);"  → child 'f' (code_snippet = entire line; comment "instantiated variable").
  2) Created from another variable's method call on the same line:
     "G g = v.create();"      → children 'g' (comment "instantiated variable") AND 'v'
                                (comment "source variable for 'g' on same line").
     Chained: "G g = v.create().tune();" → 'g' and 'v' (no 'tune').
  3) Field assignment on focus object only:
     "x.helper = new Helper();" with focus 'x' → child 'helper' only (comment "field instantiated on focus").
  4) Multi-statements SAME line:
     "Foo a = new Foo(); Bar b = a.make();"
       → 'a'(instantiated, variant 0), 'b'(instantiated, variant 0), 'a'(source for 'b', variant 1).

• EXCLUDE:
  - Instantiation used only as an argument: "call(new Foo())"   → return nothing.
  - Unbound 'new' chained only: "new Foo().init()"             → return nothing.
  - Anything inside lambda/anonymous-class bodies.
  - Trivial/logging utilities (denylist).

• FOCUS SCOPE:
  - Focus=METHOD: that method body only (exclude lambdas/anon).
  - Focus=OBJECT VAR X: smallest enclosing block where X is in scope.
    Include 'X.field = new T()' → child 'field' only.
    Include 'Var y = X.make()'  → children 'y' and 'X' on that line.
  - Focus=CALL-RESULT: the statement returning that result; include var declared from it on the same statement.

• VARIANT INDEXING:
  For each ORIGINAL source line, if the SAME name appears multiple times as separate children,
  assign 'variant' indices 0,1,... in left-to-right order on that line.

OUTPUT (STRICT JSON ONLY):
{"children":[
  {"name":"...", "code_snippet":"<ENTIRE line>", "code_block":"<smallest block>",
   "further_expand": false, "confidence": 0.0-1.0, "conditioned": false, "guards": [],
   "comment":"instantiated variable|source variable for '<other>' on same line|field instantiated on focus",
   "variant": 0}
]}
""".strip()

_PHASE1_FEWSHOTS = """
Examples:

1) Method focus
void m(){ Foo f = new Foo(); f.run(); }
→ [{"name":"f","comment":"instantiated variable","variant":0,"code_snippet":"Foo f = new Foo();"}]

2) Created from another var's method
void m(){ G g = v.create(); }
→ [{"name":"g","comment":"instantiated variable","variant":0,"code_snippet":"G g = v.create();"},
    {"name":"v","comment":"source variable for 'g' on same line","variant":0,"code_snippet":"G g = v.create();"}]

3) Multi-statements same line
void m(){ Foo a = new Foo(); Bar b = a.make(); }
→ [
  {"name":"a","comment":"instantiated variable","variant":0,"code_snippet":"Foo a = new Foo(); Bar b = a.make();"},
  {"name":"b","comment":"instantiated variable","variant":0,"code_snippet":"Foo a = new Foo(); Bar b = a.make();"},
  {"name":"a","comment":"source variable for 'b' on same line","variant":1,"code_snippet":"Foo a = new Foo(); Bar b = a.make();"}
]

4) Field on focus
void m(X x){ x.helper = new Helper(); }
Focus: x → [{"name":"helper","comment":"field instantiated on focus","variant":0,"code_snippet":"x.helper = new Helper();"}]

5) Exclusions
void m(){ call(new Foo()); new Bar().init(); items.forEach(t -> { Temp u = new Temp(); }); }
→ []
""".strip()

def _phase1_build_user(code: str, focus: str, anchor: int, anchor_content: str, chain: str, deny: List[str]) -> str:
    return (
        f"FOCUS_NAME: {focus}\n"
        f"ANCHOR_LINE (1-based): {anchor}\n"
        f"ANCHOR_LINE_CONTENT: {anchor_content}\n"
        f"ANALYTICAL_CHAIN (≤2): {chain}\n"
        f"DENYLIST: {deny}\n\n"
        "CODE:\n" + code + "\n\n" +
        _PHASE1_FEWSHOTS + "\n"
        "Return ONLY the JSON object."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase-2 — Eligible uses (no NL pass)
# ─────────────────────────────────────────────────────────────────────────────

_PHASE2_SYSTEM = """
You extract ELIGIBLE VARIABLE USES within the FOCUS SCOPE for:
  • the given Phase-1 variable names (created in-scope)
  • and (optional) variables that are EXTERNAL but used in-scope (label them).

INCLUDE (examples):
  - Receiver call on var:                   v.m(...), ((T)v).m(...)
  - Field read/write of var:                v.f, v.f = ..., x = v.f
  - Assigned-from / producer:               R r = v.make(...);   r = v;
  - Control/return use:                     if (v != null) {...}   return v;
  - Non-call array/collection assignment:   arr[i] = v;   x = v;

EXCLUDE:
  - ANY case where the variable is passed as a method/constructor ARGUMENT: call(v), sink.accept(v), new Foo(v) → exclude
  - Anything inside lambda/anonymous-class bodies
  - Trivial/logging utilities (denylist) as receivers/callees

External used in scope:
  - If include_external=true and a variable is used in this scope without an in-scope declaration/instantiation,
    emit an EC with comment "external used in scope" (combined with the specific usage comment).

Variant indexing:
  - For each ORIGINAL source line, if the SAME name appears multiple times as separate uses,
    assign variant indices 0,1,... in left-to-right order on that line.

OUTPUT (STRICT JSON ONLY):
{"children":[
  {"name":"<var>", "code_snippet":"<ENTIRE line>", "code_block":"<smallest block>",
   "further_expand": false, "confidence": 0.0-1.0, "conditioned": false, "guards": [],
   "comment":"used as receiver|used in field read|used in field write|used in assignment|used in return|used in condition|external used in scope|used as array/collection element target|used as array/collection value",
   "variant": 0}
]}
""".strip()

_PHASE2_FEWSHOTS = """
Examples:

1) Receiver call
void m(){ v.m(); } with phase1_vars=['v']
→ [{"name":"v","comment":"used as receiver","variant":0,"code_snippet":"v.m();"}]

2) Field read/write
x = v.f;        → {"name":"v","comment":"used in field read","variant":0,"code_snippet":"x = v.f;"}
v.f = y;        → {"name":"v","comment":"used in field write","variant":0,"code_snippet":"v.f = y;"}

3) Producer / assignment
R r = v.make(); → {"name":"v","comment":"used in assignment","variant":0,"code_snippet":"R r = v.make();"}
r = v;          → {"name":"v","comment":"used in assignment","variant":0,"code_snippet":"r = v;"}

4) Control / return
if (v != null) return v; → two ECs for 'v' on same line with variant 0,1

5) Array/collection (non-call)
arr[i] = v; → {"name":"v","comment":"used as array/collection value","variant":0}

6) EXCLUDE argument use
call(v); sink.accept(v); new Foo(v); → exclude

7) External used in scope
void m(){ if (p != null) p.m(); } include_external=true, phase1_vars=[]
→ emit 'p' twice with comments including "external used in scope"
""".strip()

def _phase2_build_user(
    *, code: str, focus: str, anchor: int, anchor_content: str, chain: str,
    deny: List[str], phase1_vars: List[str], include_external: bool
) -> str:
    return (
        f"FOCUS_NAME: {focus}\n"
        f"ANCHOR_LINE (1-based): {anchor}\n"
        f"ANCHOR_LINE_CONTENT: {anchor_content}\n"
        f"ANALYTICAL_CHAIN (≤2): {chain}\n"
        f"DENYLIST: {deny}\n"
        f"PHASE1_VARS: {phase1_vars}\n"
        f"INCLUDE_EXTERNAL: {include_external}\n\n"
        "CODE:\n" + code + "\n\n" +
        _PHASE2_FEWSHOTS + "\n"
        "Return ONLY the JSON object."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single relationship validator (Phase-1 + Phase-2)
# ─────────────────────────────────────────────────────────────────────────────

_VALIDATOR_SYSTEM = """
You validate the RELATIONSHIP between Phase-1 instantiations and Phase-2 eligible uses under the focus scope.

Validate the following:

A) Phase-1 instantiations:
   • Each child is a variable/field created in-scope:
     - Direct 'new' assignment → comment "instantiated variable"
     - Created from another variable's method call → TWO children: new var + source var ("source variable for '<new>' on same line")
     - Field on focus object: x.field = new T() → 'field' only ("field instantiated on focus")
   • NO entries for:
     - 'new' used only as an argument (call(new T()))
     - Unbound 'new' immediately chained (new T().init())
     - Lambdas/anonymous-class bodies
   • SAME-LINE duplicates carry variant 0,1,... left-to-right.

B) Phase-2 uses:
   • Each usage is in-scope and matches allowed categories:
     - receiver, field read/write, assignment, return, condition, array/collection (non-call)
   • EXCLUDES any argument use (call(v), new Foo(v), ...)
   • If include_external=true and var not created/declared in-scope, "external used in scope" tag in comment is acceptable.
   • SAME-LINE duplicates carry variant 0,1,... left-to-right.

C) Cross-check (relationship):
   • Every Phase-2 use of a name that was created in Phase-1 should be consistent (same scope, not argument position).
   • If a Phase-2 'name' was not in Phase-1 and include_external=true, it must be plausibly external-in-scope (param/field/outer var).
   • Variables created from 'v.create()' in Phase-1 must NOT appear as "argument use" in Phase-2 (those are excluded).
   • No class names are used as EC 'name'.

Return STRICT JSON ONLY:
{"verdicts":[{"name":"<scope-label or variable>", "valid":true|false, "confidence":0.0-1.0, "reason":"..."}]}
""".strip()


def _validator_build_user(
    *, code: str, focus: str, anchor: int, anchor_content: str, chain: str, deny: List[str],
    instantiations: List[EC], uses: List[EC], include_external: bool
) -> str:
    return (
        f"FOCUS_NAME: {focus}\n"
        f"ANCHOR_LINE (1-based): {anchor}\n"
        f"ANCHOR_LINE_CONTENT: {anchor_content}\n"
        f"ANALYTICAL_CHAIN (≤2): {chain}\n"
        f"DENYLIST: {deny}\n"
        f"INCLUDE_EXTERNAL: {include_external}\n\n"
        "PHASE1_INSTANTIATIONS (EC[]):\n" + json.dumps(instantiations, ensure_ascii=False) + "\n\n"
        "PHASE2_USES (EC[]):\n" + json.dumps(uses, ensure_ascii=False) + "\n\n"
        "CODE:\n" + code + "\n"
        "Return ONLY the JSON object."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_instantiation_usage_pipeline(
    llm: AzureChatOpenAI,
    *,
    request: PipelineInput,
    denylist: Optional[List[str]] = None,
) -> PipelineResult:
    """
    Runs Phase-1 then Phase-2. Returns {'instantiations': [...], 'uses': [...] }.
    """
    focus = request["object_name"]
    code = request["java_code"]
    anchor = int(request["java_code_line"])
    anchor_content = request.get("java_code_line_content", "")
    chain = request.get("analytical_chain", "")
    include_external = bool(request.get("include_external_uses", True))
    deny = denylist or DEFAULT_DENYLIST

    # Phase-1
    user1 = _phase1_build_user(code, focus, anchor, anchor_content, chain, deny)
    out1 = _invoke_json(llm, system=_PHASE1_SYSTEM, user=user1)
    insts = _norm_ec_list(out1.get("children", []))

    # Collect Phase-1 var names that represent actual new variables/fields
    # (We include all names from Phase-1; you can filter to comments==["instantiated variable", "field instantiated on focus"] if desired.)
    phase1_var_names: Set[str] = {ec["name"] for ec in insts}

    # Phase-2
    user2 = _phase2_build_user(
        code=code, focus=focus, anchor=anchor, anchor_content=anchor_content, chain=chain,
        deny=deny, phase1_vars=sorted(phase1_var_names), include_external=include_external
    )
    out2 = _invoke_json(llm, system=_PHASE2_SYSTEM, user=user2)
    uses = _norm_ec_list(out2.get("children", []))

    return {"instantiations": insts, "uses": uses}


def validate_instantiation_usage_relationship(
    llm: AzureChatOpenAI,
    *,
    request: PipelineInput,
    pipeline_result: PipelineResult,
    denylist: Optional[List[str]] = None,
) -> List[VerdictTD]:
    """
    Validates Phase-1 + Phase-2 relationship in one shot. Returns verdict list.
    """
    focus = request["object_name"]
    code = request["java_code"]
    anchor = int(request["java_code_line"])
    anchor_content = request.get("java_code_line_content", "")
    chain = request.get("analytical_chain", "")
    include_external = bool(request.get("include_external_uses", True))
    deny = denylist or DEFAULT_DENYLIST

    insts = pipeline_result.get("instantiations", [])
    uses = pipeline_result.get("uses", [])

    user = _validator_build_user(
        code=code, focus=focus, anchor=anchor, anchor_content=anchor_content, chain=chain,
        deny=deny, instantiations=insts, uses=uses, include_external=include_external
    )
    out = _invoke_json(llm, system=_VALIDATOR_SYSTEM, user=user)
    verdicts_raw = out.get("verdicts", [])
    verdicts: List[VerdictTD] = []
    for v in verdicts_raw:
        nm = str(v.get("name", "")).strip()
        if not nm:
            continue
        try:
            conf = float(v.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        verdicts.append({
            "name": nm,
            "valid": bool(v.get("valid", False)),
            "confidence": conf,
            "reason": str(v.get("reason", "")).strip(),
        })
    return verdicts

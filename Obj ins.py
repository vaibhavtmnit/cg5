# object_instantiation_extractor_and_validator.py
# Updated Object Instantiation — extractor + validator (TypedDict only)
# Rules implemented:
#  • Return variables created from a class (NOT class names).
#  • If created from another variable’s method call (e.g., Gome g = v.create()), return TWO children: 'g' and 'v'.
#  • If instantiation is used only as an argument (call(new Foo())), return nothing (Argument tracker handles it).
#  • For field assignment x.helper = new Helper(); return ONLY 'helper'.
#  • Handle multiple occurrences of the SAME name on the SAME line with `variant` (0,1,...) and `comment`.
#  • Two runs: original code, NL per line → merge by (name, code_snippet, comment).
#
# Requires:
#   pip install langchain langchain-openai

from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional, Tuple
import json

from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# ─────────────────────────────────────────────────────────────────────────────
# TypedDicts — EC shape now includes variant + comment
# ─────────────────────────────────────────────────────────────────────────────

class EC(TypedDict):
    name: str               # variable/field name created or used as source on that line
    code_snippet: str       # ENTIRE original line containing the creation/assignment
    code_block: str         # smallest block showing parent+child relation (can be same as snippet)
    further_expand: bool
    confidence: float
    conditioned: bool
    guards: List[str]
    # NEW:
    comment: str            # brief hint: "instantiated variable", "source variable for 'g' on same line", etc.
    variant: int            # 0-based index for same-name occurrences on the SAME line (left-to-right)

class InstantiationInput(TypedDict):
    object_name: str               # focus name (method, object var, or call-result)
    java_code: str                 # full source text
    java_code_line: int            # 1-based anchor line
    java_code_line_content: str    # exact code on that line
    analytical_chain: str          # up to two predecessors, "a->b->c"

class VerdictTD(TypedDict):
    name: str
    valid: bool
    confidence: float
    reason: str


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
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
    """Call the LLM and parse a single JSON object. Retry once with a 'strict JSON' reminder if needed."""
    msgs = [SystemMessage(content=system), HumanMessage(content=user)]
    try:
        txt = llm.invoke(msgs).content
        return json.loads(txt)
    except Exception:
        if not retry:
            raise
        user2 = user + "\n\nREMINDER: Return ONLY a valid JSON object. If nothing, return {'children': []}."
        txt = llm.invoke([SystemMessage(content=system), HumanMessage(content=user2)]).content
        return json.loads(txt)

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
        # clamp confidence
        if ec["confidence"] < 0.0: ec["confidence"] = 0.0
        if ec["confidence"] > 1.0: ec["confidence"] = 1.0
        # clamp variant
        if ec["variant"] < 0: ec["variant"] = 0
        out.append(ec)
    return out

def _merge_key(ec: EC) -> Tuple[str, str, str]:
    """Composite merge key that preserves same-name variants on the same line."""
    return (ec["name"], ec["code_snippet"], ec.get("comment", ""))

def _merge_ecs(a: List[EC], b: List[EC]) -> List[EC]:
    """
    Merge by composite key (name, code_snippet, comment) to preserve same-name duplicates on the same line.
    Keep shortest code_block, highest confidence; OR flags; union guards; prefer lower variant if conflict.
    """
    by: Dict[Tuple[str, str, str], EC] = {}
    def push(lst: List[EC]):
        for it in lst:
            key = _merge_key(it)
            if key not in by:
                by[key] = it
            else:
                cur = by[key]
                # prefer shorter code_block/snippet
                if it["code_block"] and (not cur["code_block"] or len(it["code_block"]) < len(cur["code_block"])):
                    cur["code_block"] = it["code_block"]
                if it["code_snippet"] and (not cur["code_snippet"] or len(it["code_snippet"]) < len(cur["code_snippet"])):
                    cur["code_snippet"] = it["code_snippet"]
                # max confidence
                if it["confidence"] > cur["confidence"]:
                    cur["confidence"] = it["confidence"]
                # OR flags / union guards
                cur["conditioned"] = cur["conditioned"] or it["conditioned"]
                cur["further_expand"] = cur["further_expand"] or it["further_expand"]
                cur["guards"] = list(dict.fromkeys(cur["guards"] + it["guards"]))
                # prefer smaller variant if conflict
                if it["variant"] < cur["variant"]:
                    cur["variant"] = it["variant"]
    push(a); push(b)
    # deterministic ordering: by code_snippet line first, then variant, then name
    return sorted(by.values(), key=lambda ec: (ec["code_snippet"], ec["variant"], ec["name"]))


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTOR — Run A: original code
# ─────────────────────────────────────────────────────────────────────────────

_RUNA_SYSTEM = """
Task: Extract OBJECT INSTANTIATION CHILDREN according to these strict rules. Return STRICT JSON only.

What to return as children (variables only, never class names):
1) Direct constructor assignment (declaration or reassignment):
   - Example: "Foo f = new Foo(...);" → child 'f' with code_snippet as the entire line; comment "instantiated variable".
2) Variable created FROM ANOTHER VARIABLE'S method call on the same line:
   - Example: "Gome g = v.create();" → children 'g' (comment "instantiated variable") AND 'v'
     (comment "source variable for 'g' on same line"); both use the entire line as code_snippet.
   - Chained call variant: "Gome g = v.create().tune();" → children 'g' and 'v' (no 'tune').
3) Field assignment on focused object only:
   - Example: "x.helper = new Helper();" with focus 'x' → child 'helper' only (comment "field instantiated on focus").
4) Multi-statements on the SAME line:
   - Example: "Foo a = new Foo(); Bar b = a.make();" → children: 'a' (instantiated variable), 'b' (instantiated variable),
     and 'a' again (comment "source variable for 'b' on same line").
   - For SAME-NAME children on the SAME line, assign variant indices 0,1,... in left-to-right order of their occurrences on that line.
5) EXCLUDE:
   - Instantiations used ONLY as arguments (e.g., "call(new Foo())") → return nothing here.
   - Unbound 'new' used only in a chain (e.g., "new Foo().init()") → return nothing here.
   - Anything inside lambda/anonymous-class bodies.
   - Denylisted trivial/logging utilities.
6) Focus-aware:
   - Focus=METHOD: include locals declared/reassigned in that method; do NOT include fields unless receiver is 'this' AND you can infer 'this' is the focus (otherwise exclude).
   - Focus=OBJECT VAR X: include 'X.field = new T()' as 'field'; include 'Type y = X.make()' as 'y' AND 'X' on that line.
   - Focus=CALL_RESULT: include 'Var r = <focusCall>().next()' as 'r'.

For each child produce an EC:
{"name":"<var or field>", "code_snippet":"<ENTIRE line>", "code_block":"<smallest original block>",
 "further_expand": false, "confidence": 0.0-1.0, "conditioned": false, "guards": [],
 "comment":"<reason as per the cases above>", "variant": <0-based index for same-name in same line>}
""".strip()

_RUNA_FEWSHOTS = """
Examples (diverse):

1) Method focus
void m(){ Foo f = new Foo(); f.run(); }
→ children: {"name":"f","comment":"instantiated variable","variant":0}

2) From another variable's method
void m(){ Gome g = v.create(); }
→ children: {"name":"g","comment":"instantiated variable","variant":0},
            {"name":"v","comment":"source variable for 'g' on same line","variant":0}

3) Multi-statements same line
void m(){ Foo a = new Foo(); Bar b = a.make(); }
→ children: a(instantiated, variant 0), b(instantiated, variant 0), a(source for 'b', variant 1)

4) Field on focus object
void m(X x){ x.helper = new Helper(); }
Focus: x → child: {"name":"helper","comment":"field instantiated on focus","variant":0}

5) Exclusions
void m(){ call(new Foo()); new Bar().init(); items.forEach(t -> { Temp u = new Temp(); }); }
→ no children for these lines
""".strip()

def _build_run_a_user(
    *,
    code: str,
    focus_name: str,
    anchor_line: int,
    anchor_content: str,
    chain: str,
    denylist: List[str],
) -> str:
    return (
        f"FOCUS_NAME: {focus_name}\n"
        f"ANCHOR_LINE (1-based): {anchor_line}\n"
        f"ANCHOR_LINE_CONTENT: {anchor_content}\n"
        f"ANALYTICAL_CHAIN (≤2): {chain}\n"
        f"DENYLIST: {denylist}\n\n"
        "CODE:\n"
        f"{code}\n\n"
        f"{_RUNA_FEWSHOTS}\n"
        'Output JSON: {"children":[EC,...]} ONLY.'
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTOR — Run B: NL per line → re-extract
# ─────────────────────────────────────────────────────────────────────────────

_EXPLAIN_LINES_SYSTEM = """
Convert the Java code to concise, factual natural language, ONE sentence per original line (1-based).
Preserve variable declarations/assignments, 'new' constructs, field assignments, and simple chains.
Indicate same-line multiple statements in the same original line index.
Return STRICT JSON: {"lines":[{"line":int,"text":str}, ...]}
""".strip()

_RUNB_SYSTEM = """
Using the per-line explanations, repeat the SAME extraction:
• Return only variable/field children (no class names).
• If created from another variable’s method call, include both the new variable and the source variable (with comments).
• For SAME-NAME children on the SAME line, assign variant indices 0,1,... in left-to-right order for that line.
• Exclude instantiation-only arguments, unbound 'new' chains, lambda/anon internals, denylisted utilities.

Return STRICT JSON: {"children":[EC,...]}. Prefer empty over guesses.
""".strip()

_RUNB_FEWSHOTS = """
Hints in NL:
- "line 7: declare Foo a = new Foo(); then Bar b = a.make()" → a(instantiated, variant 0), b(instantiated, v0), a(source for 'b', v1)
- "line 12: x.helper assigned new Helper()" with focus x → child 'helper'
- "line 14: call(new Foo())" → no child
""".strip()

def _build_run_b_user(
    *,
    explained_json: str,
    focus_name: str,
    anchor_line: int,
    anchor_content: str,
    chain: str,
    denylist: List[str],
) -> str:
    return (
        f"FOCUS_NAME: {focus_name}\n"
        f"ANCHOR_LINE (1-based): {anchor_line}\n"
        f"ANCHOR_LINE_CONTENT: {anchor_content}\n"
        f"ANALYTICAL_CHAIN (≤2): {chain}\n"
        f"DENYLIST: {denylist}\n\n"
        "LINES_NL (JSON array of {line:int,text:str}):\n"
        f"{explained_json}\n\n"
        f"{_RUNB_FEWSHOTS}\n"
        "Return ONLY the JSON object with schema: {'children':[EC,...]}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API — extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_object_instantiations(
    llm: AzureChatOpenAI,
    *,
    request: InstantiationInput,
    denylist: Optional[List[str]] = None,
) -> List[EC]:
    """
    Extract variable/field children created from instantiation under the specified rules.
    Two runs (original + NL per-line) → merge by (name, code_snippet, comment) to preserve same-name duplicates.
    """
    focus = request["object_name"]
    code = request["java_code"]
    anchor = int(request["java_code_line"])
    anchor_content = request.get("java_code_line_content", "")
    chain = request.get("analytical_chain", "")
    deny = denylist or DEFAULT_DENYLIST

    # Run A — original code
    user_a = _build_run_a_user(
        code=code,
        focus_name=focus,
        anchor_line=anchor,
        anchor_content=anchor_content,
        chain=chain,
        denylist=deny,
    )
    out_a = _invoke_json(llm, system=_RUNA_SYSTEM, user=user_a)
    a_children = _norm_ec_list(out_a.get("children", []))

    # Run B — explain → extract
    explained = _invoke_json(llm, system=_EXPLAIN_LINES_SYSTEM, user="CODE:\n" + code)
    explained_json = json.dumps(explained.get("lines", []), ensure_ascii=False)

    user_b = _build_run_b_user(
        explained_json=explained_json,
        focus_name=focus,
        anchor_line=anchor,
        anchor_content=anchor_content,
        chain=chain,
        denylist=deny,
    )
    out_b = _invoke_json(llm, system=_RUNB_SYSTEM, user=user_b)
    b_children = _norm_ec_list(out_b.get("children", []))

    # Merge and return
    return _merge_ecs(a_children, b_children)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR — Object Instantiation
# ─────────────────────────────────────────────────────────────────────────────

_VALIDATOR_SYSTEM = """
You validate Object Instantiation children relative to the FOCUS.
Return STRICT JSON only.

Validity rules:
• Return children ONLY for variables/fields (never class names).
• Direct constructor assignment (declaration or reassignment): child is the LHS variable ("instantiated variable").
• Created from another variable's method call: return TWO children:
   - new variable (comment "instantiated variable")
   - source variable (comment "source variable for '<newVar>' on same line")
• Field on focused object: 'x.field = new T()' with focus 'x' → child 'field' ONLY (comment "field instantiated on focus").
• Multi-statements on the SAME line: allow duplicate names; SAME-LINE duplicates must have 'variant' indices 0,1,... in left-to-right order.
• Exclusions: instantiation-only arguments (call(new T())), unbound 'new' chains (new T().init()), lambda/anon internals, denylisted utilities.

For each candidate EC, output:
{"name":"...","valid":true|false,"confidence":0..1,"reason":"..."}
""".strip()

_VALIDATOR_FEWSHOTS = """
Examples:

1) "Foo f = new Foo();" → 'f' valid ("instantiated variable")
2) "Gome g = v.create();" → 'g' valid; 'v' valid ("source variable for 'g' on same line")
3) "Foo a = new Foo(); Bar b = a.make();" → 'a'(instantiated, variant 0), 'b'(instantiated, v0), 'a'(source, variant 1) all valid
4) "x.helper = new Helper();" with focus 'x' → 'helper' valid; 'Helper' invalid
5) "call(new Foo()); new Bar().init();" → none valid
""".strip()

def validate_object_instantiations(
    llm: AzureChatOpenAI,
    *,
    request: InstantiationInput,
    candidates: List[EC],
    denylist: Optional[List[str]] = None,
) -> List[VerdictTD]:
    """
    Validate Object Instantiation candidates. Returns verdicts with name/valid/confidence/reason.
    """
    focus = request["object_name"]
    code = request["java_code"]
    anchor = int(request["java_code_line"])
    anchor_content = request.get("java_code_line_content", "")
    chain = request.get("analytical_chain", "")
    deny = denylist or DEFAULT_DENYLIST

    user = (
        f"FOCUS_NAME: {focus}\n"
        f"ANCHOR_LINE (1-based): {anchor}\n"
        f"ANCHOR_LINE_CONTENT: {anchor_content}\n"
        f"ANALYTICAL_CHAIN (≤2): {chain}\n"
        f"DENYLIST: {deny}\n\n"
        "CANDIDATES (JSON array of EC objects):\n"
        f"{candidates}\n\n"
        "CODE:\n"
        f"{code}\n\n"
        f"{_VALIDATOR_FEWSHOTS}\n"
        'Output schema: {"verdicts":[{"name":"...","valid":true|false,"confidence":0.0,"reason":"..."}]}\n'
        "Return ONLY the JSON object."
    )

    out = _invoke_json(llm, system=_VALIDATOR_SYSTEM, user=user)
    verd = out.get("verdicts", [])
    cleaned: List[VerdictTD] = []
    for v in verd:
        nm = str(v.get("name", "")).strip()
        if not nm:
            continue
        try:
            conf = float(v.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        if conf < 0.0: conf = 0.0
        if conf > 1.0: conf = 1.0
        cleaned.append({
            "name": nm,
            "valid": bool(v.get("valid", False)),
            "confidence": conf,
            "reason": str(v.get("reason", "")).strip(),
        })
    return cleaned

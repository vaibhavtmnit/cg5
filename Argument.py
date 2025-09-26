# pass_as_arg_extractor_and_validator.py
# PassAsArg (Argument used in method calls) — extractor + validator
# UPDATED: emits ALL atomic components inside argument expressions (a.b, a.b(), a::b, (Foo)obj, Foo.bar, etc.)
# - TypedDict only (no Pydantic)
# - Two independent runs (original code; NL per-line) then merge by `name`
# - Excludes lambda/anonymous-class bodies
# - Marks `conditioned` and simple `guards` when obvious (e.g., ternary)
#
# Requires:
#   pip install langchain langchain-openai

from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
import json

from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


# ─────────────────────────────────────────────────────────────────────────────
# TypedDicts — per your schema
# ─────────────────────────────────────────────────────────────────────────────

class EC(TypedDict):
    name: str               # atomic component inside the argument expression (identifier, method, class, or 'new')
    code_snippet: str       # minimal fragment showing the argument expression at the call site
    code_block: str         # smallest original block showing the call and the argument expression
    further_expand: bool    # leave False by default
    confidence: float
    conditioned: bool
    guards: List[str]       # optional guard strings ([] if none)

class PassAsArgInput(TypedDict):
    object_name: str               # focus variable/call-result name (the thing we're tracking)
    java_code: str                 # full source text
    java_code_line: int            # 1-based anchor line of the chosen occurrence
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
        }
        if not ec["name"]:
            continue
        # clamp confidence
        if ec["confidence"] < 0.0: ec["confidence"] = 0.0
        if ec["confidence"] > 1.0: ec["confidence"] = 1.0
        out.append(ec)
    return out

def _merge_by_name(a: List[EC], b: List[EC]) -> List[EC]:
    """
    Merge EC lists by 'name'. Keep shortest blocks/snippets, max confidence.
    Union guards; OR conditioned; OR further_expand.
    """
    by: Dict[str, EC] = {}
    def push(lst: List[EC]):
        for it in lst:
            nm = it["name"]
            if nm not in by:
                by[nm] = it
            else:
                cur = by[nm]
                if it["code_block"] and (not cur["code_block"] or len(it["code_block"]) < len(cur["code_block"])):
                    cur["code_block"] = it["code_block"]
                if it["code_snippet"] and (not cur["code_snippet"] or len(it["code_snippet"]) < len(cur["code_snippet"])):
                    cur["code_snippet"] = it["code_snippet"]
                if it["confidence"] > cur["confidence"]:
                    cur["confidence"] = it["confidence"]
                cur["conditioned"] = cur["conditioned"] or it["conditioned"]
                cur["guards"] = list(dict.fromkeys(cur.get("guards", []) + it.get("guards", [])))
                cur["further_expand"] = cur["further_expand"] or it["further_expand"]
    push(a); push(b)
    return [by[k] for k in sorted(by.keys())]


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTOR — Run A: original code → split ALL argument components
# ─────────────────────────────────────────────────────────────────────────────

_RUNA_SYSTEM = """
Task: Extract ALL ATOMIC COMPONENTS mentioned inside ARGUMENT EXPRESSIONS at call sites
that consume the FOCUS OBJECT anywhere within that argument expression.
Return STRICT JSON. Prefer empty results over guesses.

What counts as an atomic component:
• Identifiers (variables or class/simple names), e.g., a, Foo
• Member names in dotted or method-ref forms, e.g., in a.b, a.b(), a::b  → emit both 'a' and 'b'
• Method names as part of argument expressions, e.g., sink.accept(a.b()) → also emit 'b'
• Constructor references → emit both 'Foo' and 'new' for Foo::new
• Casts → emit both the type and the variable, e.g., (Foo)obj → 'Foo' and 'obj'

In-scope occurrences:
• A METHOD CALL whose argument list contains an expression where the FOCUS OBJECT appears (possibly nested).
• The focus object may be the whole argument (obj), or part of a larger expression (a.b, a.b(), a::b, (Foo)obj, Foo.bar(obj)).
• Prefer the occurrence nearest to the ANCHOR_LINE within the same enclosing method/initializer.

Mandatory exclusions:
• Do NOT extract items that occur only inside lambda/anonymous-class bodies.
• Do NOT extract from denylisted trivial/logging utilities (denylist provided).
• Do NOT treat receiver calls (obj.m()) as arguments (that's a different extractor).

For each kept argument site, SPLIT the expression and emit one EC per atomic component:
  {"name":"<component>", "code_snippet":"<the argument/call fragment>",
   "code_block":"<smallest block showing the call and the argument>",
   "further_expand": false, "confidence": 0.0-1.0, "conditioned": false, "guards":[]}

Return STRICT JSON: {"children":[EC,...]}
""".strip()

_RUNA_FEWSHOTS = """
Few-shot examples (synthetic, diverse):

1) Dotted member as arg
void m(){ sink.accept(a.b); }
Focus: a → children: ['a','b'] with snippet "sink.accept(a.b)"

2) Method call as arg
void m(){ sink.accept(a.b()); }
Focus: a → children: ['a','b'] with snippet "sink.accept(a.b())"

3) Method reference
void m(){ list.forEach(a::c); }
Focus: a → children: ['a','c']  (exclude anything inside lambda bodies)

4) Constructor reference
void m(){ Supplier<Foo> s = Foo::new; sink.accept(Foo::new); }
Focus: Foo → children: ['Foo','new']

5) Cast
void m(){ use((Foo)obj); }
Focus: obj → children: ['Foo','obj']

6) Static method as arg
void m(){ call(Foo.bar(obj)); }
Focus: obj → children: ['Foo','bar','obj']

7) Ternary: conditioned argument
void m(){ process(cond ? a.b : other); }
Focus: a → children: ['a','b'] and mark conditioned=true, guards=['cond']
""".strip()

def _build_run_a_user(
    *,
    code: str,
    focus_object: str,
    anchor_line: int,
    anchor_content: str,
    chain: str,
    denylist: List[str],
) -> str:
    return (
        f"FOCUS_OBJECT: {focus_object}\n"
        f"ANCHOR_LINE (1-based): {anchor_line}\n"
        f"ANCHOR_LINE_CONTENT: {anchor_content}\n"
        f"ANALYTICAL_CHAIN (≤2): {chain}\n"
        f"DENYLIST: {denylist}\n\n"
        "CODE:\n"
        f"{code}\n\n"
        f"{_RUNA_FEWSHOTS}\n"
        'Output JSON schema: {"children":[EC,...]}\n'
        "Return ONLY the JSON object."
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTOR — Run B: per-line NL → re-extract & split
# ─────────────────────────────────────────────────────────────────────────────

_EXPLAIN_LINES_SYSTEM = """
Convert the Java code to concise, factual natural language, ONE sentence per original line (1-based).
Preserve identifiers, receivers, and argument lists. No speculation. No added/removed lines.
Return STRICT JSON: {"lines":[{"line":int,"text":str}, ...]}
""".strip()

_RUNB_SYSTEM = """
Using the per-line explanations, extract ALL ATOMIC COMPONENTS inside argument expressions that consume the FOCUS OBJECT.
Apply the SAME rules as the original-code run; split dotted expressions, method calls, method references, casts, and static calls.
Anchor at the given line; prefer nearest occurrence in the same method/initializer.
Return STRICT JSON. Prefer empty results over guesses.
""".strip()

_RUNB_FEWSHOTS = """
Few-shot hints in NL form:

- "line 8: call sink.accept(a.b)" → components ['a','b']
- "line 12: call process with (cond ? a.b : other)" → components ['a','b'], conditioned=true, guards=['cond']
- "line 20: pass Foo::new" → components ['Foo','new']
""".strip()

def _build_run_b_user(
    *,
    explained_json: str,
    focus_object: str,
    anchor_line: int,
    anchor_content: str,
    chain: str,
    denylist: List[str],
) -> str:
    return (
        f"FOCUS_OBJECT: {focus_object}\n"
        f"ANCHOR_LINE (1-based): {anchor_line}\n"
        f"ANCHOR_LINE_CONTENT: {anchor_content}\n"
        f"ANALYTICAL_CHAIN (≤2): {chain}\n"
        f"DENYLIST: {denylist}\n\n"
        "LINES_NL (JSON array of {line:int,text:str}):\n"
        f"{explained_json}\n\n"
        f"{_RUNB_FEWSHOTS}\n"
        'Output JSON schema: {"children":[EC,...]}\n'
        "Return ONLY the JSON object."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API — extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_pass_as_arg(
    llm: AzureChatOpenAI,
    *,
    request: PassAsArgInput,
    denylist: Optional[List[str]] = None,
) -> List[EC]:
    """
    Extract PassAsArg components for the focus object:
    - find calls whose argument expressions include the focus (possibly nested)
    - split each such argument into ALL atomic components (identifiers, member names, method names, class names, 'new')
    - two runs (original + NL) → merge by 'name'
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
        focus_object=focus,
        anchor_line=anchor,
        anchor_content=anchor_content,
        chain=chain,
        denylist=deny,
    )
    out_a = _invoke_json(llm, system=_RUNA_SYSTEM, user=user_a)
    a_children = _norm_ec_list(out_a.get("children", []))

    # Run B — explain → extract
    explain_user = "CODE:\n" + code
    explained = _invoke_json(llm, system=_EXPLAIN_LINES_SYSTEM, user=explain_user)
    explained_json = json.dumps(explained.get("lines", []), ensure_ascii=False)

    user_b = _build_run_b_user(
        explained_json=explained_json,
        focus_object=focus,
        anchor_line=anchor,
        anchor_content=anchor_content,
        chain=chain,
        denylist=deny,
    )
    out_b = _invoke_json(llm, system=_RUNB_SYSTEM, user=user_b)
    b_children = _norm_ec_list(out_b.get("children", []))

    # Merge and return
    return _merge_by_name(a_children, b_children)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR — PassAsArg components
# ─────────────────────────────────────────────────────────────────────────────

_VALIDATOR_SYSTEM = """
You validate PassAsArg COMPONENT candidates:
Each candidate 'name' must be one of the ATOMIC COMPONENTS present inside an argument expression
at a call site where the FOCUS OBJECT appears somewhere within that argument expression.
Return STRICT JSON.

Valid components include:
• Identifiers (variables or class/simple names)
• Member names in dotted/method-call forms (a.b, a.b())
• Method references (a::b, Foo::new) → both sides are valid components
• Casts ((Foo)obj) → both 'Foo' and 'obj'
• Static method calls used as arguments (Foo.bar(obj)) → 'Foo', 'bar', and 'obj' are components

Invalid if:
• The focus object does NOT actually occur in that argument expression at the site.
• The component appears only inside a lambda/anonymous-class body.
• The only use is as a receiver (obj.m()) rather than an argument.
• The call is denylisted trivial/logging utility.

For each candidate EC, output: {"name","valid","confidence","reason"}.
Prefer empty over guesses.
""".strip()

_VALIDATOR_FEWSHOTS = """
Few-shot examples (generic):

1) sink.accept(a.b)  with focus=a  → valid: 'a', 'b'
2) sink.accept(a.b()) with focus=a → valid: 'a', 'b'
3) list.forEach(a::c) with focus=a → valid: 'a','c' (method reference)
4) use((Foo)obj) with focus=obj   → valid: 'Foo','obj'
5) call(Foo.bar(obj)) with focus=obj → valid: 'Foo','bar','obj'
6) obj.doIt()           → invalid (receiver call, not an argument)
7) inside lambda only   → invalid
""".strip()

def validate_pass_as_arg(
    llm: AzureChatOpenAI,
    *,
    request: PassAsArgInput,
    candidates: List[EC],
    denylist: Optional[List[str]] = None,
) -> List[VerdictTD]:
    """
    Validate PassAsArg component candidates. Returns verdicts with name/valid/confidence/reason.
    """
    focus = request["object_name"]
    code = request["java_code"]
    anchor = int(request["java_code_line"])
    anchor_content = request.get("java_code_line_content", "")
    chain = request.get("analytical_chain", "")
    deny = denylist or DEFAULT_DENYLIST

    user = (
        f"FOCUS_OBJECT: {focus}\n"
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

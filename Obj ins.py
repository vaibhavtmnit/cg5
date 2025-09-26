# object_instantiation_extractor_and_validator.py
# Object Instantiation — extractor + validator using TypedDict only.
# Nuances implemented:
#  • Children are all instantiations directly related to the focus.
#  • If a variable is bound to the new instance AND that variable is later used in the same scope,
#    add BOTH children: the instantiated TYPE and the VARIABLE (two ECs), each with the ENTIRE line as code_snippet.
#  • If an instantiation is used ONLY as an argument (e.g., call(new Foo())), return ONLY the TYPE child for that line.
#  • Exclude lambda/anonymous-class internals; exclude trivial/logging utilities via denylist.
#  • Two independent runs (original and NL per-line) then merge by 'name' with shortest snippet/block and max confidence.
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
    name: str               # for instantiation: class simple name (e.g., "Foo"); for variable-child: variable name (e.g., "f")
    code_snippet: str       # ENTIRE original code line containing the instantiation (or variable usage line, see rules)
    code_block: str         # smallest original block showing parent + child + relation (can be same as snippet)
    further_expand: bool    # leave False by default; orchestrator decides expansion later
    confidence: float
    conditioned: bool       # True if under explicit condition/ternary/loop/try
    guards: List[str]       # optional guard strings ([] if none)

class InstantiationInput(TypedDict):
    object_name: str               # focus (method | object variable | call_result); LLM infers relevance by rules below
    java_code: str                 # full source text
    java_code_line: int            # 1-based anchor line (occurrence to analyze)
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
# EXTRACTOR — Run A: original code
# ─────────────────────────────────────────────────────────────────────────────

_RUNA_SYSTEM = """
Task: Extract OBJECT INSTANTIATIONS directly related to the FOCUS, using ONLY the original Java code.
Return STRICT JSON. Prefer empty results over guesses.

Interpret the focus and apply relevance:
• FOCUS = METHOD → include instantiations inside that method body:
  - 'Type v = new Type(...);'  → child TYPE 'Type'
  - If 'v' is later used as an object (receiver or argument) in the same method, ALSO add child VARIABLE 'v'.
  - 'call(new Type(...))' or 'sink.accept(new Type(...))' → add ONLY child TYPE 'Type' for that line.
  - 'new Type(...).init()' (no variable bound) → add ONLY child TYPE 'Type'.
• FOCUS = OBJECT VARIABLE X → include instantiations that create or directly affect X:
  - 'XType X = new XType(...);' → child TYPE 'XType'  (do NOT add a child for 'X' here; X is the focus itself)
  - 'X.field = new Type(...);' → child TYPE 'Type'  (instantiation stored into a field of the focus)
  - 'call(new Type(X))' (focus passed into constructor) → add ONLY child TYPE 'Type'.
• FOCUS = CALL_RESULT → usually no direct instantiation; include only if the same statement builds the call result via 'new'.

Mandatory exclusions:
• Do NOT include instantiations that occur only inside lambda/anonymous-class bodies.
• Do NOT include denylisted trivial/logging utilities.
• Prefer the occurrence nearest to ANCHOR_LINE within the same enclosing method/initializer.

For each kept item, produce an EC:
  {"name":"<Type or Variable>", "code_snippet":"<ENTIRE line>", "code_block":"<smallest original block>",
   "further_expand": false, "confidence": 0.0-1.0, "conditioned": false, "guards":[]}
""".strip()

_RUNA_FEWSHOTS = """
Few-shot examples (synthetic):

1) Method focus, with later usage
void m(){
  Foo f = new Foo(a);
  f.run();
  log.debug("x");
}
Focus: m → children: ['Foo', 'f']  // 'Foo' from instantiation, 'f' because later used as object
Both EC.code_snippet should be the ENTIRE line "Foo f = new Foo(a);"

2) Method focus, instantiation as arg (no variable)
void m(){ call(new Bar()); }
Focus: m → children: ['Bar'] ONLY (from 'new Bar()'). No extra child.

3) Method focus, chained right after new
void m(){ new Baz().init(); }
Focus: m → children: ['Baz'] ONLY.

4) Object focus X, constructor of X
void m(){ X x = new X(); x.work(); }
Focus: x → children: ['X'] ONLY  // do not add a child for 'x' itself

5) Object focus X, field assigned a new
void m(){ x.helper = new Helper(); }
Focus: x → children: ['Helper']

6) Ignore lambda internals
void m(){ items.forEach(it -> { Helper h = new Helper(); }); }
Focus: m → [] (instantiation inside lambda only)
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
        'Output JSON schema: {"children":[EC,...]}\n'
        "Return ONLY the JSON object."
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTOR — Run B: per-line NL → re-extract
# ─────────────────────────────────────────────────────────────────────────────

_EXPLAIN_LINES_SYSTEM = """
Convert the Java code to concise, factual natural language, ONE sentence per original line (1-based).
Preserve identifiers and 'new Type(...)' constructs; indicate variable names and whether the line is a call, assignment, or declaration.
No speculation. No added/removed lines.
Return STRICT JSON: {"lines":[{"line":int,"text":str}, ...]}
""".strip()

_RUNB_SYSTEM = """
Using the per-line explanations, extract OBJECT INSTANTIATIONS directly related to the focus, with the same rules as in the original-code run.
Apply the special rule:
• If a variable is bound to 'new' AND the variable is later used as an object in the same method, emit TWO ECs: the TYPE and the VARIABLE (both with ENTIRE instantiation line as code_snippet).
• If the instantiation is used only as an argument or in a chained call with no variable bound, emit ONLY the TYPE EC.

Return STRICT JSON: {"children":[EC,...]}.
Prefer empty results over guesses.
""".strip()

_RUNB_FEWSHOTS = """
Few-shot hints in NL:

- "line 10: declare Foo f = new Foo(a)" and later "line 12: f.run()" → emit 'Foo' and 'f' (same instantiation line as snippet)
- "line 7: call(new Bar())" → emit 'Bar' only
- "line 5: new Baz().init()" → emit 'Baz' only
- "line 18: x.helper = new Helper()" with focus x → emit 'Helper'
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
        'Output JSON schema: {"children":[EC,...]}\n'
        "Return ONLY the JSON object."
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
    Extract Object Instantiation children according to the rules:
      • Method focus: instantiations in body; add TYPE; add VARIABLE too if later used as object (both with full line as snippet).
      • Object focus (variable): TYPE that constructs or is assigned into its fields; do NOT add the focus variable itself.
      • Call-result focus: typically none, unless the result is directly created via 'new' in the same statement.

    Two runs (original + NL) → merge by 'name'.
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
    explain_user = "CODE:\n" + code
    explained = _invoke_json(llm, system=_EXPLAIN_LINES_SYSTEM, user=explain_user)
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
    return _merge_by_name(a_children, b_children)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR — Object Instantiation
# ─────────────────────────────────────────────────────────────────────────────

_VALIDATOR_SYSTEM = """
You validate Object Instantiation candidates relative to the FOCUS.
Return STRICT JSON.

Validity rules:
• METHOD focus:
  - Valid TYPE if the method body contains a 'new Type(...)' instantiation (outside lambda/anonymous classes).
  - Valid VARIABLE if it is declared on the same line as 'new Type(...)' AND that variable is later used as an object
    (receiver or argument) within the same method. Use the ENTIRE instantiation line as the code_snippet for both ECs.
  - If an instantiation is used only as an argument or chained immediately (no variable bound), ONLY the TYPE is valid.
• OBJECT VARIABLE focus:
  - Valid TYPE if it constructs the focus (e.g., 'X x = new X(...)') or assigns a new into a field of the focus (e.g., 'x.f = new Y(...)').
  - Do NOT validate a separate VARIABLE child for the focus variable itself.
• CALL_RESULT focus:
  - Valid TYPE only if the call result is directly created via 'new' in the same statement; otherwise invalid.

Exclusions:
• Ignore instantiations inside lambda/anonymous-class bodies.
• Ignore denylisted trivial/logging utilities.

For each candidate EC, output: {"name","valid":bool,"confidence":0.0-1.0,"reason":str}.
Prefer empty over guesses.
""".strip()

_VALIDATOR_FEWSHOTS = """
Few-shot examples:

1) Method focus; var later used
void m(){ Foo f = new Foo(); f.run(); }
Candidates: ['Foo','f'] → both valid (same instantiation line as snippet)

2) Method focus; arg-only
void m(){ call(new Bar()); }
Candidates: ['Bar','temp'] → 'Bar' valid; 'temp' invalid (no variable bound)

3) Method focus; chained after new
void m(){ new Baz().init(); }
Candidates: ['Baz','x'] → 'Baz' valid; 'x' invalid

4) Object focus x
void m(){ X x = new X(); x.helper = new H(); }
Focus: x → valid: 'X','H' ; invalid: 'x' (focus itself not added)

5) Lambda-internal new ignored
void m(){ items.forEach(it -> { Q q = new Q(); }); }
Focus: m → 'Q' invalid (inside lambda only)
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

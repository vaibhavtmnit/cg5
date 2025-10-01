# lambda_children_extractor_and_validator.py
# Extract and validate "objects associated with lambdas directly related to a focus object".
#
# What we return as children (EC items):
#   • Identifiers BEFORE '->' : lambda parameters (e.g., x, (k,v)).
#   • Identifiers AFTER '->'  : objects used in the body as receivers or arguments
#                               (e.g., x.do(), call(x), a.b() → include 'x' and 'a'; for a.b() we include 'a' as the object).
#   • Method references:       obj::meth  → include 'obj' (if 'Class::meth' without object, ignore unless 'this::meth').
#   • Captured outer vars:     any identifier used inside lambda body (service, this, user) → include those identifiers.
#
# What we DO NOT return:
#   • Class names / types (Foo, Bar) unless used as an identifier variable (e.g., 'foo' vs 'Foo').
#   • Pure 'new' temporaries without identifiers (e.g., call(new Foo())).
#   • Primitives, string literals.
#
# Focus relation (which lambdas we consider “directly related”):
#   • Focus = METHOD: lambdas passed/captured within that method body.
#   • Focus = OBJECT VAR X: lambdas that are:
#         - passed to methods where receiver == X or directly chained from X (X.stream().map(...)),
#         - or assigned into fields/locals immediately derived from X’s chain on the same statement.
#   • Focus = CALL-RESULT: lambdas applied to that immediate call result in the same statement.
#
# Output EC schema (TypedDict):
#   name: str
#   code_snippet: str         # entire line containing the lambda/method-ref
#   code_block: str           # smallest block that shows the focus call + lambda context
#   further_expand: bool
#   confidence: float
#   conditioned: bool
#   guards: List[str]
#
# Requires:
#   pip install langchain langchain-openai

from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
import json
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


# ---------------------- Types ----------------------

class EC(TypedDict):
    name: str
    code_snippet: str
    code_block: str
    further_expand: bool
    confidence: float
    conditioned: bool
    guards: List[str]

class LInput(TypedDict):
    object_name: str
    java_code: str
    java_code_line: int
    java_code_line_content: str
    analytical_chain: str

class VerdictTD(TypedDict):
    name: str
    valid: bool
    confidence: float
    reason: str


# ---------------------- Helpers ----------------------

DEFAULT_DENYLIST = [
    "System.out.println",
    "logger.info",
    "logger.debug",
    "logger.trace",
    "Objects.requireNonNull",
    "Collections.emptyList",
]

def _invoke_json(llm: AzureChatOpenAI, *, system: str, user: str, retry: bool=True) -> Any:
    msgs = [SystemMessage(content=system), HumanMessage(content=user)]
    try:
        return json.loads(llm.invoke(msgs).content)
    except Exception:
        if not retry:
            raise
        user2 = user + "\n\nREMINDER: Return ONLY a valid JSON object. If nothing, return {\"children\": []}."
        return json.loads(llm.invoke([SystemMessage(content=system), HumanMessage(content=user2)]).content)

def _norm_ec_list(items: List[Dict[str,Any]]) -> List[EC]:
    out: List[EC] = []
    for it in items or []:
        ec: EC = {
            "name": str(it.get("name","")).strip(),
            "code_snippet": str(it.get("code_snippet","")).strip(),
            "code_block": str(it.get("code_block","")).strip(),
            "further_expand": bool(it.get("further_expand", False)),
            "confidence": float(it.get("confidence", 0.0)),
            "conditioned": bool(it.get("conditioned", False)),
            "guards": list(it.get("guards", []) or []),
        }
        if ec["name"]:
            ec["confidence"] = max(0.0, min(1.0, ec["confidence"]))
            out.append(ec)
    return out

def _merge_by_name_keep_best(a: List[EC], b: List[EC]) -> List[EC]:
    """
    Merge by 'name'. Keep the shortest snippet/block and highest confidence.
    OR-condition flags; union guards. Deterministic sort by name.
    """
    by: Dict[str, EC] = {}
    def push(lst: List[EC]):
        for it in lst:
            nm = it["name"]
            if nm not in by:
                by[nm] = it
            else:
                cur = by[nm]
                # prefer shorter code_block/snippet
                if it["code_block"] and (not cur["code_block"] or len(it["code_block"]) < len(cur["code_block"])):
                    cur["code_block"] = it["code_block"]
                if it["code_snippet"] and (not cur["code_snippet"] or len(it["code_snippet"]) < len(cur["code_snippet"])):
                    cur["code_snippet"] = it["code_snippet"]
                # higher confidence
                if it["confidence"] > cur["confidence"]:
                    cur["confidence"] = it["confidence"]
                # combine flags/guards
                cur["conditioned"] = cur["conditioned"] or it["conditioned"]
                cur["guards"] = list(dict.fromkeys(cur.get("guards", []) + it.get("guards", [])))
                cur["further_expand"] = cur["further_expand"] or it["further_expand"]
    push(a); push(b)
    return [by[k] for k in sorted(by.keys())]


# ---------------------- Extractor ----------------------

_SYSTEM = """
You extract OBJECT IDENTIFIERS associated with LAMBDAS that are DIRECTLY RELATED to the FOCUS.

DETERMINING "DIRECTLY RELATED":
• Focus=METHOD: consider lambdas passed/captured within that method body (exclude nested lambdas' inner bodies unrelated to the lambda itself).
• Focus=OBJECT VAR X: consider lambdas that are:
   - arguments to calls where receiver == X or chained from X (e.g., X.stream().map(...)),
   - or assigned into fields/locals that are fed directly by X's chain on that same statement.
• Focus=CALL-RESULT: consider lambdas applied to that call result in the same statement/expression chain.

FOR EACH SUCH LAMBDA, RETURN CHILDREN = OBJECT IDENTIFIERS MENTIONED BY THE LAMBDA:
• BEFORE '->'  (parameters): list each parameter identifier: e.g., x, (k, v).
• AFTER  '->'  (body): list identifiers used as:
   - receivers of calls:  a.b()  → include 'a'
   - arguments of calls:  call(a, x, y) → include 'a','x','y'
   - field owners:        a.f, a.f=g, g=a.f  → include 'a'
   - captured vars:       any non-parameter identifier referenced in the body (e.g., service, this, user)
• METHOD REFERENCES:
   - obj::meth → include 'obj'
   - this::meth → include 'this'
   - Class::meth / Foo::new (no object identifier) → ignore

DO NOT RETURN:
• Class/type names standing alone (Foo, Bar) unless they are variables.
• Pure 'new' temporaries without an identifier (e.g., call(new Foo())).
• String/number literals, operators, keywords.

OUTPUT EC (STRICT JSON):
{"children":[
  {"name":"<identifier>", "code_snippet":"<ENTIRE source line containing the lambda or method-ref>",
   "code_block":"<smallest block snippet that shows the focus call + lambda>",
   "further_expand": false, "confidence": 0.0-1.0, "conditioned": false, "guards":[]}
]}
""".strip()

_FEWSHOTS = """
Examples:

1) Focus = METHOD 'm'
void m(){ items.forEach(x -> x.trim()); helper.apply(u -> service.use(u)); }
→ for 'items.forEach(...)' lambda:  ['x'] (param), and body uses: 'x'
→ for 'helper.apply(...)' lambda:  ['u'] (param), plus 'service' (captured in body), 'u' (used as arg)

2) Focus = OBJECT VAR 'stream'
void m(){ stream.map(s -> s.toUpperCase()).filter(s -> pred.test(s)).forEach(t -> logger.info(t)); }
Focus: stream
→ lambdas directly tied to 'stream' chain → parameters 's','t' and body objects: 's','pred','t' (ignore 'logger' if denylisted)

3) Method references
list.forEach(this::tick);
→ child: 'this'
mapper.computeIfAbsent(k, v -> build(k, v));
→ children: 'k','v','mapper' (receiver in body? if body is 'build(k,v)' → add k,v; 'mapper' isn't inside body here, so not added for this lambda)

4) Captured vars
processor.run(x -> handler.handle(x, ctx));
→ children: 'x' (param), 'handler','x','ctx' (body)
""".strip()

def _build_user(code: str, focus: str, anchor: int, anchor_content: str, chain: str, deny: List[str]) -> str:
    return (
        f"FOCUS_NAME: {focus}\n"
        f"ANCHOR_LINE (1-based): {anchor}\n"
        f"ANCHOR_LINE_CONTENT: {anchor_content}\n"
        f"ANALYTICAL_CHAIN (≤2): {chain}\n"
        f"DENYLIST: {deny}\n\n"
        "CODE:\n" + code + "\n\n" +
        _FEWSHOTS + "\n"
        "Return ONLY the JSON object."
    )

def extract_lambda_children(
    llm: AzureChatOpenAI, *, request: LInput, denylist: Optional[List[str]]=None
) -> List[EC]:
    focus = request["object_name"]
    code = request["java_code"]
    anchor = int(request["java_code_line"])
    anchor_content = request.get("java_code_line_content","")
    chain = request.get("analytical_chain","")
    deny = denylist or DEFAULT_DENYLIST

    out = _invoke_json(
        llm,
        system=_SYSTEM,
        user=_build_user(code, focus, anchor, anchor_content, chain, deny),
    )
    children = _norm_ec_list(out.get("children", []))
    # De-dup by name while keeping best snippet/block/confidence
    return _merge_by_name_keep_best(children, [])


# ---------------------- Validator ----------------------

_VALIDATOR_SYSTEM = """
Validate that the returned identifiers are indeed OBJECTS associated with LAMBDAS directly related to the FOCUS.

Checks:
1) The lambda is directly related to the focus (method body / receiver == focus or chained from focus / applied to call-result).
2) Each EC.name is a valid identifier referenced by the lambda:
   - parameter before '->', OR
   - identifier used in body as receiver/argument/field-owner/captured var, OR
   - method-reference object (obj::meth or this::meth).
3) Exclude class/type names, pure 'new' temporaries, and literals.
4) code_snippet should be the entire original line containing that lambda or method-ref.
5) code_block should be a minimal snippet that shows the focus call + lambda context.
6) Apply denylist for trivial/logging utilities when they are the main object (e.g., 'logger' if you denylist it).

Return STRICT JSON ONLY:
{"verdicts":[{"name":"...","valid":true|false,"confidence":0.0-1.0,"reason":"..."}]}
""".strip()

def validate_lambda_children(
    llm: AzureChatOpenAI, *, request: LInput, candidates: List[EC], denylist: Optional[List[str]]=None
) -> List[VerdictTD]:
    focus = request["object_name"]
    code = request["java_code"]
    anchor = int(request["java_code_line"])
    anchor_content = request.get("java_code_line_content","")
    chain = request.get("analytical_chain","")
    deny = denylist or DEFAULT_DENYLIST

    user = (
        f"FOCUS_NAME: {focus}\n"
        f"ANCHOR_LINE (1-based): {anchor}\n"
        f"ANCHOR_LINE_CONTENT: {anchor_content}\n"
        f"ANALYTICAL_CHAIN (≤2): {chain}\n"
        f"DENYLIST: {deny}\n\n"
        "CANDIDATES (JSON EC[]):\n" + json.dumps(candidates, ensure_ascii=False) + "\n\n"
        "CODE:\n" + code + "\n"
        "Return ONLY the JSON object."
    )
    out = _invoke_json(llm, system=_VALIDATOR_SYSTEM, user=user)
    verdicts: List[VerdictTD] = []
    for v in out.get("verdicts", []):
        nm = str(v.get("name","")).strip()
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
            "reason": str(v.get("reason","")).strip(),
        })
    return verdicts

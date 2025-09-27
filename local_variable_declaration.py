# local_variable_declaration_extractor_and_validator.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
import json
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

class EC(TypedDict):
    name: str
    code_snippet: str
    code_block: str
    further_expand: bool
    confidence: float
    conditioned: bool
    guards: List[str]

class LVInput(TypedDict):
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

DEFAULT_DENYLIST = [
    "System.out.println","logger.info","logger.debug","logger.trace",
    "Objects.requireNonNull","Collections.emptyList",
]

def _invoke_json(llm: AzureChatOpenAI, *, system: str, user: str, retry: bool=True) -> Any:
    msgs=[SystemMessage(content=system),HumanMessage(content=user)]
    try:
        return json.loads(llm.invoke(msgs).content)
    except Exception:
        if not retry: raise
        user2=user+"\n\nREMINDER: Return ONLY a valid JSON object. If nothing, return {'children': []}."
        return json.loads(llm.invoke([SystemMessage(content=system),HumanMessage(content=user2)]).content)

def _norm_ec_list(items: List[Dict[str,Any]])->List[EC]:
    out=[]
    for it in items or []:
        ec:EC={
            "name":str(it.get("name","")).strip(),
            "code_snippet":str(it.get("code_snippet","")).strip(),
            "code_block":str(it.get("code_block","")).strip(),
            "further_expand":bool(it.get("further_expand",False)),
            "confidence":float(it.get("confidence",0.0)),
            "conditioned":bool(it.get("conditioned",False)),
            "guards":list(it.get("guards",[]) or []),
        }
        if ec["name"]:
            ec["confidence"]=max(0.0,min(1.0,ec["confidence"]))
            out.append(ec)
    return out

def _merge_by_name(a: List[EC], b: List[EC])->List[EC]:
    by:Dict[str,EC]={}
    def push(lst):
        for it in lst:
            nm=it["name"]
            if nm not in by:
                by[nm]=it
            else:
                cur=by[nm]
                if it["code_block"] and (not cur["code_block"] or len(it["code_block"])<len(cur["code_block"])): cur["code_block"]=it["code_block"]
                if it["code_snippet"] and (not cur["code_snippet"] or len(it["code_snippet"])<len(cur["code_snippet"])): cur["code_snippet"]=it["code_snippet"]
                cur["confidence"]=max(cur["confidence"],it["confidence"])
                cur["conditioned"]=cur["conditioned"] or it["conditioned"]
                cur["guards"]=list(dict.fromkeys(cur["guards"]+it["guards"]))
                cur["further_expand"]=cur["further_expand"] or it["further_expand"]
    push(a); push(b)
    return [by[k] for k in sorted(by.keys())]

# ---------- Extractor prompts ----------

_RUNA_SYSTEM = """
Task: Extract LOCAL VARIABLE DECLARATIONS directly related to the focus.
Return STRICT JSON. Prefer empty results over guesses.

Rules:
• Focus=METHOD: include locals declared in the method body (not fields); exclude lambda/anon-class internals.
• Focus=OBJECT VAR X: include locals that are directly assigned from X or produce aliases of X (e.g., R r = X.a(); var t = X;).
• Focus=CALL_RESULT: include locals that capture that call result in the same statement.
• Exclude fields, parameters, and denylisted utilities. Prefer nearest to ANCHOR_LINE within same method/initializer.

Emit EC per local:
{"name":"<localVar>", "code_snippet":"<ENTIRE declaration line>", "code_block":"<smallest block showing relation>",
 "further_expand":false, "confidence":0..1, "conditioned":false, "guards":[]}
""".strip()

_RUNA_FEWSHOTS = """
Examples:
1) METHOD focus
void m(){ int k=0; Foo f = make(); }
→ children: ['k','f']

2) OBJECT focus x
void m(){ R r = x.a(); var t = x; }
→ children: ['r','t']

3) CALL_RESULT focus 'a'
void m(){ R r = x.a(); }
→ children: ['r']

4) Exclude field vs local
int g=0; void m(){ int k=1; }
→ only 'k' for focus=m
""".strip()

def _build_run_a_user(code:str, focus:str, anchor:int, anchor_content:str, chain:str, deny:list)->str:
    return (
        f"FOCUS_NAME: {focus}\nANCHOR_LINE: {anchor}\nANCHOR_LINE_CONTENT: {anchor_content}\n"
        f"ANALYTICAL_CHAIN: {chain}\nDENYLIST: {deny}\n\nCODE:\n{code}\n\n{_RUNA_FEWSHOTS}\n"
        'Output JSON: {"children":[EC,...]} ONLY.'
    )

_EXPLAIN_LINES_SYSTEM = """
Convert Java to concise, factual NL, one sentence per line (1-based). Preserve identifiers and declarations.
Return STRICT JSON: {"lines":[{"line":int,"text":str},...]}.
""".strip()

_RUNB_SYSTEM = """
Using the NL lines, extract LOCAL VARIABLE DECLARATIONS per the same rules. Return STRICT JSON: {"children":[EC,...]}.
""".strip()

def _build_run_b_user(explained_json:str, focus:str, anchor:int, anchor_content:str, chain:str, deny:list)->str:
    return (
        f"FOCUS_NAME: {focus}\nANCHOR_LINE: {anchor}\nANCHOR_LINE_CONTENT: {anchor_content}\n"
        f"ANALYTICAL_CHAIN: {chain}\nDENYLIST: {deny}\n\nLINES_NL:\n{explained_json}\n"
        "Return ONLY the JSON object."
    )

# ---------- Public API ----------

def extract_local_variable_declarations(
    llm: AzureChatOpenAI, *, request: LVInput, denylist: Optional[List[str]]=None
)->List[EC]:
    focus=request["object_name"]; code=request["java_code"]; anchor=int(request["java_code_line"])
    anchor_content=request.get("java_code_line_content",""); chain=request.get("analytical_chain","")
    deny=denylist or DEFAULT_DENYLIST
    out_a=_invoke_json(llm, system=_RUNA_SYSTEM, user=_build_run_a_user(code,focus,anchor,anchor_content,chain,deny))
    a=_norm_ec_list(out_a.get("children",[]))
    explained=_invoke_json(llm, system=_EXPLAIN_LINES_SYSTEM, user="CODE:\n"+code)
    explained_json=json.dumps(explained.get("lines",[]), ensure_ascii=False)
    out_b=_invoke_json(llm, system=_RUNB_SYSTEM, user=_build_run_b_user(explained_json,focus,anchor,anchor_content,chain,deny))
    b=_norm_ec_list(out_b.get("children",[]))
    return _merge_by_name(a,b)

# ---------- Validator ----------

_VALIDATOR_SYSTEM = """
Validate LOCAL VARIABLE DECLARATION candidates relative to focus.
Rules:
• METHOD: name must be declared as a local inside that method body (not field/param).
• OBJECT VAR X: valid if local is assigned from X or aliases X (R r = X.a(); var t=X;).
• CALL_RESULT: valid if local captures that call result on the same statement.
Exclude lambda/anon-class internals. Return STRICT JSON: {"verdicts":[{"name":"...","valid":bool,"confidence":0..1,"reason":"..."}]}.
""".strip()

def validate_local_variable_declarations(
    llm: AzureChatOpenAI, *, request: LVInput, candidates: List[EC], denylist: Optional[List[str]]=None
)->List[VerdictTD]:
    focus=request["object_name"]; code=request["java_code"]; anchor=int(request["java_code_line"])
    anchor_content=request.get("java_code_line_content",""); chain=request.get("analytical_chain",""); deny=denylist or DEFAULT_DENYLIST
    user=(f"FOCUS_NAME: {focus}\nANCHOR_LINE: {anchor}\nANCHOR_LINE_CONTENT: {anchor_content}\nANALYTICAL_CHAIN: {chain}\n"
          f"DENYLIST: {deny}\n\nCANDIDATES:\n{candidates}\n\nCODE:\n{code}\nReturn ONLY the JSON object.")
    out=_invoke_json(llm, system=_VALIDATOR_SYSTEM, user=user)
    vs=[]
    for v in out.get("verdicts",[]):
        nm=str(v.get("name","")).strip()
        if nm:
            conf=float(v.get("confidence",0.0)); conf=max(0.0,min(1.0,conf))
            vs.append({"name":nm,"valid":bool(v.get("valid",False)),"confidence":conf,"reason":str(v.get("reason","")).strip()})
    return vs

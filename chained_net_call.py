# chained_next_call_extractor_and_validator.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
import json
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

class EC(TypedDict):
    name: str            # the immediate next method in the chain
    code_snippet: str
    code_block: str
    further_expand: bool
    confidence: float
    conditioned: bool
    guards: List[str]

class CNCInput(TypedDict):
    object_name: str          # this is the call-result focus: the callee name of the previous hop (e.g., "a" for x.a().b())
    java_code: str
    java_code_line: int
    java_code_line_content: str
    analytical_chain: str

class VerdictTD(TypedDict):
    name: str
    valid: bool
    confidence: float
    reason: str

def _invoke_json(llm: AzureChatOpenAI, *, system: str, user: str, retry: bool=True)->Any:
    msgs=[SystemMessage(content=system),HumanMessage(content=user)]
    try: return json.loads(llm.invoke(msgs).content)
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
            "confidence":max(0.0,min(1.0,float(it.get("confidence",0.0)))),
            "conditioned":bool(it.get("conditioned",False)),
            "guards":list(it.get("guards",[]) or []),
        }
        if ec["name"]: out.append(ec)
    return out

def _merge_by_name(a: List[EC], b: List[EC])->List[EC]:
    by:Dict[str,EC]={}
    def push(lst):
        for it in lst:
            nm=it["name"]
            if nm not in by: by[nm]=it
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

_RUNA_SYSTEM = """
Task: Extract the IMMEDIATE NEXT CHAINED CALL(S) for the call-result focus.
Return STRICT JSON.

Rules:
• Focus is the previous call name (e.g., 'a' in x.a().b().c()).
• Return only the next hop(s) directly chained from that call site: for focus='a' → 'b'; for focus='b' → 'c'.
• Exclude unqualified calls, calls on different receivers, deeper hops, lambda internals, denylisted utilities.
• Prefer the occurrence nearest to ANCHOR_LINE within the same method/initializer.

Emit EC per next-hop:
{"name":"<nextMethod>", "code_snippet":"<chain fragment>", "code_block":"<smallest block>",
 "further_expand":false, "confidence":0..1, "conditioned":false, "guards":[]}
""".strip()

_RUNA_FEWSHOTS = """
Examples:
x.a().b().c(); focus='a' → ['b']; focus='b' → ['c']
db.connect().query().close(); focus='connect' → ['query']
""".strip()

def _build_run_a_user(code:str, focus_call:str, anchor:int, anchor_content:str, chain:str)->str:
    return (f"FOCUS_CALL_RESULT: {focus_call}\nANCHOR_LINE: {anchor}\nANCHOR_LINE_CONTENT: {anchor_content}\nANALYTICAL_CHAIN: {chain}\n\n"
            f"CODE:\n{code}\n\n{_RUNA_FEWSHOTS}\nReturn ONLY {{\"children\":[EC,...]}}.")

_EXPLAIN_LINES_SYSTEM = """
Convert Java to concise NL, one sentence per line (1-based), showing chained calls step-by-step.
Return STRICT JSON: {"lines":[{"line":int,"text":str},...]}.
""".strip()

_RUNB_SYSTEM = """Using NL lines, extract immediate NEXT CHAINED CALLS per same rules. Strict JSON: {"children":[EC,...]}.""".strip()

def _build_run_b_user(explained_json:str, focus_call:str, anchor:int, anchor_content:str, chain:str)->str:
    return (f"FOCUS_CALL_RESULT: {focus_call}\nANCHOR_LINE: {anchor}\nANCHOR_LINE_CONTENT: {anchor_content}\nANALYTICAL_CHAIN: {chain}\n\n"
            f"LINES_NL:\n{explained_json}\nReturn ONLY the JSON object.")

def extract_chained_next_call(
    llm: AzureChatOpenAI, *, request: CNCInput
)->List[EC]:
    focus=request["object_name"]; code=request["java_code"]; anchor=int(request["java_code_line"])
    anchor_content=request.get("java_code_line_content",""); chain=request.get("analytical_chain","")
    out_a=_invoke_json(llm, system=_RUNA_SYSTEM, user=_build_run_a_user(code,focus,anchor,anchor_content,chain))
    a=_norm_ec_list(out_a.get("children",[]))
    explained=_invoke_json(llm, system=_EXPLAIN_LINES_SYSTEM, user="CODE:\n"+code)
    explained_json=json.dumps(explained.get("lines",[]), ensure_ascii=False)
    out_b=_invoke_json(llm, system=_RUNB_SYSTEM, user=_build_run_b_user(explained_json,focus,anchor,anchor_content,chain))
    b=_norm_ec_list(out_b.get("children",[]))
    return _merge_by_name(a,b)

# ---------- Validator ----------

_VALIDATOR_SYSTEM = """
Validate NEXT CHAINED CALL candidates. Valid iff each candidate is the immediate next hop chained off the focus call
at/near the anchor line in the same method. Exclude deeper hops, other receivers, lambda internals.
Return STRICT JSON: {"verdicts":[{"name":"...","valid":bool,"confidence":0..1,"reason":"..."}]}.
""".strip()

def validate_chained_next_call(
    llm: AzureChatOpenAI, *, request: CNCInput, candidates: List[EC]
)->List[VerdictTD]:
    focus=request["object_name"]; code=request["java_code"]; anchor=int(request["java_code_line"])
    anchor_content=request.get("java_code_line_content",""); chain=request.get("analytical_chain","")
    user=(f"FOCUS_CALL_RESULT: {focus}\nANCHOR_LINE: {anchor}\nANCHOR_LINE_CONTENT: {anchor_content}\nANALYTICAL_CHAIN: {chain}\n\n"
          f"CANDIDATES:\n{candidates}\n\nCODE:\n{code}\nReturn ONLY the JSON object.")
    out=_invoke_json(llm, system=_VALIDATOR_SYSTEM, user=user)
    vs=[]
    for v in out.get("verdicts",[]):
        nm=str(v.get("name","")).strip()
        if nm:
            conf=float(v.get("confidence",0.0)); conf=max(0.0,min(1.0,conf))
            vs.append({"name":nm,"valid":bool(v.get("valid",False)),"confidence":conf,"reason":str(v.get("reason","")).strip()})
    return vs

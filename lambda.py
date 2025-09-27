# lambda_expression_extractor_and_validator.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
import json
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

class EC(TypedDict):
    name: str            # emit the lambda "value" label, e.g., "lambda" or method-ref target name; simple, names-only
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

DEFAULT_DENYLIST=["System.out.println","logger.info","logger.debug","logger.trace","Objects.requireNonNull","Collections.emptyList"]

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

# ---------- Extractor ----------

_RUNA_SYSTEM = """
Task: Extract LAMBDA EXPRESSION values directly related to the focus (names-only representation).
Return STRICT JSON.

Rules:
• Focus=METHOD: include lambda/method-reference VALUES created/passed at top level in the method body,
  e.g., items.anyMatch(x -> p(x)), list.forEach(this::tick). Do NOT expose calls INSIDE lambda bodies.
• Focus=OBJECT VAR X: include lambda values where X is the receiver of the terminal op that accepts the lambda,
  e.g., X.forEach(y -> g(y)).
• Focus=CALL_RESULT: include lambda values applied to that immediate result in the same statement, e.g., a().map(x->x+1).

Emit EC per lambda value; set name to:
  - "lambda" for `args -> body`
  - referenced method simple name for `obj::meth` (e.g., "meth")
  - "new" for constructor refs `Foo::new` (or "new" if you prefer consistency)

{"name":"lambda|meth|new", "code_snippet":"<lambda/ref fragment>", "code_block":"<smallest block>",
 "further_expand":false, "confidence":0..1, "conditioned":false, "guards":[]}
""".strip()

_RUNA_FEWSHOTS = """
Examples:
1) METHOD focus
void m(){ items.forEach(x -> x.doIt()); }
→ ['lambda']

2) OBJECT focus stream
void m(){ stream.map(s -> s.trim()).collect(toList()); }
Focus: stream → ['lambda']  (do not surface 'trim' from inside)

3) Method ref
void m(){ list.forEach(this::tick); Supplier<Foo> s = Foo::new; }
→ names: ['tick','new']
""".strip()

def _build_run_a_user(code:str, focus:str, anchor:int, anchor_content:str, chain:str, deny:list)->str:
    return (f"FOCUS_NAME: {focus}\nANCHOR_LINE: {anchor}\nANCHOR_LINE_CONTENT: {anchor_content}\nANALYTICAL_CHAIN: {chain}\n"
            f"DENYLIST: {deny}\n\nCODE:\n{code}\n\n{_RUNA_FEWSHOTS}\nReturn ONLY {{\"children\":[EC,...]}}.")

_EXPLAIN_LINES_SYSTEM = """
Convert Java to concise NL, one sentence per line (1-based), preserving lambda and method references.
Return STRICT JSON: {"lines":[{"line":int,"text":str},...]}.
""".strip()

_RUNB_SYSTEM = """Using the NL lines, extract LAMBDA VALUES per the same rules. Strict JSON: {"children":[EC,...]}.""".strip()

def _build_run_b_user(explained_json:str, focus:str, anchor:int, anchor_content:str, chain:str, deny:list)->str:
    return (f"FOCUS_NAME: {focus}\nANCHOR_LINE: {anchor}\nANCHOR_LINE_CONTENT: {anchor_content}\nANALYTICAL_CHAIN: {chain}\n"
            f"DENYLIST: {deny}\n\nLINES_NL:\n{explained_json}\nReturn ONLY the JSON object.")

def extract_lambda_expressions(
    llm: AzureChatOpenAI, *, request: LInput, denylist: Optional[List[str]]=None
)->List[EC]:
    focus=request["object_name"]; code=request["java_code"]; anchor=int(request["java_code_line"])
    anchor_content=request.get("java_code_line_content",""); chain=request.get("analytical_chain",""); deny=denylist or DEFAULT_DENYLIST
    out_a=_invoke_json(llm, system=_RUNA_SYSTEM, user=_build_run_a_user(code,focus,anchor,anchor_content,chain,deny))
    a=_norm_ec_list(out_a.get("children",[]))
    explained=_invoke_json(llm, system=_EXPLAIN_LINES_SYSTEM, user="CODE:\n"+code)
    explained_json=json.dumps(explained.get("lines",[]), ensure_ascii=False)
    out_b=_invoke_json(llm, system=_RUNB_SYSTEM, user=_build_run_b_user(explained_json,focus,anchor,anchor_content,chain,deny))
    b=_norm_ec_list(out_b.get("children",[]))
    return _merge_by_name(a,b)

# ---------- Validator ----------

_VALIDATOR_SYSTEM = """
Validate LAMBDA VALUE candidates relative to focus. Valid if a lambda or method reference value
is created/passed at top level per the focus rules. Do NOT validate calls inside lambda bodies.
Return STRICT JSON: {"verdicts":[{"name":"...","valid":bool,"confidence":0..1,"reason":"..."}]}.
""".strip()

def validate_lambda_expressions(
    llm: AzureChatOpenAI, *, request: LInput, candidates: List[EC], denylist: Optional[List[str]]=None
)->List[VerdictTD]:
    focus=request["object_name"]; code=request["java_code"]; anchor=int(request["java_code_line"])
    anchor_content=request.get("java_code_line_content",""); chain=request.get("analytical_chain",""); deny=denylist or DEFAULT_DENYLIST
    user=(f"FOCUS_NAME: {focus}\nANCHOR_LINE: {anchor}\nANCHOR_LINE_CONTENT: {anchor_content}\nANALYTICAL_CHAIN: {chain}\nDENYLIST: {deny}\n\n"
          f"CANDIDATES:\n{candidates}\n\nCODE:\n{code}\nReturn ONLY the JSON object.")
    out=_invoke_json(llm, system=_VALIDATOR_SYSTEM, user=user)
    vs=[]
    for v in out.get("verdicts",[]):
        nm=str(v.get("name","")).strip()
        if nm:
            conf=float(v.get("confidence",0.0)); conf=max(0.0,min(1.0,conf))
            vs.append({"name":nm,"valid":bool(v.get("valid",False)),"confidence":conf,"reason":str(v.get("reason","")).strip()})
    return vs
# lambda_expression_extractor_and_validator.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
import json
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

class EC(TypedDict):
    name: str            # emit the lambda "value" label, e.g., "lambda" or method-ref target name; simple, names-only
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

DEFAULT_DENYLIST=["System.out.println","logger.info","logger.debug","logger.trace","Objects.requireNonNull","Collections.emptyList"]

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

# ---------- Extractor ----------

_RUNA_SYSTEM = """
Task: Extract LAMBDA EXPRESSION values directly related to the focus (names-only representation).
Return STRICT JSON.

Rules:
• Focus=METHOD: include lambda/method-reference VALUES created/passed at top level in the method body,
  e.g., items.anyMatch(x -> p(x)), list.forEach(this::tick). Do NOT expose calls INSIDE lambda bodies.
• Focus=OBJECT VAR X: include lambda values where X is the receiver of the terminal op that accepts the lambda,
  e.g., X.forEach(y -> g(y)).
• Focus=CALL_RESULT: include lambda values applied to that immediate result in the same statement, e.g., a().map(x->x+1).

Emit EC per lambda value; set name to:
  - "lambda" for `args -> body`
  - referenced method simple name for `obj::meth` (e.g., "meth")
  - "new" for constructor refs `Foo::new` (or "new" if you prefer consistency)

{"name":"lambda|meth|new", "code_snippet":"<lambda/ref fragment>", "code_block":"<smallest block>",
 "further_expand":false, "confidence":0..1, "conditioned":false, "guards":[]}
""".strip()

_RUNA_FEWSHOTS = """
Examples:
1) METHOD focus
void m(){ items.forEach(x -> x.doIt()); }
→ ['lambda']

2) OBJECT focus stream
void m(){ stream.map(s -> s.trim()).collect(toList()); }
Focus: stream → ['lambda']  (do not surface 'trim' from inside)

3) Method ref
void m(){ list.forEach(this::tick); Supplier<Foo> s = Foo::new; }
→ names: ['tick','new']
""".strip()

def _build_run_a_user(code:str, focus:str, anchor:int, anchor_content:str, chain:str, deny:list)->str:
    return (f"FOCUS_NAME: {focus}\nANCHOR_LINE: {anchor}\nANCHOR_LINE_CONTENT: {anchor_content}\nANALYTICAL_CHAIN: {chain}\n"
            f"DENYLIST: {deny}\n\nCODE:\n{code}\n\n{_RUNA_FEWSHOTS}\nReturn ONLY {{\"children\":[EC,...]}}.")

_EXPLAIN_LINES_SYSTEM = """
Convert Java to concise NL, one sentence per line (1-based), preserving lambda and method references.
Return STRICT JSON: {"lines":[{"line":int,"text":str},...]}.
""".strip()

_RUNB_SYSTEM = """Using the NL lines, extract LAMBDA VALUES per the same rules. Strict JSON: {"children":[EC,...]}.""".strip()

def _build_run_b_user(explained_json:str, focus:str, anchor:int, anchor_content:str, chain:str, deny:list)->str:
    return (f"FOCUS_NAME: {focus}\nANCHOR_LINE: {anchor}\nANCHOR_LINE_CONTENT: {anchor_content}\nANALYTICAL_CHAIN: {chain}\n"
            f"DENYLIST: {deny}\n\nLINES_NL:\n{explained_json}\nReturn ONLY the JSON object.")

def extract_lambda_expressions(
    llm: AzureChatOpenAI, *, request: LInput, denylist: Optional[List[str]]=None
)->List[EC]:
    focus=request["object_name"]; code=request["java_code"]; anchor=int(request["java_code_line"])
    anchor_content=request.get("java_code_line_content",""); chain=request.get("analytical_chain",""); deny=denylist or DEFAULT_DENYLIST
    out_a=_invoke_json(llm, system=_RUNA_SYSTEM, user=_build_run_a_user(code,focus,anchor,anchor_content,chain,deny))
    a=_norm_ec_list(out_a.get("children",[]))
    explained=_invoke_json(llm, system=_EXPLAIN_LINES_SYSTEM, user="CODE:\n"+code)
    explained_json=json.dumps(explained.get("lines",[]), ensure_ascii=False)
    out_b=_invoke_json(llm, system=_RUNB_SYSTEM, user=_build_run_b_user(explained_json,focus,anchor,anchor_content,chain,deny))
    b=_norm_ec_list(out_b.get("children",[]))
    return _merge_by_name(a,b)

# ---------- Validator ----------

_VALIDATOR_SYSTEM = """
Validate LAMBDA VALUE candidates relative to focus. Valid if a lambda or method reference value
is created/passed at top level per the focus rules. Do NOT validate calls inside lambda bodies.
Return STRICT JSON: {"verdicts":[{"name":"...","valid":bool,"confidence":0..1,"reason":"..."}]}.
""".strip()

def validate_lambda_expressions(
    llm: AzureChatOpenAI, *, request: LInput, candidates: List[EC], denylist: Optional[List[str]]=None
)->List[VerdictTD]:
    focus=request["object_name"]; code=request["java_code"]; anchor=int(request["java_code_line"])
    anchor_content=request.get("java_code_line_content",""); chain=request.get("analytical_chain",""); deny=denylist or DEFAULT_DENYLIST
    user=(f"FOCUS_NAME: {focus}\nANCHOR_LINE: {anchor}\nANCHOR_LINE_CONTENT: {anchor_content}\nANALYTICAL_CHAIN: {chain}\nDENYLIST: {deny}\n\n"
          f"CANDIDATES:\n{candidates}\n\nCODE:\n{code}\nReturn ONLY the JSON object.")
    out=_invoke_json(llm, system=_VALIDATOR_SYSTEM, user=user)
    vs=[]
    for v in out.get("verdicts",[]):
        nm=str(v.get("name","")).strip()
        if nm:
            conf=float(v.get("confidence",0.0)); conf=max(0.0,min(1.0,conf))
            vs.append({"name":nm,"valid":bool(v.get("valid",False)),"confidence":conf,"reason":str(v.get("reason","")).strip()})
    return vs

from langchain_openai import AzureChatOpenAI
from method_call_extractor import extract_method_calls, MethodCallInput, ExtractorConfig

llm_fast = AzureChatOpenAI(azure_deployment="o3-mini", temperature=0)  # you pass your own instance

req: MethodCallInput = {
    "object_name": "x",
    "java_code": """
class Demo {
  void run(){
    Worker w = new Worker();
    x.start().finish();
    x.stop();
    items.forEach(it -> it.process());
  }
}
""",
    "java_code_line": 5,            # anchor the occurrence to analyze (1-based)
    "analytical_chain": "Demo->x",  # last two nodes if any
}

cfg = ExtractorConfig(top_k=5)
children = extract_method_calls(llm_fast, request=req, config=cfg)
for c in children:
    print(c)

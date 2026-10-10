import os
import urllib.request
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import Annotated, Any, Sequence, TypedDict, List
from pathlib import Path
# LangChain & LangGraph Imports
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.graph.message import add_messages

# 1. تحميل مفاتيح البيئة (OpenAI و Tavily و LangSmith)
load_dotenv()

# 2. تهيئة العقل المدبر: استخدام نموذج OpenAI المتقدم والسريع
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# الجزء الأول: أدوات فريق البحث (Research Tools)

# أداة البحث المباشر
tavily_tool = TavilySearch(max_results=3)

# أداة سحب محتوى المواقع (Web Scraper)
@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = ""
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            html = urllib.request.urlopen(req).read()
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            # نكتفي بأول 2000 حرف من كل موقع حتى لا تنفجر ذاكرة النموذج (Context Limit)
            loader += f"\n--- Content from {url} ---\n{text[:2000]}\n"
        except Exception as e:
            loader += f"\nFailed to scrape {url}: {e}"
    return loader

research_tools = [tavily_tool, scrape_webpages]

# الجزء الثاني: أدوات فريق الكتابة (Writing Tools)

# إنشاء مجلد عمل ليحفظ فيه الوكلاء ملفاتهم
WORKING_DIRECTORY = Path("working_dir")
WORKING_DIRECTORY.mkdir(parents=True, exist_ok=True)

@tool
def read_document(file_name: str) -> str:
    """Read the specified document."""
    try:
        with open(WORKING_DIRECTORY / file_name, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@tool
def write_document(file_name: str, content: str) -> str:
    """Create and save a text document."""
    try:
        with open(WORKING_DIRECTORY / file_name, "w") as f:
            f.write(content)
        return f"Document '{file_name}' saved successfully."
    except Exception as e:
        return f"Error writing file: {e}"

repl = PythonREPL()
@tool
def python_repl(code: str):
    """Use this to execute python code."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

writing_tools = [read_document, write_document, python_repl]

# الجزء الثالث: عقدة التحويل الآمنة

# هذه الدالة ستحول ردود الوكلاء إلى رسائل بشرية وتستخرج النص بأمان لتفادي أخطاء القوائم
def agent_node(state, agent, name):
    result = agent.invoke(state)
    raw_content = result["messages"][-1].content
    if isinstance(raw_content, list):
        text_content = "".join([str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in raw_content])
    else:
        text_content = str(raw_content)
    return {"messages": [HumanMessage(content=text_content, name=name)]}


# الجزء الرابع: هندسة الذاكرة ومصنع المديرين
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
import uuid

# ذاكرة النظام (مشتركة بين كل الفرق والمديرين)
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str

# دالة سحرية (مصنع المديرين) لإنشاء أي مدير فرعي أو عام بسهولة
def create_supervisor(llm, members: list, system_prompt: str):
    options = ["FINISH"] + members
    class Route(BaseModel):
        next: str = Field(description=f"Who acts next? Options: {options}")
    
    supervisor_agent = llm.with_structured_output(Route)
    
    def supervisor_node(state):
        prompt = system_prompt + f"\nOptions: {options}"
        messages = [SystemMessage(content=prompt)] + state["messages"]
        decision = supervisor_agent.invoke(messages)
        return {"next": decision.next}
    
    return supervisor_node

# الجزء الخامس: بناء قسم الأبحاث (Research Team Sub-Graph)

# 1. إنشاء العمال
search_agent = create_react_agent(llm, tools=[tavily_tool])
scrape_agent = create_react_agent(llm, tools=[scrape_webpages])

def search_node(state):
    messages = [SystemMessage(content="You are a Searcher. Search the web for info.")] + state["messages"]
    return agent_node({"messages": messages}, search_agent, "Searcher")

def scrape_node(state):
    messages = [SystemMessage(content="You are a Scraper. Scrape URLs for detailed content.")] + state["messages"]
    return agent_node({"messages": messages}, scrape_agent, "Scraper")

# 2. إنشاء مدير قسم الأبحاث
research_supervisor = create_supervisor(
    llm, 
    ["Searcher", "Scraper"], 
    "You are the Research Supervisor. Direct the Searcher and Scraper to gather info. Output FINISH when the research is complete."
)

# 3. تجميع قسم الأبحاث في رسم بياني مستقل (Sub-Graph)
research_graph = StateGraph(AgentState)
research_graph.add_node("Searcher", search_node)
research_graph.add_node("Scraper", scrape_node)
research_graph.add_node("supervisor", research_supervisor)
research_graph.add_edge("Searcher", "supervisor")
research_graph.add_edge("Scraper", "supervisor")
research_graph.add_conditional_edges("supervisor", lambda x: x["next"], {"Searcher": "Searcher", "Scraper": "Scraper", "FINISH": END})
research_graph.set_entry_point("supervisor")
research_chain = research_graph.compile()

# الجزء السادس: بناء قسم الكتابة (Writing Team Sub-Graph)

writer_agent = create_react_agent(llm, tools=writing_tools)
coder_agent = create_react_agent(llm, tools=[python_repl])

def writer_node(state):
    messages = [SystemMessage(content="You are a Writer. Write or read documents to the working_dir.")] + state["messages"]
    return agent_node({"messages": messages}, writer_agent, "Writer")

def coder_node(state):
    messages = [SystemMessage(content="You are a Coder. Execute python scripts when calculations are needed.")] + state["messages"]
    return agent_node({"messages": messages}, coder_agent, "Coder")

writing_supervisor = create_supervisor(
    llm, 
    ["Writer", "Coder"], 
    "You are the Writing Supervisor. Direct the Writer to create/edit files or the Coder to run scripts. Output FINISH when done."
)

writing_graph = StateGraph(AgentState)
writing_graph.add_node("Writer", writer_node)
writing_graph.add_node("Coder", coder_node)
writing_graph.add_node("supervisor", writing_supervisor)
writing_graph.add_edge("Writer", "supervisor")
writing_graph.add_edge("Coder", "supervisor")
writing_graph.add_conditional_edges("supervisor", lambda x: x["next"], {"Writer": "Writer", "Coder": "Coder", "FINISH": END})
writing_graph.set_entry_point("supervisor")
writing_chain = writing_graph.compile()

# الجزء السابع: تجميع المؤسسة والمدير العام (Top-Level Graph)

# عقد الربط: لتحويل رد الفريق بالكامل إلى رسالة واحدة يقرأها المدير العام
def research_team_node(state):
    result = research_chain.invoke(state)
    return agent_node(state, research_chain, "ResearchTeam")

def writing_team_node(state):
    result = writing_chain.invoke(state)
    return agent_node(state, writing_chain, "WritingTeam")

# المدير العام للمؤسسة
top_supervisor = create_supervisor(
    llm, 
    ["ResearchTeam", "WritingTeam"], 
    "You are the General Manager. Delegate research tasks to ResearchTeam, and file creation/coding to WritingTeam. Output FINISH when the user's request is completely fulfilled."
)

super_graph = StateGraph(AgentState)
super_graph.add_node("ResearchTeam", research_team_node)
super_graph.add_node("WritingTeam", writing_team_node)
super_graph.add_node("supervisor", top_supervisor)

super_graph.add_edge("ResearchTeam", "supervisor")
super_graph.add_edge("WritingTeam", "supervisor")
super_graph.add_conditional_edges("supervisor", lambda x: x["next"], {"ResearchTeam": "ResearchTeam", "WritingTeam": "WritingTeam", "FINISH": END})
super_graph.set_entry_point("supervisor")

# تفعيل الذاكرة الدائمة (MemorySaver)
memory = MemorySaver()
enterprise_app = super_graph.compile(checkpointer=memory)


# الجزء الثامن: حلقة التشغيل التفاعلية وحواجز الحماية
if __name__ == "__main__":
    print("🏢 Welcome to the Hierarchical Enterprise System!\n")
    user_input = input("📌 What complex task do you want the enterprise to do?: ")
    
    thread_id = str(uuid.uuid4())
    # رفعنا حد الدورات إلى 25 لأن الفرق الهرمية تأخذ دورات أكثر
    run_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    initial_state = {"messages": [HumanMessage(content=user_input)]}

    try:
        for event in enterprise_app.stream(initial_state, config=run_config):
            for node_name, node_state in event.items():
                print(f"\n🔄 --- [ {node_name.upper()} HAS COMPLETED THEIR TURN ] ---")
                if "messages" in node_state:
                    print(node_state["messages"][-1].content)
                elif "next" in node_state:
                    print(f"👔 Manager decided the next team is: {node_state['next']}")
                    
    except GraphRecursionError:
        print("\n⚠️ [SYSTEM ALERT] Maximum iteration limit reached (25 cycles).")
        user_decision = input("❓ Do you want to see the enterprise's scratchpad so far? (yes/no): ").strip().lower()
        if user_decision in ['yes', 'y']:
            current_memory = enterprise_app.get_state(run_config).values.get("messages", [])
            for msg in current_memory:
                sender = msg.name if msg.name else getattr(msg, "type", "Unknown")
                raw = msg.content
                text = "".join([str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in raw]) if isinstance(raw, list) else str(raw)
                print(f"[{sender.upper()}]:\n{text}\n{'-'*30}")
                
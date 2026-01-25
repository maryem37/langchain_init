import os
import pandas as pd
from custom_pandas_engine import SimplePandasQueryEngine
from prompts import instruction_str
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.llms.ollama import Ollama
from pdf import canada_engine

# --------------------------
# LLM SETUP
# --------------------------
llm = Ollama(
    model="llama3.2:1b", 
    request_timeout=120.0,
    system_prompt="You are a helpful assistant. Provide direct, concise answers without conversational extras. Answer only what is asked."  # ✅ ADD THIS
)

# --------------------------
# POPULATION DATA
# --------------------------
population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = SimplePandasQueryEngine(
    df=population_df,
    llm=llm,
    instruction_str=instruction_str
)

# --------------------------
# SIMPLE AGENT (without ReActAgent)
# --------------------------
def route_query(query: str) -> str:
    """Route queries to appropriate tools."""
    query_lower = query.lower()
    
    # Check for note saving
    if "save note" in query_lower or "remember" in query_lower:
        note_content = query.replace("save note", "").replace("remember", "").strip()
        return note_engine.fn(note_content)
    
    # Check for population queries
    elif any(word in query_lower for word in ["population", "people", "demographic", "country"]):
        result = population_query_engine.query(query)
        return str(result)
    
    # Check for Canada queries
    elif "canada" in query_lower or "canadian" in query_lower:
        result = canada_engine.query(query)
        return str(result)
    
    # Default to asking LLM
    else:
        response = llm.complete(query)
        return str(response).strip()  # ✅ ADD .strip() to remove extra whitespace

# --------------------------
# INTERACTIVE PROMPT
# --------------------------
print("=== Local RAG System using Ollama ===")
print("Ask about population data, Canada, or save notes!")
print("Type 'q' to quit.\n")

while (prompt := input("Enter a question: ")) != "q":
    try:
        result = route_query(prompt)
        print(f"\n{result}\n")  # ✅ SIMPLIFIED: Remove "=== RESPONSE ===" wrapper
    except Exception as e:
        print(f"Error: {e}\n")
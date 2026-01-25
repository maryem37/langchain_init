import pandas as pd
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama

class SimplePandasQueryEngine(CustomQueryEngine):
    """Simple Pandas query engine."""
    
    df: pd.DataFrame
    llm: LLM
    instruction_str: str = ""
    
    def custom_query(self, query_str: str):
        """Execute query."""
        prompt = f"""
{self.instruction_str}

Given this dataframe:
{self.df.head(10).to_string()}

Query: {query_str}

Generate Python code using pandas to answer this query.
Only output the Python expression, no explanation.
The dataframe variable name is 'df'.
"""
        response = self.llm.complete(prompt)
        code = str(response).strip()
        
        try:
            # Execute the pandas code
            result = eval(code, {"df": self.df, "pd": pd})
            return str(result)
        except Exception as e:
            return f"Error executing query: {e}\nGenerated code: {code}"
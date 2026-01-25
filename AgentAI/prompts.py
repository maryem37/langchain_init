from llama_index.core import PromptTemplate

instruction_str = """\
1. Convert the query to executable Python code using Pandas.
2. The final line of code should be a Python expression that can be called with the `eval()` function.
3. The code should represent a solution to the query.
4. PRINT ONLY THE EXPRESSION.
5. Do not quote the expression."""

new_prompt = PromptTemplate(
    """\
You are working with a pandas dataframe in Python.
The name of the dataframe is `df`.
This is the result of `print(df.head())`:
{df_str}

Follow these instructions:
{instruction_str}
Query: {query_str}

Expression: """
)

context = """Purpose: The primary role of this agent is to assist users by providing accurate 
information about world population statistics and details about a country."""

coffee_context = """
## Coffee Shop Data Usage Guide
- Data Range: Surrounding the National Gallery of Canada, within a 2 km radius.
- Field Descriptions：
  - price_level: 
    - $: Budget-friendly
    - $$: Mid-range
    - $1-10: Price range $1-$10
    - $20-30: Price range $20-$30
    - [Variable range like $15-20]: Represents a variable range
  - rating: Google rating 
  - distance_km: Distance from the gallery in km
- Example Queries:
  - 'List coffee shops with rating > 4 within 1km'
  - 'What is the average price level of shops within 0.5km?'
  - 'Find the closest 3 coffee shops'
"""

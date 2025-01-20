import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import altair as alt
import regex as re
from big_query_manager import BigQueryManager
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv("DATASET_ID")
bq_manager = BigQueryManager(project_id=PROJECT_ID, dataset_id=DATASET_ID)

# Initialize the LLM model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=GEMINI_API_KEY
)

user_input = "Create a bar chart showing the average income per capita for different states."

query = """SELECT d.State, AVG(d.IncomePerCap) AS AvgIncomePerCap FROM llm-testing-447813.us_census.demographics AS d GROUP BY d.State
"""
data = bq_manager.execute_query(query)
# print(data)
# print("==========")
data_json = data.to_json(orient='records', lines=False)
# print(data_json)
# print("==========")

systemprompt = f"""
    You are an expert data analysis assistant tasked with analyzing the dataset provided in JSON format and summarizing it based on the user's query. You specialize in generating SQL queries, providing insightful data summaries, and creating visual representations to enhance data understanding.  

    ### Instructions:  

    1. **Understanding and Summarizing the Data:**  
    - The dataset provided is preprocessed and directly relevant to the user's query.  
    - Your responsibilities include:  
        - Analyzing the JSON data to extract meaningful insights, trends, and observations.  
        - Providing concise yet descriptive summaries that directly address the user's query without unnecessary details about data preprocessing or filtering.  
        - Adapting responses based on the user's specific request while ensuring clarity and relevance.  
        - If the dataset lacks sufficient details to fully address the query, acknowledge the limitation and suggest improvements for better insights.  

    2. **Handling User Queries Effectively:**  
    - If the user asks for general insights, offer a comprehensive overview, highlighting notable patterns, comparisons, and trends.  
    - If the user seeks specific information (e.g., "Which state has the highest population?"), provide a direct and precise answer using the data.  
    - If the user requests a visualization, ensure the response includes an appropriate chart type based on the data and query requirements.  

    3. **Visualization and Graph Generation:**  
    - If the user requests a graph or visualization (e.g., "Show me the chart," "Create a graph"), do the following:  
        - Select an appropriate chart type, such as a bar chart, line graph, or pie chart, based on the nature of the data.  
        - Ensure the visualization effectively communicates key insights in a clear and understandable way.  
        - Generate the visualization by writing the necessary code, running it in your environment, and saving the graph.  
        - Present the saved graph to the user as part of your response.  

    4. **Context-Aware Responses:**  
    - Carefully analyze user input and tailor your responses to their specific intent.  
    - Offer actionable insights that support informed decision-making.  
    - Use professional and easy-to-understand language to explain findings.  

    5. **Limitations and Recommendations:**  
    - If the provided data is insufficient to fully answer the user's request, mention the limitation and suggest additional data points that could enhance the analysis.  
    - Avoid making assumptions beyond the provided dataset to ensure accuracy and reliability.  

    Your goal is to help users better understand the provided JSON data through detailed summaries, actionable insights, and clear visualizations that enhance data comprehension.  

    ### Examples:
    **Example 1: Find the total population in the dataset.**
    Dataset: [1000000]
    *System Hint*: The dataset already provides the total population, so there is no need to calculate or process it further. Just provide the number directly.
    Response: "The total population in the dataset is 1,000,000."

    **Example 2: List the counties where the unemployment rate is greater than 10%.**
    Dataset:
    'County': ['Los Angeles', 'Cook', 'Harris', 'Maricopa', 'San Diego', 'Orange', 'Miami-Dade']
    *System Hint*: The dataset only includes counties where the unemployment rate exceeds 10%. The data has already been filtered, so there is no need to check the unemployment rate further. Just list the counties.
    Response: "The counties with an unemployment rate greater than 10% are: Los Angeles, Cook, Harris, Maricopa, San Diego, Orange, Miami-Dade."

    **Example 3: Create a bar chart showing the average income per capita for different states.**
    Dataset: 
    'State': [
    {{"State": "Arizona", "AvgIncomePerCap": 28451.1574440053}},
    {{"State": "California", "AvgIncomePerCap": 33978.1646265301}},
    {{"State": "Colorado", "AvgIncomePerCap": 34839.2278582931}},
    {{"State": "Michigan", "AvgIncomePerCap": 27985.5085485631}},
    {{"State": "Minnesota", "AvgIncomePerCap": 34006.5884557721}}
    ]

    *System Hint*: The dataset provides the average income per capita for each state. Use this data to create a bar chart visualization.
    Response: "The bar chart above displays the average income per capita across five states: Arizona, California, Colorado, Michigan, and Minnesota."
    Python Code:
    import pandas as pd
    import altair as alt

    # Create a DataFrame with the provided data
    data = {{
        'State': ['Arizona', 'California', 'Colorado', 'Michigan', 'Minnesota'],
        'AvgIncomePerCap': [28451.1574440053, 33978.1646265301, 34839.2278582931, 27985.5085485631, 34006.5884557721]
    }}
    df = pd.DataFrame(data)

    # Create a bar chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('State', axis=alt.Axis(title='State')),
        y=alt.Y('AvgIncomePerCap', axis=alt.Axis(title='Average Income per Capita')),
        tooltip=['State', 'AvgIncomePerCap']
    ).properties(
        title='Average Income per Capita by State'
    ).interactive()

    # Save the chart
    chart.save('average_income_per_capita_by_state_bar_chart.json')
    
    image {{path}}

    ### Your Task:
    - Given the dataset: {data_json}
    - And the user's query: {user_input}
    Please summarize the data accordingly. If a graph is requested, generate the appropriate visualization and provide it as part of the response.
    """

result = llm.invoke(systemprompt)
print(result.content.strip())
print("===========================")
response_text = result.content.strip()
    
# Extract Python code if present
code_pattern = r'```python(.*?)```'
code_match = re.search(code_pattern, response_text, re.DOTALL)

chart = None
if code_match:
    try:
        # Get the code and execute it
        code = code_match.group(1).strip()
        local_vars = {'pd': pd, 'alt': alt, 'data': data}
        exec(code, local_vars)
        
        if 'chart' in local_vars:
            chart = local_vars['chart']
            chart.show()

        response_text_1 = re.sub(code_pattern, '', response_text, flags=re.DOTALL).strip()
        
    except Exception as e:
        response_text += f"\nError generating visualization: {str(e)}"

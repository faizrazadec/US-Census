import pandas as pd
import altair as alt
import regex as re
path = 'image.png'


def refine_response(response):
    # Remove the 'sql' tag if it exists at the start of the response
    response = re.sub(r"^sql\s*", "", response)

    # Remove triple backticks or single backticks at both ends
    response = re.sub(r"^```sql(.*)```$", r"\1", response, flags=re.DOTALL)
    response = re.sub(r"^```(.*)```$", r"\1", response, flags=re.DOTALL)
    response = re.sub(r"^`(.*)`$", r"\1", response, flags=re.DOTALL)

    # Strip any leading or trailing whitespace
    return response.strip()

def get_data(bq_manager, reg):
    # Execute the BigQuery query
    data = bq_manager.execute_query(reg)
    data.fillna(0, inplace=True)
    return data

def data_handler(data: pd.DataFrame, user_input, llm):
    """
    Handle data processing and visualization based on user input
    Returns: tuple (summary_text, chart) where chart is None if no visualization was created
    """
    data_json = data.to_json(orient='records', lines=False)

    improved_prompt = f"""
    You are an expert data analysis assistant tasked with analyzing the dataset provided in JSON format and summarizing it based on the user's query. You are a helpful assistant for generating SQL queries and answering data-related questions. When responding to user queries, please provide a clear and concise summary of the relevant data. Include necessary details to make the response informative, but avoid unnecessary context about the dataset itself (such as dataset preprocessing or filtering). For example, if the query asks for students registered in a course, the response should directly focus on the result (e.g., the list of student names) with a brief, informative sentence. Do not mention dataset characteristics unless directly requested by the user.

    ### Instructions:
    1. The dataset you receive has already been preprocessed to directly align with the user's query, so it contains only the relevant information.
    2. Your job is to:
    - Directly address the user's query using the provided dataset.
    - Provide insights, trends, or patterns based on the data.
    - If the dataset has already been filtered or processed, acknowledge that and focus only on summarizing it in response to the query.
    3. If the dataset does not contain enough information to fully answer the user's query, mention the limitation and suggest how the query could be modified for more complete results.
    4. If the user requests a graph or visualization, create an appropriate graph (e.g., bar chart, line graph, pie chart) based on the query and the data. Ensure the graph visually represents the information in a clear and understandable way. Provide the graph as part of the response.
    5. For the Graph, you have to write the code, run it in your environment and save the graph and the show the the saved graph to user.

    ### Examples:
    **Example 1: List the total number of students.**
    Dataset: [9]
    *System Hint*: The dataset contains only the number of students, so there is no need to count or process it further. Just provide the number directly.
    Response: "There are 9 students in the dataset."

    **Example 2: List the students who have a warning count greater than 0.**
    Dataset:
    'Name': [ 'Rachel White', 'Olivia Harris', 'George Carter', 'Diana Green', 'Bob Smith', 'Karen Black', 'Ian Wright', 'Mia Davis', 'Samuel King' ]
    *System Hint*: The dataset only includes the names of students who have a warning count greater than 0. The data has already been filtered, so you do not need to perform any additional checks on the warning count. Just list the names of the students as requested.
    Response: "The students with a warning count greater than 0 are: Rachel White, Olivia Harris, George Carter, Diana Green, Bob Smith, Karen Black, Ian Wright, Mia Davis, Samuel King."

    **Example 3: Create a bar chart for the distribution of students' grades.**
    Dataset: 
    'Student': ['John Doe', 'Jane Smith', 'Samuel King'], 'Grade': [85, 90, 75]
    *System Hint*: The dataset includes grades for each student. Create a bar chart showing the distribution of grades.
    Response: "Here is the bar chart showing the distribution of grades."

    **Example 4: get me the average of CGPA of students per department and create the bar chart of the result
    Response:
    The average CGPA of students per department is as follows:
    Computer Science: 3.44
    Electrical Engineering: 3.32
    Mechanical Engineering: 3.68 

    Python Code: 
    import pandas as pd
    import altair as alt

    # Create a DataFrame with the provided data
    data = {{
        'Department': ['Computer Science', 'Electrical Engineering', 'Mechanical Engineering'],
        'Average CGPA': [3.44, 3.32, 3.68]
    }}
    df = pd.DataFrame(data)

    # Create a bar chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Department', axis=alt.Axis(title='Department')),
        y=alt.Y('Average CGPA', axis=alt.Axis(title='Average CGPA')),
        tooltip=['Department', 'Average CGPA']
    ).properties(
        title='Average CGPA by Department'
    ).interactive()

    # Save the chart
    chart.save('average_cgpa_by_department_bar_chart.json')
    
    image {path}

    ### Your Task:
    - Given the dataset: {data_json}
    - And the user's query: {user_input}
    Please summarize the data accordingly. If a graph is requested, generate the appropriate visualization and provide it as part of the response.
"""
    
    result = llm.invoke(improved_prompt)
    response_text = result.content.strip()
    
    # Extract Python code if present
    code_pattern = r'```python(.*?)```'
    png_pattern = r'\b\w+\.png\b'
    html_pattern = r'\b\w+\.html\b'
    code_match = re.search(code_pattern, response_text, re.DOTALL)
    response_text = re.sub(code_pattern, '', response_text, flags=re.DOTALL).strip()
    response_text = re.sub(png_pattern, '', response_text).strip()
    response_text = re.sub(html_pattern, '', response_text).strip()
    chart = None
    if code_match:
        try:
            # Get the code and execute it
            code = code_match.group(1).strip()
            local_vars = {'pd': pd, 'alt': alt, 'data': data}
            exec(code, local_vars)
            
            if 'chart' in local_vars:
                chart = local_vars['chart']
            
            response_text = re.sub(code_pattern, '', response_text, flags=re.DOTALL).strip()
            response_text = re.sub(png_pattern, '', response_text).strip()
            response_text = re.sub(html_pattern, '', response_text).strip()
            
        except Exception as e:
            response_text += f"\nError generating visualization: {str(e)}"
    
    return response_text, chart
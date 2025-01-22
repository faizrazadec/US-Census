import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from system_prompt import SYSTEM_PROMPT
from big_query_manager import BigQueryManager
import regex as re
import pandas as pd
import altair as alt
from logger import setup_logger

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Get the configured logger
logger = setup_logger()

PROJECT_ID = os.getenv('PROJECT_ID')
DATASET_ID = os.getenv("DATASET_ID")
bq_manager = BigQueryManager(project_id=PROJECT_ID, dataset_id=DATASET_ID)

# Initialize the embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY,
    task_type="retrieval_document"
)
# Initialize the LLM model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=GEMINI_API_KEY
)
# Initialize the Chroma vector store (assumed to be stored in './chroma_langchain_db')
vector_store = Chroma(
    collection_name="Demographics_Schema_Collection",
    embedding_function=embeddings,
    persist_directory="./Chroma_db"  # Path where Chroma DB is persisted
)

def generate_initial_response(user_input, llm, vector_store, k):
    """Generate the initial response from the LLM based on user input and schema context."""
    try:
        logger.info("Function generate_initial_response.")
        # Retrieve relevant schema information from ChromaDB
        results = vector_store.similarity_search(user_input, k=k)

        # Flatten the list of results to extract content
        flattened_context = [item.page_content for item in results]

        # Concatenate retrieved schema context
        context = "\n".join(flattened_context)

        # Initial system prompt and message
        system_message = SystemMessage(content=f"{SYSTEM_PROMPT}\nSchema Context:\n{context}")
        human_message = HumanMessage(content=user_input)

        # Generate initial response
        response = llm.invoke([system_message, human_message])

        # Debugging output
        # print("Initial Response generated:")
        # print(response.content.strip())

        return response.content.strip()
    except Exception as e:
        logger.error("Error generating response...")
        print(f"Error generating response: {e}")
        return "An error occurred while processing your request. Please try again later."

def trigger_fallback_logic(user_input, llm, context, human_message):
    """Trigger the fallback logic when the initial response cannot generate a SQL query."""
    try:
        logger.info("Fallback Logic triggering...")
        print("Triggering fallback logic")

        # Fallback system prompt
        SYSTEM_PROMPT_2 = f"""
        You are an assistant tasked with refining natural language queries for better SQL generation. 

        **IMPORTANT INSTRUCTIONS:**
        1. **You are strictly bounded NOT to generate or include any SQL queries under any circumstances.**
        2. Your task is to:
        - Explain why the user's original query could not generate a valid SQL query.
        - You are strictly bound to return at least 3 refined natural language prompts that address the issues in the original query.
        3. Your response must strictly contain:
        - A short explanation of why the original query failed.
        - Refined natural language prompts, formatted as bullet points.
        4. **Do NOT explain how to write SQL queries.**
        5. **Do NOT mention SQL query structures, examples, or any SQL-related code in your response.**

        **Here is the user's query that needs refinement:**  
        {user_input}

        **Schema Context:**  
        {context}

        **Response Format:**
        1. **Why the Query Failed:**  
        - [Brief explanation of failure]

        2. **Refined Prompts:**  
        - Refined Prompt 1: [First refined query]  
        - Refined Prompt 2: [Second refined query]  
        - Refined Prompt 3: [Third refined query]  

        **Strict Reminder:**  
        - You are strictly bound NOT to generate or include SQL queries in your response.
        - You are strictly bound NOT to NOT discuss SQL syntax, query examples, or anything related to SQL query writing.
        - You are strictly bound not to **Suggested SQL Query:**
        - You are strictly bound not to return Improved SQL (based on Refined Prompt)
        """
        
        # Use fallback system prompt
        refined_system_message = SystemMessage(content=SYSTEM_PROMPT_2)
        refined_response = llm.invoke([refined_system_message, human_message])

        # Debugging output
        print("Refined Response generated:")
        print(refined_response.content.strip())
        
        # Return refined response (this means no further processing or BigQuery execution)
        return refined_response.content.strip()

    except Exception as e:
        print(f"Error triggering fallback logic: {e}")
        logger.error("Error triggering fallback logic.")
        return "An error occurred while processing the fallback logic. Please try again later."
    
def get_response(user_input, llm, vector_store, k):
    """Main function to get response and handle fallback logic if needed."""
    try:
        logger.info("Function get_response...")
        # Generate initial response
        response = generate_initial_response(user_input, llm, vector_store, k)

        if "I cannot generate a SQL query for this request based on the provided schema." in response:
            # If the response indicates fallback is needed, trigger fallback logic
            # print("Fallback triggered.")
            logger.info("Fallback triggered...")
            # Retrieve schema context from ChromaDB again to pass to the fallback logic
            results = vector_store.similarity_search(user_input, k=k)
            flattened_context = [item.page_content for item in results]
            context = "\n".join(flattened_context)
            # Call the fallback logic
            return trigger_fallback_logic(user_input, llm, context, HumanMessage(content=user_input))
        
        logger.critical("TERMINATED")
        # Return the initial response if successful (i.e., SQL query generation)
        return response

    except Exception as e:
        print(f"Error in get_response: {e}")
        logger.error("An error occurred while processing your request")
        logger.critical("TERMINATED")
        return "An error occurred while processing your request. Please try again later."
    
def refine_response(response):
    try:
        logger.info("Refining Resonse...")
        # Remove the 'sql' tag if it exists at the start of the response
        response = re.sub(r"^sql\s*", "", response)

        # Remove triple backticks or single backticks at both ends
        response = re.sub(r"^```sql(.*)```$", r"\1", response, flags=re.DOTALL)
        response = re.sub(r"^```(.*)```$", r"\1", response, flags=re.DOTALL)
        response = re.sub(r"^`(.*)`$", r"\1", response, flags=re.DOTALL)
    except:
        logger.error("Error in refine_response.")
    # Strip any leading or trailing whitespace
    return response.strip()

def get_data(bq_manager, reg):
    try:
        # Execute the BigQuery query
        logger.info("Hitting BigQuery...")
        data = bq_manager.execute_query(reg)
    except:
        logger.error("SQL Query Problem.")
    return data

def preprocess_data(data: pd.DataFrame):
    logger.info("Preporcessing Data...")
    data.fillna(0, inplace=True)
    data = data[(data != 0).any(axis=1)].reset_index(drop=True)
    return data

def save_json(data, filename='data.json'):
    logger.info("Saving to Json...")
    data.to_json(filename, orient='records', indent=4)
    return filename

def get_head(data, rows=10):
    logger.info("Fetching Head...")
    return data.head(rows).to_json(orient='records', indent=4)

def short_data(data:pd.DataFrame, user_input, llm):
    logger.info("Fucntion short_data.")
    preprocessed_data = preprocess_data(data)
    data_json = preprocessed_data.to_json(orient='records', lines=False, indent=4)

    system_prompt = f"""
    You are an expert data analysis assistant tasked with analyzing the dataset provided in JSON format and summarizing it based on the user's query. When responding to user queries, please provide a clear and concise summary of the relevant data. Include necessary details to make the response informative, but avoid unnecessary context about the dataset itself (such as dataset preprocessing or filtering). For example, if the query asks for students registered in a course, the response should directly focus on the result (e.g., the list of student names) with a brief, informative sentence. Do not mention dataset characteristics unless directly requested by the user.

    ### Instructions:
    1. The dataset you receive has already been preprocessed to directly align with the user's query, so it contains only the relevant information.
    2. Your job is to:
    - Directly address the user's query using the provided dataset.
    - Provide insights, trends, or patterns based on the data.
    - If the dataset has already been filtered or processed, acknowledge that and focus only on summarizing it in response to the query.
    3. If the dataset does not contain enough information to fully answer the user's query, then you have to respond on the provided data as the provided data is the only relevent data that user asked for. and don't mention the limitation and suggest how the query could be modified for more complete results.
    4. If the user requests a graph or visualization, create an appropriate graph (e.g., bar chart, line graph, pie chart) based on the query and the data. Ensure the graph visually represents the information in a clear and understandable way. Provide the graph as part of the response.
    5. For the Graph, you have to write the code, run it in your environment and save the graph and the show the the saved graph to user.

    ### Examples:
    **Example 1: Find the total population in the dataset.**
    Dataset: [1000000]
    *System Hint*: The dataset already provides the total population, so there is no need to calculate or process it further. Just provide the number directly.
    Response: "The total population is 1,000,000."

    **Example 2: List the counties where the unemployment rate is greater than 10%.**
    Dataset:
    'County': ['Los Angeles', 'Cook', 'Harris', 'Maricopa', 'San Diego', 'Orange', 'Miami-Dade']
    *System Hint*: The dataset only includes counties where the unemployment rate exceeds 10%. The data has already been filtered, so there is no need to check the unemployment rate further. Just list the counties.
    Response: "The counties with an unemployment rate greater than 10% are: Los Angeles, Cook, Harris, Maricopa, San Diego, Orange, Miami-Dade."

    **Example 3: Create a bar chart showing the average income per capita for different states.**
    Dataset: 
    'State': ['California', 'Texas', 'Florida'], IncomePerCap: [40000, 35000, 30000]
    *System Hint*: The dataset provides the average income per capita for each state. Use this data to create a bar chart visualization.
    Response: "The income per capita values are as follows:
        California: $40,000
        Texas: $35,000
        Florida: $30,000
    The  California has the highest average income per capita among the three states, followed by Texas, and then Florida with the lowest average income per capita. Here is the bar chart showing the average income per capita for different states.
    Python Code:
    import pandas as pd
    import altair as alt
    # Make dataframe with columns: `State` and `IncomePerCap` using the data given.
    df = pd.DataFrame({{'State': ['California', 'Texas', 'Florida'], 'IncomePerCap': [40000, 35000, 30000]}})
    # Create a bar chart
    chart = alt.Chart(df).mark_bar().encode(
        x='State',
        y='IncomePerCap',
        tooltip=['State', 'IncomePerCap']
    ).properties(
        title='Average Income per Capita by State'
    ).interactive()
    # Save the chart
    chart.save('average_income_per_capita_by_state_bar_chart.json')"

    **Example 4: Calculate the percentage of people working from home in each state and create a bar chart of the result.
    Dataset:
    [{{"State":"Arizona","AvgWorkFromHomePercentage":6.1744554455}},{{"State":"California","AvgWorkFromHomePercentage":5.7201201201}},{{"State":"Colorado","AvgWorkFromHomePercentage":7.4054971706}},{{"State":"Michigan","AvgWorkFromHomePercentage":3.8448489261}},{{"State":"Minnesota","AvgWorkFromHomePercentage":5.5103603604}},{{"State":"Montana","AvgWorkFromHomePercentage":7.873605948}},{{"State":"Nebraska","AvgWorkFromHomePercentage":4.9155009452}},{{"State":"North Carolina","AvgWorkFromHomePercentage":5.0084870849}},{{"State":"Utah","AvgWorkFromHomePercentage":5.8878424658}},{{"State":"Wisconsin","AvgWorkFromHomePercentage":4.3570812365}},{{"State":"Alabama","AvgWorkFromHomePercentage":2.8529361702}},{{"State":"Alaska","AvgWorkFromHomePercentage":5.4102409639}},{{"State":"Arkansas","AvgWorkFromHomePercentage":3.2213138686}},{{"State":"Connecticut","AvgWorkFromHomePercentage":4.8723095526}},{{"State":"Delaware","AvgWorkFromHomePercentage":4.7831775701}},{{"State":"Florida","AvgWorkFromHomePercentage":6.1060009602}},{{"State":"Georgia","AvgWorkFromHomePercentage":4.7996927803}},{{"State":"Hawaii","AvgWorkFromHomePercentage":4.980952381}},{{"State":"Illinois","AvgWorkFromHomePercentage":4.3616051364}},{{"State":"Indiana","AvgWorkFromHomePercentage":3.3805444887}},{{"State":"Iowa","AvgWorkFromHomePercentage":4.7150668287}},{{"State":"Kansas","AvgWorkFromHomePercentage":4.2932806324}},{{"State":"Kentucky","AvgWorkFromHomePercentage":3.3713898917}},{{"State":"Louisiana","AvgWorkFromHomePercentage":2.8303730018}},{{"State":"Maine","AvgWorkFromHomePercentage":5.7005698006}},{{"State":"Maryland","AvgWorkFromHomePercentage":4.461299639}},{{"State":"Massachusetts","AvgWorkFromHomePercentage":4.9449453552}},{{"State":"Mississippi","AvgWorkFromHomePercentage":2.0687975647}},{{"State":"Missouri","AvgWorkFromHomePercentage":4.5036717063}},{{"State":"Nevada","AvgWorkFromHomePercentage":4.0932153392}},{{"State":"New Hampshire","AvgWorkFromHomePercentage":6.1482876712}},{{"State":"New Jersey","AvgWorkFromHomePercentage":4.1807518797}},{{"State":"New Mexico","AvgWorkFromHomePercentage":5.2076305221}},{{"State":"New York","AvgWorkFromHomePercentage":4.0684275691}},{{"State":"Ohio","AvgWorkFromHomePercentage":3.5995915589}},{{"State":"Oklahoma","AvgWorkFromHomePercentage":3.7250717703}},{{"State":"Oregon","AvgWorkFromHomePercentage":6.8875151515}},{{"State":"Pennsylvania","AvgWorkFromHomePercentage":4.2565842978}},{{"State":"Rhode Island","AvgWorkFromHomePercentage":3.945}},{{"State":"South Carolina","AvgWorkFromHomePercentage":3.7717171717}},{{"State":"Tennessee","AvgWorkFromHomePercentage":3.9021636241}},{{"State":"Texas","AvgWorkFromHomePercentage":4.2865620214}},{{"State":"Vermont","AvgWorkFromHomePercentage":7.4601092896}},{{"State":"Virginia","AvgWorkFromHomePercentage":4.6888}},{{"State":"Washington","AvgWorkFromHomePercentage":5.8076177285}},{{"State":"Wyoming","AvgWorkFromHomePercentage":4.9648854962}},{{"State":"Puerto Rico","AvgWorkFromHomePercentage":2.2007900677}},{{"State":"Idaho","AvgWorkFromHomePercentage":6.0047138047}},{{"State":"District of Columbia","AvgWorkFromHomePercentage":5.282122905}},{{"State":"North Dakota","AvgWorkFromHomePercentage":6.0541463415}},{{"State":"South Dakota","AvgWorkFromHomePercentage":6.8779279279}},{{"State":"West Virginia","AvgWorkFromHomePercentage":3.1214876033}}]
    *System Hint*: The dataset provides the average workathome percentage of the states, use this data to create the bar graph.
    Response:
    "The states with the highest average work-from-home percentages are Montana (7.87%), Vermont (7.46%), and Colorado (7.41%), while the states with the lowest percentages are Mississippi (2.07%), Puerto Rico (2.20%), and Louisiana (2.83%).
    The average work-from-home percentages for different states are as follows:
    Montana: 7.87%
    Vermont: 7.46%
    Colorado: 7.41%
    South Dakota: 6.88%
    Oregon: 6.89%
    Arizona: 6.17%
    New Hampshire: 6.15%
    Florida: 6.11%
    North Dakota: 6.05%
    Idaho: 6.00%
    Washington: 5.81%
    Utah: 5.89%
    Maine: 5.70%
    California: 5.72%
    Minnesota: 5.51%
    Alaska: 5.41%
    New Mexico: 5.21%
    District of Columbia: 5.28%
    North Carolina: 5.01%
    Hawaii: 4.98%
    Wisconsin: 4.36%
    Illinois: 4.36%
    Texas: 4.29%
    Kansas: 4.29%
    Missouri: 4.50%
    Maryland: 4.46%
    Pennsylvania: 4.26%
    Nevada: 4.09%
    New Jersey: 4.18%
    New York: 4.07%
    Rhode Island: 3.95%
    Tennessee: 3.90%
    South Carolina: 3.77%
    Oklahoma: 3.73%
    Ohio: 3.60%
    Michigan: 3.84%
    Indiana: 3.38%
    Kentucky: 3.37%
    Arkansas: 3.22%
    West Virginia: 3.12%
    Alabama: 2.85%
    Louisiana: 2.83%
    Puerto Rico: 2.20%
    Mississippi: 2.07%

    Python Code: 
    import pandas as pd
    import altair as alt
    # Define the data using a dictionary
    data = [
        {{"State": "Arizona", "AvgWorkFromHomePercentage": 6.1744554455}},
        {{"State": "California", "AvgWorkFromHomePercentage": 5.7201201201}},
        {{"State": "Colorado", "AvgWorkFromHomePercentage": 7.4054971706}},
        {{"State": "Michigan", "AvgWorkFromHomePercentage": 3.8448489261}},
        {{"State": "Minnesota", "AvgWorkFromHomePercentage": 5.5103603604}},
        {{"State": "Montana", "AvgWorkFromHomePercentage": 7.873605948}},
        {{"State": "Nebraska", "AvgWorkFromHomePercentage": 4.9155009452}},
        {{"State": "North Carolina", "AvgWorkFromHomePercentage": 5.0084870849}},
        {{"State": "Utah", "AvgWorkFromHomePercentage": 5.8878424658}},
        {{"State": "Wisconsin", "AvgWorkFromHomePercentage": 4.3570812365}},
        {{"State": "Alabama", "AvgWorkFromHomePercentage": 2.8529361702}},
        {{"State": "Alaska", "AvgWorkFromHomePercentage": 5.4102409639}},
        {{"State": "Arkansas", "AvgWorkFromHomePercentage": 3.2213138686}},
        {{"State": "Connecticut", "AvgWorkFromHomePercentage": 4.8723095526}},
        {{"State": "Delaware", "AvgWorkFromHomePercentage": 4.7831775701}},
        {{"State": "Florida", "AvgWorkFromHomePercentage": 6.1060009602}},
        {{"State": "Georgia", "AvgWorkFromHomePercentage": 4.7996927803}},
        {{"State": "Hawaii", "AvgWorkFromHomePercentage": 4.980952381}},
        {{"State": "Illinois", "AvgWorkFromHomePercentage": 4.3616051364}},
        {{"State": "Indiana", "AvgWorkFromHomePercentage": 3.3805444887}},
        {{"State": "Iowa", "AvgWorkFromHomePercentage": 4.7150668287}},
        {{"State": "Kansas", "AvgWorkFromHomePercentage": 4.2932806324}},
        {{"State": "Kentucky", "AvgWorkFromHomePercentage": 3.3713898917}},
        {{"State": "Louisiana", "AvgWorkFromHomePercentage": 2.8303730018}},
        {{"State": "Maine", "AvgWorkFromHomePercentage": 5.7005698006}},
        {{"State": "Maryland", "AvgWorkFromHomePercentage": 4.461299639}},
        {{"State": "Massachusetts", "AvgWorkFromHomePercentage": 4.9449453552}},
        {{"State": "Mississippi", "AvgWorkFromHomePercentage": 2.0687975647}},
        {{"State": "Missouri", "AvgWorkFromHomePercentage": 4.5036717063}},
        {{"State": "Nevada", "AvgWorkFromHomePercentage": 4.0932153392}},
        {{"State": "New Hampshire", "AvgWorkFromHomePercentage": 6.1482876712}},
        {{"State": "New Jersey", "AvgWorkFromHomePercentage": 4.1807518797}},
        {{"State": "New Mexico", "AvgWorkFromHomePercentage": 5.2076305221}},
        {{"State": "New York", "AvgWorkFromHomePercentage": 4.0684275691}},
        {{"State": "Ohio", "AvgWorkFromHomePercentage": 3.5995915589}},
        {{"State": "Oklahoma", "AvgWorkFromHomePercentage": 3.7250717703}},
        {{"State": "Oregon", "AvgWorkFromHomePercentage": 6.8875151515}},
        {{"State": "Pennsylvania", "AvgWorkFromHomePercentage": 4.2565842978}},
        {{"State": "Rhode Island", "AvgWorkFromHomePercentage": 3.945}},
        {{"State": "South Carolina", "AvgWorkFromHomePercentage": 3.7717171717}},
        {{"State": "Tennessee", "AvgWorkFromHomePercentage": 3.9021636241}},
        {{"State": "Texas", "AvgWorkFromHomePercentage": 4.2865620214}},
        {{"State": "Vermont", "AvgWorkFromHomePercentage": 7.4601092896}},
        {{"State": "Virginia", "AvgWorkFromHomePercentage": 4.6888}},
        {{"State": "Washington", "AvgWorkFromHomePercentage": 5.8076177285}},
        {{"State": "Wyoming", "AvgWorkFromHomePercentage": 4.9648854962}},
        {{"State": "Puerto Rico", "AvgWorkFromHomePercentage": 2.2007900677}},
        {{"State": "Idaho", "AvgWorkFromHomePercentage": 6.0047138047}},
        {{"State": "District of Columbia", "AvgWorkFromHomePercentage": 5.282122905}},
        {{"State": "North Dakota", "AvgWorkFromHomePercentage": 6.0541463415}},
        {{"State": "South Dakota", "AvgWorkFromHomePercentage": 6.8779279279}},
        {{"State": "West Virginia", "AvgWorkFromHomePercentage": 3.1214876033}},
    ]
    # Create dataframe
    df = pd.DataFrame(data)
    # Create bar plot
    chart = (
        alt.Chart(df).mark_bar().encode(
            x=alt.X('State', axis=alt.Axis(title='State', labelAngle=-45)),
            y=alt.Y('AvgWorkFromHomePercentage', axis=alt.Axis(title='Average Work From Home Percentage')),
            tooltip=['State', 'AvgWorkFromHomePercentage']
        ).properties(
            title='Average Work From Home Percentage by State'
        ).interactive()
    )
    # Save chart
    chart.save('average_work_from_home_percentage_by_state_bar_chart.json')
    
    image {{path}}

    ### Your Task:
    - Given the dataset: {data_json}
    - And the user's query: {user_input}
    Please summarize the data accordingly. If a graph is requested, generate the appropriate visualization and provide it as part of the response.
    """

    # Call LLM to get a refined response based on the dataset and user query
    result = llm.invoke(system_prompt)
    return result.content.strip()

def large_data(data:pd.DataFrame, user_input, llm, filename='data.json', rows=10):
    """
    Handle data processing and visualization based on user input
    Returns: tuple (summary_text, chart) where chart is None if no visualization was created
    """
    logger.info("Function large_data")
    # Step 1: Preprocess the data
    preprocessed_data = preprocess_data(data)
    # Step 2: Save full preprocessed data to a JSON file
    json_filename = save_json(preprocessed_data, filename)
    # Step 3: Get a preview (head) of the dataset for metadata
    data_preview = get_head(preprocessed_data, rows)

    system_prompt = f"""
    You are an expert data analysis assistant tasked with analyzing the dataset provided in JSON format and summarizing it based on the user's query. When responding to user queries, please provide a clear and concise summary of the relevant data. Include necessary details to make the response informative, but avoid unnecessary context about the dataset itself (such as dataset preprocessing or filtering). For example, if the query asks for students registered in a course, the response should directly focus on the result (e.g., the list of student names) with a brief, informative sentence. Do not mention dataset characteristics unless directly requested by the user.

    ### Instructions:
    - Instead of receiving the entire dataset, you will be provided with:
    1. The **filename** of the saved JSON data file.
    2. A **preview (head)** of the dataset, which contains the first few rows to help you understand the structure.
    - Use this preview to analyze the structure and generate the necessary code to process the full dataset stored in the given file.
    - You must analyze the data by **reading it from the provided file**, ensuring your insights are based on the complete dataset.
    3. The dataset you receive has already been preprocessed to directly align with the user's query, so it contains only the relevant information.
    4. Your job is to:
    - Directly address the user's query using the provided dataset.
    - Provide insights, trends, or patterns based on the data.
    - If the dataset has already been filtered or processed, acknowledge that and focus only on summarizing it in response to the query.
    5. If the dataset does not contain enough information to fully answer the user's query, then you have to respond on the provided data as the provided data is the only relevent data that user asked for. and don't mention the limitation and suggest how the query could be modified for more complete results.
    6. If the user requests a graph or visualization, create an appropriate graph (e.g., bar chart, line graph, pie chart) based on the query and the data. Ensure the graph visually represents the information in a clear and understandable way. Provide the graph as part of the response.
    7. For the Graph, you have to write the code, run it in your environment and save the graph and the show the the saved graph to user.

    ### Your Responsibilities:
    1. **Understanding the Dataset:**
    - Use the head of the dataset to infer structure (columns, data types).
    - Write appropriate code to analyze the full dataset by reading it from the provided file and Generate accurate summaries and responses based on the user's query.
    
    2. **Generating Insights:**
    - Directly address the user's query by processing the data accordingly.
    - If requested, generate visualizations using appropriate Python libraries(altair).
    - Provide concise yet informative summaries based on the retrieved data.
    
    3. **Handling Missing Information:**
    - If the provided preview doesn't contain enough information to answer the query, don't panic.
    - The dataset you have is already filtered on the base of the user query, e.g. user query is "Visualize the racial distribution (White, Black, Asian, Hispanic) for New York." and the data you have just contain the columns of the (White, Black, Asian, Hispanic). It means it's just the data of the New York. You don't need the city column for filtering.

    4. **Visualization Instructions:**
    - When requested, generate visualization code that reads the data from the JSON file and creates appropriate graphs (bar charts, line graphs, etc.).
    - Ensure the generated code is executable, and the chart is saved for display.

    ### Examples:
    **Example 1: Create a bar chart for the racial distribution in New York.**
    - Filename: `data.json`
    - Preview:
    ```json
        [{{"White":55.9,"Black":4.0,"Native":1.0,"Asian":14.9,"Pacific":0.9,"Hispanic":17.0}},{{"White":95.5,"Black":0.1,"Native":0.6,"Asian":0.4,"Pacific":0.0,"Hispanic":2.2}},{{"White":1.2,"Black":27.2,"Native":1.3,"Asian":0.0,"Pacific":0.0,"Hispanic":69.4}},{{"White":0.7,"Black":16.1,"Native":0.0,"Asian":1.0,"Pacific":0.0,"Hispanic":78.8}},{{"White":13.8,"Black":74.8,"Native":0.0,"Asian":0.4,"Pacific":0.0,"Hispanic":6.5}},{{"White":71.3,"Black":0.9,"Native":0.0,"Asian":19.5,"Pacific":0.0,"Hispanic":2.8}},{{"White":2.2,"Black":92.0,"Native":0.0,"Asian":0.0,"Pacific":0.0,"Hispanic":4.7}},{{"White":75.2,"Black":6.1,"Native":0.0,"Asian":0.0,"Pacific":0.0,"Hispanic":18.6}},{{"White":95.2,"Black":1.6,"Native":0.0,"Asian":0.3,"Pacific":0.0,"Hispanic":1.4}},{{"White":8.4,"Black":82.1,"Native":0.3,"Asian":0.0,"Pacific":0.0,"Hispanic":7.3}}]
        ```
    *System Hint*: The dataset already provides the distribution of recial in New York, so there's no need to filter it further. Just use it to response.
    Response: "The racial distribution across different areas of New York is as follows:
        White Population:
            Highest: 95.5%
            Lowest: 0.7%
            Observed in areas with significant variations.
        Black Population:
            Highest: 92.0%
            Lowest: 0.1%
            Some areas have a predominantly Black population, while others have minimal representation.
        Native Population:
            Highest: 1.3%
            Lowest: 0.0%
            The Native population remains relatively low across all areas.
        Asian Population:
            Highest: 19.5%
            Lowest: 0.0%
            Certain areas have a notable Asian presence, particularly up to 19.5%.
        Pacific Population:
            Highest: 0.9%
            Lowest: 0.0%
            The Pacific Islander population is minimal across the dataset.
        Hispanic Population:
            Highest: 78.8%
            Lowest: 1.4%
            Some areas show a very high concentration of Hispanic residents.
    Here's the bar graph for visulization
    Python Code:
    import pandas as pd
    import altair as alt

    # Load the JSON file
    df = pd.read_json('data.json')

    # Transpose the DataFrame.
    df = df.T

    # Calculate the average of each column.
    df_mean = pd.DataFrame(df.mean(axis=1), columns=['Mean'])

    # Add a column `Race` to `df_mean`.
    df_mean['Race'] = df_mean.index

    # Create a bar plot using columns `Race` and `Mean` from `df_mean`.
    chart = alt.Chart(df_mean).mark_bar().encode(
        x='Race',
        y='Mean',
        tooltip=['Race', 'Mean']
    ).properties(
        title='Racial Distribution in New York'
    )

    # Save the chart
    chart.save('racial_distribution_new_york.json')"

    **Example 2: List the counties where the unemployment rate is greater than 10%.**
    - Filename: `data.json`
    - Preview: 
    ```json
        [{{"County":"Trimble County"}},{{"County":"Lea County"}},{{"County":"Walla Walla County"}},{{"County":"Modoc County"}},{{"County":"Catahoula Parish"}},{{"County":"Banks County"}},{{"County":"Stokes County"}},{{"County":"Buckingham County"}},{{"County":"Ziebach County"}},{{"County":"Gladwin County"}}]
    ```
    *System Hint*: The dataset only includes counties where the unemployment rate exceeds 10%. The data has already been filtered, so there is no need to check the unemployment rate further. Just list the counties.
    Response: "The counties with an unemployment rate greater than 10% are: 
        Trimble County
        Lea County
        Walla Walla County
        Modoc County
        Catahoula Parish
        Banks County
        Stokes County
        Buckingham County
        Ziebach County
        Gladwin County
    Here is a visualization representing the listed counties.
    Python Code:
    import pandas as pd
    import altair as alt

    # Load the JSON file
    df = pd.read_json('data.json')

    # Calculate the frequency of each county.
    df['Frequency'] = df.groupby('County')['County'].transform('count')

    # Create the bar plot.
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('County', axis=alt.Axis(title='County', labelAngle=-45)),
        y=alt.Y('Frequency', axis=alt.Axis(title='Frequency')),
        tooltip=['County', 'Frequency']
    ).properties(
        title='Frequency of Counties'
    )

    # Save the chart.
    chart.save('frequency_of_counties.json')"

    **Example 3: Create a bar chart showing the average income per capita for different states.**
    - Filename: `data.json`
    - Preview: 
    ```json
        [{{"State":"Vermont","AvgIncomePerCap":31682.6612021858}},{{"State":"Virginia","AvgIncomePerCap":36048.3336886993}},{{"State":"Washington","AvgIncomePerCap":34461.5671745153}},{{"State":"Wyoming","AvgIncomePerCap":30673.8015267176}},{{"State":"Puerto Rico","AvgIncomePerCap":12043.7898305085}},{{"State":"Idaho","AvgIncomePerCap":25361.5838926174}},{{"State":"District of Columbia","AvgIncomePerCap":49815.7486033519}},{{"State":"North Dakota","AvgIncomePerCap":33524.8634146341}},{{"State":"South Dakota","AvgIncomePerCap":28359.0045045045}},{{"State":"West Virginia","AvgIncomePerCap":24130.3719008264}}]
    ```
    *System Hint*: The dataset contains information about the average income per capita for various states and territories. This data can be used to analyze and compare economic conditions across different regions. Use it to create visualizations, such as bar charts, to better understand income disparities and trends across states.
    Response: ""The average income per capita for the listed states and territories is as follows:
        Vermont: $31,682
        Virginia: $36,048
        Washington: $34,461
        Wyoming: $30,673
        Puerto Rico: $12,043
        Idaho: $25,361
        District of Columbia: $49,815
        North Dakota: $33,524
        South Dakota: $28,359
        West Virginia: $24,130
    Among the listed states, the District of Columbia has the highest average income per capita at $49,815, whereas Puerto Rico has the lowest at $12,043. Most states fall within the range of $25,000 to $36,000.
    Here is a bar chart visualization representing the average income per capita across different states for better comparison and analysis.
    Python Code:
    import pandas as pd
    import altair as alt

    # Load the JSON file
    df = pd.read_json('data.json')

    # Create a bar chart with `State` on the x-axis and `AvgIncomePerCap` on the y-axis.
    chart = alt.Chart(df).mark_bar().encode(
        x='State',
        y='AvgIncomePerCap',
        tooltip=['State', 'AvgIncomePerCap']
    ).properties(
        title='Average Income per Capita by State'
    )

    # Save the chart
    chart.save('average_income_per_capita_by_state_bar_chart.json')"

    **Example 4: Calculate the percentage of people working from home in each state and create a bar chart of the result.
    - Filename: `data.json`
    - Preview: 
    ```json
        [{{"State":"Vermont","AvgWorkFromHomePercentage":7.4601092896}},{{"State":"Virginia","AvgWorkFromHomePercentage":4.6888}},{{"State":"Washington","AvgWorkFromHomePercentage":5.8076177285}},{{"State":"Wyoming","AvgWorkFromHomePercentage":4.9648854962}},{{"State":"Puerto Rico","AvgWorkFromHomePercentage":2.2007900677}},{{"State":"Idaho","AvgWorkFromHomePercentage":6.0047138047}},{{"State":"District of Columbia","AvgWorkFromHomePercentage":5.282122905}},{{"State":"North Dakota","AvgWorkFromHomePercentage":6.0541463415}},{{"State":"South Dakota","AvgWorkFromHomePercentage":6.8779279279}},{{"State":"West Virginia","AvgWorkFromHomePercentage":3.1214876033}}]
    ```
    *System Hint*: The dataset provides the average workathome percentage of the states, use this data to create the bar graph.
    Response:
    "The states with the highest average work-from-home percentages are Montana (7.87%), Vermont (7.46%), and Colorado (7.41%), while the states with the lowest percentages are Mississippi (2.07%), Puerto Rico (2.20%), and Louisiana (2.83%).
    The average work-from-home percentages for different states are as follows:
    Montana: 7.87%
    Vermont: 7.46%
    Colorado: 7.41%
    South Dakota: 6.88%
    Oregon: 6.89%
    Arizona: 6.17%
    New Hampshire: 6.15%
    Florida: 6.11%
    North Dakota: 6.05%
    Idaho: 6.00%
    Washington: 5.81%
    Utah: 5.89%
    Maine: 5.70%
    California: 5.72%
    Minnesota: 5.51%
    Alaska: 5.41%
    New Mexico: 5.21%
    District of Columbia: 5.28%
    North Carolina: 5.01%
    Hawaii: 4.98%
    Wisconsin: 4.36%
    Illinois: 4.36%
    Texas: 4.29%
    Kansas: 4.29%
    Missouri: 4.50%
    Maryland: 4.46%
    Pennsylvania: 4.26%
    Nevada: 4.09%
    New Jersey: 4.18%
    New York: 4.07%
    Rhode Island: 3.95%
    Tennessee: 3.90%
    South Carolina: 3.77%
    Oklahoma: 3.73%
    Ohio: 3.60%
    Michigan: 3.84%
    Indiana: 3.38%
    Kentucky: 3.37%
    Arkansas: 3.22%
    West Virginia: 3.12%
    Alabama: 2.85%
    Louisiana: 2.83%
    Puerto Rico: 2.20%
    Mississippi: 2.07%

    Python Code: 
    import pandas as pd
    import altair as alt
    # Load the JSON file
    df = pd.read_json('data.json')
    # Create bar plot
    chart = (
        alt.Chart(df).mark_bar().encode(
            x=alt.X('State', axis=alt.Axis(title='State', labelAngle=-45)),
            y=alt.Y('AvgWorkFromHomePercentage', axis=alt.Axis(title='Average Work From Home Percentage')),
            tooltip=['State', 'AvgWorkFromHomePercentage']
        ).properties(
            title='Average Work From Home Percentage by State'
        ).interactive()
    )
    # Save chart
    chart.save('average_work_from_home_percentage_by_state_bar_chart.json')
    
    image {{path}}

    ### Your Task:
    - Given the dataset filename: {json_filename}
    - And the preview of the dataset: {data_preview}
    - And the user's query: {user_input}
    Please summarize the data accordingly. If a graph is requested, generate the appropriate visualization and provide it as part of the response.
    """

    # Call LLM to get a refined response based on the dataset and user query
    result = llm.invoke(system_prompt)
    return result.content.strip()

def process_llm_response(response_text, data):
    logger.info("Function process_llm_response")
    # Step 2: Define patterns for code and file references (PNG, HTML)
    code_pattern = r'```python(.*?)```'
    nocode_pattern = r'```(.*?)```'
    pattern = r'Python Code:\s*'
    png_pattern = r'\b\w+\.png\b'
    html_pattern = r'\b\w+\.html\b'
    json_pattern = r'\b\w+\.json\b'

    # Step 3: Extract the Python code from the response (if present)
    code_match = re.search(code_pattern, response_text, re.DOTALL)

    # Step 4: Clean the response by removing file references (PNG, HTML) from the text
    response_text = re.sub(code_pattern, '', response_text, flags=re.DOTALL).strip()
    response_text = re.sub(png_pattern, '', response_text).strip()
    response_text = re.sub(pattern, '', response_text).strip()
    response_text = re.sub(html_pattern, '', response_text).strip()
    response_text = re.sub(json_pattern, '', response_text).strip()
    response_text = re.sub(nocode_pattern, '', response_text).strip()

    # Initialize the chart variable to None
    chart = None

    # Step 5: If Python code exists, try executing it safely
    if code_match:
        try:
            logger.info("Python Code found in the response.")
            # Extract the Python code from the match and clean it
            code = code_match.group(1).strip()
            # Define local variables for execution
            local_vars = {'pd': pd, 'alt': alt, 'data': data}  # Assuming 'data' is provided elsewhere
            exec(code, local_vars)
            
            # Check if a chart object was created
            if 'chart' in local_vars:
                logger.info("Chart found in the local_vars.")
                chart = local_vars['chart']
                logger.critical("TERMINATED")
            
            # Clean the response by removing any remaining file references after code execution
            response_text = re.sub(code_pattern, '', response_text, flags=re.DOTALL).strip()
            response_text = re.sub(png_pattern, '', response_text).strip()
            response_text = re.sub(html_pattern, '', response_text).strip()
            response_text = re.sub(json_pattern, '', response_text).strip()
            response_text = re.sub(nocode_pattern, '', response_text).strip()
            response_text = re.sub(pattern, '', response_text).strip()

        except Exception as e:
            logger.error("Failed to Generate Chart.")
            # Append error message if code execution fails
            response_text += f"\nError generating visualization: {str(e)}"
            logger.critical("TERMINATED")

    # Return the cleaned response text and the chart (if generated)
    return response_text, chart

def data_handle(data, user_input, llm, filename='data.json', rows=10):
    data_rows = len(data)

    if data_rows > 100:
        logger.info("Function called for large dataset")
        response = large_data(data, user_input, llm, filename='data.json', rows=10)

        response_text, chart = process_llm_response(response, data)

        return data_rows, response_text, chart
    
    elif data_rows <= 100:
        logger.info("Function Called for short dataset")
        response = short_data(data, user_input, llm)

        response_text, chart = process_llm_response(response, data)

        return data_rows, response_text, chart

def data_handler(data: pd.DataFrame, user_input, llm, filename='data.json', rows=10):
    """
    Handle data processing and visualization based on user input
    Returns: tuple (summary_text, chart) where chart is None if no visualization was created
    """
    # Step 1: Preprocess the data
    preprocessed_data = preprocess_data(data)

    # Step 2: Save full preprocessed data to a JSON file
    json_filename = save_json(preprocessed_data, filename)

    # Step 3: Get a preview (head) of the dataset for metadata
    data_preview = get_head(preprocessed_data, rows)

    improved_prompt = f"""
    You are an expert data analysis assistant tasked with analyzing the dataset provided in JSON format and summarizing it based on the user's query. When responding to user queries, please provide a clear and concise summary of the relevant data. Include necessary details to make the response informative, but avoid unnecessary context about the dataset itself (such as dataset preprocessing or filtering). For example, if the query asks for students registered in a course, the response should directly focus on the result (e.g., the list of student names) with a brief, informative sentence. Do not mention dataset characteristics unless directly requested by the user.

    ### Instructions:
    - Instead of receiving the entire dataset, you will be provided with:
    1. The **filename** of the saved JSON data file.
    2. A **preview (head)** of the dataset, which contains the first few rows to help you understand the structure.
    - Use this preview to analyze the structure and generate the necessary code to process the full dataset stored in the given file.
    - You must analyze the data by **reading it from the provided file**, ensuring your insights are based on the complete dataset.
    3. The dataset you receive has already been preprocessed to directly align with the user's query, so it contains only the relevant information.
    4. Your job is to:
    - Directly address the user's query using the provided dataset.
    - Provide insights, trends, or patterns based on the data.
    - If the dataset has already been filtered or processed, acknowledge that and focus only on summarizing it in response to the query.
    5. If the dataset does not contain enough information to fully answer the user's query, then you have to respond on the provided data as the provided data is the only relevent data that user asked for. and don't mention the limitation and suggest how the query could be modified for more complete results.
    6. If the user requests a graph or visualization, create an appropriate graph (e.g., bar chart, line graph, pie chart) based on the query and the data. Ensure the graph visually represents the information in a clear and understandable way. Provide the graph as part of the response.
    7. For the Graph, you have to write the code, run it in your environment and save the graph and the show the the saved graph to user.

    ### Your Responsibilities:
    1. **Understanding the Dataset:**
    - Use the head of the dataset to infer structure (columns, data types).
    - Write appropriate code to analyze the full dataset by reading it from the provided file and Generate accurate summaries and responses based on the user's query.
    
    2. **Generating Insights:**
    - Directly address the user's query by processing the data accordingly.
    - If requested, generate visualizations using appropriate Python libraries(altair).
    - Provide concise yet informative summaries based on the retrieved data.
    
    3. **Handling Missing Information:**
    - If the provided preview doesn't contain enough information to answer the query, don't panic.
    - The dataset you have is already filtered on the base of the user query, e.g. user query is "Visualize the racial distribution (White, Black, Asian, Hispanic) for New York." and the data you have just contain the columns of the (White, Black, Asian, Hispanic). It means it's just the data of the New York. You don't need the city column for filtering.

    4. **Visualization Instructions:**
    - When requested, generate visualization code that reads the data from the JSON file and creates appropriate graphs (bar charts, line graphs, etc.).
    - Ensure the generated code is executable, and the chart is saved for display.
    
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

    **Example 1: Find the total revenue from the dataset.**
    - Filename: `sales_data.json`
    - Preview: 
    ```json
    [
        {{"Product": "Laptop", "Price": 1000, "Quantity": 5}},
        {{"Product": "Phone", "Price": 500, "Quantity": 10}}
    ]
    ```

    *System Hint*: The dataset contains columns for Product, Price, and Quantity. The total revenue can be calculated as the sum of `Price * Quantity`.

    Response: 
    "The total revenue in the dataset is 2000, in which the revenue genereated from laptop is the highest."

    **Python Code:**
    ```python
    import pandas as pd

    # Load the JSON file
    df = pd.read_json('sales_data.json')

    # Calculate total revenue
    total_revenue = (df['Price'] * df['Quantity']).sum()
    print("Total Revenue:", total_revenue)
    ```

    --- 

    **Example 2: Create a bar chart for the racial distribution in New York.**
    - Filename: `demographics.json`
    - Preview:
    ```json
    [
        {{"City": "New York", "White": 500000, "Black": 200000, "Asian": 150000, "Hispanic": 250000}},
        {{"City": "Los Angeles", "White": 600000, "Black": 300000, "Asian": 200000, "Hispanic": 350000}}
    ]
    ```

    *System Hint*: The dataset provides racial distribution per city. Focus only on New York and create a bar chart.

    Response: 
    "The racial distribution in New York is as follows: 500,000 White, 200,000 Black, 150,000 Asian, and 250,000 Hispanic. Here is the visualization:"

    **Python Code:**
    ```python
    import pandas as pd
    import altair as alt

    # Load the JSON file
    df = pd.read_json('demographics.json')

    # Filter data for New York
    ny_data = df[df['City'] == 'New York']

    # Prepare data for visualization
    melted_df = ny_data.melt(id_vars=['City'], var_name='Race', value_name='Population')

    # Create bar chart
    chart = alt.Chart(melted_df).mark_bar().encode(
        x='Race',
        y='Population',
        color='Race'
    ).properties(title='Racial Distribution in New York').interactive()

    # Save the chart
    chart.save('ny_racial_distribution.json')
    ```

    --- 

    **Example 3: Show the top 5 products by sales volume.**
    - Filename: `ecommerce_sales.json`
    - Preview:
    ```json
    [
        {{"Product": "TV", "Units_Sold": 100}},
        {{"Product": "Laptop", "Units_Sold": 200}},
        {{"Product": "Phone", "Units_Sold": 300}}
    ]
    ```

    *System Hint*: The dataset contains `Product` and `Units_Sold` columns. Find the top 5 products by sorting the data.

    Response:  
    "Here are the top 5 products by sales volume: Phone, Laptop, TV."

    **Python Code:**
    ```python
    import pandas as pd

    # Load the JSON file
    df = pd.read_json('ecommerce_sales.json')

    # Get top 5 products by units sold
    top_products = df.sort_values(by='Units_Sold', ascending=False).head(5)

    print("Top 5 Products by Sales Volume:")
    print(top_products[['Product', 'Units_Sold']])
    ```

    ### Your Task:
    - Given the dataset filename: {json_filename}
    - And the preview of the dataset: {data_preview}
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

# Example usage
if __name__ == "__main__":
    user_query = "Find the top 5 counties with the highest percentage of self-employed individuals and show their average income."
    
    try:
        logger.info("Generating SQL query...")
        initial_response = generate_initial_response(user_query, llm, vector_store, k=1)
        logger.info("Initial Response from LLM:")
        # logger.info(initial_response)

        if re.search(r"cannot generate.*SQL", initial_response, re.IGNORECASE):
            logger.info("Fallback triggered.")
            fallback_response = trigger_fallback_logic(user_query, llm, "", HumanMessage(content=user_query))
            logger.info("Fallback Response:")
            logger.info(fallback_response)
        else:
            refined_response = refine_response(initial_response)
            logger.info("Refined SQL Query:")
            logger.info(refined_response)

            data = get_data(bq_manager, refined_response)
            logger.info("Data retrieved from BigQuery:")
            # logger.info(data.head())

            if isinstance(data, pd.DataFrame) and not data.empty:
                filename = save_json(data, 'data.json')
                logger.info(f"Data saved to {filename}")
                data_preview = preprocess_data(data)
                logger.info("Data Preview Sent to LLM:")
                logger.info(data_preview)

                summary, chart = data_handle(data, user_query, llm, filename='data.json', rows=10)
                logger.info("Data Summary:")
                logger.info(summary)

                if chart:
                    chart.save('chart_output.png')
                    logger.info("Chart saved as chart_output.png")
            else:
                logger.warning("No relevant data found.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
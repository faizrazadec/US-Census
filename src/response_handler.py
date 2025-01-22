"""
This module contains functions for generating SQL queries or refining user input in 
natural language based on schema information stored in a Chroma vector store. The module 
uses a large language model (LLM) to process the user input and interact with the schema 
context, providing either a valid SQL query or refined natural language prompts if the 
query cannot be processed.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from src.system_prompt import SYSTEM_PROMPT
from src.logger import setup_logger

# Get the configured logger
logger = setup_logger()


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
        system_message = SystemMessage(
            content=f"{SYSTEM_PROMPT}\nSchema Context:\n{context}"
        )
        human_message = HumanMessage(content=user_input)

        # Generate initial response
        response = llm.invoke([system_message, human_message])

        # Debugging output
        # print("Initial Response generated:")
        # print(response.content.strip())
        logger.info(response.content.strip())

        return response.content.strip()
    except Exception as e:
        logger.error("Error generating response...")
        print(f"Error generating response: {e}")
        return (
            "An error occurred while processing your request. Please try again later."
        )


def trigger_fallback_logic(user_input, llm, context, human_message):
    """Trigger the fallback logic when the initial response cannot generate a SQL query."""
    try:
        logger.info("Fallback Logic triggered.")
        # print("Triggering fallback logic")

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
        logger.info("Improved Prompts Generated.")
        # print("Refined Response generated:")
        # print(refined_response.content.strip())

        # Return refined response (this means no further processing or BigQuery execution)
        logger.critical("TERMINATED")
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

        if (
            "I cannot generate a SQL query for this request based on the provided schema."
            in response
        ):
            # If the response indicates fallback is needed, trigger fallback logic
            # print("Fallback triggered.")
            logger.info("Fallback triggered")
            # Retrieve schema context from ChromaDB again to pass to the fallback logic
            results = vector_store.similarity_search(user_input, k=k)
            flattened_context = [item.page_content for item in results]
            context = "\n".join(flattened_context)
            # Call the fallback logic
            return trigger_fallback_logic(
                user_input, llm, context, HumanMessage(content=user_input)
            )

        logger.critical("TERMINATED")
        # Return the initial response if successful (i.e., SQL query generation)
        return response

    except Exception as e:
        print(f"Error in get_response: {e}")
        logger.error("An error occurred while processing your request")
        logger.critical("TERMINATED")
        return (
            "An error occurred while processing your request. Please try again later."
        )

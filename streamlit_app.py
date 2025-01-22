"""Streamlit App"""

import asyncio
import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage
from src.components import initialize_components
from src.response_handler import generate_initial_response, trigger_fallback_logic
from src.data_handler import refine_response, get_data, data_handle

async def main():
    # Configure the page
    st.set_page_config(page_title="DIVS", page_icon="🔍", layout="wide")

    # Application title and description
    st.title("Intelligent Data Insights and Visualization System")
    st.write(
        """Please enter your query in natural language,
        and I will assist you in generating the appropriate
        SQL query to retrieve the data and provide meaningful visualizations."""
    )

    # Add custom CSS for chart styling
    st.markdown(
        """
        <style>
        .stChart {
            margin: auto;
            width: 80% !important;
        }
        .chart-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 2rem auto;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Initialize components
    try:
        if (
            "llm" not in st.session_state
            or "vector_store" not in st.session_state
            or "bq_manager" not in st.session_state
        ):
            llm, vector_store, bq_manager = await initialize_components()
            st.session_state.llm = llm
            st.session_state.vector_store = vector_store
            st.session_state.bq_manager = bq_manager
        else:
            llm = st.session_state.llm
            vector_store = st.session_state.vector_store
            bq_manager = st.session_state.bq_manager
    except Exception as e:
        st.error(f"Failed to initialize components. Error: {e}")
        return

    # Initialize progress bar
    # if "progress" not in st.session_state:
    #     st.session_state.progress = 0

    # User input
    user_query = st.text_area(
        "Enter your query:",
        height=68,
        placeholder="Example: Provide me the names of the students that have paid their challan form and are in semester Fall 2025",
    )

    # Generate SQL Query Button
    if st.button("Submit"):
        if user_query:
            # Create placeholder for results
            with st.container():
                with st.spinner("Processing your query... Please wait."):
                    try:
                        # progress_bar = st.progress(st.session_state.progress)
                        # Step 1: Get initial response from LLM
                        initial_response = generate_initial_response(
                            user_query, llm, vector_store, k=5
                        )
                        # st.write("Initial Response from LLM:")
                        # st.write(initial_response)
                        # st.session_state.progress += 20 # Increase the progress by 10%
                        # progress_bar.progress(st.session_state.progress) #update the progress bar

                        # Step 2: Check if initial response indicates fallback is needed
                        if (
                            """I cannot generate a SQL query
                            for this request based on the provided schema."""
                            in initial_response
                        ):
                            # st.write("Fallback response generated.")
                            fallback_response = trigger_fallback_logic(
                                user_query, llm, "", HumanMessage(content=user_query)
                            )
                            # st.session_state.progress += 80 # Increase the progress by 10%
                            # progress_bar.progress(st.session_state.progress)
                            # st.write("Fallback Response:")
                            st.write(fallback_response)
                        else:
                            # Step 3: Refine the response to remove backticks if any
                            refined_response = refine_response(initial_response)
                            # st.write("Refined Response:")
                            # st.write(refined_response)
                            # st.session_state.progress += 20 # Increase the progress by 10%
                            # progress_bar.progress(st.session_state.progress)

                            # Step 4: Get data from BigQuery
                            data = get_data(bq_manager, refined_response)
                            # st.write("Data retrieved from BigQuery:")
                            # st.write(data)
                            # st.session_state.progress += 20 # Increase the progress by 10%
                            # progress_bar.progress(st.session_state.progress)

                            # Step 5: Handle and summarize the data
                            if isinstance(data, pd.DataFrame) and not data.empty:
                                summary_text, chart = data_handle(
                                    data, user_query, llm, filename="data.json", rows=10
                                )
                                # st.write(data_rows)
                                if summary_text:
                                    # st.write(preprocessed_data)

                                    st.write("Data Summary:")
                                    with st.expander(
                                        "Click to view the Data Summary", expanded=True
                                    ):
                                        st.write(summary_text)
                                        # st.session_state.progress += 20 
                                        # progress_bar.progress(st.session_state.progress)
                                    # Display the chart if one was generated
                                    if chart is not None:
                                        # Create columns for centered layout
                                        col1, col2, col3 = st.columns([1, 2, 1])
                                        with col2:
                                            st.altair_chart(
                                                chart, use_container_width=True
                                            )
                                else:
                                    st.write("No relevant data found.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
        else:
            st.write("Please enter a query.")

    # Sidebar with additional information
    with st.sidebar:
        st.header("About")
        st.write(
            """
        This application uses LangChain and Google's Gemini model to:
        1. Understand your natural language query
        2. Search relevant schema context
        3. Generate appropriate SQL queries
        """
        )

        st.header("Tips")
        st.write(
            """
        - Be specific about the data you want to retrieve.
        - Include relevant time periods or conditions.
        - Mention specific table names if you know them.
        - Use clear and concise language.
        """
        )


if __name__ == "__main__":
    asyncio.run(main())

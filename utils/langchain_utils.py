import warnings
import duckdb_engine

warnings.filterwarnings("ignore", category=duckdb_engine.DuckDBEngineWarning)

# import streamlit as st # Removed to fix circular import
# import pandas as pd # Removed as data formatting for prompt should be done outside this module
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
# LangChain also has memory components, but for a direct SQL chain,
# memory is often handled in the main application logic or by specific agent types.
# We will adapt the main.py logic to pass history if needed by the chain.

# Helper function to clean the generated SQL string
def clean_sql_query(query: str) -> str:
    # Remove common markdown code fences (```sql, ```) and leading/trailing whitespace
    cleaned_query = query.strip()
    if cleaned_query.startswith('```sql'):
        cleaned_query = cleaned_query[len('```sql'):].strip()
    elif cleaned_query.startswith('```'):
         cleaned_query = cleaned_query[len('```'):].strip()

    if cleaned_query.endswith('```'):
        cleaned_query = cleaned_query[:-len('```')].strip()

    # Also remove leading 'sql ' if it's still there after removing fences
    if cleaned_query.lower().startswith("sql"):
        cleaned_query = cleaned_query[3:].strip()

    return cleaned_query

# Helper function to format chat history for the prompt
def format_chat_history_for_prompt(chat_history: list) -> str:
    """Formats relevant parts of chat history, including feedback, for the prompt."""
    if not chat_history:
        return "No chat history available."

    formatted_history = "Recent Chat History (with Corrections and Column Usage):\n"
    # Iterate through the last few messages (e.g., last 5-10 turns)
    # To avoid making the prompt too long
    # Only include user questions and assistant responses with SQL/feedback
    relevant_history = []
    for entry in reversed(chat_history): # Iterate in reverse to get recent turns
        if len(relevant_history) >= 10: # Limit to last 10 relevant entries
            break
        if entry["role"] == "user":
            relevant_history.append(entry)
        elif entry["role"] == "assistant" and isinstance(entry["content"], dict):
            # Always include entries with feedback or corrections
            if "feedback" in entry["content"] or "sql_query" in entry["content"]:
                relevant_history.append(entry)

    # Now format the relevant history in chronological order
    for entry in reversed(relevant_history):
        role = entry["role"]
        content = entry["content"]

        if role == "user":
            formatted_history += f"User: {content}\n"
        elif role == "assistant" and isinstance(content, dict):
            formatted_history += "Assistant:\n"
            if "sql_query" in content:
                formatted_history += f"  Generated SQL: {content['sql_query']}\n"
                if "columns_used" in content and content["columns_used"]:
                    formatted_history += f"  Columns Used: {', '.join(content['columns_used'])}\n"
                if "column_mapping" in content and content["column_mapping"]:
                    formatted_history += f"  Column Mapping: {', '.join(f'{col} ({count})' for col, count in content['column_mapping'].items())}\n"
            if "feedback" in content:
                if content["feedback"] == "corrected":
                    formatted_history += f"  ❌ INCORRECT QUERY - User provided correction:\n"
                    if "corrected_query" in content:
                        formatted_history += f"  ✅ CORRECTED SQL: {content['corrected_query']}\n"
                        # Extract and show columns from corrected query
                        if "corrected_columns" in content and content["corrected_columns"]:
                            formatted_history += f"  ✅ CORRECTED Columns: {', '.join(content['corrected_columns'])}\n"
                        # Add explicit mapping of incorrect to correct columns
                        if "columns_used" in content and content["columns_used"]:
                            formatted_history += f"  ❌ INCORRECT Columns Used: {', '.join(content['columns_used'])}\n"
                            formatted_history += f"  🔄 Column Mapping Correction:\n"
                            # Map incorrect columns to correct ones
                            incorrect_cols = set(content["columns_used"])
                            correct_cols = set(content.get("corrected_columns", []))
                            for col in incorrect_cols:
                                if col not in correct_cols:
                                    formatted_history += f"    - '{col}' was incorrect, should not be used\n"
                            for col in correct_cols:
                                if col not in incorrect_cols:
                                    formatted_history += f"    - '{col}' is the correct column to use\n"
                elif content["feedback"] == "incorrect":
                    formatted_history += f"  ❌ INCORRECT QUERY - User marked as incorrect\n"
                    if "columns_used" in content and content["columns_used"]:
                        formatted_history += f"  ❌ INCORRECT Columns Used: {', '.join(content['columns_used'])}\n"
                        formatted_history += f"  ⚠️ These columns should be avoided for similar questions\n"
                elif content["feedback"] == "correct":
                    formatted_history += f"  ✅ CORRECT QUERY - User confirmed\n"
                    if "columns_used" in content and content["columns_used"]:
                        formatted_history += f"  ✅ CORRECT Columns Used: {', '.join(content['columns_used'])}\n"
                        formatted_history += f"  💡 These columns are correct for similar questions\n"
            formatted_history += "\n" # Add a newline after assistant turn

    formatted_history += "---\n" # Separator
    formatted_history += "IMPORTANT INSTRUCTIONS:\n"
    formatted_history += "1. If the current question is similar to any previous question that was corrected, you MUST use the corrected SQL query or a logically equivalent version.\n"
    formatted_history += "2. Pay special attention to the columns used in corrected queries - these are the correct columns to use for similar questions.\n"
    formatted_history += "3. If you see a pattern of certain columns being used correctly in multiple queries, prefer those columns for similar questions.\n"
    formatted_history += "4. If a previous query was marked as incorrect, DO NOT use the same columns or approach for similar questions.\n"
    formatted_history += "5. For each new question, first check if it's similar to any previous questions in the chat history.\n"
    formatted_history += "6. If you find a similar question, use its corrected query or successful approach.\n"
    formatted_history += "7. If you find a similar question that was marked as incorrect, avoid using its approach.\n"
    formatted_history += "8. Always prioritize using columns that were marked as correct in previous queries.\n"
    formatted_history += "9. If you see a pattern of certain columns being used incorrectly, avoid using them.\n"
    formatted_history += "10. When in doubt, prefer columns that have been frequently used in correct queries.\n"
    formatted_history += "11. If you see a correction that maps one column to another (e.g., 'developer_organization_name' was incorrect, 'developer_website' is correct), you MUST use the correct column.\n"
    formatted_history += "12. If a correction shows that certain columns should not be used, you MUST avoid using those columns in similar questions.\n"
    formatted_history += "13. Pay special attention to column mapping corrections - they show exactly which columns to use and which to avoid.\n"
    formatted_history += "14. If a correction shows that a specific value (like 'TRUST') should be used with a specific column, you MUST use that exact combination.\n"
    formatted_history += "15. When you see a correction, treat it as a rule that must be followed for all similar questions.\n"
    formatted_history += "16. **CRITICAL:** The only available tables are: ytflow_api_tickets, external_aggregated_api_usage. DO NOT use any other table names.\n"
    formatted_history += "17. **CRITICAL:** For quota-related queries, you MUST use the ytflow_api_tickets table.\n"
    formatted_history += "18. **CRITICAL:** If a previous query tried to use a non-existent table (like 'quota'), you MUST NOT use that table in similar queries.\n"
    formatted_history += "19. **CRITICAL:** If you see a correction that shows a table was incorrect, you MUST use the correct table for similar queries.\n"
    formatted_history += "20. **CRITICAL:** Always check the chat history for table name corrections before generating a query.\n"

    return formatted_history


CUSTOM_SQL_PROMPT = PromptTemplate.from_template("""
You are an elite data analyst and DuckDB SQL expert. Your task is to generate precise and highly optimized DuckDB SQL queries in response to user questions. You must fully leverage the provided database schema, sample data, categorical value distributions, historical column usage information, and recent chat history.

Database Context:
{table_info}

{column_usage_info}

{chat_history_context}

Instructions:
- Read and understand the Database Schema, Column Types, Sample Values, and Table Relationships from {table_info}.
- Pay close attention to the {column_usage_info} section to understand which columns have been frequently used in past queries.
- **CRITICAL INSTRUCTION:** Review the "Recent Chat History:" provided in {chat_history_context}. This history includes previous questions, generated SQL queries, and user feedback/corrections. 
  - If you see any ❌ INCORRECT QUERY followed by a ✅ CORRECTED SQL, you MUST use that corrected SQL (or a logically equivalent version).
  - If the current question is semantically similar to a previous question that was marked as incorrect, you MUST NOT repeat the same incorrect approach.
  - If the current question is semantically similar to a previous question that was marked as correct, you SHOULD follow the same successful approach.
  - **MOST IMPORTANT:** If you see a pattern of certain columns being used correctly in multiple queries, you MUST use those columns for similar questions.
  - **CRITICAL:** If you see a pattern of certain columns being used incorrectly, you MUST avoid using those columns.
  - **CRITICAL:** If you see a correction about table names, you MUST use the correct table names for similar queries.

Query Rules:
1. 🔍 **Table Names:**
   - **NEVER** infer table names from the user's question unless they explicitly mention them.
   - **ONLY** use tables that are explicitly listed in the schema.
   - **NEVER** create or reference tables that don't exist in the schema.
   - If unsure which table to use, use the most semantically relevant table from the schema.

2. 🔍 **Column Usage & Filtering:**
   - **CRITICAL:** First check the chat history for similar questions and their feedback.
   - If you find a similar question that was marked as correct, use the same columns and approach.
   - If you find a similar question that was marked as incorrect, avoid using those columns.
   - Prioritize using columns listed under "Column Usage:" if they are semantically relevant to the user's question.
   - **CRITICAL FOR WORD MATCHING:** When filtering on words or text:
     - **ALWAYS** use `LIKE '%word%'` with `LOWER()` for word matching to catch variations
     - Example: If user asks for "analytics", use `LOWER(column) LIKE '%analytics%'` to match "analytics.com", "myanalytics", etc.
     - Only use exact match (`=`) when the value is a specific identifier (like project_number)
     - For partial matches, always use `LIKE` with wildcards to ensure we don't miss variations
   - Always prefer column names based on semantic meaning inferred from the user's question.

3. 📆 **Date and Time Handling:**
   - **CRITICAL:** When dealing with dates and times:
     - Check if both `*_time` and `*_date` columns exist (e.g., `create_time` and `create_date`)
     - If both exist, prefer the `*_date` column for date-based filtering
     - Use `EXTRACT(YEAR FROM column)` for year-based filtering
     - Use `CAST()` to align data types when comparing timestamps
     - Use filters like `WHERE date_column >= CURRENT_DATE - INTERVAL '90 days'` when asked for "past 90 days"
     - For time-based analysis, use appropriate date columns based on the context
   - **NEVER** use a date/time column name as a table name

4. 🧠 **Semantic Matching:**
   - **ALWAYS** check the Categorical Column Distributions section for any values mentioned in the user's question.
   - If the user mentions ANY value that exists in the distributions, you MUST:
     1. Find the exact column where that value appears in the distributions
     2. Use that column in your WHERE clause, even if the user didn't mention the column name
     3. **CRITICAL:** Use `LOWER(column) LIKE '%value%'` for text matching to catch all variations
   - If a value appears in multiple columns, use the most semantically relevant one based on the question context.
   - If unsure about the correct column, prefer columns that have been frequently used (from Column Usage History).

5. 📊 **Aggregations:**
   - Use `COUNT`, `AVG`, `SUM`, `MAX`, `MIN` with appropriate `GROUP BY` when summarization is requested.
   - Use meaningful aliases for aggregated columns.
   - For time-based analysis, group by date columns and use appropriate date functions.

6. 🧠 **Advanced SQL Logic:**
   - You can use `CASE WHEN`, `ROW_NUMBER()`, `WINDOW` functions, `CTEs`, and `JOIN` statements if the query complexity requires it.
   - When needing ranked results or deduplication, use `ROW_NUMBER() OVER (PARTITION BY ...)`, ensuring you select necessary columns for partitioning and ordering.
   - For time-based analysis, use window functions to calculate running totals or averages.

7. ⚠️ **Always:**
   - Select only the necessary columns based on the user's question.
   - Avoid `SELECT *` unless explicitly needed.
   - Ensure correctness of joins using foreign keys or matching fields across tables.
   - Format clean, readable SQL with correct indentation.
   - **MOST IMPORTANTLY:** If you see a similar question in the chat history that was corrected, you MUST use the corrected SQL query or a logically equivalent version.
   - **CRITICAL:** Before writing the query, scan the Categorical Column Distributions for ANY values mentioned in the user's question and map them to the correct columns.
   - **CRITICAL:** If you see a pattern of certain columns being used correctly in multiple queries, you MUST use those columns for similar questions.
   - **CRITICAL:** If you see a pattern of certain columns being used incorrectly, you MUST avoid using those columns.
   - **CRITICAL:** DO NOT create or reference tables that don't exist in the schema.
   - **CRITICAL:** If you see a correction about table names in the chat history, you MUST use the correct table names for similar queries.
   - **CRITICAL:** If a previous query tried to use a non-existent table, you MUST NOT use that table in similar queries.
   - **CRITICAL:** NEVER infer table names from the user's question. Only use tables that are explicitly listed in the schema.
   - **CRITICAL:** Learn from the chat history to understand which tables and columns are appropriate for different types of queries.
   - **CRITICAL:** Pay special attention to corrections in the chat history to understand the correct table and column usage patterns.
   - **CRITICAL:** For word matching, ALWAYS use `LIKE '%word%'` with `LOWER()` to catch all variations of the word.
   - **CRITICAL:** When dealing with dates, check for both `*_time` and `*_date` columns and use the appropriate one.

8. ✂️ **Final Output:**
   - Output only the executable SQL query.
   - Do not include comments, explanations, or notes outside the query.

User question: {input}
Limit the result to {top_k} rows if not specified by the user.
SQL query:
""")


# Helper function to format samples for the prompt (should be called before passing to this function)
def format_samples_for_prompt(samples):
    formatted_samples = ""
    if samples:
        formatted_samples += "\nSample Data:\n"
        for table, df in samples.items():
            formatted_samples += f"Table: {table}\n"
            formatted_samples += str(df) + "\n\n" # Using str() as a fallback

    return formatted_samples

# Helper function to format distributions for the prompt (should be called before passing to this function)
def format_distributions_for_prompt(distributions):
    distribution_desc = ""
    if distributions:
        distribution_desc += "\nCategorical Column Distributions:\n"
        lines = []
        for sheet, cols in distributions.items():
            for col, vc in cols.items():
                # Limit value counts displayed
                # Assuming vc is a dictionary or similar structure
                dist_str = ", ".join([f"{k} ({v})" for k, v in list(vc.items())[:10]]) # Limit to top 10 values
                lines.append(f"- {sheet}.{col}: {dist_str}")
        if lines:
            distribution_desc += "\n".join(lines)
    return distribution_desc


# Helper function to format column usage counts for the prompt
def format_column_usage_for_prompt(column_usage_counts: dict) -> str:
    """Formats column usage counts into a string for the prompt."""
    if not column_usage_counts:
        return "Column Usage History: No column usage data available yet."

    usage_info = "Column Usage History (Most Frequent First):\n"
    # Sort columns by count in descending order
    sorted_columns = sorted(column_usage_counts.items(), key=lambda item: item[1], reverse=True)

    # Format the top N columns for the prompt (e.g., top 20)
    top_n = 20
    lines = []
    for col, count in sorted_columns[:top_n]:
        lines.append(f"- `{col}` (used {count} times)")

    if lines:
        usage_info += "\n".join(lines)
    else:
        usage_info += "No recent significant column usage."

    return usage_info


# Import SentenceTransformer and cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the sentence transformer model (load once)
# Use the same model as in db_utils for consistency
try:
    # Attempt to load the model from the default cache directory first
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading sentence transformer model in langchain_utils: {e}")
    print("Attempting to download the model...")
    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence transformer model downloaded and loaded successfully in langchain_utils.")
    except Exception as download_error:
        print(f"Error downloading and loading sentence transformer model in langchain_utils: {download_error}")
        sentence_model = None # Set to None if model cannot be loaded


def get_langchain_chain(engine, api_key=None, schema=None, samples=None, distributions=None, metadata_embeddings=None, chat_history=None, column_usage_counts=None):
    # This function receives schema, samples, distributions, metadata_embeddings, AND column_usage_counts
    # The formatting of samples and distributions should ideally be done in main.py
    # before passing them as strings to this function to avoid pandas dependency here.

    # 1. Set up the LangChain SQLDatabase
    db = SQLDatabase(engine=engine) # LangChain\'s SQLDatabase takes the SQLAlchemy engine

    # 2. Set up the Gemini LLM with lower temperature for more consistent results
    llm = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-2.0-flash",
        temperature=0.1,  # Lower temperature for more consistent results
        max_output_tokens=2048,  # Increase max tokens for complex queries
        top_p=0.8,  # Slightly lower top_p for more focused sampling
        top_k=40  # Lower top_k for more focused sampling
    )

    # Format column usage data for the prompt
    formatted_column_usage = format_column_usage_for_prompt(column_usage_counts if column_usage_counts is not None else {})

    # Format chat history for the prompt
    formatted_chat_history = format_chat_history_for_prompt(chat_history if chat_history is not None else [])

    # Format samples and distributions if they were not pre-formatted strings
    formatted_samples_str = format_samples_for_prompt(samples) if not isinstance(samples, str) else samples
    formatted_distributions_str = format_distributions_for_prompt(distributions) if not isinstance(distributions, str) else distributions

    # 3. Implement semantic search to filter context based on user question
    if sentence_model and metadata_embeddings and chat_history and chat_history[-1]['role'] == 'user':
        user_question = chat_history[-1]['content']
        # Generate embedding for the user question
        question_embedding = sentence_model.encode([user_question], convert_to_numpy=True)[0]

        # Calculate similarity between question embedding and metadata embeddings
        similarity_scores = {}
        for key, item in metadata_embeddings.items():
            if 'embedding' in item and item['embedding'] is not None:
                embedding_pair = np.array([question_embedding, item['embedding']])
                score = cosine_similarity(embedding_pair)[0, 1]
                similarity_scores[key] = score

        # Sort metadata keys by similarity score in descending order
        sorted_metadata_keys = sorted(similarity_scores, key=similarity_scores.get, reverse=True)

        # Select top-k relevant metadata keys (e.g., top 15 or adjust as needed)
        top_k = 15 # Number of top relevant items to include
        relevant_metadata_keys = sorted_metadata_keys[:top_k]

        # Filter schema, samples, and distributions to include only relevant items
        filtered_schema = {table: cols for table, cols in schema.items() if table in relevant_metadata_keys or any(f'{table}.{col}' in relevant_metadata_keys for col in cols)}
        filtered_samples = {table: df for table, df in (samples.items() if isinstance(samples, dict) else []) if table in filtered_schema}
        filtered_distributions = {table: dist for table, dist in (distributions.items() if isinstance(distributions, dict) else []) if table in filtered_schema}

        # Construct filtered table_info for the prompt
        filtered_table_info_str = db.get_table_info(table_names=list(filtered_schema.keys())) + format_distributions_for_prompt(filtered_distributions) + format_samples_for_prompt(filtered_samples)

        # Create the SQL query generation chain piece with filtered table_info
        sql_generator = create_sql_query_chain(llm, db, prompt=CUSTOM_SQL_PROMPT)

        # Define the chain that prepares context and generates the SQL query string
        sql_query_generation_chain = (
            RunnablePassthrough.assign(
                table_info=lambda x: filtered_table_info_str,
                input=lambda x: x['question'],
                top_k=lambda x: 100,
                column_usage_info=lambda x: formatted_column_usage,
                chat_history_context=lambda x: formatted_chat_history
            )
            | sql_generator
            | StrOutputParser()
            | clean_sql_query
        )

        return sql_query_generation_chain

    else:
        # If semantic search is not used (model/embeddings/history missing), use full context
        full_table_info_str = db.get_table_info() + format_distributions_for_prompt(distributions) + format_samples_for_prompt(samples)

        # Create the SQL query generation chain piece with full context
        sql_generator = create_sql_query_chain(llm, db, prompt=CUSTOM_SQL_PROMPT)

        # Define the chain
        sql_query_generation_chain = (
            RunnablePassthrough.assign(
                table_info=lambda x: full_table_info_str,
                input=lambda x: x['question'],
                top_k=lambda x: 100,
                column_usage_info=lambda x: formatted_column_usage,
                chat_history_context=lambda x: formatted_chat_history
            )
            | sql_generator
            | StrOutputParser()
            | clean_sql_query
        )
        return sql_query_generation_chain

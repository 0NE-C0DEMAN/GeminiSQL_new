import re
from typing import List, Dict, Set, Tuple
import json
from pathlib import Path

def extract_column_names(sql_query: str, schema: dict) -> List[str]:
    """
    Extracts actual column names used in the SELECT clause and WHERE clause of a SQL query
    by checking against the provided schema.
    Handles complex SQL including CTEs, subqueries, and joins.
    """
    extracted_columns = set()

    # Create a set of all valid column names from the schema
    valid_columns_lower = set()
    table_column_map = {}  # Map to store table -> columns mapping
    for table, columns in schema.items():
        table_column_map[table.lower()] = [col.lower() for col in columns]
        for col in columns:
            valid_columns_lower.add(col.lower())

    # Normalize query for processing
    query = sql_query.strip()
    query_upper = query.upper()

    # Handle CTEs first
    if 'WITH' in query_upper:
        # Extract the final SELECT statement after all CTEs
        final_select_idx = query_upper.rindex('SELECT')
        query = query[final_select_idx:]

    # Extract table aliases
    table_aliases = {}
    from_clause_match = re.search(r'\bFROM\b.*?(?=\bWHERE\b|\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|$)', query_upper, re.IGNORECASE | re.DOTALL)
    if from_clause_match:
        from_clause = query[from_clause_match.start():from_clause_match.end()]
        # Find table aliases in FROM clause
        alias_matches = re.finditer(r'\bFROM\b\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)|JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)', from_clause, re.IGNORECASE)
        for match in alias_matches:
            table_name = match.group(1) or match.group(3)
            alias = match.group(2) or match.group(4)
            if table_name and alias:
                table_aliases[alias.lower()] = table_name.lower()

    # Extract columns from SELECT clause
    select_match = re.search(r'\bSELECT\b(.*?)(?=\bFROM\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|$)', query, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_clause = select_match.group(1)
        
        # Split by commas, handling nested expressions
        parts = []
        current_part = ""
        paren_count = 0
        for char in select_clause:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                parts.append(current_part.strip())
                current_part = ""
                continue
            current_part += char
        if current_part:
            parts.append(current_part.strip())

        # Process each part
        for part in parts:
            # Handle column aliases
            if ' AS ' in part.upper():
                col_expr, alias = part.split(' AS ', 1)
                col_expr = col_expr.strip()
            else:
                col_expr = part.strip()

            # Skip if it's a function or expression
            if '(' in col_expr and not col_expr.startswith('('):
                continue

            # Handle table-qualified columns
            if '.' in col_expr:
                table_alias, col = col_expr.split('.', 1)
                table_alias = table_alias.strip().lower()
                col = col.strip().strip('"').strip("'").strip('`')
                
                # Resolve table alias to actual table name
                actual_table = table_aliases.get(table_alias, table_alias)
                if actual_table in table_column_map and col.lower() in table_column_map[actual_table]:
                    extracted_columns.add(col)
            else:
                # Handle unqualified columns
                col = col_expr.strip().strip('"').strip("'").strip('`')
                if col.lower() in valid_columns_lower:
                    extracted_columns.add(col)

    # Extract columns from WHERE clause
    where_match = re.search(r'\bWHERE\b(.*?)(?=\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|$)', query, re.IGNORECASE | re.DOTALL)
    if where_match:
        where_clause = where_match.group(1)
        
        # Find all column references in the WHERE clause
        # This regex looks for column names that are:
        # 1. Not part of a function call (unless it's EXTRACT)
        # 2. Not part of a string literal
        # 3. Not part of a number
        column_pattern = r'(?<!\w)([a-zA-Z_][a-zA-Z0-9_]*)(?!\s*\()'
        column_matches = re.finditer(column_pattern, where_clause)
        
        for match in column_matches:
            col = match.group(1)
            # Skip if it's a SQL keyword or function
            if col.upper() in {'AND', 'OR', 'NOT', 'IN', 'BETWEEN', 'LIKE', 'IS', 'NULL', 'TRUE', 'FALSE', 'EXTRACT', 'YEAR', 'FROM'}:
                continue
                
            # Handle table-qualified columns
            if '.' in col:
                table_alias, col = col.split('.', 1)
                table_alias = table_alias.strip().lower()
                col = col.strip().strip('"').strip("'").strip('`')
                
                # Resolve table alias to actual table name
                actual_table = table_aliases.get(table_alias, table_alias)
                if actual_table in table_column_map and col.lower() in table_column_map[actual_table]:
                    extracted_columns.add(col)
            else:
                # Handle unqualified columns
                col = col.strip().strip('"').strip("'").strip('`')
                if col.lower() in valid_columns_lower:
                    extracted_columns.add(col)

    return list(extracted_columns)

def extract_table_names(sql_query: str) -> List[str]:
    """
    Extracts table names from a SQL query.
    Handles CTEs properly by not treating CTE aliases as table names.
    """
    tables = set()
    query_upper = sql_query.upper()
    
    # First, extract all CTE aliases to exclude them from table validation
    cte_aliases = set()
    if 'WITH' in query_upper:
        cte_matches = re.finditer(r'\bWITH\b\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(', query_upper)
        for match in cte_matches:
            cte_aliases.add(match.group(1).lower())
    
    # Extract tables from FROM and JOIN clauses, excluding CTE aliases
    from_join_matches = re.finditer(r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*))?', query_upper)
    for match in from_join_matches:
        table_name = match.group(1)
        if table_name and table_name.lower() not in cte_aliases:
            tables.add(table_name.lower())
    
    return list(tables)

def load_schema_metadata(metadata_path: Path) -> dict:
    """
    Load schema metadata from JSON file.
    """
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

def validate_column_against_schema(column: str, table: str, schema_metadata: dict) -> bool:
    """
    Validate if a column exists in the specified table according to schema metadata.
    """
    if not schema_metadata:
        return False
    
    table_info = schema_metadata.get("tables", {}).get(table, {})
    return column in table_info.get("columns", [])

def validate_table_relationship(table1: str, table2: str, schema_metadata: dict) -> bool:
    """
    Validate if there's a relationship between two tables.
    """
    if not schema_metadata:
        return False
    
    # Check if both tables exist
    if table1 not in schema_metadata.get("tables", {}) or table2 not in schema_metadata.get("tables", {}):
        return False
    
    # Check for relationships in metadata
    relationships = schema_metadata.get("relationships", [])
    for rel in relationships:
        if (rel.get("from_table") == table1 and rel.get("to_table") == table2) or \
           (rel.get("from_table") == table2 and rel.get("to_table") == table1):
            return True
    
    return False

def validate_sql_query(query: str, schema: dict, chat_history: list = None) -> dict:
    """
    Basic validation of SQL query against schema.
    Returns dict with validation result and error message if any.
    """
    try:
        # Basic validation
        if not query or not query.strip():
            return {"valid": False, "error": "Empty query"}

        # Extract table names and aliases using improved regex
        # This pattern matches both "FROM table" and "FROM table AS alias"
        table_pattern = r'(?i)FROM\s+([a-zA-Z0-9_]+)(?:\s+AS\s+[a-zA-Z0-9_]+)?'
        tables = re.findall(table_pattern, query)
        
        # Also check for table references in JOIN clauses
        join_pattern = r'(?i)JOIN\s+([a-zA-Z0-9_]+)(?:\s+AS\s+[a-zA-Z0-9_]+)?'
        join_tables = re.findall(join_pattern, query)
        tables.extend(join_tables)
        
        # Remove duplicates and empty strings
        tables = list(set([t for t in tables if t]))
        
        # Validate tables exist
        for table in tables:
            if table.lower() not in [t.lower() for t in schema.keys()]:
                return {
                    "valid": False, 
                    "error": f"Table '{table}' not found in schema. Available tables: {', '.join(schema.keys())}"
                }

        # Let the AI handle column validation through the LLM chain
        return {"valid": True, "error": None}

    except Exception as e:
        print(f"Validation error: {str(e)}")
        return {"valid": False, "error": f"Error validating query: {str(e)}"}

def is_similar_query(query1: str, query2: str) -> bool:
    """
    Check if two queries are semantically similar.
    """
    # Normalize queries
    q1 = query1.lower().strip()
    q2 = query2.lower().strip()
    
    # Extract key components
    def extract_components(q):
        components = {
            'tables': set(),
            'columns': set(),
            'conditions': set()
        }
        
        # Extract tables
        table_matches = list(re.finditer(r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*))?', q.upper()))
        for match in table_matches:
            if match.group(1):
                components['tables'].add(match.group(1).lower())
        
        # Extract columns
        select_matches = list(re.finditer(r'\bSELECT\b(.*?)(?=\bFROM\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|$)', q, re.IGNORECASE | re.DOTALL))
        if select_matches:
            select_clause = select_matches[0].group(1)
            col_matches = list(re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+AS\s+[a-zA-Z_][a-zA-Z0-9_]*)?', select_clause))
            for match in col_matches:
                if match.group(1):
                    components['columns'].add(match.group(1).lower())
        
        # Extract conditions
        where_matches = list(re.finditer(r'\bWHERE\b(.*?)(?=\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|$)', q, re.IGNORECASE | re.DOTALL))
        if where_matches:
            where_clause = where_matches[0].group(1)
            condition_matches = list(re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=|>|<|>=|<=|!=|LIKE|IN)\s*', where_clause))
            for match in condition_matches:
                if match.group(1):
                    components['conditions'].add(match.group(1).lower())
        
        return components
    
    # Compare components
    c1 = extract_components(q1)
    c2 = extract_components(q2)
    
    # Calculate similarity scores
    table_similarity = len(c1['tables'] & c2['tables']) / max(len(c1['tables'] | c2['tables']), 1)
    column_similarity = len(c1['columns'] & c2['columns']) / max(len(c1['columns'] | c2['columns']), 1)
    condition_similarity = len(c1['conditions'] & c2['conditions']) / max(len(c1['conditions'] | c2['conditions']), 1)
    
    # Weighted average of similarities
    similarity = (table_similarity * 0.4 + column_similarity * 0.4 + condition_similarity * 0.2)
    
    return similarity > 0.7  # Threshold for considering queries similar

def matches_corrected_pattern(query: str, corrected_query: str) -> bool:
    """
    Check if a query matches the pattern of a corrected query.
    """
    # Normalize queries
    q = query.lower().strip()
    cq = corrected_query.lower().strip()
    
    # Extract table patterns
    def extract_table_pattern(q):
        pattern = []
        table_matches = list(re.finditer(r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*))?', q.upper()))
        for match in table_matches:
            if match.group(1):
                pattern.append(match.group(1).lower())
        return pattern
    
    # Extract column patterns
    def extract_column_pattern(q):
        pattern = []
        select_matches = list(re.finditer(r'\bSELECT\b(.*?)(?=\bFROM\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|$)', q, re.IGNORECASE | re.DOTALL))
        if select_matches:
            select_clause = select_matches[0].group(1)
            col_matches = list(re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+AS\s+[a-zA-Z_][a-zA-Z0-9_]*)?', select_clause))
            for match in col_matches:
                if match.group(1):
                    pattern.append(match.group(1).lower())
        return pattern
    
    # Compare patterns
    q_tables = extract_table_pattern(q)
    cq_tables = extract_table_pattern(cq)
    q_columns = extract_column_pattern(q)
    cq_columns = extract_column_pattern(cq)
    
    # Check if tables match
    if set(q_tables) != set(cq_tables):
        return False
    
    # Check if columns match (allowing for some flexibility in order)
    if len(set(q_columns) & set(cq_columns)) < len(set(cq_columns)) * 0.8:  # 80% match required
        return False
    
    return True 
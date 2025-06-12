def get_custom_css():
    return """
    <style>
        /* Padding for the main block container */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Style for chat messages */
        .stChatMessage {
            padding: 1em;
            border-radius: 8px;
            margin-bottom: 0.5em;
        }

        /* Style for scrollable chat history container */
        /* This targets the container created by st.container */
        div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stChatMessage"]) {
            overflow-y: auto;
            max-height: 400px; /* Match the height of the chat container */
            display: flex;
            flex-direction: column; /* Display messages in chronological order top-down */
        }

        /* Ensure chat messages themselves don't add extra scrollbars */
        div[data-testid="stChatMessage"] {
             overflow: visible;
        }

        /* Style for expanders */
        .stExpander {
            margin-bottom: 1em;
        }

        /* Style for sample data tabs - decrease top margin */
        div[data-testid="stTabs"] {
             margin-top: -1em;
        }
         /* Style for dataframe in tabs - decrease top margin */
        div[data-testid="stDataFrame"] {
            margin-top: -1em;
        }
    </style>
    """ 
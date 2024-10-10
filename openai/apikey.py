import os
apikey = os.getenv('OPENAI_API_KEY')

if apikey is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
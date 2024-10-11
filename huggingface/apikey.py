import os
apikey = os.getenv('HF_API_KEY')

if apikey is None:
    raise ValueError("HF_API_KEY not found in environment variables")
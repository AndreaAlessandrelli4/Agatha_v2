from dotenv import load_dotenv
load_dotenv()

import os
print("ENV key:", os.getenv("OPENAI_API_KEY"))
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
 
load_dotenv()
 
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

print(AZURE_OPENAI_KEY)
print(AZURE_OPENAI_ENDPOINT)
print(AZURE_OPENAI_DEPLOYMENT)
print(AZURE_API_VERSION)
 
client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
)
 
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Ans",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=AZURE_OPENAI_DEPLOYMENT
)
 
print(response.choices[0].message.content)
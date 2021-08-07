from dotenv import load_dotenv
import os
import openai

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

examples = """
          Facebook: Social media, Technology\n
          LinkedIn: Social media, Technology, Enterprise, Careers\n
          Uber: Transportation, Technology, Marketplace\n
          Unilever: Conglomerate, Consumer Goods\n
          Mcdonalds: Food, Fast Food, Logistics, Restaurants\n
        """
query = "FedEx:"
prompt = examples+query

response = openai.Completion.create(
  engine="davinci",
  prompt=prompt,
  temperature=0,
  max_tokens=6,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\n"]
)

print(query, response['choices'][0]['text'])
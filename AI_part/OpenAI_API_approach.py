from dotenv import load_dotenv
import os
import openai
import pandas as pd
from sklearn.model_selection import train_test_split
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

class text_classification_OpenAI():
  def init(self, data):
    """
      input: 
        data: the data imported into the class --> .csv with label 'content', 'label'
    """
    self.content = self.data['content']
    self.label = self.data['label']
    
  def create_example(self, content, label):
    """
      input:
        content: the content --> list
        label: the label correspond to the content, one per content --> list
      
      output:
        return organized example for prediction
        content: label \n
        ...
    """
    examples = ""
    for i in range(len(content)):
      examples += content[i],": ", label[i], "\n"
      
    return examples

  def predict(self,examples, query):
    """
      input: 
        examples: the example sentences (content: label...\n)
        query: the content that need to be classified
      
      output: the predicted label of the query
    """
    prompt = examples+query+':'
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
    predicted_label = response['choices'][0]['text']
    
    return predicted_label
  
  def mulitiple_query(self, query_contents):
    example = self.create_example(self.content, self.label)
    predicted_label = []
    for i in range(len(query_contents)):
      predicted_label.append([self.predict(example, query_contents[i])])
    
    return predicted_label

data = pd.read_csv('.csv')
train_content, train_label, test_content, test_label = train_test_split(data['content'], data['label'])
df_train = {'content': train_content, 'label': train_label}
print(df_train)
df_train = pd.DataFrame(df_train)

text_classification = text_classification_OpenAI(df_train)
predicted_label = text_classification.mulitiple_query(test_content)
error = 0
for i in range(len(test_label)):
  if predicted_label != test_label:
    error += 1
print('error rate:', error/len(test_label)*100, '%')
##################################################################################

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
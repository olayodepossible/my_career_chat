import os
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
from pydantic import BaseModel

load_dotenv(override=True)
openai_api_key = os.getenv('OPENROUTER_API_KEY')
base_url = os.getenv('OPEN_ROUTER_BASE_URL')
name = os.getenv('RESUME_NAME')
resume = os.getenv('RESUME_FILE_NAME')
summary = os.getenv('RESUME_SUMMARY_FILE_NAME')
openai = OpenAI(base_url=base_url, api_key=openai_api_key)



reader = PdfReader(f"my_data/{resume}.pdf")
resume_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        resume_text += text

with open(f"my_data/{summary}.txt", 'r', encoding='utf-8') as file:
   summary_text = file.read()

system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
particularly questions related to {name}'s career, background, skills and experience. \
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer, say so."

system_prompt += f"\n\n## Summary:\n{summary_text}\n\n## LinkedIn Profile:\n{resume_text}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."

def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content

# Define the evaluation model to evaluate the  model response
class EvaluationModel(BaseModel):
    is_acceptable: bool
    feedback: str

evaluator_system_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality. \
The Agent is playing the role of {name} and is representing {name} on their website. \
The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website. \
The Agent has been provided with context on {name} in the form of their summary and LinkedIn details. Here's the information:"

evaluator_system_prompt += f"\n\n## Summary:\n{summary_text}\n\n## LinkedIn Profile:\n{resume_text}\n\n"
evaluator_system_prompt += f"With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."

def evaluator_user_prompt(reply, message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
    user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
    user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
    user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
    return user_prompt


def evaluate(reply, message, history) -> EvaluationModel:

    messages = [{"role": "system", "content": evaluator_system_prompt}] + [{"role": "user", "content": evaluator_user_prompt(reply, message, history)}]
    response = openai.chat.completions.parse(model="google/gemini-2.5-flash", messages=messages, response_format=Evaluation)
    return response.choices[0].message.parsed

def main():
    
    gr.ChatInterface(chat).launch()


if __name__ == "__main__":
    main()

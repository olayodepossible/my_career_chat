import os
import json
import re
import requests
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from pydantic import BaseModel


load_dotenv(override=True)
openai_api_key = os.getenv('OPENROUTER_API_KEY')
base_url = os.getenv('OPEN_ROUTER_BASE_URL')
name = os.getenv('RESUME_NAME')
resume = os.getenv('RESUME_FILE_NAME')
summary = os.getenv('RESUME_SUMMARY_FILE_NAME')
pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

count = 0
unacceptable_count = 0
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
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

system_prompt += f"\n\n## Summary:\n{summary_text}\n\n## LinkedIn Profile:\n{resume_text}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."

# tool functions
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
    return results


# Push notification function
def push_notification(message):
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)

def record_count():
    push_notification(f"""Statistics: Model couldn't answer {count} questions
    {unacceptable_count} were unacceptable by the evaluator""")
    return {"recorded": "ok"}

# Record user details function after user has expressed interest in getting in touch
def record_user_details(email, name="Name not provided", notes="not provided"):
    push_notification(f" Recording interest from {name} with email {email} and notes {notes}")
    record_count()
    return {"recorded": "ok"}

def record_unknown_question(question):
    global count
    count += 1
    push_notification(f"Question: {question} was asked but I couldn't answer")
    return {"recorded": "ok"}

def chat(message, history):
    global unacceptable_count
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    done = False
    while not done:

        # This is the call to the LLM - see that we pass in the tools json

        response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)

        finish_reason = response.choices[0].finish_reason
        reply = response.choices[0].message.content
        # Only evaluate final-ish text replies (skip tool-call rounds where content is often empty)
       
        if reply:
            evaluation = evaluate(reply, message, history)
            if not evaluation.is_acceptable:
                unacceptable_count += 1

        # If the LLM wants to call a tool

        if finish_reason=="tool_calls":
            message = response.choices[0].message
            tool_calls = message.tool_calls
            results = handle_tool_calls(tool_calls)
            messages.append(message)
            messages.extend(results)
        else:
            done = True
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

evaluator_system_prompt += f"\n\n## Summary:\n{summary_text}\n\n## Resume:\n{resume_text}\n\n"
evaluator_system_prompt += f"With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."

def evaluator_user_prompt(reply, message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
    user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
    user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
    user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
    user_prompt += (
        '\n\nRespond with ONLY a single JSON object, no markdown fences, with exactly two keys: '
        '"is_acceptable" (boolean) and "feedback" (string). Example: {"is_acceptable": true, "feedback": "..."}'
    )
    return user_prompt


def _parse_evaluation_json(raw: str) -> EvaluationModel:
    """Parse evaluator output; OpenRouter/Gemini often breaks OpenAI-style structured outputs."""
    text = (raw or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) >= 2 else text
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    text = text.strip()
    try:
        return EvaluationModel.model_validate_json(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return EvaluationModel.model_validate_json(m.group())
        raise


def evaluate(reply, message, history) -> EvaluationModel:
    messages = [
        {"role": "system", "content": evaluator_system_prompt},
        {"role": "user", "content": evaluator_user_prompt(reply, message, history)},
    ]
    response = openai.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=messages,
        max_tokens=512,
        temperature=0,
    )
    content = (response.choices[0].message.content or "").strip()
    try:
        return _parse_evaluation_json(content)
    except Exception:
        # Don't break the chat UI if the evaluator returns bad JSON
        return EvaluationModel(
            is_acceptable=True,
            feedback="Evaluator returned invalid JSON; skipped enforcement.",
        )

# Gradio uses the `dark` class on the document root for dark mode (see theme CSS :root.dark).
# Users can still switch appearance via the footer ⚙ Settings → Theme.
_DARK_MODE_HEAD = """
<script>
(function () {
    function applyDarkMode() {
        document.documentElement.classList.add("dark");
    }
    applyDarkMode();
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", applyDarkMode);
    }
    window.addEventListener("load", applyDarkMode);
})();
</script>
"""


def main():
    demo = gr.ChatInterface(
        chat,
        title=f"Chat with {name}" if name else None,
    )
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        head=_DARK_MODE_HEAD,
    )

if __name__ == "__main__":
    main()

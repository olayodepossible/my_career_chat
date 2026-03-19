import os
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr

load_dotenv(override=True)
openai_api_key = os.getenv('OPENROUTER_API_KEY')
base_url = os.getenv('OPEN_ROUTER_BASE_URL')
name = os.getenv('RESUME_NAME')
openai = OpenAI(base_url=base_url, api_key=openai_api_key)




def get_resume_text(resume_path):
    reader = PdfReader(resume_path)
    resume_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            resume_text += text
    return resume_text

def open_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def main():
    resume_text = get_resume_text("my_data/Abayomi_resume.pdf")
    summary_text = open_text_file("my_data/Abayomi-summary.txt")
    print(summary_text)


if __name__ == "__main__":
    main()

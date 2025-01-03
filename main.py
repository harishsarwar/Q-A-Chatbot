import os
from flask import Flask, request, render_template
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Fetch HuggingFace API key
api_key = os.getenv("HUGGINGFACE_API_KEY")
if not api_key:
    raise ValueError("HUGGINGFACE_API_KEY is not set in the environment variables.")

# Model repository
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

# Initialize HuggingFaceEndpoint
model_loading = HuggingFaceEndpoint(
    task="text-generation",
    repo_id=repo_id,
    max_new_tokens=200,
    temperature=0.7,
    huggingfacehub_api_token=api_key,
)

# Define prompt template
template = """Question: {question}

Answer: Let's think step by step."""
prompting = PromptTemplate(template=template, input_variables=["question"])

# Create LangChain
llm_chain = LLMChain(prompt=prompting, llm=model_loading)

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    if request.method == "POST":
        question = request.form.get("question")
        # Get response from the model
        response = llm_chain.invoke({"question": question})
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)

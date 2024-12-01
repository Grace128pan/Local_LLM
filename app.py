from flask import Flask, request, jsonify, render_template
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# Initialize Llama and prompt
template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Initialize conversation context
context = ""


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global context
    user_input = request.json.get("question", "")
    if not user_input:
        return jsonify({"error": "No question provided"}), 400

    # Generate a response using the chain
    result = chain.invoke({"context": context, "question": user_input})
    context += f"User: {user_input}\nBot: {result}\n"

    return jsonify({"response": result})

if __name__ == "__main__":
    app.run(debug=True)

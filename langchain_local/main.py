from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from vector import retriever  # your Chroma retriever

# Initialise le modèle Ollama
model = OllamaLLM(model="llama3.2")

# Template pour le prompt
template = (
    "You are an expert in answering questions about a pizza restaurant.\n"
    "Here are some relevant reviews:\n{reviews}\n"
    "Here is the question to answer: {question}"
)

prompt = ChatPromptTemplate.from_template(template)

# Crée la chaîne LLMChain
chain = LLMChain(llm=model, prompt=prompt)

# Invoque la chaîne
while True:
    print("\n-----------------")
    question = input("Ask your question (q to quit): ")
    if question.lower() == "q":
        break

    # Get the top 5 relevant reviews
    retrieved_docs = retriever.get_relevant_documents(question)
    reviews_text = "\n".join([doc.page_content for doc in retrieved_docs])

    # Pass the reviews to the chain
    result = chain.invoke({"reviews": reviews_text, "question": question})

    # Print only the text
    if isinstance(result, dict):
        print("\nAnswer:\n", result.get("text", ""))
    else:
        print("\nAnswer:\n", result)

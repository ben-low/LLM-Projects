import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#load_dotenv()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="./knowledge_base.csv")
documents = loader.load()
# print(len(documents))

# embeddings = OpenAIEmbeddings()
# db = FAISS.from_documents(documents, embeddings)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    #print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
llm = LlamaCpp(
    model_path="../models/llama-2-7b-chat.gguf.q2_K.bin",
    callback_manager=callback_manager,
    verbose=False,
    temperature=0.7,
)

#I will share a security vulnerability with you and you will give me a description and impact of the security vulnerability based on past report

template = """

Below is a security vulnerability:
{security_vulnerability}

Here is an example of description for the security vulnerabilities:
{past_report}

Given the example description, generate an executive summary on the security vulnerabilities and provide mitigation:
"""

prompt = PromptTemplate(
    input_variables=["security_vulnerability", "past_report"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt, verbose=True)


# 4. Retrieval augmented generation
def generate_response(security_vulnerability):
    past_report = retrieve_info(security_vulnerability)
    response = chain.run(security_vulnerability=security_vulnerability, past_report=past_report)
    print(response)
    return response


# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Executive Summary Generator", page_icon=":robot_face:")

    st.header("Executive Summary Generator  :robot_face:")
    message = st.text_area("Provide a security vulnerability  :thought_balloon:")

    if message:
        st.write("Generating executive summary...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()

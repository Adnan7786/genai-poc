import warnings

warnings.filterwarnings(action="ignore")


def clean_text(text):
    import re
    # Replace multiple \n with a single \n
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing spaces
    text = text.strip()
    return text


def read_from_csv(csv_file, skip_header=True):
    import csv
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        if skip_header:
            next(reader)
        rows = list(reader)
    return rows


def write_to_csv(csv_file, rows, header_row=None):
    import csv
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        if header_row:
            writer.writerow(header_row)
        writer.writerows(rows)


def read_from_json(json_file):
    import json
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data


def get_documents(loader):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs


def get_documents_from_web(url):
    from langchain_community.document_loaders import WebBaseLoader
    print(f"Loading docs from url: {url}")
    loader = WebBaseLoader(url)
    return get_documents(loader)


def get_documents_from_web_urls(urls):
    from langchain_community.document_loaders import WebBaseLoader
    print(f"Loading docs from below urls:\n{'\n'.join(urls)}")
    loader = WebBaseLoader(urls)
    return get_documents(loader)


def get_documents_from_pdf(pdf_path):
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    print(f"Loading docs from pdf: {pdf_path}")
    loader = PyPDFDirectoryLoader(pdf_path)
    return get_documents(loader)


def is_model_available(model_type, source, model):
    global available_models
    try:
        if not available_models:
            available_models = read_from_json(
                json_file="available_models.json")
        sel_model_type = available_models.get(model_type)
        if not sel_model_type:
            return False
        sel_source = sel_model_type.get("sources").get(source)
        if not sel_source:
            return False
        models = sel_source.get("models")
        if model not in models:
            return False
        return True
    except Exception as e:
        print(f"Error in check_if_model_available: {e}")
        return False


def get_embeddings(embed_source="Ollama", embed_model="mxbai-embed-large:latest"):
    if not is_model_available("embed", embed_source, embed_model):
        return
    if embed_source == "Ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=embed_model)
    elif embed_source == "OpenAI":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=embed_model)


def create_chroma_vectorstore(docs, embeddings, chroma_path):
    from langchain_chroma import Chroma
    vectorStore = Chroma.from_documents(
        docs, embeddings, persist_directory=chroma_path)
    return vectorStore


def get_chroma_vectorstore(embeddings, chroma_path):
    from langchain_chroma import Chroma
    vectorStore = Chroma(persist_directory=chroma_path,
                         embedding_function=embeddings)
    return vectorStore


def clear_chroma_vectorstore(chroma_path):
    import os
    import shutil
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)


def get_llm(genai_source="Ollama", genai_model="qwen2.5-coder:3b"):
    temperature = 0
    max_retries = 2
    verbose = True
    if not is_model_available("genai", genai_source, genai_model):
        raise Exception(f"The provided {genai_source} model {
                        genai_model} is not available")
    if genai_source == "Ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=genai_model, temperature=temperature, verbose=verbose)
    elif genai_source == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=genai_model, temperature=temperature, max_retries=max_retries, verbose=verbose)
    elif genai_source == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=genai_model, temperature=temperature, max_retries=max_retries, verbose=verbose)
    elif genai_source == "Google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=genai_model, temperature=temperature, max_retries=max_retries, verbose=verbose)
    else:
        raise Exception(f"The provided {genai_source} model {
                        genai_model} is not connfigured.")


def get_chat_prompt():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.prompts import MessagesPlaceholder
    return ChatPromptTemplate.from_messages([
        ("system",
         "Answer the user's questions based on the below context: {context}"),
        ("system",
         "Do not hallucinate, if the information is not available in the given context, reply with 'This question falls beyond my expertise.'"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}.")
    ])


def get_retriever_prompt():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.prompts import MessagesPlaceholder
    return ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human",
         '''Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.
         ''')
    ])


def create_chain(vectorstore, llm):
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.history_aware_retriever import create_history_aware_retriever
    chat_prompt = get_chat_prompt()
    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=chat_prompt
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retriever_prompt = get_retriever_prompt()
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=retriever_prompt
    )
    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )
    return retrieval_chain


def process_chat(chain, question, chat_history):
    from langchain_core.messages import AIMessage
    response = chain.invoke({
        "chat_history": chat_history,
        "input": question,
    })
    answer = response["answer"]
    return answer.content if type(answer) == AIMessage else answer


def get_evaluation_prompt():
    from langchain_core.prompts import ChatPromptTemplate
    return ChatPromptTemplate.from_messages([
        ("system",
         "If the AI response logically matches the expected response, respond with true otherwise respond with false"),
        ("human", '''
         Expected Response: {expected_response}
         AI Response: {ai_response}''')
    ])


def get_test_result_filename(embed_model, genai_model, accuracy):
    accuracy = int(accuracy)
    return f"test_results/{embed_model}_{genai_model}_{accuracy}.csv"


def get_match_result(evaluation_results):
    from langchain_core.messages import AIMessage
    evaluation_results_str = evaluation_results.content if type(
        evaluation_results) == AIMessage else evaluation_results
    cleaned_result = evaluation_results_str.strip().lower()
    if "true" in cleaned_result:
        return True
    elif "false" in cleaned_result:
        return False
    return None


def calculate_accuracy(test_result_rows: list):
    total_testcases = len(test_result_rows)
    passed_testcases = len(
        [row for row in test_result_rows if row[-1] == True])
    print(f"Passed Test Cases: {passed_testcases}/{total_testcases}")
    return (passed_testcases/total_testcases) * 100


def get_vectorstore(create_new=False):
    import os
    embeddings = get_embeddings(
        embed_source=embed_source, embed_model=embed_model)

    chroma_path_exists = os.path.exists(chroma_path)

    if not create_new and chroma_path_exists:
        print(f"Reading existing vectorstore {chroma_path}")
        return get_chroma_vectorstore(
            embeddings=embeddings, chroma_path=chroma_path)

    web_urls_json = read_from_json("gen_ai_web_urls.json")
    web_urls = web_urls_json and web_urls_json.get("web_urls")
    if not web_urls:
        return

    # docs = []
    # for url in web_urls:
    #     doc = get_documents_from_web(url)
    #     cleaned_content = clean_text(doc.page_content)
    #     doc.page_content = cleaned_content
    #     docs.extend(doc)

    docs = get_documents_from_web_urls(web_urls)
    # print("Docs listed below:")
    # for doc in docs:
    #     print(doc)
    cleaned_docs = []
    for doc in docs:
        cleaned_content = clean_text(doc.page_content)
        doc.page_content = cleaned_content
        cleaned_docs.append(doc)
    # print("\nCleaned Docs lited below:")
    # for doc in cleaned_docs:
    #     print(doc)
    if chroma_path_exists:
        clear_chroma_vectorstore(chroma_path=chroma_path)
    print(f"Creating new vectorstore {chroma_path}...")
    return create_chroma_vectorstore(
        docs=cleaned_docs, embeddings=embeddings, chroma_path=chroma_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    embed_source = "OpenAI"
    embed_model = "text-embedding-3-large"
    genai_source = "Ollama"
    genai_model = "qwen2.5-coder:latest"

    chroma_path = f"{embed_source}-{embed_model}-chroma"

    available_models = read_from_json(json_file="available_models.json")

    vectorstore = get_vectorstore(create_new=False)
    llm = get_llm(genai_source=genai_source, genai_model=genai_model)
    chain = create_chain(vectorstore=vectorstore, llm=llm)

    chat_history = []
    evaluation_prompt = get_evaluation_prompt()

    test_result_header_row = ["Question",
                              "Expected Response", "AI Response", "Match"]
    test_result_rows = []
    testcases_rows = read_from_csv("gen_ai_test_cases.csv")
    total_testcases = len(testcases_rows)
    for index, row in enumerate(testcases_rows):
        try:
            print(f"\nTest Case: {index+1}/{total_testcases}")
            question, expected_response = row
            print(f"Question: {question}")
            print(f"Expected Answer: {expected_response}")
            response_text = process_chat(chain, question, chat_history)
            print(f"AI Answer: {response_text}")
            evail_chain = get_evaluation_prompt() | llm
            evaluation_results_str = evail_chain.invoke({
                "expected_response": expected_response, "ai_response": response_text
            })
            is_match = get_match_result(evaluation_results_str)
            print(f"Match: {is_match}")
            print("\n")
            test_result_rows.append(
                [question, expected_response, response_text, is_match])
        except Exception as e:
            print(f"Error occured in main method: {e}")
            test_result_rows.append(
                [question, expected_response, response_text, f"Exception Occurred: {e}"])
    accuracy = calculate_accuracy(test_result_rows)
    print(f"Accuracy: {accuracy}%")
    result_filename = get_test_result_filename(
        embed_model=embed_model, genai_model=genai_model, accuracy=accuracy)
    write_to_csv(result_filename, test_result_rows, test_result_header_row)

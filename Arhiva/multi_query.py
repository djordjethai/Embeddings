import logging

class StringLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_records = []

    def emit(self, record):
        log_entry = self.format(record)
        self.log_records.append(log_entry)



class MultiQueryDocumentRetriever:
    def __init__(self, document_url, question, temperature=0, chunk_size=500, chunk_overlap=0):

        self.log_handler = StringLogHandler()
        self.log_handler.setLevel(logging.INFO)
        logger = logging.getLogger("langchain.retrievers.multi_query")
        logger.setLevel(logging.INFO)
        logger.addHandler(self.log_handler)

        self.document_url = document_url
        self.question = question
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectordb = None
        self.llm = None
        self.retriever = None
        self.setup_vector_db()
        self.setup_llm()
        self.setup_retriever()

    def setup_vector_db(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings

        loader = WebBaseLoader(self.document_url)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        splits = text_splitter.split_documents(data)
        embedding = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

    def setup_llm(self):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(temperature=self.temperature)

    def setup_retriever(self):
        from langchain.retrievers.multi_query import MultiQueryRetriever
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=self.vectordb.as_retriever(), llm=self.llm
        )

    def get_relevant_documents(self, custom_question=None):
        if custom_question is None:
            custom_question = self.question
        return self.retriever.get_relevant_documents(query=custom_question)


    def extract_generated_queries(self):
        generated_queries = []
        for log_record in self.log_handler.log_records:
            if "Generated queries:" in log_record:
                # Extracting the queries part from the log message
                start = log_record.find("[")
                end = log_record.find("]") + 1
                if start != -1 and end != -1:
                    queries_str = log_record[start:end]
                    try:
                        # Assuming the queries are logged in a list format
                        queries = eval(queries_str)
                        generated_queries.extend(queries)
                    except NameError:
                        # Handle possible eval errors (e.g., due to undefined names in the string)
                        continue
        return generated_queries


# Replace 'document_url' with the actual URL of your document
document_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
question = "What are the approaches to Task Decomposition?"
retriever_instance = MultiQueryDocumentRetriever(document_url, question)

# To get documents relevant to the original question
unique_docs = retriever_instance.get_relevant_documents()
print(f"Number of unique documents retrieved: {len(unique_docs)}")

# To query a different question without creating a new instance
other_question = "What does the course say about classification?"
unique_docs_for_other_question = retriever_instance.get_relevant_documents(custom_question=other_question)
print(f"Number of unique documents retrieved for the other question: {len(unique_docs_for_other_question)}")

generated_queries = retriever_instance.extract_generated_queries()
print("Generated Queries:", generated_queries)

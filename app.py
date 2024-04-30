from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.helper import *
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

## loading the data
loader = DirectoryLoader(path='data/', 
                         glob = '*.pdf',
                         loader_cls=PyPDFLoader)

documents = loader.load()

## Chunking the data
txt_splitter= RecursiveCharacterTextSplitter(chunk_size=500,
                                             chunk_overlap=50
                                             )

txt_chunks = txt_splitter.split_documents(documents)

## Embedding model
emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                           model_kwargs={'device':'cpu'}
                           )

## convert the text chunks into embeddings and store it in FAISS vector store
vector_store = FAISS.from_documents(txt_chunks, emb)

llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q4_0.bin',
                    model_type='llama',
                    config={'max_new_tokens':128,'temperature':0.01}
                    )

qa_prompt = PromptTemplate(input_variables=['context','question'], 
                           template=template)
chain = RetrievalQA.from_chain_type(llm=llm,
                    chain_type='stuff',
                    retriever=vector_store.as_retriever(search_kwargs={'k':2}),
                    return_source_documents=False,
                    chain_type_kwargs={'prompt': qa_prompt})

# user_input='Tell me about Ontology'

# result = chain({'query':user_input})
# print(f'answer:{result["result"]}')

@app.route('/', methods=["GET","POST"])
def index():
    return render_template('index.html', **locals())

@app.route('/chatbot', methods=['GET','POST'])
def chatbotResponse():

    if request.method == 'POST':
        user_input=request.form['question']
        print(user_input)

        result = chain({'query':user_input})
        print(f'answer:{result["result"]}')
    
    return jsonify({'response':str(result["result"])})





if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)




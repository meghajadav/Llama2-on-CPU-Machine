DEFAULT_SYSTEM_PROMPT='You are an advance assisstant who summarize the given book'

template = """Use the following pieces of information to answer the user's question.
If you do not know the answer just say you know, do not try to make up an answer.

Context:{context}
Question:{question}

Only return the helpful answer below and nothing else
Helpful answer 
"""
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)

# Define an instruction to guide the embeddings
instruction = "Represent this for current job matching: "

# Sentences to encode with the instruction prepended
sentences = [
    instruction + "Python"
]

docs = ["Current job: Frontend",  "Current job: Python Developer", "Previous job: Python Engineer"]
embeddings = model.encode(sentences)
docs_embeddings = model.encode(docs)

similarities = model.similarity(embeddings, docs_embeddings)
print(similarities)
# [4, 4]
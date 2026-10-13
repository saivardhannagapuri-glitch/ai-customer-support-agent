from retriever import retrieve_documents
from generator import generate_answer


query = "Where can I see my order status?"


# Retrieve relevant information
retrieved_chunks = retrieve_documents(query)

print("\nRetrieved chunks:")
for chunk in retrieved_chunks:
    print(chunk)


# Generate final answer
answer = generate_answer(query, retrieved_chunks)

print("\nFinal answer:")
print(answer)
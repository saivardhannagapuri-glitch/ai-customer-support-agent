import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text, chunk_size=500, chunk_overlap=50):
    """
    Split FAQ text into natural FAQ sections first,
    then split long sections if necessary.
    """

    # Split before each numbered FAQ question
    sections = re.split(r"\n(?=\d+\.)", text.strip())[1:]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []

    for section in sections:
        section_chunks = text_splitter.split_text(section)
        chunks.extend(section_chunks)

    return chunks


# Read the FAQ file
with open("data/faq.txt", "r", encoding="utf-8") as file:
    text = file.read()


# Create chunks
chunks = split_text(text)


print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)
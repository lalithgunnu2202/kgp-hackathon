def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Splits a long text into overlapping chunks.

    text        : full novel text (string)
    chunk_size  : number of characters per chunk
    overlap     : number of overlapping characters

    returns     : list of text chunks
    """

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # move start forward with overlap
        start = end - overlap

        if start < 0:
            start = 0

    return chunks

import pathway as pw


# Define the schema (like a table structure)
class ChunkSchema(pw.Schema):
    chunk_id: int
    text: str


def create_chunk_table(chunks):
    """
    Takes a list of text chunks and creates a Pathway table.
    """

    rows = []
    for i, chunk in enumerate(chunks):
        rows.append((i, chunk))

    # Create Pathway table
    table = pw.debug.table_from_rows(
        rows,
        schema=ChunkSchema
    )

    return table

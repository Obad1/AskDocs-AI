from text_processor import chunk_text


def test_chunk_text_basic():
    text = ("Sentence one. " * 200).strip()
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    # Overlap implies the second chunk should share words with first
    first_words = chunks[0].split()[-10:]
    second_words = chunks[1].split()[:10]
    assert first_words == second_words



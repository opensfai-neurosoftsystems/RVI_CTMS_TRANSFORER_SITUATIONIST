class ByteTokenizer:
    """Minimal reversible UTF-8 byte tokenizer for research prototyping."""
    vocab_size = 256

    def encode(self, text: str):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        b = bytes(int(i) % 256 for i in ids)
        return b.decode("utf-8", errors="replace")

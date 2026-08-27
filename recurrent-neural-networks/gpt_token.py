# pip install tiktoken
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")     # GPT-4 用的编码

tests = ["hello world", "tokenization", "深度学习很有意思",
         "the", " the", "1234567", "ChatGPT"]

for s in tests:
    ids = enc.encode(s)
    pieces = [enc.decode([i]) for i in ids]
    print(f'{s!r:20} {len(ids):>2} tokens  {pieces}')
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
tokens = tokenizer("Hello, don't")['input_ids']
print(tokens)
print(tokenizer.convert_ids_to_tokens(tokens))
print(tokenizer("Hello world in java .")['input_ids'])
print(tokenizer(" Hello world in python .")['input_ids'])
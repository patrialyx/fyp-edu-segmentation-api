from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "She sells"
# if we tokenize it, this becomes:
encoding = tokenizer(text, return_tensors="pt") # this creates a dictionary with keys 'input_ids' etc.
# we add the pos_tag_ids to the dictionary
# pos_tags = [NNP, VNP]
encoding['pos_tag_ids'] = torch.tensor([0, 1])

# next, we can provide this to our modified BertModel:

model = BertModel.from_pretrained("bert-base-uncased")
outputs = model(**encoding)
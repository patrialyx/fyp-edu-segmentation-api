from edu_segmentation.download import download_models
from edu_segmentation.main import ModelFactory, EDUSegmentation

download_models()

# Create a BERT Uncased model
model = ModelFactory.create_model("bert_uncased")

# Create an instance of EDUSegmentation using the model
edu_segmenter = EDUSegmentation(model)

# Segment the text using the conjunction-based segmentation strategy
text = "The food is good, but the service is bad."
granularity = "conjunction_words"
conjunctions = ["and", "but", "however"]
device = 'cpu'

segmented_output = edu_segmenter.run(text, granularity, conjunctions, device)
print(segmented_output)

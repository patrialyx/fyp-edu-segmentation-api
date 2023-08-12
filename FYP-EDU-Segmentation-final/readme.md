Final Year Project on EDU Segmentation:

To improve EDU segmentation performance using Segbot. As Segbot has an encoder-decoder model architecture, we can replace bidirectional GRU encoder with generative pretraining models such as BART and T5. Evaluate the new model using the RST dataset by using few-shot based settings (e.g. 100 examples) to train the model, instead of using the full dataset.

Segbot: <br>
http://138.197.118.157:8000/segbot/ <br>
https://www.ijcai.org/proceedings/2018/0579.pdf

----
### Installation

To use the EDUSegmentation module, follow these steps:

1. Import the `download` module to download all models:<br>
```
from edu_segmentation import download
download.download_models()
```

2. Import the `edu_segmentation` module and its related classes<br>
```
from edu_segmentation.main import EDUSegmentation, ModelFactory, BERTUncasedModel, BERTCasedModel, BARTModel
```

### Usage
The edu_segmentation module provides an easy-to-use interface to perform EDU segmentation using different strategies and models. Follow these steps to use it:

1. Create a segmentation strategy:<br><br>
You can choose between the default segmentation strategy or a conjunction-based segmentation strategy. <br><br>
<strong>Conjunction-based segmentation strategy:</strong> After the text has been EDU-segmented, if there are conjunctions at the start or end of each segment, the conjunctions will be isolated as its own segment.<br><br>
<strong>Default segmentation strategy: </strong> No post-processing occurs after the text has been EDU-segmented <br><br>
```
from edu_segmentation import DefaultSegmentation, ConjunctionSegmentation
```

2. Create a model using the `ModelFactory`. <br><br>
Choose from BERT Uncased, BERT Cased, or BART models.

```
model_type = "bert_uncased"  # or "bert_cased", "bart"
model = ModelFactory.create_model(model_type)
```

3. create an instance of `EDUSegmentation` using the chosen model: <br>
```
edu_segmenter = EDUSegmentation(model)
```

4. Segment the text using the chosen strategy: <br>
```
text = "Your input text here."
granularity = "conjunction_words"  # or "default"
conjunctions = ["and", "but", "however"]  # Customize conjunctions if needed
device = 'cpu'  # Choose your device, e.g., 'cuda:0'

segmented_output = edu_segmenter.run(text, granularity, conjunctions, device)
```


### Example

Here's a simple example demonstrating how to use the edu_segmentation module:

```
from EDUSegmentation import EDUSegmentation, ModelFactory, BERTUncasedModel, ConjunctionSegmentation

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
```
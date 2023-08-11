Final Year Project on EDU Segmentation:

To improve EDU segmentation performance using Segbot. As Segbot has an encoder-decoder model architecture, we can replace bidirectional GRU encoder with generative pretraining models such as BART and T5. Evaluate the new model using the RST dataset by using few-shot based settings (e.g. 100 examples) to train the model, instead of using the full dataset.

Segbot: <br>
http://138.197.118.157:8000/segbot/ <br>
https://www.ijcai.org/proceedings/2018/0579.pdf

----
### Authors
Liu Qingyi, Patria Lim

### How to Use
<li> `from edu_segmentation import download`: use `download.download_models()` to download all models
<li> `from edu_segmentation import main`: use `main.run_segbot(sent, granularity_level="default", model="bart", conjunctions=["and", "but", "however"], device='cuda')` to perform edu-segmentation
<li> Options:
<li> granularity level = ["default", "conjunction_words"]
<li> model = ["bart", "bert_uncased", "bert_cased"]
<li> device = ["cuda", "cpu"]

<li> conjunctions = <user can input their choice of conjunctions into a list>
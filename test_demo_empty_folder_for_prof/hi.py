from edu_segmentation.download import download_models
from edu_segmentation.main import ModelFactory, EDUSegmentation

download_models()


class test1:
    def test1(self):
        # Create a BERT Uncased model
        model = ModelFactory.create_model("bart")

        # Create an instance of EDUSegmentation using the model
        edu_segmenter = EDUSegmentation(model)

        # Segment the text using the conjunction-based segmentation strategy
        text = "The food is good, but the service is bad."
        granularity = "default"
        conjunctions = ["and", "but", "however"]
        device = 'cpu'

        segmented_output = edu_segmenter.run(text, granularity, conjunctions, device)
        print(segmented_output)

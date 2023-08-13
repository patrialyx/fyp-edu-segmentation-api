import unittest
from unittest.mock import Mock
from edu_segmentation.main import (
    ConjunctionSegmentation,
    DefaultSegmentation,
    BERTUncasedModel,
    BERTCasedModel,
    BARTModel,
    ModelFactory,
    EDUSegmentation,
)
from edu_segmentation.download import download_models

class TestConjunctionSegmentation(unittest.TestCase):
    def test_segment(self):
        conjunctions = ["and", "but", "however"]
        segmentation = ConjunctionSegmentation()

        # Test case 1: Only start with a conjunction
        model_output = [
            ("0, 5", "and this is a test."),
            ("6, 13", "But here's another."),
        ]
        expected_output = [['0, 1', 'and'], ['2,  5', 'this is a test'], ['6, 7', 'but'], ['8,  13', "here's another"]]
        self.assertEqual(segmentation.segment(model_output, conjunctions), expected_output)

        # Test case 2: Only end with a conjunction
        model_output = [
            ("0, 4", "This is a test and"),
            ("5, 12", "another one however."),
        ]
        expected_output = [['0, 4', 'this is a test'], ['5,  4', 'and'], ['5, 7', 'another one'], ['8,  12', 'however']]
        self.assertEqual(segmentation.segment(model_output, conjunctions), expected_output)

        # Test case 3: Start and end with a conjunction
        model_output = [
            ("0, 6", "And this is a but"),
            ("7, 15", "however test."),
        ]
        expected_output = [['0, 1', 'and'], ['1, 4', 'this is a'], ['5,  6', 'but'], ['7, 8', 'however'], ['9,  15', 'test']]
        self.assertEqual(segmentation.segment(model_output, conjunctions), expected_output)

        # Test case 4: No conjunctions
        model_output = [
            ("0, 4", "This is a test."),
            ("5, 12", "No conjunctions here."),
        ]
        expected_output = [('0, 4', 'This is a test.'), ('5, 12', 'No conjunctions here.')]
        self.assertEqual(segmentation.segment(model_output, conjunctions), expected_output)

class TestDefaultSegmentation(unittest.TestCase):
    def test_segment(self):
        segmentation = DefaultSegmentation()

        # Test case 1: No change in segmentation
        model_output = [
            ("0, 4", "This is a test."),
            ("5, 12", "No change here."),
        ]
        expected_output = [
            ("0, 4", "This is a test."),
            ("5, 12", "No change here."),
        ]
        self.assertEqual(segmentation.segment(model_output), expected_output)

        # Test case 2: Another example with no change
        model_output = [
            ("0, 5", "Another"),
            ("6, 14", "example."),
        ]
        expected_output = [
            ("0, 5", "Another"),
            ("6, 14", "example."),
        ]
        self.assertEqual(segmentation.segment(model_output), expected_output)


class TestModel(unittest.TestCase):
    def test_run_segbot(self):

        # bert_uncased
        model = ModelFactory.create_model("bert_uncased")
        edu_segmenter = EDUSegmentation(model)
        sentence = "The food is good, but the service is bad."
        device = "cpu"  # or "cuda" if applicable
        expected_output = [['0,4', 'the food is good,'], ['5,10', 'but the service is bad.']]
        self.assertEqual(edu_segmenter.run(sentence, device), expected_output)

        # bert_cased
        model = ModelFactory.create_model("bert_cased")
        edu_segmenter = EDUSegmentation(model)
        sentence = "The food is good, but the service is bad."
        device = "cpu"  # or "cuda" if applicable
        expected_output = [['0,4', 'The food is good,'], ['5,10', 'but the service is bad.']]
        self.assertEqual(edu_segmenter.run(sentence, device), expected_output)

        # bart
        model = ModelFactory.create_model("bart")
        edu_segmenter = EDUSegmentation(model)
        sentence = "The food is good, but the service is bad."
        device = "cpu"  # or "cuda" if applicable
        expected_output = [['0,5', 'The food is good,'], ['5,11', ' but the service is bad.']]
        self.assertEqual(edu_segmenter.run(sentence, device), expected_output)

    

download_models()

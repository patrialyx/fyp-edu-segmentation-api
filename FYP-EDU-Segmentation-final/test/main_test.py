import unittest
from edu_segmentation import main 

class TestRunSegbot(unittest.TestCase):

    # unit tests

    def test_bart(self):
        sent = "This is a test sentence. Another sentence follows."
        result = main.run_segbot(sent)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        
    def test_bert_uncased(self):
        sent = "Testing BERT Uncased model. More text here."
        result = main.run_segbot(sent, model="bert_uncased")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_bert_cased(self):
        sent = "Using BERT Cased model. Additional content."
        result = main.run_segbot(sent, granularity_level="conjunction_words")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    # integration tests
    # bart

    def test_conjunction_granularity_start_with_conjunction_bart(self):
        sent = "The food is good, however the service is bad."
        result = main.run_segbot(sent, granularity_level="conjunction_words")
        expected_output = [['0,5', 'The food is good,'], ['5, 6', 'however'], ['7, 11', 'the service is bad.']]
        assert result == expected_output
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        
    def test_conjunction_granularity_end_with_conjunction_bart(self):
        sent = "The food is good but the service is bad however."
        result = main.run_segbot(sent, granularity_level="conjunction_words")
        expected_output = [['0,4', 'The food is good'], ['4, 5', 'but'], ['5, 9', 'the service is bad'], ['10, 11', 'however.']]
        assert result == expected_output
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_conjunction_granularity_custom_bart(self):
        sent = "The food is good, but the service is bad. The ambience is good though."
        custom_conjunctions = ["though"]
        result = main.run_segbot(sent, granularity_level="conjunction_words", conjunctions=custom_conjunctions)
        expected_output = [['0,5', 'The food is good,'], ['5,12', ' but the service is bad. '], ['12, 16', 'The ambience is good'], ['17, 19', 'though.']]
        assert result == expected_output
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    # bert_uncased

    def test_conjunction_granularity_start_with_conjunction_bert_uncased(self):
        sent = "The food is good, however the service is bad."
        result = main.run_segbot(sent, granularity_level="conjunction_words", model="bert_uncased")
        expected_output = [['0, 9', 'The food is good, however the service is bad.']]
        assert result == expected_output
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        
    def test_conjunction_granularity_end_with_conjunction_bert_uncased(self):
        sent = "The food is good but the service is bad however."
        result = main.run_segbot(sent, granularity_level="conjunction_words", model="bert_uncased")
        expected_output = [['0, 9', 'The food is good but the service is bad'], ['10,  10', 'however.']]
        assert result == expected_output
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_conjunction_granularity_custom_bert_uncased(self):
        sent = "The food is good, but the service is bad. The ambience is good though."
        custom_conjunctions = ["though"]
        result = main.run_segbot(sent, granularity_level="conjunction_words", model="bert_uncased", conjunctions=custom_conjunctions)
        expected_output = [['0, 13', 'The food is good, but the service is bad. The ambience is good'], ['14,  14', 'though.']]
        assert result == expected_output
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    # bert_cased

    def test_conjunction_granularity_start_with_conjunction_bert_cased(self):
        sent = "The food is good, however the service is bad."
        result = main.run_segbot(sent, granularity_level="conjunction_words", model="bert_cased")
        expected_output = [['0,4', 'The food is good,'], ['5, 6', 'however'], ['7, 10', 'the service is bad.']]
        assert result == expected_output
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        
    def test_conjunction_granularity_end_with_conjunction_bert_cased(self):
        sent = "The food is good but the service is bad however."
        result = main.run_segbot(sent, granularity_level="conjunction_words", model="bert_cased")
        expected_output = [['0, 9', 'The food is good but the service is bad'], ['10, 10', 'however.']]
        assert result == expected_output
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_conjunction_granularity_custom_bert_cased(self):
        sent = "The food is good, but the service is bad. The ambience is good though."
        custom_conjunctions = ["though"]
        result = main.run_segbot(sent, granularity_level="conjunction_words", model="bert_cased", conjunctions=custom_conjunctions)
        expected_output = [['0,4', 'The food is good,'], ['5,10', 'but the service is bad.'], ['11, 15', 'The ambience is good'], ['16, 18', 'though.']]
        assert result == expected_output
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)


if __name__ == '__main__':
    unittest.main()
from abc import ABC, abstractmethod
from .BARTTokenClassification.run_segbot_bart import run_segbot_bart
from .BERTTokenClassification.run_bert import run_segbot_bert_cased, run_segbot_bert_uncased
import warnings

class SegmentationStrategy(ABC):
    """
    An abstract base class defining the interface for segmentation strategies.
    """
    @abstractmethod
    def segment(self, model_output):
        """
        Abstract method for segmenting the model's output.

        Args:
            model_output (list): The output of the token classification model.

        Returns:
            list: The segmented output.
        """
        pass

class ConjunctionSegmentation(SegmentationStrategy):
    """
    A segmentation strategy that splits segments at conjunctions.
    """

    def segment(self, model_output, conjunctions):
        """
        Segment the model's output by splitting segments at specified conjunctions.

        Args:
            model_output (list): The output of the token classification model.
            conjunctions (list): A list of conjunction words to split segments at.

        Returns:
            list: The segmented output.
        """
        results = []
        for segment in model_output:
            index_str = segment[0].split(",")
            index_begin = index_str[0]
            index_end = index_str[1]
            word_str = segment[1]
            word_str = word_str.strip()
            word_str = word_str.rstrip(".").lower()
            
            if word_str.startswith(tuple(conjunctions)) and not word_str.endswith(tuple(conjunctions)):
                splitted = word_str.split()
                first_word = splitted[0]
                remaining_words = " ".join(splitted[1:])
                results.append([f'{index_begin}, {int(index_begin)+1}', first_word])
                results.append([f'{int(index_begin)+2}, {index_end}', remaining_words])
            elif word_str.endswith(tuple(conjunctions)) and not word_str.startswith(tuple(conjunctions)):
                splitted = word_str.split()
                remaining_words = " ".join(splitted[:-1])
                last_word = splitted[-1]
                results.append([f'{index_begin}, {int(index_begin)+len(splitted)-1}', remaining_words])
                results.append([f'{int(index_begin)+len(splitted)}, {index_end}', last_word])
            elif word_str.endswith(tuple(conjunctions)) and word_str.startswith(tuple(conjunctions)):
                splitted = word_str.split()
                first_word = splitted[0]
                remaining_words = " ".join(splitted[1:-1])
                last_word = splitted[-1]
                results.append([f'{index_begin}, {int(index_begin)+1}', first_word])
                results.append([f'{int(index_begin)+1}, {int(index_begin)+len(splitted)-1}', remaining_words])
                results.append([f'{int(index_begin)+len(splitted)}, {index_end}', last_word])
            else:
                results.append(segment)
        return results

class DefaultSegmentation(SegmentationStrategy):
    """
    A default segmentation strategy that returns the model's output as is.
    """
    def segment(self, model_output):
        """
        Return the model's output without any segmentation.

        Args:
            model_output (list): The output of the token classification model.

        Returns:
            list: The input model output.
        """
        return model_output

class SegbotModel:
    """
    An abstract base class defining the interface for Segbot models.
    """
    @abstractmethod
    def run_segbot(self, sent, device):
        """
        Abstract method for running the Segbot model.

        Args:
            sent (str): The input sentence.
            device (str): The device to run the model on (e.g., 'cpu', 'cuda').

        Returns:
            list: The model's output.
        """
        pass

class BARTModel(SegbotModel):
    """
    Implementation of a Segbot model using BART architecture.
    """
    def run_segbot(self, sent, device):
        """
        Run the BART-based Segbot model on the input sentence.

        Args:
            sent (str): The input sentence.
            device (str): The device to run the model on (e.g., 'cpu', 'cuda').

        Returns:
            list: The model's output.
        """
        return run_segbot_bart(sent, device)

class BERTUncasedModel(SegbotModel):
    """
    Implementation of a Segbot model using BERT Uncased architecture.
    """
    def run_segbot(self, sent, device):
        """
        Run the BERT-Uncased-based Segbot model on the input sentence.

        Args:
            sent (str): The input sentence.
            device (str): The device to run the model on (e.g., 'cpu', 'cuda').

        Returns:
            list: The model's output.
        """
        return run_segbot_bert_uncased(sent, device)

class BERTCasedModel(SegbotModel):
    """
    Implementation of a Segbot model using BERT Cased architecture.
    """
    def run_segbot(self, sent, device):
        """
        Run the BERT-Cased-based Segbot model on the input sentence.

        Args:
            sent (str): The input sentence.
            device (str): The device to run the model on (e.g., 'cpu', 'cuda').

        Returns:
            list: The model's output.
        """
        return run_segbot_bert_cased(sent, device)

class ModelFactory:
    """
    A factory class responsible for creating SegbotModel instances based on model types.
    """
    @staticmethod
    def create_model(model_type):
        """
        Create and return a SegbotModel instance based on the specified model type.

        Args:
            model_type (str): The model type. Possible values: "bert_uncased", "bert_cased", "bart".

        Returns:
            SegbotModel or str: An instance of the appropriate SegbotModel subclass if the model type is valid,
                                or a string indicating that the model type is invalid.
        """
        if model_type == "bert_uncased":
            return BERTUncasedModel()
        elif model_type == "bert_cased":
            return BERTCasedModel()
        elif model_type == "bart":
            return BARTModel()
        else:
            return "This model does not exist"

class EDUSegmentation:
    """
    A class for performing EDU segmentation using different models and strategies.
    """
    def __init__(self, model):
        """
        Initialize the EDUSegmentation instance with a model.

        Args:
            model (SegbotModel): A Segbot model instance to perform segmentation.
        """
        self.model = model

    def run(self, sent, granularity="default", conjunctions=["and", "but", "however"], device='cpu'):
        """
        Run EDU segmentation on the input sentence using the specified strategy and model.

        Args:
            sent (str): The input sentence.
            granularity (str): The segmentation granularity ('default' or 'conjunction_words').
            conjunctions (list): A list of conjunction words for 'conjunction_words' granularity.
            device (str): The device to run the model on (e.g., 'cpu', 'cuda').

        Returns:
            list: The segmented output.
        """
        warnings.filterwarnings('ignore')
        print('self.model', self.model)
        output = self.model.run_segbot(sent, device)
        if granularity=="default":
            output = DefaultSegmentation().segment(output)
        elif granularity=="conjunction_words":
            output = ConjunctionSegmentation().segment(output, conjunctions)
        return output
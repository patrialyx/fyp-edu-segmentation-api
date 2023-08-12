from abc import ABC, abstractmethod
from .BARTTokenClassification.run_segbot_bart import run_segbot_bart
from .BERTTokenClassification.run_bert import run_segbot_bert_cased, run_segbot_bert_uncased
import warnings

class SegmentationStrategy(ABC):
    @abstractmethod
    def segment(self, model_output):
        pass

class ConjunctionSegmentation(SegmentationStrategy):

    def segment(self, model_output, conjunctions):
        results = []
        for segment in model_output:
            index_str = segment[0].split(",")
            index_begin = index_str[0]
            index_end = index_str[1]
            word_str = segment[1]
            word_str = word_str.strip()
            word_str = word_str.rstrip(".")
            
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
    def segment(self, model_output):
        return model_output

class SegbotModel:
    @abstractmethod
    def run_segbot(self, sent, device):
        pass

class BARTModel(SegbotModel):
    def run_segbot(self, sent, device):
        return run_segbot_bart(sent, device)

class BERTUncasedModel(SegbotModel):
    def run_segbot(self, sent, device):
        return run_segbot_bert_uncased(sent, device)

class BERTCasedModel(SegbotModel):
    def run_segbot(self, sent, device):
        return run_segbot_bert_cased(sent, device)

class ModelFactory:
    @staticmethod
    def create_model(model_type):
        if model_type == "bert_uncased":
            return BERTUncasedModel()
        elif model_type == "bert_cased":
            return BERTCasedModel()
        elif model_type == "bart":
            return BARTModel()
        else:
            return "This model does not exist"

class EDUSegmentation:
    def __init__(self, model):
        self.model = model

    def run(self, sent, granularity="default", conjunctions=["and", "but", "however"], device='cpu'):
        warnings.filterwarnings('ignore')
        output = self.model.run_segbot(sent, device)
        if granularity=="default":
            output = DefaultSegmentation().segment(output)
        elif granularity=="conjunction_words":
            output = ConjunctionSegmentation().segment(output, conjunctions)
        return output



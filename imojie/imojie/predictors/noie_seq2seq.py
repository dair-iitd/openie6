from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

from imojie import bert_utils

import ipdb

@Predictor.register('noie_seq2seq')
class MemSeq2SeqPredictor(Predictor):
    """
    Predictor for sequence to sequence models, including
    :class:`~allennlp.models.encoder_decoder.mem_seq2seq`.
    """

    # def predict(self, source: str) -> JsonDict:
        # return self.predict_json({"source" : source})

    @overrides
    def load_line(self, line: str) -> JsonDict:  # pylint: disable=no-self-use
        json_line = {"source": line}
        return json_line

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)

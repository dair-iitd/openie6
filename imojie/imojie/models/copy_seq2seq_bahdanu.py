import logging
from typing import Dict, Tuple, List, Any, Union
import argparse
import numpy as np
import time
from overrides import overrides
import ipdb
import pdb
import copy

import torch
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell, GRUCell
from torch.nn import LSTM

import allennlp
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Metric, BLEU
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.div_beam_search import DivBeamSearch
from allennlp.nn.cov_beam_search import CoverageBeamSearch

from imojie import bert_utils

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class KeyDict(dict):
    def __missing__(self, key):
        return key

@Model.register("copy_seq2seq_bahdanu")
class CopyNetSeq2Seq(Model):
    """
    This is an implementation of `CopyNet <https://arxiv.org/pdf/1603.06393>`_.
    CopyNet is a sequence-to-sequence encoder-decoder model with a copying mechanism
    that can copy tokens from the source sentence into the target sentence instead of
    generating all target tokens only from the target vocabulary.

    It is very similar to a typical seq2seq model used in neural machine translation
    tasks, for example, except that in addition to providing a "generation" score at each timestep
    for the tokens in the target vocabulary, it also provides a "copy" score for each
    token that appears in the source sentence. In other words, you can think of CopyNet
    as a seq2seq model with a dynamic target vocabulary that changes based on the tokens
    in the source sentence, allowing it to predict tokens that are out-of-vocabulary (OOV)
    with respect to the actual target vocab.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    attention : ``Attention``, required
        This is used to get a dynamic summary of encoder outputs at each timestep
        when producing the "generation" scores for the target vocab.
    beam_size : ``int``, required
        Beam width to use for beam search prediction.
    max_decoding_steps : ``int``, required
        Maximum sequence length of target predictions.
    target_embedding_dim : ``int``, optional (default = 30)
        The size of the embeddings for the target vocabulary.
    copy_token : ``str``, optional (default = '@COPY@')
        The token used to indicate that a target token was copied from the source.
        If this token is not already in your target vocabulary, it will be added.
    source_namespace : ``str``, optional (default = 'source_tokens')
        The namespace for the source vocabulary.
    target_namespace : ``str``, optional (default = 'target_tokens')
        The namespace for the target vocabulary.
    tensor_based_metric : ``Metric``, optional (default = BLEU)
        A metric to track on validation data that takes raw tensors when its called.
        This metric must accept two arguments when called: a batched tensor
        of predicted token indices, and a batched tensor of gold token indices.
    token_based_metric : ``Metric``, optional (default = None)
        A metric to track on validation data that takes lists of lists of tokens
        as input. This metric must accept two arguments when called, both
        of type `List[List[str]]`. The first is a predicted sequence for each item
        in the batch and the second is a gold sequence for each item in the batch.
    initializer : ``InitializerApplicator``, optional
        An initialization strategy for the model weights.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 beam_size: int,
                 max_decoding_steps: int,
                 target_embedding_dim: int = 30,
                 decoder_layers: int = 3,
                 copy_token: str = "@COPY@",
                 source_namespace: str = "source_tokens",
                 target_namespace: str = "target_tokens",
                 tensor_based_metric: Metric = None,
                 token_based_metric: Metric = None,
                 lambda_diversity: int = 5,
                 beam_search_type: str="beam_search",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 bert: bool = False,
                 append: bool = False,
                 coverage: bool = False,
                 max_extractions: int = -1,
                 decoder_config: str='',
                 decoder_type: str='lstm',
                 teacher_forcing: bool = True) -> None:
        super().__init__(vocab)
        self._decoder_type = decoder_type
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace
        self._bert = bert
        self._append = append
        self._coverage = coverage
        global START_SYMBOL,END_SYMBOL
        # Needed for lstm_append - could be done for all - but issue is lstm_single is trained with old START, END symbols
        if self._append: 
            START_SYMBOL,END_SYMBOL = bert_utils.init_globals()
            
        if self._bert:
            START_SYMBOL,END_SYMBOL = bert_utils.init_globals()
            self._target_vocab_size = 28996 
            self.token_mapping = bert_utils.mapping
        else:
            if self.vocab.get_token_index(copy_token, self._target_namespace) == 1:
                # +1 is for the copy token, which is added later on in initialize()
                # self._target_vocab_size = self.vocab.get_vocab_size(self._target_namespace) + 1
                self._target_vocab_size = self.vocab.get_vocab_size(self._target_namespace) + 1
            else:
                # copy_token already in dictionary, as an existing vocabulary is being loaded
                self._target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)
            self.token_mapping = KeyDict()
        # Encoding modules.
        self._source_embedder = source_embedder
        self._encoder = encoder
        self._attention = attention
        self._beam_size = beam_size
        self._max_decoding_steps = max_decoding_steps
        self._target_embedding_dim = target_embedding_dim
        self._decoder_layers = decoder_layers
        self._copy_token = copy_token
        self._tensor_based_metric = tensor_based_metric
        self._token_based_metric = token_based_metric
        self._lambda_diversity = lambda_diversity
        self._beam_search_type = beam_search_type
        self._initializer = initializer
        self._max_extractions = max_extractions
        self._decoder_config = decoder_config
        self._decoder_type = decoder_type
        self._teacher_forcing = teacher_forcing

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        # We arbitrarily set the decoder's input dimension to be the same as the output dimension.
        self.encoder_output_dim = self._encoder.get_output_dim()
        self.decoder_output_dim = self.encoder_output_dim

        if self._coverage:
            coverage_dim = self.decoder_output_dim
            self.coverage_dim = coverage_dim
            self._coverage_cell = GRUCell(self.decoder_output_dim + self.encoder_output_dim + 1, coverage_dim)

        self.decoder_input_dim = self.decoder_output_dim
        # The decoder input will be a function of the embedding of the previous predicted token,
        # an attended encoder hidden state called the "attentive read", and another
        # weighted sum of the encoder hidden state called the "selective read".
        # While the weights for the attentive read are calculated by an `Attention` module,
        # the weights for the selective read are simply the predicted probabilities
        # corresponding to each token in the source sentence that matches the target
        # token from the previous timestep.
        self._target_embedder = Embedding(self._target_vocab_size, self._target_embedding_dim)
        self._input_projection_layer = Linear(
                self._target_embedding_dim + self.encoder_output_dim * 2,
                self.decoder_input_dim)

        # We then run the projected decoder input through an LSTM cell to produce
        # the next hidden state.
        if self._decoder_type == 'lstm':
            self._decoder_cell = LSTM(self.decoder_input_dim, self.decoder_output_dim, num_layers=self._decoder_layers, batch_first=True)
        elif self._decoder_type == 'transformer':
            decoder_layer = torch.nn.TransformerDecoderLayer(d_model=256, nhead=4)
            self._decoder_cell = torch.nn.TransformerDecoder(decoder_layer, num_layers=1)

        # We create a "generation" score for each token in the target vocab
        # with a linear projection of the decoder hidden state.
        self._output_generation_layer = Linear(self.decoder_output_dim, self._target_vocab_size)

        # We create a "copying" score for each source token by applying a non-linearity
        # (tanh) to a linear projection of the encoded hidden state for that token,
        # and then taking the dot product of the result with the decoder hidden state.
        self._output_copying_layer = Linear(self.encoder_output_dim, self.decoder_output_dim)

        # No relation to the initialized in the next instruction
        self._initializer(self)

        ## Initialize all the required network parameters after the first call to forward
        ## Bert requires this as vocabulary for BERT is filled up only after the first call to forward
        ## Required for all the indexes
        self._initialized = False 



    def _tokens_to_ids(self, tokens) :
        ids: Dict[str, int] = {}
        out: List[int] = []
        for token in tokens:
            out.append(ids.setdefault(token, len(ids)))
        return out

    def _remove_padding(self, batch_tokens, index=-1):
        if(index == -1):
            index = self._pad_index

        unpadded_batch = []
        for tokens in batch_tokens:
            tokens = tokens.tolist()
            tokens = tokens + [index]
            unpadded_batch.append(tokens[:tokens.index(index)])
        return unpadded_batch

    def _leave_one(self, tokens, index):
        # Remove all 'index' tokens except one
        # In-order to allow extraction of separate triples
        # does not operate on batches
        unpadded_tokens = []
        prev_token = -1
        for token in tokens:
            if prev_token == index and token == index:
                continue
            unpadded_tokens.append(token)
            prev_token = token
        return unpadded_tokens

    def _insert_padding(self, batch_tokens, device, max_length=-1):
        ## For append batching

        if(max_length == -1):
            max_length = max([len(bt) for bt in batch_tokens])
        padded_batch = []
        for tokens in batch_tokens:
            padded_batch.append(tokens + [self._pad_index] * (max_length - len(tokens)))
        return torch.tensor(padded_batch).to(device)       

    def initialize(self):
        ## Initilaization which require the vocabulary to be built
        ## If module requires to be placed on GPU, should be created in __init__ function itself

        self._src_start_index = self.vocab.get_token_index(START_SYMBOL, self._source_namespace)
        self._src_end_index = self.vocab.get_token_index(END_SYMBOL, self._source_namespace)
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        if self._bert:
            # source and target vocabulary are same - no need to use instance dictionary to get the word (it is unk there as well)
            # therefore map oov_token to a random token - not supposed to be used at all in case of bert
            self.vocab._oov_token = '[unused99]' 
            self.vocab._padding_token = '[PAD]'
        self._oov_index = self.vocab.get_token_index(self.vocab._oov_token, self._target_namespace)  # pylint: disable=protected-access
        self._pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)  # pylint: disable=protected-access

        self._copy_index = self.vocab.add_token_to_namespace(self.token_mapping[self._copy_token], self._target_namespace)
        self._eoe_index = self.vocab.get_token_index(self.token_mapping['EOE'], self._target_namespace)

        self.start_arg1 = self.vocab.get_token_index(self.token_mapping['<arg1>'], self._target_namespace)
        self.end_arg1 = self.vocab.get_token_index(self.token_mapping['</arg1>'], self._target_namespace)
        self.start_arg2 = self.vocab.get_token_index(self.token_mapping['<arg2>'], self._target_namespace)
        self.end_arg2 = self.vocab.get_token_index(self.token_mapping['</arg2>'], self._target_namespace)
        self.start_rel = self.vocab.get_token_index(self.token_mapping['<rel>'], self._target_namespace)
        self.end_rel = self.vocab.get_token_index(self.token_mapping['</rel>'], self._target_namespace)

        # At prediction time, we'll use a beam search to find the best target sequence.
        if self._beam_search_type == 'beam_search':
            self._beam_search = BeamSearch(self._end_index, max_steps=self._max_decoding_steps, beam_size=self._beam_size)
        elif self._beam_search_type == 'div_beam_search':
            self._beam_search = DivBeamSearch(self._end_index, max_steps=self._max_decoding_steps, beam_size=self._beam_size, lambda_diversity=self._lambda_diversity, \
                ignore_indices=[self.start_arg1, self.start_arg2, self.start_rel, self.end_arg1, self.end_arg2, self.end_rel])
        elif self._beam_search_type == 'cov_beam_search':
            self._beam_search = CoverageBeamSearch(self._end_index, max_steps=self._max_decoding_steps, beam_size=self._beam_size)

    def forward_append(self, source_tokens, source_token_ids, source_to_target, metadata, target_tokens, target_token_ids, optimizer):
        if target_tokens:
            # output_dict = self.train_append(source_tokens, source_token_ids, source_to_target, metadata, target_tokens, target_token_ids, optimizer=optimizer)
            if self.training:
                output_dict = self.train_append(source_tokens, source_token_ids, source_to_target, metadata, target_tokens, target_token_ids, optimizer=optimizer)
            else:
                output_dict = {}

                state = self._encode(source_tokens)
                state["source_token_ids"] = source_token_ids
                state["source_to_target"] = source_to_target
                state = self._init_decoder_state(state)
                target_tokens['tokens'] = target_tokens['tokens'][:,:1,:].squeeze(1)
                target_token_ids = target_token_ids[:,:1,:].squeeze(1)
                output_dict = self._forward_loss(target_tokens, target_token_ids, state)
        else:
            output_dict = {}
            predictions = self.test_append(source_tokens, source_token_ids, source_to_target)
            output_dict.update(predictions)
        output_dict["metadata"] = metadata
        return output_dict

    def forward_single(self, source_tokens, source_token_ids, source_to_target, metadata, target_tokens, target_token_ids, optimizer):
        state = self._encode(source_tokens)
        state["source_token_ids"] = source_token_ids
        state["source_to_target"] = source_to_target
        if self._coverage:
            state["m_t"] = None

        if target_tokens:
            state = self._init_decoder_state(state)
            output_dict = self._forward_loss(target_tokens, target_token_ids, state)
        else:
            output_dict = {}
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)        
            output_dict.update(predictions)

        output_dict["metadata"] = metadata
         
        return output_dict


    global gm
    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                source_token_ids: torch.Tensor,
                source_to_target: torch.Tensor,
                metadata: List[Dict[str, Any]],
                target_tokens: Dict[str, torch.LongTensor] = None,
                target_token_ids: torch.Tensor = None,
                optimizer = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``, required
            Source tokens with source vocabulary 

            The output of `TextField.as_array()` applied on the source `TextField`. This will be
            passed through a `TextFieldEmbedder` and then through an encoder.
        source_token_ids : ``torch.Tensor``, required
            Source tokens with example vocabulary

            Tensor containing IDs that indicate which source tokens match each other.
            Has shape: `(batch_size, trimmed_source_length)`.
        source_to_target : ``torch.Tensor``, required
            Source tokens with target vocabulary

            Tensor containing vocab index of each source token with respect to the
            target vocab namespace. Shape: `(batch_size, trimmed_source_length)`.
        metadata : ``List[Dict[str, Any]]``, required
            Metadata field that contains the original source tokens with key 'source_tokens'
            and any other meta fields. When 'target_tokens' is also passed, the metadata
            should also contain the original target tokens with key 'target_tokens'.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
            Target tokens with example vocabulary

            Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
            target tokens are also represented as a `TextField` which must contain a "tokens"
            key that uses single ids.
        target_token_ids : ``torch.Tensor``, optional (default = None)
            Target tokens with sentence vocabulary

            A tensor of shape `(batch_size, target_sequence_length)` which indicates which
            tokens in the target sequence match tokens in the source sequence.

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        if not self._initialized:
            # initialization which require the vocabulary to be built
            # not built at __init__ function
            # so need to introduce another function
            self.initialize()
            self._initialized = True 

        if self.training and not self._decoder_type == 'transformer': # useful when ngpus > 1 # produces undesirable warnings during testing - but can't avoid it
            self._decoder_cell.flatten_parameters()

        if self._append:
            output_dict = self.forward_append(source_tokens, source_token_ids, source_to_target, metadata, target_tokens, target_token_ids, optimizer)
        else:
            output_dict = self.forward_single(source_tokens, source_token_ids, source_to_target, metadata, target_tokens, target_token_ids, optimizer)

        if metadata[0]['validation']: 
            predicted_tokens = self._get_predicted_tokens(output_dict["predictions"],
                                                            metadata,
                                                            n_best=5)
            predicted_confidences = output_dict['predicted_log_probs']
            self._token_based_metric(predicted_tokens, predicted_confidences, # type: ignore
                                        [x["example_ids"] for x in metadata],
                                        self._append,
                                        self._coverage)

        return output_dict

    def test_append(self, source_tokens, source_token_ids, source_to_target):
        # (batch_size, source_tokens_length)
        append_tokens = source_tokens['tokens']
        source_tokens_cpu = source_tokens['tokens'].cpu()
        # unpadded_append_tokens
        unp_append_tokens = self._remove_padding(append_tokens)
        batch_size, source_tokens_length = source_tokens['tokens'].shape
        device = source_tokens['tokens'].device
        predictions, predictions_log_probs = [], []
        stop = False
        num_extractions = 0
        completed = np.array([0]*batch_size)
        other_predictions = []
        other_log_probs = []
    
        while not stop:
            append_token_ids = []
            append_to_target = []
            for bs in range(batch_size):
                # BERT can handle maximum of 512 sequence - But seen errors with 511
                # Goes this high only in exceptional cases
                unp_append_tokens[bs] = unp_append_tokens[bs][:510]
                append_token_ids.append(self._tokens_to_ids(unp_append_tokens[bs][1:-1]))
                append_to_target.append(unp_append_tokens[bs][1:-1])

            append_token_ids = self._insert_padding(append_token_ids, device)
            append_to_target = self._insert_padding(append_to_target, device)

            append_tokens = self._insert_padding(unp_append_tokens, device)
            append_tokens_cpu = append_tokens.cpu()
            state = self._encode({'tokens': append_tokens})

            state["source_token_ids"] = append_token_ids
            state["source_to_target"] = append_to_target
            state["source_tokens"] = append_tokens

            state = self._init_decoder_state(state)
            # source_to_target_cpu = state['source_to_target'].cpu()
            predictions_dict = self._forward_beam_search(state)
            # (batch_size, max_predicted_sequence_length)
            prediction = predictions_dict['predictions'][:, 0, :] # Get the best prediction
            # prediction = predictions_dict['predictions'][:, 0, :].unsqueeze(1)
            prediction_log_prob = predictions_dict['predicted_log_probs'][:, 0].unsqueeze(dim=1)

            batch_size = prediction.shape[0]
            end_column = torch.tensor([self._end_index]*batch_size).unsqueeze(1).to(device) 
            # Add a [SEP] token at the end - useful to extract the different extractions
            prediction = torch.cat((prediction, end_column), dim=1)
            predictions.append(prediction)
            predictions_log_probs.append(prediction_log_prob)

            completed = completed | ((prediction == self._eoe_index).sum(dim=1) != 0).to('cpu').numpy().astype(int)
            prediction_cpu = prediction.cpu()

            if sum(completed) == len(completed):
                stop = True
            unp_prediction_tokens = self._remove_padding(prediction, index=self._end_index)

            # (batch_size, num_appended_tokens)
            # START token from append_tokens and END token from prediction is needed
            new_unp_append_tokens = []
            for bs in range(batch_size):
                unp_prediction_tokens_bs = np.array(unp_prediction_tokens[bs])
                if not self._bert:
                    unp_prediction_tokens_bs[unp_prediction_tokens_bs >= len(self.vocab._token_to_index['target_tokens'])] = self._oov_index
                new_unp_append_tokens.append(unp_append_tokens[bs][:-1] + unp_prediction_tokens_bs.tolist())
            unp_append_tokens = new_unp_append_tokens

            num_extractions += 1
            if(num_extractions >= self._max_extractions):
                stop = True

        predictions = {'predictions': torch.cat(predictions, dim=1).unsqueeze(dim=1), 'predicted_log_probs': torch.cat(predictions_log_probs, dim=1)}
        return predictions 

    def train_append(self, source_tokens, source_token_ids, source_to_target, metadata, target_tokens, target_token_ids, optimizer):
        # (batch_size, source_tokens_length)
        append_tokens = source_tokens['tokens']
        unp_append_tokens = self._remove_padding(append_tokens)
        batch_size, num_extractions, target_tokens_length = target_tokens['tokens'].shape
        device = source_tokens['tokens'].device
        probs = list()

        for extraction_num in range(num_extractions):
            current_target = target_tokens['tokens'][:, extraction_num, :]
            unp_current_target = self._remove_padding(current_target)
        
            append_token_ids = []
            append_to_target = []
            current_target_token_ids = []
            for bs in range(batch_size):
                # BERT can handle maximum of 512 sequence - But seen errors with 511
                # Goes this high only in exceptional cases
                unp_append_tokens[bs] = unp_append_tokens[bs][:510]
                append_and_target_ids = self._tokens_to_ids(unp_append_tokens[bs][1:-1] + unp_current_target[bs])
                append_token_ids.append(append_and_target_ids[:len(unp_append_tokens[bs])-2])
                append_to_target.append(unp_append_tokens[bs][1:-1])
                current_target_token_ids.append(append_and_target_ids[len(unp_append_tokens[bs])-2:])

            append_token_ids = self._insert_padding(append_token_ids, device)
            current_target_token_ids = self._insert_padding(current_target_token_ids, device, max_length=target_tokens_length)
            append_to_target = self._insert_padding(append_to_target, device)

            append_tokens = self._insert_padding(unp_append_tokens, device)
            state = self._encode({'tokens': append_tokens})
            state["source_token_ids"] = append_token_ids
            state["source_to_target"] = append_to_target

            state = self._init_decoder_state(state)
            loss_dict = self._forward_loss({'tokens': current_target}, current_target_token_ids, state)
            probs.append(loss_dict['probs'])
            
            if metadata[0]['gradients'] and extraction_num != num_extractions-1:
                loss_dict['loss'].backward()
            
            if self._teacher_forcing:
                prediction = current_target # Has both the START and END tokens
                prediction = prediction[:,1:] # Remove the START token
                unp_prediction_tokens = self._remove_padding(prediction)
            else:
                with torch.no_grad():
                    state = self._init_decoder_state(state)
                    predictions_dict = self._forward_beam_search(state)
                    # (batch_size, max_predicted_sequence_length)
                    prediction = predictions_dict['predictions'][:, 0, :] # Has the END token but not the start token
                    unp_prediction_tokens = self._remove_padding(prediction, index=self._end_index)

            # (batch_size, num_appended_tokens)
            # START token from append_tokens and END token from prediction is needed
            new_unp_append_tokens = []
            for bs in range(batch_size):
                new_unp_append_tokens.append(unp_append_tokens[bs][:-1] + unp_prediction_tokens[bs])
            unp_append_tokens = new_unp_append_tokens
        
        probs = torch.cat([torch.tensor(p).unsqueeze(1) for p in probs], dim=1)
        output_dict = {'loss': loss_dict['loss'], 'probs': probs}
        return output_dict


    def _gather_extended_gold_tokens(self,
                                     target_tokens: torch.Tensor,
                                     source_token_ids: torch.Tensor,
                                     target_token_ids: torch.Tensor) -> torch.LongTensor:
        """
        Modify the gold target tokens relative to the extended vocabulary.

        For gold targets that are OOV but were copied from the source, the OOV index
        will be changed to the index of the first occurence in the source sentence,
        offset by the size of the target vocabulary.

        Parameters
        ----------
        target_tokens : ``torch.Tensor``
            Shape: `(batch_size, target_sequence_length)`.
        source_token_ids : ``torch.Tensor``
            Shape: `(batch_size, trimmed_source_length)`.
        target_token_ids : ``torch.Tensor``
            Shape: `(batch_size, target_sequence_length)`.

        Returns
        -------
        torch.Tensor
            Modified `target_tokens` with OOV indices replaced by offset index
            of first match in source sentence.
        """
        batch_size, target_sequence_length = target_tokens.size()
        trimmed_source_length = source_token_ids.size(1)
        # Only change indices for tokens that were OOV in target vocab but copied from source.
        # shape: (batch_size, target_sequence_length)
        oov = (target_tokens == self._oov_index)
        # shape: (batch_size, target_sequence_length, trimmed_source_length)
        expanded_source_token_ids = source_token_ids\
            .unsqueeze(1)\
            .expand(batch_size, target_sequence_length, trimmed_source_length)
        # shape: (batch_size, target_sequence_length, trimmed_source_length)
        expanded_target_token_ids = target_token_ids\
            .unsqueeze(-1)\
            .expand(batch_size, target_sequence_length, trimmed_source_length)
        # shape: (batch_size, target_sequence_length, trimmed_source_length)
        matches = (expanded_source_token_ids == expanded_target_token_ids)
        # shape: (batch_size, target_sequence_length)
        copied = (matches.sum(-1) > 0)
        # shape: (batch_size, target_sequence_length)
        mask = (oov & copied).long()
        # shape: (batch_size, target_sequence_length)
        first_match = ((matches.cumsum(-1) == 1) * matches).argmax(-1)
        # shape: (batch_size, target_sequence_length)
        new_target_tokens = target_tokens * (1 - mask) + (first_match.long() + self._target_vocab_size) * mask
        return new_target_tokens

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Initialize the encoded state to be passed to the first decoding time step.
        """
        batch_size, _ = state["source_mask"].size()

        # shape: (batch_size, encoder_output_dim)
        if not isinstance(self._encoder, allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper.PytorchSeq2SeqWrapper):## Assuming only LSTM as the Pytorch wrapper we will us 
            ## Taking the intial hidden state corresponding to the CLS token
            final_encoder_output = state["encoder_outputs"][:, 0, :]
        else:
            # Initialize the decoder hidden state with the final output of the encoder,
            # and the decoder context with zeros.
            final_encoder_output = util.get_final_encoder_states(
                    state["encoder_outputs"],
                    state["source_mask"],
                    self._encoder.is_bidirectional())

        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["encoder_outputs"].new_zeros(batch_size, self.decoder_output_dim)

        state["decoder_hidden_all"] = final_encoder_output.unsqueeze(0).repeat(self._decoder_layers, 1, 1)
        state["decoder_context_all"] = state["encoder_outputs"].new_zeros(batch_size, self._decoder_layers, self.decoder_output_dim)

        state["decoder_hidden_all"] = state["decoder_hidden_all"].transpose(0,1).contiguous().view(-1, self._decoder_layers*self.decoder_output_dim)
        state["decoder_context_all"] = state["decoder_context_all"].transpose(0,1).contiguous().view(-1, self._decoder_layers*self.decoder_output_dim)

        return state

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode source input sentences.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        try:
            source_tokens_cpu = source_tokens['tokens'].cpu()
            embedded_input = self._source_embedder(source_tokens)
            embedded_input_cpu = embedded_input.cpu()
        except:
            ipdb.set_trace()
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _decoder_step(self,
                      last_predictions: torch.Tensor,
                      selective_weights: torch.Tensor,
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        batch_size, num_inp_words, _ = state['encoder_outputs'].shape
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs_mask = state["source_mask"].float()
        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions) 
        # use embedded_input.cpu() for debugging index out of range errors

        if self._decoder_type == 'lstm':
            if self._coverage :
                if type(state['m_t']) == type(None):
                    # (batch_size, num_inp_words, coverage_dim)
                    state['m_t'] = embedded_input.new_zeros(batch_size, num_inp_words, self.coverage_dim)
                # (batch_size, num_inp_words, encoder_output_dim + coverage_dim)
                enc_hidden_cov = torch.cat((state['encoder_outputs'], state['m_t']), dim=-1)

                attentive_weights = self._attention(
                        state["decoder_hidden"], enc_hidden_cov, encoder_outputs_mask)

                # (batch_size, decoder_hidden_dim) --> (batch_size, num_inp_words, decoder_hidden_dim)
                exp_dec_hidden = state['decoder_hidden'].unsqueeze(1).repeat(1, num_inp_words, 1)
                # (batch_size, num_inp_words, decoder_hidden_dim + encoder_hidden_dim + 1)
                comb_hidden = torch.cat((exp_dec_hidden, state['encoder_outputs'], attentive_weights.unsqueeze(2)), dim=-1)

                # m_t: (batch_size * num_inp_words, coverage_dim)
                m_t = self._coverage_cell(comb_hidden.view(batch_size * num_inp_words, -1), state['m_t'].contiguous().view(batch_size * num_inp_words, -1))
                # state['m_t']: (batch_size, num_inp_words, coverage_dim)
                state['m_t'] = m_t.view(batch_size, num_inp_words, -1)
            else:
                # shape: (group_size, max_input_sequence_length)
                attentive_weights = self._attention(
                        state["decoder_hidden"], state["encoder_outputs"], encoder_outputs_mask)
        
            # shape: (group_size, encoder_output_dim)
            attentive_read = util.weighted_sum(state["encoder_outputs"], attentive_weights)
            # shape: (group_size, encoder_output_dim)
            selective_read = util.weighted_sum(state["encoder_outputs"][:, 1:-1], selective_weights)
            # shape: (group_size, target_embedding_dim + encoder_output_dim * 2)
            decoder_input = torch.cat((embedded_input, attentive_read, selective_read), -1)
            # shape: (group_size, decoder_input_dim)
            projected_decoder_input = self._input_projection_layer(decoder_input)

            # (batch_size, decoder_layers * output_dim) --> (decoder_layers, batch_size, output_dim)
            state["decoder_hidden_all"] = state["decoder_hidden_all"].view(-1, self._decoder_layers, self.decoder_output_dim).transpose(0,1).contiguous()
            state["decoder_context_all"] = state["decoder_context_all"].view(-1, self._decoder_layers, self.decoder_output_dim).transpose(0,1).contiguous()

            _, (state["decoder_hidden_all"], state["decoder_context_all"]) = self._decoder_cell(
                    projected_decoder_input.unsqueeze(1),
                    (state["decoder_hidden_all"], state["decoder_context_all"]))

            state["decoder_hidden"], state["decoder_context"] = state["decoder_hidden_all"][-1], state["decoder_context_all"][-1]
            # (decoder_layers, batch_size, output_dim) --> (batch_size, decoder_layers * output_dim)
            state["decoder_hidden_all"] = state["decoder_hidden_all"].transpose(0,1).contiguous().view(-1, self._decoder_layers*self.decoder_output_dim)
            state["decoder_context_all"] = state["decoder_context_all"].transpose(0,1).contiguous().view(-1, self._decoder_layers*self.decoder_output_dim)
        elif self._decoder_type == 'transformer':
            if "inputs_so_far" not in state:
                state["inputs_so_far"] = embedded_input.unsqueeze(1)
            else:
                state["inputs_so_far"] = torch.cat((state["inputs_so_far"], embedded_input.unsqueeze(1)), dim=1)
            # transformer accepts sequence-first
            outputs = self._decoder_cell(state["inputs_so_far"].transpose(0,1), state["encoder_outputs"].transpose(0,1))
            # consider only the hidden state corresponding to the last token
            # state["decoder_hidden"] = outputs[-1,:,:]
            state["decoder_hidden"] = outputs.transpose(0,1)[:,-1,:]
        return state

    def _get_generation_scores(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._output_generation_layer(state["decoder_hidden"])

    def _get_copy_scores(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        # shape: (batch_size, max_input_sequence_length - 2, encoder_output_dim)
        trimmed_encoder_outputs = state["encoder_outputs"][:, 1:-1]
        # shape: (batch_size, max_input_sequence_length - 2, decoder_output_dim)
        copy_projection = self._output_copying_layer(trimmed_encoder_outputs)
        # shape: (batch_size, max_input_sequence_length - 2, decoder_output_dim)
        copy_projection = torch.tanh(copy_projection)
        # shape: (batch_size, max_input_sequence_length - 2)
        copy_scores = copy_projection.bmm(state["decoder_hidden"].unsqueeze(-1)).squeeze(-1)
        return copy_scores

    def _get_ll_contrib(self,
                        generation_scores: torch.Tensor,
                        generation_scores_mask: torch.Tensor,
                        copy_scores: torch.Tensor,
                        target_tokens: torch.Tensor,
                        target_to_source: torch.Tensor,
                        copy_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the log-likelihood contribution from a single timestep.

        Parameters
        ----------
        generation_scores : ``torch.Tensor``
            Shape: `(batch_size, target_vocab_size)`
        generation_scores_mask : ``torch.Tensor``
            Shape: `(batch_size, target_vocab_size)`. This is just a tensor of 1's.
        copy_scores : ``torch.Tensor``
            Shape: `(batch_size, trimmed_source_length)`
        target_tokens : ``torch.Tensor``
            Shape: `(batch_size,)`
        target_to_source : ``torch.Tensor``
            Shape: `(batch_size, trimmed_source_length)`
        copy_mask : ``torch.Tensor``
            Shape: `(batch_size, trimmed_source_length)`

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Shape: `(batch_size,), (batch_size, max_input_sequence_length)`
        """
        _, target_size = generation_scores.size()

        # The point of this mask is to just mask out all source token scores
        # that just represent padding. We apply the mask to the concatenation
        # of the generation scores and the copy scores to normalize the scores
        # correctly during the softmax.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        mask = torch.cat((generation_scores_mask, copy_mask), dim=-1)
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        all_scores = torch.cat((generation_scores, copy_scores), dim=-1)
        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        log_probs = util.masked_log_softmax(all_scores, mask)
        # Calculate the log probability (`copy_log_probs`) for each token in the source sentence
        # that matches the current target token. We use the sum of these copy probabilities
        # for matching tokens in the source sentence to get the total probability
        # for the target token. We also need to normalize the individual copy probabilities
        # to create `selective_weights`, which are used in the next timestep to create
        # a selective read state.
        # shape: (batch_size, trimmed_source_length)
        copy_log_probs = log_probs[:, target_size:] + (target_to_source.float() + 1e-45).log()
        # Since `log_probs[:, target_size]` gives us the raw copy log probabilities,
        # we use a non-log softmax to get the normalized non-log copy probabilities.
        selective_weights = util.masked_softmax(log_probs[:, target_size:], target_to_source)
        # This mask ensures that item in the batch has a non-zero generation probabilities
        # for this timestep only when the gold target token is not OOV or there are no
        # matching tokens in the source sentence.
        # shape: (batch_size, 1)
        gen_mask = ((target_tokens != self._oov_index) | (target_to_source.sum(-1) == 0)).float()
        log_gen_mask = (gen_mask + 1e-45).log().unsqueeze(-1)
        # Now we get the generation score for the gold target token.
        # shape: (batch_size, 1)
        generation_log_probs = log_probs.gather(1, target_tokens.unsqueeze(1)) + log_gen_mask
        # ... and add the copy score to get the step log likelihood.
        # shape: (batch_size, 1 + trimmed_source_length)
        combined_gen_and_copy = torch.cat((generation_log_probs, copy_log_probs), dim=-1)
        # shape: (batch_size,)
        step_log_likelihood = util.logsumexp(combined_gen_and_copy)

        return step_log_likelihood, selective_weights

    def _forward_loss(self,
                      target_tokens: Dict[str, torch.LongTensor],
                      target_token_ids: torch.Tensor,
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate the loss against gold targets.
        """
        batch_size, target_sequence_length = target_tokens["tokens"].size()

        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        num_decoding_steps = target_sequence_length - 1
        # We use this to fill in the copy index when the previous input was copied.
        # shape: (batch_size,)
        copy_input_choices = source_mask.new_full((batch_size,), fill_value=self._copy_index)
        # shape: (batch_size, trimmed_source_length)
        copy_mask = source_mask[:, 1:-1].float()
        # We need to keep track of the probabilities assigned to tokens in the source
        # sentence that were copied during the previous timestep, since we use
        # those probabilities as weights when calculating the "selective read".
        # shape: (batch_size, trimmed_source_length)
        selective_weights = state["decoder_hidden"].new_zeros(copy_mask.size())

        # Indicates which tokens in the source sentence match the current target token.
        # shape: (batch_size, trimmed_source_length)
        target_to_source = state["source_token_ids"].new_zeros(copy_mask.size())

        # This is just a tensor of ones which we use repeatedly in `self._get_ll_contrib`,
        # so we create it once here to avoid doing it over-and-over.
        generation_scores_mask = state["decoder_hidden"].new_full((batch_size, self._target_vocab_size),
                                                                  fill_value=1.0)

        step_log_likelihoods = []
        for timestep in range(num_decoding_steps):
            # shape: (batch_size,)
            input_choices = target_tokens["tokens"][:, timestep]
            # If the previous target token was copied, we use the special copy token.
            # But the end target token will always be THE end token, so we know
            # it was not copied.
            if timestep < num_decoding_steps - 1:
                # Get mask tensor indicating which instances were copied.
                # shape: (batch_size,)
                copied = ((input_choices == self._oov_index) &
                          (target_to_source.sum(-1) > 0)).long()
                # shape: (batch_size,)
                input_choices = input_choices * (1 - copied) + copy_input_choices * copied
                # shape: (batch_size, trimmed_source_length)
                target_to_source = state["source_token_ids"] == target_token_ids[:, timestep+1].unsqueeze(-1)
            # Update the decoder state by taking a step through the RNN.
            state = self._decoder_step(input_choices, selective_weights, state)
            # Get generation scores for each token in the target vocab.
            # shape: (batch_size, target_vocab_size)
            generation_scores = self._get_generation_scores(state)
            # Get copy scores for each token in the source sentence, excluding the start
            # and end tokens.
            # shape: (batch_size, trimmed_source_length)
            copy_scores = self._get_copy_scores(state)
            # shape: (batch_size,)
            step_target_tokens = target_tokens["tokens"][:, timestep + 1]
            step_log_likelihood, selective_weights = self._get_ll_contrib(
                    generation_scores,
                    generation_scores_mask,
                    copy_scores,
                    step_target_tokens,
                    target_to_source,
                    copy_mask)
            step_log_likelihoods.append(step_log_likelihood.unsqueeze(1))

        # Gather step log-likelihoods.
        # shape: (batch_size, num_decoding_steps = target_sequence_length - 1)
        log_likelihoods = torch.cat(step_log_likelihoods, 1)
        # Get target mask to exclude likelihood contributions from timesteps after
        # the END token.
        # shape: (batch_size, target_sequence_length)
        target_mask = util.get_text_field_mask(target_tokens)
        # The first timestep is just the START token, which is not included in the likelihoods.
        # shape: (batch_size, num_decoding_steps)
        target_mask = target_mask[:, 1:].float()
        # Sum of step log-likelihoods.
        # shape: (batch_size,)
        log_likelihood = (log_likelihoods * target_mask).sum(dim=-1)
        # The loss is the negative log-likelihood, averaged over the batch.
        loss = - log_likelihood.sum() / batch_size

        return {"loss": loss, "probs": (log_likelihood/target_mask.sum(dim=1)).tolist()}
        # return {"loss": loss, "probs": log_likelihood.tolist()}

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, source_length = state["source_mask"].size()
        trimmed_source_length = source_length - 2
        # Initialize the copy scores to zero.
        state["copy_log_probs"] = \
                (state["decoder_hidden"].new_zeros((batch_size, trimmed_source_length)) + 1e-45).log()
        # shape: (batch_size,)
        start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self._start_index)
        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        if(self._beam_search_type == 'cov_beam_search'):
            all_top_k_predictions, log_probabilities, all_top_k_word_log_probabilities = self._beam_search.search(
                    start_predictions, state, self.take_search_step)
            output_dict = {"predicted_log_probs": all_top_k_word_log_probabilities, "predictions": all_top_k_predictions, 'token_scores': all_top_k_word_log_probabilities}
        else:
            all_top_k_predictions, log_probabilities = self._beam_search.search(
                    start_predictions, state, self.take_search_step)
            target_mask = all_top_k_predictions != self._end_index
            log_probabilities = log_probabilities / (target_mask.sum(dim=2).float()+1) # +1 for predicting the last SEP token
            output_dict = {"predicted_log_probs": log_probabilities, "predictions": all_top_k_predictions}
        return output_dict

    def _get_input_and_selective_weights(self,
                                         last_predictions: torch.LongTensor,
                                         state: Dict[str, torch.Tensor]) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        Get input choices for the decoder and the selective copy weights.

        The decoder input choices are simply the `last_predictions`, except for
        target OOV predictions that were copied from source tokens, in which case
        the prediction will be changed to the COPY symbol in the target namespace.

        The selective weights are just the probabilities assigned to source
        tokens that were copied, normalized to sum to 1. If no source tokens were copied,
        there will be all zeros.

        Parameters
        ----------
        last_predictions : ``torch.LongTensor``
            Shape: `(group_size,)`
        state : ``Dict[str, torch.Tensor]``

        Returns
        -------
        Tuple[torch.LongTensor, torch.Tensor]
            `input_choices` (shape `(group_size,)`) and `selective_weights`
            (shape `(group_size, trimmed_source_length)`).
        """
        group_size, trimmed_source_length = state["source_to_target"].size()

        # This is a mask indicating which last predictions were copied from the
        # the source AND not in the target vocabulary (OOV).
        # (group_size,)
        only_copied_mask = (last_predictions >= self._target_vocab_size).long()

        # If the last prediction was in the target vocab or OOV but not copied,
        # we use that as input, otherwise we use the COPY token.
        # shape: (group_size,)
        copy_input_choices = only_copied_mask.new_full((group_size,), fill_value=self._copy_index)
        input_choices = last_predictions * (1 - only_copied_mask) + copy_input_choices * only_copied_mask

        # In order to get the `selective_weights`, we need to find out which predictions
        # were copied or copied AND generated, which is the case when a prediction appears
        # in both the source sentence and the target vocab. But whenever a prediction
        # is in the target vocab (even if it also appeared in the source sentence),
        # its index will be the corresponding target vocab index, not its index in
        # the source sentence offset by the target vocab size. So we first
        # use `state["source_to_target"]` to get an indicator of every source token
        # that matches the predicted target token.
        # shape: (group_size, trimmed_source_length)
        expanded_last_predictions = last_predictions.unsqueeze(-1).expand(group_size, trimmed_source_length)
        # shape: (group_size, trimmed_source_length)
        source_copied_and_generated = (state["source_to_target"] == expanded_last_predictions).long()

        # In order to get indicators for copied source tokens that are OOV with respect
        # to the target vocab, we'll make use of `state["source_token_ids"]`.
        # First we adjust predictions relative to the start of the source tokens.
        # This makes sense because predictions for copied tokens are given by the index of the copied
        # token in the source sentence, offset by the size of the target vocabulary.
        # shape: (group_size,)
        adjusted_predictions = last_predictions - self._target_vocab_size
        # The adjusted indices for items that were not copied will be negative numbers,
        # and therefore invalid. So we zero them out.
        adjusted_predictions = adjusted_predictions * only_copied_mask
        # shape: (group_size, trimmed_source_length)
        source_token_ids = state["source_token_ids"]
        # shape: (group_size, trimmed_source_length)
        adjusted_prediction_ids = source_token_ids.gather(-1, adjusted_predictions.unsqueeze(-1))
        # This mask will contain indicators for source tokens that were copied
        # during the last timestep.
        # shape: (group_size, trimmed_source_length)
        source_only_copied = (source_token_ids == adjusted_prediction_ids).long()
        # Since we zero'd-out indices for predictions that were not copied,
        # we need to zero out all entries of this mask corresponding to those predictions.
        source_only_copied = source_only_copied * only_copied_mask.unsqueeze(-1)

        # shape: (group_size, trimmed_source_length)
        mask = source_only_copied | source_copied_and_generated
        # shape: (group_size, trimmed_source_length)
        selective_weights = util.masked_softmax(state["copy_log_probs"], mask)

        return input_choices, selective_weights

    def _gather_final_log_probs(self,
                                generation_log_probs: torch.Tensor,
                                copy_log_probs: torch.Tensor,
                                state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combine copy probabilities with generation probabilities for matching tokens.

        Parameters
        ----------
        generation_log_probs : ``torch.Tensor``
            Shape: `(group_size, target_vocab_size)`
        copy_log_probs : ``torch.Tensor``
            Shape: `(group_size, trimmed_source_length)`
        state : ``Dict[str, torch.Tensor]``

        Returns
        -------
        torch.Tensor
            Shape: `(group_size, target_vocab_size + trimmed_source_length)`.
        """
        _, trimmed_source_length = state["source_to_target"].size()
        source_token_ids = state["source_token_ids"]


        # shape: [(batch_size, *)]
        modified_log_probs_list: List[torch.Tensor] = []
        for i in range(trimmed_source_length):
            # shape: (group_size,)
            copy_log_probs_slice = copy_log_probs[:, i]
            # `source_to_target` is a matrix of shape (group_size, trimmed_source_length)
            # where element (i, j) is the vocab index of the target token that matches the jth
            # source token in the ith group, if there is one, or the index of the OOV symbol otherwise.
            # We'll use this to add copy scores to corresponding generation scores.
            # shape: (group_size,)
            source_to_target_slice = state["source_to_target"][:, i]
            # The OOV index in the source_to_target_slice indicates that the source
            # token is not in the target vocab, so we don't want to add that copy score
            # to the OOV token.
            copy_log_probs_to_add_mask = (source_to_target_slice != self._oov_index).float()
            copy_log_probs_to_add = copy_log_probs_slice + (copy_log_probs_to_add_mask + 1e-45).log()
            # shape: (batch_size, 1)
            copy_log_probs_to_add = copy_log_probs_to_add.unsqueeze(-1)
            # shape: (batch_size, 1)
            selected_generation_log_probs = generation_log_probs.gather(1, source_to_target_slice.unsqueeze(-1))
            combined_scores = util.logsumexp(
                    torch.cat((selected_generation_log_probs, copy_log_probs_to_add), dim=1))
            generation_log_probs = generation_log_probs.scatter(-1,
                                                                source_to_target_slice.unsqueeze(-1),
                                                                combined_scores.unsqueeze(-1))
            copy_log_probs_cpu = copy_log_probs.cpu()
            # We have to combine copy scores for duplicate source tokens so that
            # we can find the overall most likely source token. So, if this is the first
            # occurence of this particular source token, we add the log_probs from all other
            # occurences, otherwise we zero it out since it was already accounted for.
            if i < (trimmed_source_length - 1):
                # Sum copy scores from future occurences of source token.
                # shape: (group_size, trimmed_source_length - i)
                source_future_occurences = (source_token_ids[:, (i+1):] == source_token_ids[:, i].unsqueeze(-1)).float()  # pylint: disable=line-too-long
                # shape: (group_size, trimmed_source_length - i)
                future_copy_log_probs = copy_log_probs[:, (i+1):] + (source_future_occurences + 1e-45).log()
                # shape: (group_size, 1 + trimmed_source_length - i)
                combined = torch.cat((copy_log_probs_slice.unsqueeze(-1), future_copy_log_probs), dim=-1)
                # shape: (group_size,)
                copy_log_probs_slice = util.logsumexp(combined)
            if i > 0:
                # Remove copy log_probs that we have already accounted for.
                # shape: (group_size, i)
                source_previous_occurences = source_token_ids[:, 0:i] == source_token_ids[:, i].unsqueeze(-1)
                # shape: (group_size,)
                duplicate_mask = (source_previous_occurences.sum(dim=-1) == 0).float()
                copy_log_probs_slice = copy_log_probs_slice + (duplicate_mask + 1e-45).log()

            # Finally, we zero-out copy scores that we added to the generation scores
            # above so that we don't double-count them.
            # shape: (group_size,)
            left_over_copy_log_probs = copy_log_probs_slice + (1.0 - copy_log_probs_to_add_mask + 1e-45).log()
            modified_log_probs_list.append(left_over_copy_log_probs.unsqueeze(-1))
        modified_log_probs_list.insert(0, generation_log_probs)

        # shape: (group_size, target_vocab_size + trimmed_source_length)
        modified_log_probs = torch.cat(modified_log_probs_list, dim=-1)

        return modified_log_probs

    def take_search_step(self,
                         last_predictions: torch.Tensor,
                         state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take step during beam search.

        This function is what gets passed to the `BeamSearch.search` method. It takes
        predictions from the last timestep and the current state and outputs
        the log probabilities assigned to tokens for the next timestep, as well as the updated
        state.

        Since we are predicting tokens out of the extended vocab (target vocab + all unique
        tokens from the source sentence), this is a little more complicated that just
        making a forward pass through the model. The output log probs will have
        shape `(group_size, target_vocab_size + trimmed_source_length)` so that each
        token in the target vocab and source sentence are assigned a probability.

        Note that copy scores are assigned to each source token based on their position, not unique value.
        So if a token appears more than once in the source sentence, it will have more than one score.
        Further, if a source token is also part of the target vocab, its final score
        will be the sum of the generation and copy scores. Therefore, in order to
        get the score for all tokens in the extended vocab at this step,
        we have to combine copy scores for re-occuring source tokens and potentially
        add them to the generation scores for the matching token in the target vocab, if
        there is one.

        So we can break down the final log probs output as the concatenation of two
        matrices, A: `(group_size, target_vocab_size)`, and B: `(group_size, trimmed_source_length)`.
        Matrix A contains the sum of the generation score and copy scores (possibly 0)
        for each target token. Matrix B contains left-over copy scores for source tokens
        that do NOT appear in the target vocab, with zeros everywhere else. But since
        a source token may appear more than once in the source sentence, we also have to
        sum the scores for each appearance of each unique source token. So matrix B
        actually only has non-zero values at the first occurence of each source token
        that is not in the target vocab.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            Shape: `(group_size,)`

        state : ``Dict[str, torch.Tensor]``
            Contains all state tensors necessary to produce generation and copy scores
            for next step.

        Notes
        -----
        `group_size` != `batch_size`. In fact, `group_size` = `batch_size * beam_size`.
        """
        _, trimmed_source_length = state["source_to_target"].size()

        # Get input to the decoder RNN and the selective weights. `input_choices`
        # is the result of replacing target OOV tokens in `last_predictions` with the
        # copy symbol. `selective_weights` consist of the normalized copy probabilities
        # assigned to the source tokens that were copied. If no tokens were copied,
        # there will be all zeros.
        # shape: (group_size,), (group_size, trimmed_source_length)
        input_choices, selective_weights = self._get_input_and_selective_weights(last_predictions, state)
        # Update the decoder state by taking a step through the RNN.
        state = self._decoder_step(input_choices, selective_weights, state)
        # Get the un-normalized generation scores for each token in the target vocab.
        # shape: (group_size, target_vocab_size)
        generation_scores = self._get_generation_scores(state)
        # Get the un-normalized copy scores for each token in the source sentence,
        # excluding the start and end tokens.
        # shape: (group_size, trimmed_source_length)
        copy_scores = self._get_copy_scores(state)
        # Concat un-normalized generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        all_scores = torch.cat((generation_scores, copy_scores), dim=-1)
        # shape: (group_size, trimmed_source_length)
        copy_mask = state["source_mask"][:, 1:-1].float()
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        mask = torch.cat((generation_scores.new_full(generation_scores.size(), 1.0), copy_mask), dim=-1)
        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        log_probs = util.masked_log_softmax(all_scores, mask)
        # shape: (group_size, target_vocab_size), (group_size, trimmed_source_length)
        generation_log_probs, copy_log_probs = log_probs.split(
                [self._target_vocab_size, trimmed_source_length], dim=-1)
        # Update copy_probs needed for getting the `selective_weights` at the next timestep.
        state["copy_log_probs"] = copy_log_probs
        # We now have normalized generation and copy scores, but to produce the final
        # score for each token in the extended vocab, we have to go through and add
        # the copy scores to the generation scores of matching target tokens, and sum
        # the copy scores of duplicate source tokens.
        # shape: (group_size, target_vocab_size + trimmed_source_length)
        final_log_probs = self._gather_final_log_probs(generation_log_probs, copy_log_probs, state)

        return final_log_probs, state

    def _get_predicted_tokens(self,
                              predicted_indices: Union[torch.Tensor, np.ndarray],
                              batch_metadata: List[Any],
                              n_best: int = None) -> List[Union[List[List[str]], List[str]]]:
        """
        Convert predicted indices into tokens.

        If `n_best = 1`, the result type will be `List[List[str]]`. Otherwise the result
        type will be `List[List[List[str]]]`.
        """
        if not isinstance(predicted_indices, np.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        predicted_tokens: List[Union[List[List[str]], List[str]]] = []
        for top_k_predictions, metadata in zip(predicted_indices, batch_metadata):
            batch_predicted_tokens: List[List[str]] = []
            for indices in top_k_predictions[:n_best]:
                tokens: List[str] = []
                if self._append:
                    dummy = True
                    # expected that eoe occurs only at the last index 
                    # ignore all words after that even if the model generates something
                    indices = list(indices)
                    if self._eoe_index in indices: # will not be true if max number of extractions already reached
                        indices = indices[:indices.index(self._eoe_index)]
                    indices = self._leave_one(indices, self._end_index)
                elif self._end_index in indices:
                    # indices = indices[:indices.index(self._end_index)]
                    # append decoder produces lots of end-indices in-between must remove all of them 
                    # this captures the earlier case as well
                    indices = indices[indices != self._end_index] 
                    indices = list(indices)
                for index in indices:
                    if index >= self._target_vocab_size:
                        adjusted_index = index - self._target_vocab_size
                        token = metadata["source_tokens"][adjusted_index]
                    else:
                        token = self.vocab.get_token_from_index(index, self._target_namespace)
                    tokens.append(token)
                batch_predicted_tokens.append(tokens)
            if n_best == 1:
                predicted_tokens.append(batch_predicted_tokens[0])
            else:
                predicted_tokens.append(batch_predicted_tokens)
        return predicted_tokens

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Finalize predictions.

        After a beam search, the predicted indices correspond to tokens in the target vocabulary
        OR tokens in source sentence. Here we gather the actual tokens corresponding to
        the indices.
        """
        predicted_tokens = self._get_predicted_tokens(output_dict["predictions"],
                                                      output_dict["metadata"])
        output_dict["predicted_tokens"] = predicted_tokens
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(self._tensor_based_metric.get_metric(reset=reset))  # type: ignore
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))  # type: ignore
        return all_metrics

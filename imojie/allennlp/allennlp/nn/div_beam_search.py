from typing import List, Callable, Tuple, Dict
import warnings

import torch
import ipdb

from allennlp.common.checks import ConfigurationError


StateType = Dict[str, torch.Tensor]  # pylint: disable=invalid-name
StepFunctionType = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]  # pylint: disable=invalid-name


class DivBeamSearch:
    """
    Implements the beam search algorithm for decoding the most likely sequences.

    Parameters
    ----------
    end_index : ``int``
        The index of the "stop" or "end" token in the target vocabulary.
    max_steps : ``int``, optional (default = 50)
        The maximum number of decoding steps to take, i.e. the maximum length
        of the predicted sequences.
    beam_size : ``int``, optional (default = 10)
        The width of the beam used.
    per_node_beam_size : ``int``, optional (default = beam_size)
        The maximum number of candidates to consider per node, at each step in the search.
        If not given, this just defaults to ``beam_size``. Setting this parameter
        to a number smaller than ``beam_size`` may give better results, as it can introduce
        more diversity into the search. See `Beam Search Strategies for Neural Machine Translation.
        Freitag and Al-Onaizan, 2017 <http://arxiv.org/abs/1702.01806>`_.
    """

    def __init__(self,
                 end_index: int,
                 max_steps: int = 50,
                 beam_size: int = 10,
                 per_node_beam_size: int = None,
                 lambda_diversity: int=1,
                 ignore_indices: list=None) -> None:
        self._end_index = end_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size
        self.lambda_diversity = lambda_diversity
        self.ignore_indices = ignore_indices
        self.ignore_indices.append(self._end_index)

    def search(self,
               start_predictions: torch.Tensor,
               start_state: StateType,
               step: StepFunctionType) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a starting state and a step function, apply beam search to find the
        most likely target sequences.

        Notes
        -----
        If your step function returns ``-inf`` for some log probabilities
        (like if you're using a masked log-softmax) then some of the "best"
        sequences returned may also have ``-inf`` log probability. Specifically
        this happens when the beam size is smaller than the number of actions
        with finite log probability (non-zero probability) returned by the step function.
        Therefore if you're using a mask you may want to check the results from ``search``
        and potentially discard sequences with non-finite log probability.

        Parameters
        ----------
        start_predictions : ``torch.Tensor``
            A tensor containing the initial predictions with shape ``(batch_size,)``.
            Usually the initial predictions are just the index of the "start" token
            in the target vocabulary.
        start_state : ``StateType``
            The initial state passed to the ``step`` function. Each value of the state dict
            should be a tensor of shape ``(batch_size, *)``, where ``*`` means any other
            number of dimensions.
        step : ``StepFunctionType``
            A function that is responsible for computing the next most likely tokens,
            given the current state and the predictions from the last time step.
            The function should accept two arguments. The first being a tensor
            of shape ``(group_size,)``, representing the index of the predicted
            tokens from the last time step, and the second being the current state.
            The ``group_size`` will be ``batch_size * beam_size``, except in the initial
            step, for which it will just be ``batch_size``.
            The function is expected to return a tuple, where the first element
            is a tensor of shape ``(group_size, target_vocab_size)`` containing
            the log probabilities of the tokens for the next step, and the second
            element is the updated state. The tensor in the state should have shape
            ``(group_size, *)``, where ``*`` means any other number of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, log_probabilities)``, where ``predictions``
            has shape ``(batch_size, beam_size, max_steps)`` and ``log_probabilities``
            has shape ``(batch_size, beam_size)``.
        """
        batch_size = start_predictions.size()[0]

        # List of (batch_size, beam_size) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []

        # List of (batch_size, beam_size) tensors. One for each time step. None for
        # the first.  Stores the index n for the parent prediction, i.e.
        # predictions[t-1][i][n], that it came from.
        # backpointers: List[torch.Tensor] = []

        # start_class_log_probabilities, state = step(start_predictions, start_state)

        state = start_state
        last_log_probabilities = None

        beam_states = [dict() for i in range(self.beam_size)]
        # Set the same state for each element in the beam.
        for key, state_tensor in state.items():
            # _, *last_dims = state_tensor.size()
            # last_dims_ones = [1] * len(last_dims)
            # shape: (batch_size, beam_size, *)
            # state[key] = state_tensor.\
                    # unsqueeze(1).\
                    # repeat(1, self.beam_size, *last_dims_ones)
            for beam_indx in range(self.beam_size):
                beam_states[beam_indx][key] = state_tensor.clone()
        
        for timestep in range(self.max_steps):

            current_predictions = start_predictions.new_zeros(self.beam_size, batch_size)
            if(type(last_log_probabilities) != type(None)): # Executed at all timesteps except timestep == 0
                counts = last_log_probabilities.new_zeros(batch_size, num_classes)
                # If every predicted token from the last step is `self._end_index`,
                # then we can stop early.
                if (predictions[-1].reshape(batch_size * self.beam_size) == self._end_index).all():
                    break
            for beam_indx in range(self.beam_size):
                # shape: (batch_size,)
                if(len(predictions) > 0):
                    last_predictions = predictions[-1][:, beam_indx]
                else:
                    last_predictions = start_predictions

                # Take a step. This get the predicted log probs of the next classes
                # and updates the state.
                # shape: (batch_size, num_classes)
                class_log_probabilities, beam_states[beam_indx] = step(last_predictions, beam_states[beam_indx])
                if(type(last_log_probabilities) == type(None)): # Executed only at timestep == 0 and beam_indx == 0
                    num_classes = class_log_probabilities.size()[1]
                    # Log probability tensor that mandates that the end token is selected.
                    # shape: (batch_size, num_classes)
                    log_probs_after_end = class_log_probabilities.new_full(
                            (batch_size, num_classes),
                            float("-inf")
                    )
                    log_probs_after_end[:, self._end_index] = 0.

                    last_log_probabilities = class_log_probabilities.new_zeros(batch_size, self.beam_size)
                    counts = last_log_probabilities.new_zeros(batch_size, num_classes)



                # shape: (batch_size, num_classes)
                last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                        batch_size,
                        num_classes
                )

                # Here we are finding any beams where we predicted the end token in
                # the previous timestep and replacing the distribution with a
                # one-hot distribution, forcing the beam to predict the end token
                # this timestep as well.
                # shape: (batch_size, num_classes)
                cleaned_log_probabilities = torch.where(
                        last_predictions_expanded == self._end_index,
                        log_probs_after_end,
                        class_log_probabilities
                )

                aug_log_probabilities = cleaned_log_probabilities - counts * self.lambda_diversity

                # Choosing only top 1 from V for each beam instead of the original bV-->b^2-->b
                # shape (both): (batch_size,)
                _, predicted_classes = \
                    aug_log_probabilities.topk(1)
                predicted_classes = predicted_classes.squeeze(1)

                current_predictions[beam_indx] = predicted_classes
                # (batch_size,)
                top_log_probabilities = torch.gather(cleaned_log_probabilities, 1, predicted_classes.unsqueeze(1)).squeeze(1)
                # (batch_size,)
                last_log_probabilities[:, beam_indx] = last_log_probabilities[:, beam_indx] + top_log_probabilities

                for t in range(len(predictions)):
                    for b in range(batch_size):
                        counts[b][predictions[t][b, beam_indx]] += 1

                for b in range(batch_size):
                    counts[b][predicted_classes[b]] += 1
                    for indx in self.ignore_indices:
                        counts[b][indx] = 0

            predictions.append(current_predictions.transpose(0,1))

        if not torch.isfinite(last_log_probabilities).all():
            warnings.warn("Infinite log probabilities encountered. Some final sequences may not make sense. "
                          "This can happen when the beam size is larger than the number of valid (non-zero "
                          "probability) transitions that the step function produces.",
                          RuntimeWarning)

        predictions = [prediction.unsqueeze(2) for prediction in predictions]
        all_predictions = torch.cat(predictions, 2)

        return all_predictions, last_log_probabilities

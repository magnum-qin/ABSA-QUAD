"""
Conditional Random Field (CRF) for BIO Sequence Labeling
Enforces valid BIO transition constraints:
  - I-ASP can only follow B-ASP or I-ASP
  - I-OPN can only follow B-OPN or I-OPN
  - B-* and O can follow any tag
"""

import torch
import torch.nn as nn


class CRF(nn.Module):
    """Linear-chain CRF layer for BIO tagging.

    Labels: 0=O, 1=B-ASP, 2=I-ASP, 3=B-OPN, 4=I-OPN
    """

    IMPOSSIBLE = -10000.0

    def __init__(self, num_tags: int = 5):
        super().__init__()
        self.num_tags = num_tags

        # Learnable transition parameters (small init)
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

        # Fixed constraint masks (not trainable)
        # trans_mask[prev, next] = IMPOSSIBLE means prev->next is forbidden
        # In forward/Viterbi: inner[b, prev, next] = alpha[prev] + trans[prev, next] + emit[next]
        trans_mask = torch.zeros(num_tags, num_tags)
        start_mask = torch.zeros(num_tags)

        # I-ASP(2) can only follow B-ASP(1) or I-ASP(2)
        for prev in range(num_tags):
            if prev not in (1, 2):
                trans_mask[prev, 2] = self.IMPOSSIBLE

        # I-OPN(4) can only follow B-OPN(3) or I-OPN(4)
        for prev in range(num_tags):
            if prev not in (3, 4):
                trans_mask[prev, 4] = self.IMPOSSIBLE

        # Cannot start with I-*
        start_mask[2] = self.IMPOSSIBLE
        start_mask[4] = self.IMPOSSIBLE

        self.register_buffer('trans_constraint', trans_mask)
        self.register_buffer('start_constraint', start_mask)

    def _get_transitions(self):
        """Get transitions with constraints applied (every forward call)."""
        return self.transitions + self.trans_constraint

    def _get_start_transitions(self):
        """Get start transitions with constraints applied."""
        return self.start_transitions + self.start_constraint

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss.

        Args:
            emissions: [batch_size, seq_len, num_tags]
            tags: [batch_size, seq_len] ground truth tag indices
            mask: [batch_size, seq_len] 1 for valid, 0 for padding (float)

        Returns:
            Scalar loss (mean over batch)
        """
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_algorithm(emissions, mask)
        # Normalize by sequence length to stabilize loss magnitude
        lengths = mask.sum(dim=1).clamp(min=1)
        nll = (forward_score - gold_score) / lengths
        return nll.mean()

    def _score_sentence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Score of the gold tag sequence."""
        transitions = self._get_transitions()
        start_trans = self._get_start_transitions()

        score = start_trans[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for t in range(1, emissions.size(1)):
            trans = transitions[tags[:, t - 1], tags[:, t]]
            emit = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            score += (trans + emit) * mask[:, t]

        last_tag = self._get_last_tag(tags, mask)
        score += self.end_transitions.gather(0, last_tag)

        return score

    def _forward_algorithm(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward algorithm to compute log partition function."""
        transitions = self._get_transitions()
        start_trans = self._get_start_transitions()
        batch_size, seq_len, num_tags = emissions.shape

        alpha = start_trans.unsqueeze(0) + emissions[:, 0]  # [B, T]

        for t in range(1, seq_len):
            emit = emissions[:, t].unsqueeze(1)  # [B, 1, T]
            trans = transitions.unsqueeze(0)  # [1, T, T]
            alpha_expand = alpha.unsqueeze(2)  # [B, T, 1]

            inner = alpha_expand + trans + emit  # [B, prev, next]
            new_alpha = torch.logsumexp(inner, dim=1)  # [B, T]

            m = mask[:, t].unsqueeze(1)
            alpha = new_alpha * m + alpha * (1 - m)

        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)  # [B]

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Viterbi decoding to find best tag sequence.

        Args:
            emissions: [batch_size, seq_len, num_tags]
            mask: [batch_size, seq_len] float mask

        Returns:
            best_tags: [batch_size, seq_len] best tag sequence
        """
        transitions = self._get_transitions()
        start_trans = self._get_start_transitions()
        batch_size, seq_len, num_tags = emissions.shape

        viterbi = start_trans.unsqueeze(0) + emissions[:, 0]  # [B, T]
        backpointers = []

        for t in range(1, seq_len):
            viterbi_expand = viterbi.unsqueeze(2)  # [B, T, 1]
            trans = transitions.unsqueeze(0)  # [1, T, T]
            emit = emissions[:, t].unsqueeze(1)  # [B, 1, T]

            inner = viterbi_expand + trans + emit  # [B, prev, next]
            best_prev = inner.argmax(dim=1)  # [B, T]
            best_score = inner.max(dim=1).values  # [B, T]

            m = mask[:, t].unsqueeze(1)
            viterbi = best_score * m + viterbi * (1 - m)
            backpointers.append(best_prev)

        viterbi = viterbi + self.end_transitions.unsqueeze(0)

        best_tags_list = []
        best_last = viterbi.argmax(dim=1)  # [B]
        best_tags_list.append(best_last)

        for bp in reversed(backpointers):
            best_last = bp.gather(1, best_last.unsqueeze(1)).squeeze(1)
            best_tags_list.append(best_last)

        best_tags_list.reverse()
        best_tags = torch.stack(best_tags_list, dim=1)  # [B, L]

        return best_tags

    def _get_last_tag(
        self,
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Get the tag at the last valid position for each sample."""
        lengths = mask.long().sum(dim=1) - 1
        last_tag = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
        return last_tag

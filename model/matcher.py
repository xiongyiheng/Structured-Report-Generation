# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_disease. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_disease: float = 1, cost_organ: float = 1, cost_location: float = 1):
        """Creates the matcher

        Params:
            cost_disease: This is the relative weight of the disease error in the matching cost
            cost_organ: This is the relative weight of the organ error in the matching cost
            cost_location: This is the relative weight of the location error in the matching cost
        """
        super().__init__()
        self.cost_disease = cost_disease
        self.cost_organ = cost_organ
        self.cost_location = cost_location
        assert cost_disease != 0 or cost_organ != 0 or cost_location != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_disease_logits": Tensor of dim [batch_size, num_queries, num_diseases + 1] with the disease logits
                 "pred_organ_logits": Tensor of dim [batch_size, num_queries, num_organs + 1] with the organ logits
                 "pred_location_logits": Tensor of dim [batch_size, num_queries, num_locations + 1] with the location logits

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "diseases": Tensor of dim [num_target_diseases] (where num_target_diseases is the number of ground-truth
                           diseases in the target) containing the diseases labels
                 "organs": Tensor of dim [num_target_diseases] containing the target organ labels
                 "locations" Tensor of dim [num_target_diseases] containing the target location labels

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_diseases)
        """
        bs, num_queries = outputs["pred_disease_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_disease_prob = outputs["pred_disease_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_diseases+1]
        out_organ_prob = outputs["pred_organ_logits"].flatten(0, 1).softmax(-1) # [batch_size * num_queries, num_organs+1]
        out_location_prob = outputs["pred_location_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_locations+1]

        # Also concat the target diseases/organs/locations
        tgt_disease_ids = torch.cat([v["diseases"] for v in targets])
        tgt_organ_ids = torch.cat([v["organs"] for v in targets])
        tgt_location_ids = torch.cat([v["locations"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_disease = -out_disease_prob[:, tgt_disease_ids]
        cost_organ = -out_organ_prob[:, tgt_organ_ids]
        cost_location = -out_location_prob[:, tgt_location_ids]

        # Final cost matrix
        C = self.cost_disease * cost_disease + self.cost_organ * cost_organ + self.cost_location * cost_location
        C = C.view(bs, num_queries, -1).detach().cpu()
        sizes = [len(v["diseases"]) for v in targets] # the key corresponds to min(num_queries, num_target_diseases)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_disease=args.set_cost_disease, cost_organ=args.set_cost_organ, cost_location=args.set_cost_location)

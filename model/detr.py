# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, is_dist_avail_and_initialized)

from .resnet import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs structured report generation """
    def __init__(self, backbone, transformer, num_diseases, num_organs, num_locations, num_queries):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_diseases: number of disease classes
            num_organs: number of organ classes
            num_locations: number of location classes
            num_queries: number of disease queries. This is the maximal number of diseases
                         DETR can find in a single X-ray image.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.disease_embed = nn.Linear(hidden_dim, num_diseases + 1)
        self.organ_embed = MLP(hidden_dim, hidden_dim, num_organs + 1, 3)
        self.location_embed = MLP(hidden_dim, hidden_dim, num_locations + 1, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_disease_logits": the classification logits (including no-disease) for all queries.
                                Shape= [batch_size x num_queries x (num_diseases + 1)]
               - "pred_organ_logits": the classification logits for organs
               - "pred_location_logits": the classification logits for organs
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_disease = self.disease_embed(hs)
        outputs_organ = self.organ_embed(hs)
        outputs_location = self.location_embed(hs)
        out = {'pred_disease_logits': outputs_disease[-1], 'pred_organ_logits': outputs_organ[-1], 'pred_location_logits': outputs_location[-1]}
        return out


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth token and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise disease, organ and location)
    """
    def __init__(self, num_diseases, num_organs, num_locations, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_diseases: number of disease categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_diseases = num_diseases
        self.num_organs = num_organs
        self.num_locations = num_locations
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        with open('datasets/radgraph/fre_occur_dis_organ_loc.npy', 'rb') as f:
            fre_occur_dis = np.load(f)
            fre_occur_organ = np.load(f)
            fre_occur_loc = np.load(f)
        weight_disease = torch.ones(self.num_diseases + 1)
        #weight_disease[:self.num_diseases] = torch.FloatTensor(1 / (fre_occur_dis + 1e-5))
        #weight_disease[weight_disease > 1e4] = 0.0
        #weight_disease = torch.sqrt(torch.sqrt(weight_disease))
        weight_disease[-1] = self.eos_coef

        weight_organ = torch.zeros(self.num_organs + 1)
        weight_organ[:num_organs] = torch.FloatTensor(1 / (fre_occur_organ + 1e-5))
        weight_organ[weight_organ > 1e4] = 0.0
        weight_organ = torch.sqrt(torch.sqrt(weight_organ))

        weight_location = torch.zeros(self.num_locations + 1)
        weight_location[:num_locations] = torch.FloatTensor(1 / (fre_occur_loc + 1e-5))
        weight_location[weight_location > 1e4] = 0.0
        weight_location = torch.sqrt(torch.sqrt(weight_location))
        self.register_buffer('weight_disease', weight_disease)
        self.register_buffer('weight_organ', weight_organ)
        self.register_buffer('weight_location', weight_location)

    def loss_diseases(self, outputs, targets, indices, num_diseases, log=True):
        """Classification loss (NLL) for disease
        targets dicts must contain the key "diseases" containing a tensor of dim [nb_target_diseases] [1, 5, 6, ..] TODO
        """
        assert 'pred_disease_logits' in outputs
        src_logits = outputs['pred_disease_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["diseases"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_diseases,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_disease = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.weight_disease)
        losses = {'loss_disease': loss_disease}
        return losses

    def loss_organs(self, outputs, targets, indices, num_diseases):
        """Classification loss (NLL) for organ
        targets dicts must contain the key "organs" containing a tensor of dim [nb_target_diseases] TODO
        """
        assert 'pred_organ_logits' in outputs # shape  [batch_size, num_queries, num_organs]
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['pred_organ_logits']
        target_classes_o = torch.cat([t["organs"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_organs,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_organ = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_organ': loss_organ / 1}
        return losses

    def loss_locations(self, outputs, targets, indices, num_diseases):
        """Classification loss (NLL) for location
        targets dicts must contain the key "locations" containing a tensor of dim [nb_target_diseases] TODO   targets = {'diseases':tensor[1,5,3...],'organs':tensor[6,3,7...],'locations':[4,77,90...] }
        """
        assert 'pred_location_logits' in outputs  # [batch_size, num_queries, num_locations]
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['pred_location_logits']
        target_classes_o = torch.cat([t["locations"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_locations,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_location = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_location': loss_location / 1}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_diseases, **kwargs):
        loss_map = {
            'diseases': self.loss_diseases,
            'organs': self.loss_organs,
            'locations': self.loss_locations
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_diseases, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target diseases across all nodes, for normalization purposes
        num_diseases = sum(len(t["diseases"]) for t in targets)
        num_diseases = torch.as_tensor([num_diseases], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_diseases))

        return losses, indices, self._get_src_permutation_idx(indices)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_diseases`corresponds to `max_disease_id + 1`, where max_disease_id
    # is the maximum id for a disease class in your datasets.
    num_diseases = 126 # if args.dataset_file != 'radgraph' else 91
    num_organs = 47 # if args.dataset_file != 'radgraph' else 91
    num_locations = 80 # if args.dataset_file != 'radgraph' else 91

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_diseases=num_diseases,
        num_organs=num_organs,
        num_locations=num_locations,
        num_queries=args.num_queries,
    )
    matcher = build_matcher(args)
    weight_dict = {'loss_disease': args.disease_loss_coef, 'loss_organ': args.organ_loss_coef, 'loss_location': args.location_loss_coef}

    losses = ['diseases', 'organs', 'locations']
    criterion = SetCriterion(num_diseases, num_organs, num_locations, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)

    return model, criterion

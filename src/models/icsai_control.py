"""Explicit BIO-feedback intervention for the frozen ICSAI A/B/C experiment.

The historical model and its state-dict layout remain unchanged. The experiment
factory must also be used when restoring its checkpoints: the arm is configuration,
not a learned tensor, and cannot be inferred from the weights alone.
"""
import hashlib

import torch

from src.models.feature_rethink import LabelEmbeddingModule
from scripts.inference import _create_model


class ControlledLabelEmbedding(LabelEmbeddingModule):
    """Compute the original soft embedding, then optionally zero its content."""

    def embed_bio_labels(self, bio_probs):
        embedding = super().embed_bio_labels(bio_probs)
        return embedding if self.bio_feedback_enabled else embedding * 0.0

    @classmethod
    def from_original(cls, original, enabled):
        # Replacement construction must not advance the training RNG stream.
        if original.bio_embedding.weight.device.type != 'cpu':
            raise ValueError('Install the feedback control before moving the model to GPU')
        with torch.random.fork_rng(devices=[]):
            result = cls(
                num_bio_labels=original.bio_embedding.num_embeddings,
                num_entity_labels=original.entity_embedding.num_embeddings,
                num_attribute_labels=original.attribute_embedding.num_embeddings,
                embedding_dim=original.embedding_dim,
                hidden_dim=original.token_fusion.out_features,
                pair_dim=original.pair_fusion.in_features - 4 * original.embedding_dim,
            )
        result.load_state_dict(original.state_dict(), strict=True)
        result.train(original.training)
        result.bio_feedback_enabled = enabled
        return result


def build_control_model(config):
    arm = config['icsai']['arm']
    if arm not in ('A', 'B', 'C'):
        raise ValueError(f'Unknown ICSAI arm: {arm}')
    if config['model']['rethink']['enabled'] != (arm != 'A'):
        raise ValueError('Arm and rethink.enabled disagree')
    if arm != 'A' and config['model']['rethink']['num_iterations'] != 2:
        raise ValueError('Frozen B/C design requires two refinement iterations')
    model = _create_model(config)
    if arm != 'A':
        original = model.rethink_module.label_embedding
        model.rethink_module.label_embedding = ControlledLabelEmbedding.from_original(
            original, enabled=arm == 'B'
        )
    return model


def state_fingerprint(model):
    """Hash actual initial tensor bytes, including key/shape/dtype boundaries."""
    full = hashlib.sha256()
    shared = hashlib.sha256()
    shared_prefixes = ('encoder.', 'span_extractor.', 'pair_classifier.', 'va_regressor.')
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        header = f'{name}:{list(tensor.shape)}:{tensor.dtype}\n'.encode()
        data = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
        full.update(header); full.update(data)
        if name.startswith(shared_prefixes):
            shared.update(header); shared.update(data)
    return {
        'full_sha256': full.hexdigest(),
        'shared_sha256': shared.hexdigest(),
        'total_parameters': sum(p.numel() for p in model.parameters()),
    }


def load_control_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = build_control_model(config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    shadow = checkpoint.get('ema_state_dict', {}).get('shadow')
    if shadow is not None:
        parameters = dict(model.named_parameters())
        if set(shadow) != set(parameters):
            raise ValueError('Incomplete EMA parameter set')
        with torch.no_grad():
            for name, p in parameters.items():
                p.copy_(shadow[name])
    model.to(device).eval()
    return model, config


def validate_seed_predictions(predictions, expected_ids):
    """Reject misalignment or duplicate triples before any ensemble voting."""
    if [r['ID'] for r in predictions] != list(expected_ids):
        raise ValueError('Prediction ID order differs from the input')
    for row in predictions:
        keys = [(q['Aspect'], q['Category'], q['Opinion']) for q in row['Quadruplet']]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate categorical triple within one seed: {row['ID']}")

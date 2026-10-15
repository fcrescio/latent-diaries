"""Position-sharing image-prefix reader and training-only ridge length control."""
import torch
from torch import nn

from .image_reader import GlobalImageReader


class ImageQueryReader(GlobalImageReader):
    def __init__(self, maximum_tokens, mean, scale, length_coef, width=128):
        super().__init__(maximum_tokens, mean, scale, length_coef, width)
        del self.network
        self.input_norm = nn.LayerNorm(768)
        self.memory = nn.Linear(768, width * 4)
        self.queries = nn.Parameter(torch.randn(1, maximum_tokens, width) * .02)
        layer = nn.TransformerDecoderLayer(width, 4, width * 4, dropout=0,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, 4, norm=nn.LayerNorm(width))
        self.output = nn.Linear(width, 1024)
        self.register_buffer('causal_mask', torch.ones(maximum_tokens, maximum_tokens,
                                                       dtype=torch.bool).triu(1))

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 768:
            raise ValueError('Expected [B,768] global image vectors')
        if not torch.is_floating_point(features) or not torch.isfinite(features).all():
            raise ValueError('Finite floating-point image vectors required')
        memory = self.memory(self.input_norm(features)).reshape(features.shape[0], 4, -1)
        query = self.queries.expand(features.shape[0], -1, -1)
        hidden = self.decoder(query, memory, tgt_mask=self.causal_mask)
        prefix = self.output(hidden) * self.scale + self.mean
        design = torch.cat([features.new_ones(features.shape[0], 1), features], -1)
        lengths = (design @ self.length_coef).round().long().clamp(1, self.maximum_tokens)
        mask = torch.arange(self.maximum_tokens, device=features.device)[None] < lengths[:, None]
        return prefix, mask, lengths


def ridge_length(features, lengths, alphas=(.0001, .001, .01, .1, 1., 10., 100.)):
    """Fit an unpenalized intercept; select ridge alpha by training-only LOOCV."""
    if features.ndim != 2 or lengths.shape != (features.shape[0],) or len(lengths) < 3:
        raise ValueError('Expected at least three paired training rows')
    x, y = features.double(), lengths.double()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Finite training data required')
    xm, ym = x.mean(0), y.mean()
    xc, yc = x - xm, y - ym
    kernel = xc @ xc.T
    identity = torch.eye(len(y), device=x.device, dtype=x.dtype)
    fits = []
    for alpha in alphas:
        if alpha <= 0:
            raise ValueError('Positive ridge penalty required')
        inverse = torch.linalg.solve(kernel + alpha * identity, identity)
        hat = kernel @ inverse
        predicted = ym + hat @ yc
        leverage = hat.diagonal() + 1 / len(y)
        loo_error = ((y - predicted) / (1 - leverage).clamp_min(1e-12)).square().mean()
        coefficients = xc.T @ inverse @ yc
        bias = ym - xm @ coefficients
        fits.append((float(loo_error), float(alpha), torch.cat([bias[None], coefficients])))
    score, alpha, coefficients = min(fits, key=lambda fit: (fit[0], fit[1]))
    return coefficients.float(), {'alpha': alpha, 'training_loocv_mse': score,
                                  'candidates': [{'alpha': a, 'loocv_mse': s} for s, a, _ in fits]}

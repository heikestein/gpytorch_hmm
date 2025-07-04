#!/usr/bin/env python3

import warnings
from typing import Any, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Poisson

from ..distributions import base_distributions, MultitaskMultivariateNormal, Distribution
from ..priors import Prior
from .likelihood import Likelihood




class PoissonLikelihood(Likelihood):
    r"""
    Implements the Softmax (multiclass) likelihood used for GP classification.

    .. math::
        p(\mathbf y \mid \mathbf f) = \text{Softmax} \left( \mathbf W \mathbf f \right)

    :math:`\mathbf W` is a set of linear mixing weights applied to the latent functions :math:`\mathbf f`.

    :param num_features: Dimensionality of latent function :math:`\mathbf f`.
    :param num_classes: Number of classes.
    :param mixing_weights: (Default: `True`) Whether to learn a linear mixing weight :math:`\mathbf W` applied to
        the latent function :math:`\mathbf f`. If `False`, then :math:`\mathbf W = \mathbf I`.
    :param mixing_weights_prior: Prior to use over the mixing weights :math:`\mathbf W`.

    :ivar torch.Tensor mixing_weights: (Optional) mixing weights.
    """

    def __init__(
        self,
        num_states = 1, 
        x = None,
        C = None,
        d = None, 
    ) -> None:
        super().__init__()
        self.C = C
        self.d = d
        self.x = x
        self.num_states = num_states

    def forward(self, function_samples: Tensor, **kwargs: Any):
        r"""
        Args:
            function_samples (torch.Tensor): Samples from the latent function f (e.g., predictive mean of the GP).
        Returns:
            torch.Tensor: Samples from the Poisson distribution.
        """
        
        n_states = self.num_states
        rates = torch.zeros([n_states, function_samples.size(0), function_samples.size(1), self.C.size(1)])

        for s in range(self.num_states):
            state_idx = torch.arange(function_samples.size(-1))[s*n_states:(s+1)*n_states]

            f = function_samples[..., state_idx] # actual samples of the multivariate normal

            rates[s] = torch.exp((self.x + f) @ self.C + self.d).clamp_min(1e-6) # Avoid zero rates

        pred = Poisson(rates).sample()

        return pred.mean(1)

    def __call__(self, input: Union[Tensor, MultitaskMultivariateNormal], *args: Any, **kwargs: Any) -> Distribution:
        if isinstance(input, Distribution) and not isinstance(input, MultitaskMultivariateNormal):
            warnings.warn(
                "The input to Poisson should be a MultitaskMultivariateNormal (num_data x num_tasks). "
                "Batch MultivariateNormal inputs (num_tasks x num_data) will be deprectated.",
                DeprecationWarning,
            )
            input = MultitaskMultivariateNormal.from_batch_mvn(input)
        return super().__call__(input, *args, **kwargs)


    def expected_log_prob(self, y: torch.Tensor, function_samples: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Args:
            y (torch.Tensor): The observed data (e.g., Poisson counts).
            function_samples (torch.Tensor): Samples from the latent function f (e.g., predictive mean of the GP).
        Returns:
            torch.Tensor: The expected log probability of the observed data under the latent function samples.
        """
        f = function_samples.mean
        
        # Calculate the rate parameter lambda = exp(C(x + f) + d)
        rates = torch.exp((self.x + f) @ self.C + self.d).clamp_min(1e-6)  # Ensure rates are positive

        # Compute the log probability for Poisson distribution
        log_prob = y * torch.log(rates) - rates - torch.lgamma(y+1)  # Factorial for Poisson log likelihood

        # Return the mean log prob over all samples and all timepoints
        return log_prob.sum(dim=-1)/y.size(-1)
    

    # def log_marginal(self, observations, function_dist, **kwargs):
    #     """
    #     Compute the log marginal likelihood (approximation).
    #     Args:
    #         observations (torch.Tensor): Observed values y.
    #         function_dist (gpytorch.distributions.MultivariateNormal): Latent function distribution p(f).
    #     Returns:
    #         torch.Tensor: Log marginal likelihood.
    #     """
    #     # Use Monte Carlo approximation if needed
    #     mean = function_dist.mean.clamp_min(1e-6)
    #     variance = function_dist.variance
    #     sample_mean = mean
    #     sample_variance = mean  # Approximation for Poisson variance = mean

    #     # Gaussian approximation to log-marginal likelihood
    #     log_likelihood = observations * sample_mean.log() - sample_mean - torch.lgamma(observations + 1)
    #     return log_likelihood.sum()
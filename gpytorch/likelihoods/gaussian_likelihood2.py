import warnings
from typing import Any, Union

import torch
from torch import Tensor
from torch.distributions import Normal

from ..distributions import MultitaskMultivariateNormal, Distribution
from .likelihood import Likelihood


class GaussianLikelihood2(Likelihood):
    def __init__(
        self,
        num_states=1,
        x=None,
        C=None,
        d=None,
        noise_variance=.01,  # Default noise variance for Gaussian likelihood
    ) -> None:
        super().__init__()
        self.C = C
        self.d = d
        self.x = x
        self.noise = torch.tensor([noise_variance])
        self.num_states = num_states
        self.register_buffer("noise_variance", torch.tensor(noise_variance))  # Learnable or fixed variance

    def forward(self, function_samples: Tensor, **kwargs: Any):
        """
        Args:
            function_samples (torch.Tensor): Samples from the latent function f.
        Returns:
            torch.Tensor: Samples from the Gaussian distribution.
        """
        n_states = self.num_states
        means = torch.zeros([n_states, function_samples.size(0), function_samples.size(1), self.C.size(1)])
        variances = torch.ones_like(means) * self.noise_variance

        for s in range(self.num_states):
            state_idx = torch.arange(function_samples.size(-1))[s * n_states : (s + 1) * n_states]

            f = function_samples[..., state_idx]  # Actual samples of the multivariate normal

            means[s] = (self.x + f) @ self.C + self.d  # Linear mapping to compute means

        # Sample from the Gaussian
        pred = Normal(means, variances.sqrt()).sample()

        return pred.mean(1)

    def __call__(self, input: Union[Tensor, MultitaskMultivariateNormal], *args: Any, **kwargs: Any) -> Distribution:
        if isinstance(input, Distribution) and not isinstance(input, MultitaskMultivariateNormal):
            warnings.warn(
                "The input to GaussianLikelihood should be a MultitaskMultivariateNormal (num_data x num_tasks). "
                "Batch MultivariateNormal inputs (num_tasks x num_data) will be deprecated.",
                DeprecationWarning,
            )
            input = MultitaskMultivariateNormal.from_batch_mvn(input)
        return super().__call__(input, *args, **kwargs)

    def expected_log_prob(self, y: torch.Tensor, function_samples: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Args:
            y (torch.Tensor): Observed data.
            function_samples (torch.Tensor): Samples from the latent function f.
        Returns:
            torch.Tensor: The expected log probability of the observed data under the latent function samples.
        """
        mu = function_samples.mean
        Q  = function_samples.lazy_covariance_matrix
        Q  = torch.stack([Q.to_dense()[d::mu.shape[-1],d::mu.shape[-1]] 
                            for d in range(mu.shape[-1])], dim=-1)
        
        # Calculate mean from latent function
        # means = (self.x + f) @ self.C + self.d

        # print(means.shape, f.shape)

        # Gaussian log-likelihood
        # log_prob = -0.5 * torch.log(2 * torch.pi * self.noise_variance) - 0.5 * ((y - means) ** 2 + var) / self.noise_variance

        x = self.x
        C = self.C.T
        d = self.d.T
        noise = self.noise

        # precompute C @ C^T
        C_trans_C = C.T @ C    # dim_x x dim_x

        log_LLs = []
        # Loop over datapoints
        for i in range(len(y)):
            # Extract single-datapoint inputs
            y_i = y[i]                      # Observation (d_y)
            x_i = x[i]                      # Input (2)
            mu_i = mu[i]                    # Mean of f (2)
            
            # Centered observation
            Cx_i = torch.matmul(C, x_i)               # Transform input (d_y)
            mu_0_i = y_i - Cx_i - d.squeeze()         # Centered observation (d_y)

            # Trace terms
            trace_term = torch.trace(Q[i,i]*C_trans_C)  # Tr(Q1 * C1^T C1) # scalar

            # Mean terms # mu^T C^T C mu # scalar
            mean_quadratic_term = torch.dot(mu_i, torch.matmul(C.T, torch.matmul(C, mu_i)))

            # Quadratic and linear terms # scalars
            quadratic_term = torch.dot(mu_0_i, mu_0_i)                        # mu_0^T mu_0
            linear_term = -2 * torch.dot(mu_0_i, torch.matmul(C, mu_i))       # -2 * mu_0^T C mu

            # Combine terms for this datapoint
            log_LLs.append((
                -0.5 * y.shape[-1] * torch.log(2 * torch.pi * noise)
                - 0.5 / noise * (quadratic_term + linear_term + trace_term + mean_quadratic_term)
            )/y.shape[-1])

        return torch.tensor(log_LLs, requires_grad=True)


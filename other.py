import abc
import torch
import torch.nn as nn
import numpy as np
from nupic.torch.duty_cycle_metrics import binary_entropy, max_entropy


def update_boost_strength(m):
    """Function used to update KWinner modules boost strength. This is typically done
    during training at the beginning of each epoch.

    Call using :meth:`torch.nn.Module.apply` after each epoch if required
    For example: ``m.apply(update_boost_strength)``

    :param m: KWinner module
    """
    if isinstance(m, KWinnersBase):
        m.update_boost_strength()


class KWinnersBase(nn.Module, metaclass=abc.ABCMeta):
    """Base KWinners class.

    :param percent_on:
      The activity of the top k = percent_on * number of input units will be
      allowed to remain, the rest are set to zero.
    :type percent_on: float

    :param k_inference_factor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * k_inference_factor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type k_inference_factor: float

    :param boost_strength:
      boost strength (0.0 implies no boosting). Must be >= 0.0
    :type boost_strength: float

    :param boost_strength_factor:
      Boost strength factor to use [0..1]
    :type boost_strength_factor: float

    :param duty_cycle_period:
      The period used to calculate duty cycles
    :type duty_cycle_period: int
    """

    def __init__(
            self,
            percent_on,
            k_inference_factor=1.0,
            boost_strength=1.0,
            boost_strength_factor=1.0,
            duty_cycle_period=1000,
    ):
        super(KWinnersBase, self).__init__()
        assert boost_strength >= 0.0
        assert 0.0 <= boost_strength_factor <= 1.0
        assert 0.0 < percent_on < 1.0
        assert 0.0 < percent_on * k_inference_factor < 1.0

        self.percent_on = percent_on
        self.percent_on_inference = percent_on * k_inference_factor
        self.k_inference_factor = k_inference_factor
        self.learning_iterations = 0
        self.n = 0
        self.k = 0
        self.k_inference = 0

        # Boosting related parameters. Put boost_strength in a buffer so that it
        # is saved in the state_dict. Keep a copy that remains a Python float so
        # that its value can be accessed in 'if' statements without blocking to
        # fetch from GPU memory.
        self.register_buffer("boost_strength", torch.tensor(boost_strength,
                                                            dtype=torch.float))
        self._cached_boost_strength = boost_strength

        self.boost_strength_factor = boost_strength_factor
        self.duty_cycle_period = duty_cycle_period

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._cached_boost_strength = self.boost_strength.item()

    def update_boost_strength(self):
        """
        Update boost strength by multiplying by the boost strength factor.
        This is typically done during training at the beginning of each epoch.
        """
        self._cached_boost_strength *= self.boost_strength_factor
        self.boost_strength.fill_(self._cached_boost_strength)

    def entropy(self):
        """Returns the current total entropy of this layer."""
        _, entropy = binary_entropy(self.duty_cycle)
        return entropy

    def max_entropy(self):
        """Returns the maximum total entropy we can expect from this layer."""
        return max_entropy(self.n, int(self.n * self.percent_on))


class KWinners(KWinnersBase):
    """Applies K-Winner function to the input tensor.

    :param k_inference_factor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * k_inference_factor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type k_inference_factor: float
    """

    def __init__(
            self,
            n,  # Number of units
            percent_on,
            k_inference_factor=1.5,
            boost_strength=1.0,  # boost strength (0.0 implies no boosting).
            boost_strength_factor=0.9,  # Boost strength factor to use [0..1]
            duty_cycle_period=1000,  # The period used to calculate duty cycles
    ):

        super(KWinners, self).__init__(
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
        )

        self.n = n
        self.k = int(round(n * percent_on))
        self.k_inference = int(self.k * self.k_inference_factor)
        self.duty_cycle = torch.zeros(self.n)

    def forward(self, x):

        boosted = x.detach() * torch.exp(-self._cached_boost_strength * self.duty_cycle)

        if self.training:
            indices = torch.topk(boosted, k=self.k, dim=1, sorted=False)[1]
        else:
            indices = torch.topk(boosted, k=self.k_inference, dim=1, sorted=False)[1]
        off_mask = torch.ones_like(boosted, dtype=torch.bool)
        off_mask.scatter_(1, indices, 0)

        out = x.masked_fill(off_mask, 0)

        if self.training: self.update_duty_cycle(out)

        return out

    def update_duty_cycle(self, x):
        r"""Updates our duty cycle estimates with the new value. Duty cycles are
        updated according to the following formula:

        .. math::
            dutyCycle = \frac{dutyCycle \times \left( period - batchSize \right)
                                + newValue}{period}

        :param x:
          Current activity of each unit
        """
        batch_size = x.shape[0]
        self.learning_iterations += batch_size
        period = min(self.duty_cycle_period, self.learning_iterations)
        self.duty_cycle = torch.mul(self.duty_cycle, (period - batch_size))
        self.duty_cycle = torch.add(self.duty_cycle, (x.gt(0).sum(dim=0, dtype=torch.float)))
        self.duty_cycle.div_(period)

    def extra_repr(self):
        s = (
            "n={0}, percent_on={1}, boost_strength={2}, boost_strength_factor={3}, "
            "k_inference_factor={4}, duty_cycle_period={5}".format(
                self.n, self.percent_on, self._cached_boost_strength,
                self.boost_strength_factor, self.k_inference_factor,
                self.duty_cycle_period
            )
        )
        s += f", break_ties={self.break_ties}"
        return s

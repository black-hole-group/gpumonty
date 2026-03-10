Scattering Bias Tuning
======================

This document explains how the bias tuning mechanism works in the scattering loop implemented in ``scattering_flow_control`` in ``scattering.cu``. The goal of this mechanism is to dynamically adjust the scattering bias so that the number of scattered photons remains close to a desired target ratio.

Motivation
----------

In Monte Carlo radiative transfer simulations, scattering events may produce **too many** or **too few** photons depending on the bias parameter

If too many photons are produced:

- memory usage increases
- computational cost grows quickly

If too few photons are produced:

- statistical noise increases
- scattering events are poorly sampled

To control this, the code dynamically adjusts a **bias tuning parameter** (``biasTuning``) to maintain a desired ratio between:

.. math::

   \mathrm{Ratio} = \frac{N_{\rm scat}}{N_{\rm in}}

where:

- :math:`N_{\rm scat}` is the number of scattered photons
- :math:`N_{\rm in}` is the number of incoming photons

Parameters
--------------------

``params.biasTuning``
~~~~~~~~~~~~~~~~~~~~~

Scaling factor that modifies the scattering probability.

``params.targetRatio``
~~~~~~~~~~~~~~~~~~~~~~

Desired ratio between scattered photons and incoming photons. For example, a target ratio of 0.5 means the algorithm attempts to produce roughly **0.5 scattered photons per incoming photon**. In this
algorithm, we accept the tuning if it falls within ±20% tolerance.


Overview of the Algorithm
-------------------------

For each scattering round ``n``:

1. The photons from the previous round are copied into ``CurrentLayerScattering``.
2. A kernel (``track_scat``) is launched to simulate scattering.
3. The number of produced scattered photons is measured.
4. The ratio between scattered and incoming photons is computed.
5. If the ratio is outside the allowed interval, the bias is adjusted and the photons are re-tracked.

This process repeats until:

- the ratio falls within the acceptable range, or
- the maximum number of bias tuning iterations is reached.
- the ratio becomes extremely low (indicating that scattering is physically rare).

In the case of an extremely low ratio, the algorithm accepts the current bias and stops tuning to avoid excessive computational costs. 

For the purposes of this algorithm,we define an extremely low ratio as one that is less than 10⁻³. 
If the ratio falls below this threshold, it is likely that scattering events are rare in the physical scenario being simulated, and increasing the bias would be computational inefficient.
Therefore, if the ratio is less than the threshold, the algorithm will accept the current bias without further tuning.

.. note::
    In the implementation, each scattering layer has its own bias tuning parameter.
    The values are stored on the device in an array so that every scattering round :math:`n` uses its own value
    :math:`\mathrm{b}_n`. This allows the algorithm to adapt the
    scattering bias independently for each layer, since the probability of
    scattering and the number of generated photons can vary significantly
    between different layers.
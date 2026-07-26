"""Parameter-driven GERG/EOS-CG multifluid Helmholtz kernel.

This module implements the published multifluid *structure*. A model is only
GERG-2008 or EoS-CG when supplied with the complete, versioned coefficient
set for that model; the constructors deliberately require such parameters
instead of silently substituting a generic mixture backend.

The named GERG-2008 and EOS-CG-2021 parameterizations are defined by Kunz and
Wagner (2012), doi:10.1021/je300655b, and Neumann et al. (2023),
doi:10.1007/s10765-023-03263-6, respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import torch
from torch import Tensor, nn

from torch_flash.constants import R
from torch_flash.exceptions import ConvergenceError
from torch_flash.types import PhaseKind, normalize_composition


@dataclass(frozen=True)
class MultifluidMetadata:
    """Identity and validation scope of one multifluid coefficient set.

    Attributes
    ----------
    model
        Exact published model or fitted-parameter identity.
    reference
        Defining bibliographic reference.
    version
        Version of the serialized coefficient set.
    validated_components
        Component inventory for which the parameterization is defined.
    """

    model: str
    reference: str
    version: str
    validated_components: tuple[str, ...]


@dataclass(frozen=True)
class HelmholtzTerms:
    """Power, exponential, Gaussian, and GERG-special terms.

    The full form is ``n*delta^d*tau^t*exp(-delta^l
    -eta*(delta-epsilon)^2-beta*(tau-gamma)^2
    -linear_density*(delta-linear_shift))``. Setting the optional arrays to
    zero recovers the power-exponential subset. The final linear-density
    factor is the binary departure form used by GERG-2008.

    Attributes
    ----------
    n, d, t, decay
        Term amplitudes, reduced-density exponents, inverse-reduced-temperature
        exponents, and exponential density powers.
    eta, epsilon
        Optional Gaussian density width and center arrays.
    beta, gamma
        Optional Gaussian temperature width and center arrays.
    linear_density, linear_shift
        Optional GERG linear-density exponential parameters.

    Notes
    -----
    Every present tensor has the same term-table shape. Pure tables typically
    use component rows; departure tables use component-pair rows.
    """

    n: Tensor
    d: Tensor
    t: Tensor
    decay: Tensor
    eta: Tensor | None = None
    epsilon: Tensor | None = None
    beta: Tensor | None = None
    gamma: Tensor | None = None
    linear_density: Tensor | None = None
    linear_shift: Tensor | None = None

    def __post_init__(self) -> None:
        if not (self.n.shape == self.d.shape == self.t.shape == self.decay.shape):
            raise ValueError("all Helmholtz term arrays must have equal shape")
        gaussian = (self.eta, self.epsilon, self.beta, self.gamma)
        if any(value is not None for value in gaussian) and not all(
            value is not None and value.shape == self.n.shape for value in gaussian
        ):
            raise ValueError("Gaussian arrays must all be present with the term-table shape")
        linear = (self.linear_density, self.linear_shift)
        if any(value is not None for value in linear) and not all(
            value is not None and value.shape == self.n.shape for value in linear
        ):
            raise ValueError("linear-density arrays must both have the term-table shape")


@dataclass(frozen=True)
class GaoBTerms:
    """Gao-B critical-region residual Helmholtz coefficient arrays.

    Attributes
    ----------
    n, d, t
        Term amplitude and reduced-density/reduced-temperature exponents.
    eta, epsilon, beta, gamma, b
        Critical-region shape parameters. Every tensor must have the same
        shape as ``n``.
    """

    n: Tensor
    d: Tensor
    t: Tensor
    eta: Tensor
    epsilon: Tensor
    beta: Tensor
    gamma: Tensor
    b: Tensor

    def __post_init__(self) -> None:
        arrays = (self.d, self.t, self.eta, self.epsilon, self.beta, self.gamma, self.b)
        if not all(value.shape == self.n.shape for value in arrays):
            raise ValueError("all Gao-B term arrays must have equal shape")


@dataclass(frozen=True)
class NonAnalyticTerms:
    """Span--Wagner non-analytic critical-region coefficient arrays.

    Attributes
    ----------
    n
        Term amplitudes.
    capital_a, capital_b, capital_c, capital_d
        Non-analytic shape coefficients.
    a, b, beta
        Critical exponents and crossover parameters. Every tensor must have
        the same shape as ``n``.
    """

    n: Tensor
    capital_a: Tensor
    capital_b: Tensor
    capital_c: Tensor
    capital_d: Tensor
    a: Tensor
    b: Tensor
    beta: Tensor

    def __post_init__(self) -> None:
        arrays = (
            self.capital_a,
            self.capital_b,
            self.capital_c,
            self.capital_d,
            self.a,
            self.b,
            self.beta,
        )
        if not all(value.shape == self.n.shape for value in arrays):
            raise ValueError("all non-analytic term arrays must have equal shape")


@dataclass(frozen=True)
class IdealHelmholtzTerms:
    """Canonical pure-fluid ideal Helmholtz term tables.

    Every component has lead, logarithmic, power, Planck--Einstein, and
    optional GERG sinh/cosh terms. ``gas_scale`` is the ratio between the gas
    constant of the original pure-fluid equation and the common mixture gas
    constant. It multiplies the density-independent part only, preserving the
    exact ideal-gas pressure limit.

    Attributes
    ----------
    lead_constant, lead_tau, log_tau, tau_log_tau
        One leading coefficient of each analytic form per component.
    power_n, power_t
        Amplitudes and exponents for pure power terms.
    planck_n, planck_theta
        Planck--Einstein amplitudes and characteristic-temperature ratios.
    gerg_n, gerg_theta, gerg_sign
        GERG hyperbolic-term amplitudes, arguments, and sinh/cosh selectors.
    gas_scale
        Per-component ratio of the source pure-fluid gas constant to the
        common mixture gas constant.
    """

    lead_constant: Tensor
    lead_tau: Tensor
    log_tau: Tensor
    tau_log_tau: Tensor
    power_n: Tensor
    power_t: Tensor
    planck_n: Tensor
    planck_theta: Tensor
    gerg_n: Tensor
    gerg_theta: Tensor
    gerg_sign: Tensor
    gas_scale: Tensor

    def __post_init__(self) -> None:
        component_shape = self.lead_constant.shape
        vectors = (
            self.lead_tau,
            self.log_tau,
            self.tau_log_tau,
            self.gas_scale,
        )
        if len(component_shape) != 1 or not all(
            value.shape == component_shape for value in vectors
        ):
            raise ValueError("all ideal Helmholtz lead arrays must have one value per component")
        paired = (
            (self.power_n, self.power_t),
            (self.planck_n, self.planck_theta),
            (self.gerg_n, self.gerg_theta, self.gerg_sign),
        )
        if not all(
            all(value.ndim == 2 and value.shape[0] == component_shape[0] for value in group)
            and all(value.shape == group[0].shape for value in group)
            for group in paired
        ):
            raise ValueError("ideal Helmholtz term tables must have one row per component")


class MultiFluidEOS(nn.Module):
    """Autodifferentiable multifluid Helmholtz equation of state.

    Parameters
    ----------
    names
        Canonical component names defining the final tensor axis.
    critical_temperature, critical_density, molar_mass
        Pure-component reducing constants in K, mol/m3, and kg/mol.
    pure_terms, departure_terms
        Pure residual and binary departure Helmholtz coefficient tables.
    beta_temperature, gamma_temperature, beta_volume, gamma_volume
        Symmetric binary reducing-function matrices.
    departure_scale
        Symmetric binary departure-function multipliers.
    metadata
        Exact model identity and supported-component scope.
    trainable
        Register coefficient arrays as trainable PyTorch parameters.
    gas_constant
        Model gas constant in J/(mol K).

    Notes
    -----
    Property kernels preserve leading batches and gradients. Density inversion
    and root selection are iterative; failure raises
    :class:`~torch_flash.exceptions.ConvergenceError`. Float64 is the
    reference precision, especially near critical or coalescing roots.
    """

    critical_temperature: Tensor
    critical_density: Tensor
    critical_pressure: Tensor
    acentric_factor: Tensor
    molar_mass: Tensor
    pure_n: Tensor
    pure_d: Tensor
    pure_t: Tensor
    pure_decay: Tensor
    pure_eta: Tensor
    pure_epsilon: Tensor
    pure_beta: Tensor
    pure_gamma: Tensor
    pure_linear_density: Tensor
    pure_linear_shift: Tensor
    departure_n: Tensor
    departure_d: Tensor
    departure_t: Tensor
    departure_decay: Tensor
    departure_eta: Tensor
    departure_epsilon: Tensor
    departure_beta: Tensor
    departure_gamma: Tensor
    departure_linear_density: Tensor
    departure_linear_shift: Tensor
    pure_gaob_n: Tensor
    pure_gaob_d: Tensor
    pure_gaob_t: Tensor
    pure_gaob_eta: Tensor
    pure_gaob_epsilon: Tensor
    pure_gaob_beta: Tensor
    pure_gaob_gamma: Tensor
    pure_gaob_b: Tensor
    pure_nonanalytic_n: Tensor
    pure_nonanalytic_capital_a: Tensor
    pure_nonanalytic_capital_b: Tensor
    pure_nonanalytic_capital_c: Tensor
    pure_nonanalytic_capital_d: Tensor
    pure_nonanalytic_a: Tensor
    pure_nonanalytic_b: Tensor
    pure_nonanalytic_beta: Tensor
    beta_temperature: Tensor
    gamma_temperature: Tensor
    beta_volume: Tensor
    gamma_volume: Tensor
    departure_scale: Tensor
    gas_constant: Tensor
    ideal_lead_constant: Tensor
    ideal_lead_tau: Tensor
    ideal_log_tau: Tensor
    ideal_tau_log_tau: Tensor
    ideal_power_n: Tensor
    ideal_power_t: Tensor
    ideal_planck_n: Tensor
    ideal_planck_theta: Tensor
    ideal_gerg_n: Tensor
    ideal_gerg_theta: Tensor
    ideal_gerg_sign: Tensor
    ideal_gas_scale: Tensor
    _upper_triangle: Tensor
    _critical_temperature_pair: Tensor
    _inverse_density_pair: Tensor

    def __init__(
        self,
        names: tuple[str, ...],
        critical_temperature: Tensor,
        critical_density: Tensor,
        molar_mass: Tensor,
        pure_terms: HelmholtzTerms,
        departure_terms: HelmholtzTerms,
        beta_temperature: Tensor,
        gamma_temperature: Tensor,
        beta_volume: Tensor,
        gamma_volume: Tensor,
        departure_scale: Tensor,
        metadata: MultifluidMetadata,
        *,
        trainable: bool = False,
        gas_constant: float = R,
        pure_gaob_terms: GaoBTerms | None = None,
        pure_nonanalytic_terms: NonAnalyticTerms | None = None,
        ideal_terms: IdealHelmholtzTerms | None = None,
        critical_pressure: Tensor | None = None,
        acentric_factor: Tensor | None = None,
    ) -> None:
        super().__init__()
        ncomponents = len(names)
        if pure_terms.n.shape[0] != ncomponents:
            raise ValueError("pure term table must have one row per component")
        expected_pair = (ncomponents, ncomponents)
        if departure_terms.n.shape[:2] != expected_pair:
            raise ValueError("departure term table must start with (component, component)")
        for matrix in (
            beta_temperature,
            gamma_temperature,
            beta_volume,
            gamma_volume,
            departure_scale,
        ):
            if matrix.shape != expected_pair:
                raise ValueError("all multifluid interaction matrices must be square")
        self.names = names
        self.metadata = metadata
        self.register_buffer(
            "gas_constant",
            critical_temperature.new_tensor(gas_constant),
        )
        self.register_buffer("critical_temperature", critical_temperature.clone())
        self.register_buffer("critical_density", critical_density.clone())
        self.register_buffer(
            "critical_pressure",
            (
                torch.full_like(critical_temperature, torch.nan)
                if critical_pressure is None
                else critical_pressure.clone()
            ),
        )
        self.register_buffer(
            "acentric_factor",
            (
                torch.full_like(critical_temperature, torch.nan)
                if acentric_factor is None
                else acentric_factor.clone()
            ),
        )
        self.register_buffer("molar_mass", molar_mass.clone())
        self._store_terms("pure", pure_terms, trainable)
        self._store_terms("departure", departure_terms, trainable)
        self._store_gaob_terms(pure_gaob_terms, ncomponents, critical_temperature, trainable)
        self._store_nonanalytic_terms(
            pure_nonanalytic_terms, ncomponents, critical_temperature, trainable
        )
        # Padded absent critical terms are common in GERG parameter documents.
        # Keep trainable tables active because a caller may intentionally fit a
        # coefficient initialized at zero; immutable all-zero tables can bypass
        # otherwise relatively expensive critical-term algebra.
        self._has_gaob_terms = trainable or bool(torch.count_nonzero(self.pure_gaob_n))
        self._has_nonanalytic_terms = trainable or bool(
            torch.count_nonzero(self.pure_nonanalytic_n)
        )
        self._store_ideal_terms(ideal_terms, ncomponents, critical_temperature, trainable)
        self._store_parameter("beta_temperature", beta_temperature, trainable)
        self._store_parameter("gamma_temperature", gamma_temperature, trainable)
        self._store_parameter("beta_volume", beta_volume, trainable)
        self._store_parameter("gamma_volume", gamma_volume, trainable)
        self._store_parameter("departure_scale", departure_scale, trainable)
        self.register_buffer(
            "_upper_triangle",
            torch.triu(torch.ones_like(beta_temperature), diagonal=1),
        )
        self.register_buffer(
            "_critical_temperature_pair",
            torch.sqrt(critical_temperature[:, None] * critical_temperature[None, :]),
        )
        self.register_buffer(
            "_inverse_density_pair",
            0.125
            * (
                critical_density[:, None].pow(-1.0 / 3.0)
                + critical_density[None, :].pow(-1.0 / 3.0)
            ).pow(3),
        )

    def _store_parameter(self, name: str, value: Tensor, trainable: bool) -> None:
        if trainable:
            setattr(self, name, nn.Parameter(value.clone()))
        else:
            self.register_buffer(name, value.clone())

    def _store_terms(self, prefix: str, terms: HelmholtzTerms, trainable: bool) -> None:
        self._store_parameter(f"{prefix}_n", terms.n, trainable)
        self.register_buffer(f"{prefix}_d", terms.d.clone())
        self.register_buffer(f"{prefix}_t", terms.t.clone())
        self.register_buffer(f"{prefix}_decay", terms.decay.clone())
        for name in ("eta", "epsilon", "beta", "gamma", "linear_density", "linear_shift"):
            value = getattr(terms, name)
            self.register_buffer(
                f"{prefix}_{name}",
                torch.zeros_like(terms.n) if value is None else value.clone(),
            )

    def _store_gaob_terms(
        self,
        terms: GaoBTerms | None,
        ncomponents: int,
        like: Tensor,
        trainable: bool,
    ) -> None:
        if terms is None:
            zeros = like.new_zeros((ncomponents, 1))
            terms = GaoBTerms(zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros + 1.0)
        if terms.n.shape[0] != ncomponents:
            raise ValueError("Gao-B term table must have one row per component")
        self._store_parameter("pure_gaob_n", terms.n, trainable)
        for name in ("d", "t", "eta", "epsilon", "beta", "gamma", "b"):
            self.register_buffer(f"pure_gaob_{name}", getattr(terms, name).clone())

    def _store_nonanalytic_terms(
        self,
        terms: NonAnalyticTerms | None,
        ncomponents: int,
        like: Tensor,
        trainable: bool,
    ) -> None:
        if terms is None:
            zeros = like.new_zeros((ncomponents, 1))
            ones = zeros + 1.0
            terms = NonAnalyticTerms(zeros, zeros, zeros, zeros, zeros, ones, ones, ones)
        if terms.n.shape[0] != ncomponents:
            raise ValueError("non-analytic term table must have one row per component")
        self._store_parameter("pure_nonanalytic_n", terms.n, trainable)
        for name in ("capital_a", "capital_b", "capital_c", "capital_d", "a", "b", "beta"):
            self.register_buffer(f"pure_nonanalytic_{name}", getattr(terms, name).clone())

    def _store_ideal_terms(
        self,
        terms: IdealHelmholtzTerms | None,
        ncomponents: int,
        like: Tensor,
        trainable: bool,
    ) -> None:
        self.has_ideal_terms = terms is not None
        if terms is None:
            vector = like.new_zeros(ncomponents)
            table = like.new_zeros((ncomponents, 1))
            terms = IdealHelmholtzTerms(
                vector,
                vector,
                vector,
                vector,
                table,
                table,
                table,
                table + 1.0,
                table,
                table + 1.0,
                table,
                vector + 1.0,
            )
        if terms.lead_constant.shape != (ncomponents,):
            raise ValueError("ideal Helmholtz tables must have one row per component")
        for name in (
            "lead_constant",
            "lead_tau",
            "log_tau",
            "tau_log_tau",
            "power_n",
            "planck_n",
            "gerg_n",
        ):
            self._store_parameter(f"ideal_{name}", getattr(terms, name), trainable)
        for name in (
            "power_t",
            "planck_theta",
            "gerg_theta",
            "gerg_sign",
            "gas_scale",
        ):
            self.register_buffer(f"ideal_{name}", getattr(terms, name).clone())

    def _reducing_functions_normalized(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate reducing functions for an already normalized composition."""
        xi = x[..., :, None]
        xj = x[..., None, :]
        pair_fraction_t = (
            2.0
            * xi
            * xj
            * self.beta_temperature
            * self.gamma_temperature
            * (xi + xj)
            / (self.beta_temperature.square() * xi + xj)
        )
        pair_fraction_v = (
            2.0
            * xi
            * xj
            * self.beta_volume
            * self.gamma_volume
            * (xi + xj)
            / (self.beta_volume.square() * xi + xj)
        )
        reducing_temperature = torch.sum(
            x.square() * self.critical_temperature,
            dim=-1,
        ) + torch.sum(
            self._upper_triangle * pair_fraction_t * self._critical_temperature_pair,
            dim=(-2, -1),
        )
        inverse_reducing_density = torch.sum(
            x.square() / self.critical_density,
            dim=-1,
        ) + torch.sum(
            self._upper_triangle * pair_fraction_v * self._inverse_density_pair,
            dim=(-2, -1),
        )
        return reducing_temperature, inverse_reducing_density.reciprocal()

    def reducing_functions(self, composition: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate composition-dependent mixture reducing functions.

        Parameters
        ----------
        composition
            Mole fractions with components on the final axis.

        Returns
        -------
        tuple
            Reducing temperature in K and reducing molar density in mol/m3,
            with the broadcast leading shape.
        """
        return self._reducing_functions_normalized(normalize_composition(composition))

    @staticmethod
    def _evaluate_terms(
        n: Tensor,
        d: Tensor,
        t: Tensor,
        decay: Tensor,
        eta: Tensor,
        epsilon: Tensor,
        beta: Tensor,
        gamma: Tensor,
        linear_density: Tensor,
        linear_shift: Tensor,
        delta: Tensor,
        tau: Tensor,
    ) -> Tensor:
        term_axes = (None,) * n.ndim
        delta = delta[(..., *term_axes)]
        tau = tau[(..., *term_axes)]
        exponent = (
            -delta.pow(decay) * (decay != 0.0)
            - eta * (delta - epsilon).square()
            - beta * (tau - gamma).square()
            - linear_density * (delta - linear_shift)
        )
        return torch.sum(
            n * delta.pow(d) * tau.pow(t) * torch.exp(exponent),
            dim=-1,
        )

    @staticmethod
    def _evaluate_terms_delta(
        n: Tensor,
        d: Tensor,
        t: Tensor,
        decay: Tensor,
        eta: Tensor,
        epsilon: Tensor,
        beta: Tensor,
        gamma: Tensor,
        linear_density: Tensor,
        linear_shift: Tensor,
        delta: Tensor,
        tau: Tensor,
    ) -> Tensor:
        """Evaluate the derivative of standard residual terms with respect to delta."""
        term_axes = (None,) * n.ndim
        delta = delta[(..., *term_axes)]
        tau = tau[(..., *term_axes)]
        active_decay = decay != 0.0
        exponent = (
            -delta.pow(decay) * active_decay
            - eta * (delta - epsilon).square()
            - beta * (tau - gamma).square()
            - linear_density * (delta - linear_shift)
        )
        terms = n * delta.pow(d) * tau.pow(t) * torch.exp(exponent)
        logarithmic_derivative = (
            d / delta
            - decay * delta.pow(decay - 1.0) * active_decay
            - 2.0 * eta * (delta - epsilon)
            - linear_density
        )
        return torch.sum(terms * logarithmic_derivative, dim=-1)

    def _evaluate_gaob(self, delta: Tensor, tau: Tensor) -> Tensor:
        delta = delta[(..., None, None)]
        tau = tau[(..., None, None)]
        exponent = (
            self.pure_gaob_d * torch.log(delta)
            + self.pure_gaob_t * torch.log(tau)
            + self.pure_gaob_eta * (delta - self.pure_gaob_epsilon).square()
            + (
                self.pure_gaob_beta * (tau - self.pure_gaob_gamma).square() + self.pure_gaob_b
            ).reciprocal()
        )
        return torch.sum(self.pure_gaob_n * torch.exp(exponent), dim=-1)

    def _evaluate_gaob_delta(self, delta: Tensor, tau: Tensor) -> Tensor:
        """Evaluate the Gao-B residual derivative with respect to delta."""
        delta = delta[(..., None, None)]
        tau = tau[(..., None, None)]
        exponent = (
            self.pure_gaob_d * torch.log(delta)
            + self.pure_gaob_t * torch.log(tau)
            + self.pure_gaob_eta * (delta - self.pure_gaob_epsilon).square()
            + (
                self.pure_gaob_beta * (tau - self.pure_gaob_gamma).square() + self.pure_gaob_b
            ).reciprocal()
        )
        terms = self.pure_gaob_n * torch.exp(exponent)
        logarithmic_derivative = self.pure_gaob_d / delta + 2.0 * self.pure_gaob_eta * (
            delta - self.pure_gaob_epsilon
        )
        return torch.sum(terms * logarithmic_derivative, dim=-1)

    def _evaluate_nonanalytic(self, delta: Tensor, tau: Tensor) -> Tensor:
        delta = delta[(..., None, None)]
        tau = tau[(..., None, None)]
        delta_offset = delta - 1.0
        delta_squared = delta_offset.square().clamp_min(torch.finfo(delta.dtype).tiny)
        tau_offset = tau - 1.0
        psi = torch.exp(
            -self.pure_nonanalytic_capital_c * delta_squared
            - self.pure_nonanalytic_capital_d * tau_offset.square()
        )
        theta = -tau_offset + self.pure_nonanalytic_capital_a * delta_squared.pow(
            0.5 / self.pure_nonanalytic_beta
        )
        capital_delta = theta.square() + self.pure_nonanalytic_capital_b * delta_squared.pow(
            self.pure_nonanalytic_a
        )
        return torch.sum(
            self.pure_nonanalytic_n * delta * psi * capital_delta.pow(self.pure_nonanalytic_b),
            dim=-1,
        )

    def _evaluate_nonanalytic_delta(self, delta: Tensor, tau: Tensor) -> Tensor:
        """Evaluate the nonanalytic critical-term derivative with respect to delta."""
        delta = delta[(..., None, None)]
        tau = tau[(..., None, None)]
        delta_offset = delta - 1.0
        raw_delta_squared = delta_offset.square()
        tiny = torch.finfo(delta.dtype).tiny
        delta_squared = raw_delta_squared.clamp_min(tiny)
        delta_squared_derivative = torch.where(
            raw_delta_squared > tiny,
            2.0 * delta_offset,
            torch.zeros_like(delta_offset),
        )
        tau_offset = tau - 1.0
        psi = torch.exp(
            -self.pure_nonanalytic_capital_c * delta_squared
            - self.pure_nonanalytic_capital_d * tau_offset.square()
        )
        theta_power = 0.5 / self.pure_nonanalytic_beta
        theta = -tau_offset + self.pure_nonanalytic_capital_a * delta_squared.pow(theta_power)
        theta_derivative = (
            self.pure_nonanalytic_capital_a
            * theta_power
            * delta_squared.pow(theta_power - 1.0)
            * delta_squared_derivative
        )
        capital_delta = theta.square() + self.pure_nonanalytic_capital_b * delta_squared.pow(
            self.pure_nonanalytic_a
        )
        capital_delta_derivative = (
            2.0 * theta * theta_derivative
            + self.pure_nonanalytic_capital_b
            * self.pure_nonanalytic_a
            * delta_squared.pow(self.pure_nonanalytic_a - 1.0)
            * delta_squared_derivative
        )
        capital_delta_safe = capital_delta.clamp_min(tiny)
        capital_delta_power = capital_delta_safe.pow(self.pure_nonanalytic_b)
        psi_log_derivative = -self.pure_nonanalytic_capital_c * delta_squared_derivative
        derivative = (
            self.pure_nonanalytic_n
            * psi
            * (
                capital_delta_power
                + delta
                * (
                    psi_log_derivative * capital_delta_power
                    + self.pure_nonanalytic_b
                    * capital_delta_safe.pow(self.pure_nonanalytic_b - 1.0)
                    * capital_delta_derivative
                )
            )
        )
        return torch.sum(derivative, dim=-1)

    def alpha_residual(
        self, temperature: Tensor, molar_density: Tensor, composition: Tensor
    ) -> Tensor:
        """Evaluate dimensionless molar residual Helmholtz energy.

        Parameters
        ----------
        temperature
            Temperature in K.
        molar_density
            Molar density in mol/m3.
        composition
            Mole fractions with components on the final axis.

        Returns
        -------
        Tensor
            Dimensionless ``alpha^r = a^r/(RT)``.
        """
        x = normalize_composition(composition)
        reducing_temperature, reducing_density = self._reducing_functions_normalized(x)
        tau = reducing_temperature / temperature
        delta = molar_density / reducing_density
        pure_values = self._evaluate_terms(
            self.pure_n,
            self.pure_d,
            self.pure_t,
            self.pure_decay,
            self.pure_eta,
            self.pure_epsilon,
            self.pure_beta,
            self.pure_gamma,
            self.pure_linear_density,
            self.pure_linear_shift,
            delta,
            tau,
        )
        if self._has_gaob_terms:
            pure_values = pure_values + self._evaluate_gaob(delta, tau)
        if self._has_nonanalytic_terms:
            pure_values = pure_values + self._evaluate_nonanalytic(delta, tau)
        departure_values = self._evaluate_terms(
            self.departure_n,
            self.departure_d,
            self.departure_t,
            self.departure_decay,
            self.departure_eta,
            self.departure_epsilon,
            self.departure_beta,
            self.departure_gamma,
            self.departure_linear_density,
            self.departure_linear_shift,
            delta,
            tau,
        )
        pure = torch.sum(x * pure_values, dim=-1)
        pair_weights = (
            x[..., :, None] * x[..., None, :] * self.departure_scale * self._upper_triangle
        )
        return pure + torch.sum(pair_weights * departure_values, dim=(-2, -1))

    def alpha_residual_delta(
        self, temperature: Tensor, molar_density: Tensor, composition: Tensor
    ) -> Tensor:
        """Return the residual Helmholtz derivative with respect to reduced density."""
        x = normalize_composition(composition)
        reducing_temperature, reducing_density = self._reducing_functions_normalized(x)
        tau = reducing_temperature / temperature
        delta = molar_density / reducing_density
        return self._alpha_residual_delta_reduced(delta, tau, x)

    def _alpha_residual_delta_reduced(
        self,
        delta: Tensor,
        tau: Tensor,
        x: Tensor,
    ) -> Tensor:
        """Evaluate ``d(alpha_r)/d(delta)`` from reduced state variables."""
        pure_derivatives = self._evaluate_terms_delta(
            self.pure_n,
            self.pure_d,
            self.pure_t,
            self.pure_decay,
            self.pure_eta,
            self.pure_epsilon,
            self.pure_beta,
            self.pure_gamma,
            self.pure_linear_density,
            self.pure_linear_shift,
            delta,
            tau,
        )
        if self._has_gaob_terms:
            pure_derivatives = pure_derivatives + self._evaluate_gaob_delta(delta, tau)
        if self._has_nonanalytic_terms:
            pure_derivatives = pure_derivatives + self._evaluate_nonanalytic_delta(delta, tau)
        departure_derivatives = self._evaluate_terms_delta(
            self.departure_n,
            self.departure_d,
            self.departure_t,
            self.departure_decay,
            self.departure_eta,
            self.departure_epsilon,
            self.departure_beta,
            self.departure_gamma,
            self.departure_linear_density,
            self.departure_linear_shift,
            delta,
            tau,
        )
        pure = torch.sum(x * pure_derivatives, dim=-1)
        pair_weights = (
            x[..., :, None] * x[..., None, :] * self.departure_scale * self._upper_triangle
        )
        return pure + torch.sum(pair_weights * departure_derivatives, dim=(-2, -1))

    def alpha_ideal(
        self, temperature: Tensor, molar_density: Tensor, composition: Tensor
    ) -> Tensor:
        """Evaluate dimensionless molar ideal-gas Helmholtz energy.

        Parameters
        ----------
        temperature
            Temperature in K.
        molar_density
            Molar density in mol/m3.
        composition
            Mole fractions with components on the final axis.

        Returns
        -------
        Tensor
            Dimensionless ideal contribution ``alpha^0``.

        Raises
        ------
        RuntimeError
            If the model was constructed without ideal Helmholtz terms.
        """
        if not self.has_ideal_terms:
            raise RuntimeError("this multifluid model has no ideal Helmholtz coefficient table")
        x = normalize_composition(composition)
        tau = self.critical_temperature / temperature[..., None]
        thermal = (
            self.ideal_lead_constant
            + self.ideal_lead_tau * tau
            + self.ideal_log_tau * torch.log(tau)
            + self.ideal_tau_log_tau * tau * torch.log(tau)
            + torch.sum(
                self.ideal_power_n * tau[..., :, None].pow(self.ideal_power_t),
                dim=-1,
            )
        )
        planck_argument = self.ideal_planck_theta * tau[..., :, None]
        thermal = thermal + torch.sum(
            self.ideal_planck_n * torch.log(-torch.expm1(-planck_argument)),
            dim=-1,
        )
        gerg_argument = (self.ideal_gerg_theta * tau[..., :, None]).abs()
        safe_argument = gerg_argument.clamp_min(torch.finfo(gerg_argument.dtype).tiny)
        log_sinh = (
            safe_argument
            + torch.log1p(-torch.exp(-2.0 * safe_argument))
            - torch.log(safe_argument.new_tensor(2.0))
        )
        log_cosh = torch.logaddexp(safe_argument, -safe_argument) - torch.log(
            safe_argument.new_tensor(2.0)
        )
        gerg_value = torch.where(self.ideal_gerg_sign > 0.0, log_sinh, -log_cosh)
        thermal = thermal + torch.sum(self.ideal_gerg_n * gerg_value, dim=-1)
        pure = (
            torch.log(molar_density[..., None] / self.critical_density)
            + self.ideal_gas_scale * thermal
        )
        return torch.sum(x * pure + torch.special.xlogy(x, x), dim=-1)

    def alpha_total(
        self, temperature: Tensor, molar_density: Tensor, composition: Tensor
    ) -> Tensor:
        """Evaluate total dimensionless molar Helmholtz energy.

        Parameters
        ----------
        temperature
            Temperature in K.
        molar_density
            Molar density in mol/m3.
        composition
            Mole fractions on the final axis.

        Returns
        -------
        Tensor
            ``alpha^0 + alpha^r``.
        """
        return self.alpha_ideal(temperature, molar_density, composition) + self.alpha_residual(
            temperature, molar_density, composition
        )

    def residual_helmholtz_rt(self, temperature: Tensor, volume: Tensor, moles: Tensor) -> Tensor:
        """Evaluate extensive reduced residual Helmholtz energy.

        Parameters
        ----------
        temperature
            Temperature in K.
        volume
            Total volume in m3.
        moles
            Component amounts in mol on the final axis.

        Returns
        -------
        Tensor
            Dimensionless extensive ``A^R/(RT)``.
        """
        total = moles.sum(dim=-1)
        x = moles / total[..., None]
        density = total / volume
        return total * self.alpha_residual(temperature, density, x)

    def helmholtz_rt(self, temperature: Tensor, volume: Tensor, moles: Tensor) -> Tensor:
        """Evaluate extensive total reduced Helmholtz energy.

        Parameters
        ----------
        temperature
            Temperature in K.
        volume
            Total volume in m3.
        moles
            Component amounts in mol on the final axis.

        Returns
        -------
        Tensor
            Dimensionless extensive ``A/(RT)``.
        """
        total = moles.sum(dim=-1)
        x = moles / total[..., None]
        return total * self.alpha_total(temperature, total / volume, x)

    def molar_helmholtz_energy(
        self, temperature: Tensor, molar_density: Tensor, composition: Tensor
    ) -> Tensor:
        """Return total molar Helmholtz energy.

        Parameters
        ----------
        temperature, molar_density, composition
            Temperature in K, density in mol/m3, and final-axis mole fractions.

        Returns
        -------
        Tensor
            Molar Helmholtz energy in J/mol.
        """
        return (
            self.gas_constant
            * temperature
            * self.alpha_total(temperature, molar_density, composition)
        )

    def molar_entropy(
        self, temperature: Tensor, molar_density: Tensor, composition: Tensor
    ) -> Tensor:
        """Return total molar entropy by temperature differentiation.

        Parameters
        ----------
        temperature, molar_density, composition
            Temperature in K, fixed density in mol/m3, and mole fractions.

        Returns
        -------
        Tensor
            Molar entropy in J/(mol K).
        """
        derivative: Tensor = torch.func.grad(
            lambda current: self.molar_helmholtz_energy(
                current,
                molar_density,
                composition,
            ).sum()
        )(temperature)
        return -derivative

    def molar_internal_energy(
        self, temperature: Tensor, molar_density: Tensor, composition: Tensor
    ) -> Tensor:
        """Return total molar internal energy.

        Parameters
        ----------
        temperature, molar_density, composition
            Temperature in K, density in mol/m3, and mole fractions.

        Returns
        -------
        Tensor
            Molar internal energy in J/mol.
        """
        helmholtz = self.molar_helmholtz_energy(temperature, molar_density, composition)
        return helmholtz + temperature * self.molar_entropy(temperature, molar_density, composition)

    def molar_heat_capacity_cv(
        self, temperature: Tensor, molar_density: Tensor, composition: Tensor
    ) -> Tensor:
        """Return isochoric molar heat capacity.

        Parameters
        ----------
        temperature, molar_density, composition
            Temperature in K, fixed density in mol/m3, and mole fractions.

        Returns
        -------
        Tensor
            ``C_v`` in J/(mol K).
        """
        derivative: Tensor = torch.func.grad(
            lambda current: self.molar_internal_energy(
                current,
                molar_density,
                composition,
            ).sum()
        )(temperature)
        return derivative

    def molar_heat_capacity_cp(
        self, temperature: Tensor, molar_density: Tensor, composition: Tensor
    ) -> Tensor:
        """Return isobaric molar heat capacity from Helmholtz derivatives.

        Parameters
        ----------
        temperature, molar_density, composition
            Temperature in K, density in mol/m3, and mole fractions.

        Returns
        -------
        Tensor
            ``C_p`` in J/(mol K).
        """
        volume = molar_density.reciprocal()
        cv = self.molar_heat_capacity_cv(temperature, molar_density, composition)
        dp_dt: Tensor = torch.func.grad(
            lambda current: self.pressure(current, volume, composition).sum()
        )(temperature)
        dp_drho: Tensor = torch.func.grad(
            lambda density: self.pressure(
                temperature,
                density.reciprocal(),
                composition,
            ).sum()
        )(molar_density)
        return cv + temperature * dp_dt.square() / (molar_density.square() * dp_drho)

    def molar_enthalpy(
        self, temperature: Tensor, molar_density: Tensor, composition: Tensor
    ) -> Tensor:
        """Return total molar enthalpy.

        Parameters
        ----------
        temperature, molar_density, composition
            Temperature in K, density in mol/m3, and mole fractions.

        Returns
        -------
        Tensor
            Molar enthalpy in J/mol.
        """
        volume = molar_density.reciprocal()
        return (
            self.molar_internal_energy(temperature, molar_density, composition)
            + self.pressure(temperature, volume, composition) * volume
        )

    def molar_gibbs_energy(
        self, temperature: Tensor, molar_density: Tensor, composition: Tensor
    ) -> Tensor:
        """Return total molar Gibbs energy.

        Parameters
        ----------
        temperature, molar_density, composition
            Temperature in K, density in mol/m3, and mole fractions.

        Returns
        -------
        Tensor
            Molar Gibbs energy in J/mol.
        """
        volume = molar_density.reciprocal()
        return (
            self.molar_helmholtz_energy(temperature, molar_density, composition)
            + self.pressure(temperature, volume, composition) * volume
        )

    def chemical_potentials(
        self, temperature: Tensor, molar_volume: Tensor, composition: Tensor
    ) -> Tensor:
        """Return total component chemical potentials.

        Parameters
        ----------
        temperature
            Temperature in K.
        molar_volume
            Homogeneous molar volume in m3/mol.
        composition
            Mole fractions on the final axis.

        Returns
        -------
        Tensor
            Component chemical potentials in J/mol.
        """
        x = normalize_composition(composition)
        reduced: Tensor = torch.func.grad(
            lambda moles: self.helmholtz_rt(
                temperature,
                molar_volume,
                moles,
            ).sum()
        )(x)
        return self.gas_constant * temperature * reduced

    def speed_of_sound(
        self, temperature: Tensor, molar_density: Tensor, composition: Tensor
    ) -> Tensor:
        """Return homogeneous-phase speed of sound.

        Parameters
        ----------
        temperature, molar_density, composition
            Temperature in K, density in mol/m3, and mole fractions.

        Returns
        -------
        Tensor
            Speed of sound in m/s.
        """
        x = normalize_composition(composition)
        dp_drho: Tensor = torch.func.grad(
            lambda density: self.pressure(
                temperature,
                density.reciprocal(),
                x,
            ).sum()
        )(molar_density)
        cv = self.molar_heat_capacity_cv(temperature, molar_density, x)
        cp = self.molar_heat_capacity_cp(temperature, molar_density, x)
        mixture_molar_mass = torch.sum(x * self.molar_mass, dim=-1)
        return torch.sqrt((cp / cv) * dp_drho / mixture_molar_mass)

    def pressure(self, temperature: Tensor, molar_volume: Tensor, composition: Tensor) -> Tensor:
        """Return pressure from the exact residual-Helmholtz density derivative."""
        x = normalize_composition(composition)
        molar_density = molar_volume.reciprocal()
        reducing_temperature, reducing_density = self._reducing_functions_normalized(x)
        delta = molar_density / reducing_density
        tau = reducing_temperature / temperature
        alpha_delta = self._alpha_residual_delta_reduced(delta, tau, x)
        return self.gas_constant * temperature * molar_density * (1.0 + delta * alpha_delta)

    def molar_volume(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Solve and select a homogeneous density root.

        Phase-specific states first use a differentiable logarithmic-density
        Newton solve.  If it fails, a logarithmic scan brackets every
        mechanically admissible positive root before bisection and Newton
        polishing.  This conservative fallback matters because Helmholtz
        mixture models can have three density roots below the mixture critical
        locus, and an initial guess can cross a spinodal during phase-envelope
        calculations.
        """
        x = normalize_composition(composition)
        if x.shape[-1] != len(self.names):
            raise ValueError("multifluid composition has the wrong number of components")
        if phase not in ("vapor", "liquid", "stable"):
            raise ValueError(f"unknown phase root {phase!r}")
        batch_shape = torch.broadcast_shapes(
            temperature.shape,
            pressure.shape,
            x.shape[:-1],
        )
        if batch_shape:
            temperature = torch.broadcast_to(temperature, batch_shape)
            pressure = torch.broadcast_to(pressure, batch_shape)
            x = torch.broadcast_to(x, (*batch_shape, len(self.names)))
            if phase in ("vapor", "liquid"):
                return self._batched_phase_volume(temperature, pressure, x, phase)
            if phase == "stable":
                return self._batched_stable_volume(temperature, pressure, x)

        ideal_density = pressure / (self.gas_constant * temperature)
        density_scale = torch.sum(x * self.critical_density)
        reducing_temperature, _ = self.reducing_functions(x)

        def residual(current: Tensor) -> Tensor:
            volume = torch.exp(-current)
            return (self.pressure(temperature, volume, x) - pressure) / pressure

        needs_implicit_gradient = (
            temperature.requires_grad
            or pressure.requires_grad
            or x.requires_grad
            or any(parameter.requires_grad for parameter in self.parameters())
        )

        def volume_with_implicit_gradient(current: Tensor) -> Tensor:
            """Restore root derivatives after detached numerical iteration."""
            if needs_implicit_gradient:
                value = residual(current)
                derivative: Tensor = torch.func.grad(residual)(current)
                current = current - value / derivative
            return torch.exp(-current)

        # Locate a phase-specific root with detached, safeguarded secant steps.
        # One centered slope initializes the iteration; subsequent slopes reuse
        # adjacent residual values instead of evaluating pressure twice more at
        # every step. One exact correction at the converged root restores the
        # implicit state and parameter gradients. The scan below remains the
        # conservative fallback near spinodals or when the conventional seed
        # lies in the basin of another root.
        if phase != "stable":
            density = (
                ideal_density
                if phase == "vapor"
                else torch.maximum(ideal_density, 3.0 * density_scale)
            )
            log_density = torch.log(density)
            slope_offset = log_density.new_tensor(1.0e-4)
            value = residual(log_density)
            derivative = (
                residual(log_density + slope_offset) - residual(log_density - slope_offset)
            ) / (2.0 * slope_offset)
            for _ in range(20):
                if float(value.detach().abs()) <= 1.0e-10:
                    solved_density = torch.exp(log_density)
                    phase_consistent = (
                        solved_density <= density_scale
                        if phase == "vapor"
                        else solved_density >= density_scale
                    )
                    if bool((temperature >= reducing_temperature) | phase_consistent):
                        return volume_with_implicit_gradient(log_density)
                    break
                step = torch.clamp(-value / derivative, -0.5, 0.5)
                if not bool(torch.isfinite(step)):
                    break
                next_log_density = (log_density + step).detach()
                next_value = residual(next_log_density)
                log_density_step = next_log_density - log_density
                next_derivative = (next_value - value) / log_density_step
                if not bool(
                    torch.isfinite(next_derivative)
                    & (next_derivative.abs() > torch.finfo(next_derivative.dtype).tiny)
                ):
                    next_derivative = (
                        residual(next_log_density + slope_offset)
                        - residual(next_log_density - slope_offset)
                    ) / (2.0 * slope_offset)
                log_density = next_log_density
                value = next_value
                derivative = next_derivative

        minimum = torch.minimum(ideal_density * 1.0e-4, density_scale * 1.0e-8)
        maximum = torch.maximum(
            ideal_density * 10.0,
            100.0 * torch.max(self.critical_density),
        )
        grid = torch.logspace(
            float(torch.log10(minimum.detach())),
            float(torch.log10(maximum.detach())),
            96,
            dtype=temperature.dtype,
            device=temperature.device,
        )

        brackets: list[tuple[Tensor, Tensor]] = []
        left = torch.log(grid[0])
        left_value = residual(left)
        for density in grid[1:]:
            right = torch.log(density)
            right_value = residual(right)
            finite = bool(torch.isfinite(left_value) & torch.isfinite(right_value))
            changes_sign = bool(torch.signbit(left_value) != torch.signbit(right_value))
            if finite and (
                float(left_value.detach()) == 0.0
                or float(right_value.detach()) == 0.0
                or changes_sign
            ):
                brackets.append((left, right))
            left, left_value = right, right_value
        if not brackets:
            raise ConvergenceError("multifluid density solve did not converge")

        roots: list[Tensor] = []
        for left, right in brackets:
            left_value = residual(left)
            midpoint = 0.5 * (left + right)
            for _ in range(80):
                midpoint = 0.5 * (left + right)
                midpoint_value = residual(midpoint)
                if float(midpoint_value.detach().abs()) <= 1.0e-12:
                    break
                if bool(torch.signbit(left_value) != torch.signbit(midpoint_value)):
                    right = midpoint
                else:
                    left = midpoint
                    left_value = midpoint_value

            root = torch.exp(midpoint)
            root_residual = residual(midpoint)
            if not bool(torch.isfinite(root_residual) & (root_residual.abs() <= 1.0e-8)):
                # A sign change across a discontinuity or overflow is not a
                # density root. This occurs in very cold, dense Helmholtz
                # states and must not enter the Gibbs-root comparison.
                continue
            if not roots or abs(float((root / roots[-1] - 1.0).detach())) > 1.0e-8:
                roots.append(root)

        if not roots:
            raise ConvergenceError("multifluid density solve did not find an admissible root")
        if phase == "vapor":
            density = roots[0]
        elif phase == "liquid":
            density = roots[-1]
        elif phase == "stable":
            gibbs: list[Tensor] = []
            for density_root in roots:
                volume = density_root.reciprocal()
                z = pressure * volume / (self.gas_constant * temperature)
                residual_helmholtz = self.residual_helmholtz_rt(temperature, volume, x)
                gibbs.append(residual_helmholtz + z - 1.0 - torch.log(z))
            density = roots[int(torch.argmin(torch.stack(gibbs)).detach())]
        return volume_with_implicit_gradient(torch.log(density))

    def _batched_stable_volume(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
    ) -> Tensor:
        """Select the lower-Gibbs outer root for a batch of stable states."""
        vapor_volume, vapor_converged = self._batched_phase_volume(
            temperature,
            pressure,
            composition,
            "vapor",
            return_convergence=True,
        )
        liquid_volume, liquid_converged = self._batched_phase_volume(
            temperature,
            pressure,
            composition,
            "liquid",
            return_convergence=True,
        )

        def reduced_gibbs(volume: Tensor) -> Tensor:
            compressibility = pressure * volume / (self.gas_constant * temperature)
            residual_helmholtz = self.residual_helmholtz_rt(
                temperature,
                volume,
                composition,
            )
            return residual_helmholtz + compressibility - 1.0 - torch.log(compressibility)

        vapor_gibbs = reduced_gibbs(vapor_volume)
        liquid_gibbs = reduced_gibbs(liquid_volume)
        stable_volume = torch.where(
            liquid_converged & (~vapor_converged | (liquid_gibbs < vapor_gibbs)),
            liquid_volume,
            vapor_volume,
        )
        # The sub-reducing candidate ensemble above is used by both searches,
        # so a lone converged outer root is admissible for a single-root state.
        # When both exist, the Gibbs comparison selects the stable branch.
        stable_converged = liquid_converged | vapor_converged
        if bool(stable_converged.all()):
            return stable_volume

        flat_temperature = temperature.reshape(-1)
        flat_pressure = pressure.reshape(-1)
        flat_composition = composition.reshape(-1, len(self.names))
        flat_volume = stable_volume.reshape(-1)
        failed_indices = torch.nonzero(
            ~stable_converged.reshape(-1),
            as_tuple=False,
        ).flatten()
        fallback = torch.stack(
            [
                self.molar_volume(
                    flat_temperature[index],
                    flat_pressure[index],
                    flat_composition[index],
                    "stable",
                )
                for index in failed_indices.tolist()
            ]
        )
        return flat_volume.index_copy(0, failed_indices, fallback).reshape(temperature.shape)

    @overload
    def _batched_phase_volume(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: Literal["liquid", "vapor"],
        *,
        return_convergence: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def _batched_phase_volume(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: Literal["liquid", "vapor"],
        *,
        return_convergence: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def _batched_phase_volume(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: Literal["liquid", "vapor"],
        *,
        return_convergence: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Solve independent phase-specific states with one tensor workload.

        A conventional phase-specific seed is advanced for the complete batch.
        Extra density seeds are evaluated for failed states and below the
        reducing temperature, where a conventional seed can converge to a
        middle root. Selecting the lowest or highest converged root handles
        multiple-root states and cases in which the requested vapor-like
        branch is the sole, dense root above saturation pressure. Only states
        for which every seed fails use the conservative scalar root scan.
        """
        batch_shape = temperature.shape
        flat_temperature = temperature.reshape(-1)
        flat_pressure = pressure.reshape(-1)
        flat_composition = composition.reshape(-1, len(self.names))
        ideal_density = flat_pressure / (self.gas_constant * flat_temperature)
        density_scale = torch.sum(flat_composition * self.critical_density, dim=-1)
        reducing_temperature, _ = self.reducing_functions(flat_composition)
        density = (
            ideal_density if phase == "vapor" else torch.maximum(ideal_density, 3.0 * density_scale)
        )
        log_density = torch.log(density)

        def residual(current: Tensor) -> Tensor:
            volume = torch.exp(-current)
            return (
                self.pressure(
                    flat_temperature,
                    volume,
                    flat_composition,
                )
                - flat_pressure
            ) / flat_pressure

        # A fixed workload avoids per-state Python dispatch and device
        # synchronization. A centered log-density slope is much faster here
        # than nesting reverse-mode AD through the already differentiated
        # pressure kernel. It is used only to locate the root.
        slope_offset = log_density.new_tensor(1.0e-4)
        for _ in range(10):
            value = residual(log_density)
            derivative = (
                residual(log_density + slope_offset) - residual(log_density - slope_offset)
            ) / (2.0 * slope_offset)
            step = torch.clamp(-value / derivative, -0.5, 0.5)
            step = torch.where(torch.isfinite(step), step, torch.zeros_like(step))
            log_density = (log_density + step).detach()

        primary_residual = residual(log_density)
        primary_density = torch.exp(log_density)
        phase_consistent = (
            primary_density <= density_scale
            if phase == "vapor"
            else primary_density >= density_scale
        )
        primary_converged = (
            torch.isfinite(primary_residual)
            & (primary_residual.abs() <= 1.0e-8)
            & ((flat_temperature >= reducing_temperature) | phase_consistent)
        )
        preliminary_converged = primary_converged

        needs_candidates = ~primary_converged | (flat_temperature < reducing_temperature)
        if bool(needs_candidates.any()):
            candidate_indices = torch.nonzero(
                needs_candidates,
                as_tuple=False,
            ).flatten()
            candidate_temperature = flat_temperature[candidate_indices]
            candidate_pressure = flat_pressure[candidate_indices]
            candidate_composition = flat_composition[candidate_indices]
            candidate_ideal_density = ideal_density[candidate_indices]
            candidate_density_scale = density_scale[candidate_indices]
            candidate_density = torch.stack(
                (
                    0.2 * candidate_ideal_density,
                    candidate_ideal_density,
                    2.0 * candidate_ideal_density,
                    0.25 * candidate_density_scale,
                    0.5 * candidate_density_scale,
                    0.75 * candidate_density_scale,
                    candidate_density_scale,
                    1.5 * candidate_density_scale,
                    3.0 * candidate_density_scale,
                    10.0 * candidate_density_scale,
                    primary_density[candidate_indices],
                ),
                dim=-1,
            )
            candidate_log_density = torch.log(candidate_density)

            def candidate_residual(current: Tensor) -> Tensor:
                volume = torch.exp(-current)
                return (
                    self.pressure(
                        candidate_temperature[:, None],
                        volume,
                        candidate_composition[:, None, :],
                    )
                    - candidate_pressure[:, None]
                ) / candidate_pressure[:, None]

            for _ in range(10):
                value = candidate_residual(candidate_log_density)
                derivative = (
                    candidate_residual(candidate_log_density + slope_offset)
                    - candidate_residual(candidate_log_density - slope_offset)
                ) / (2.0 * slope_offset)
                step = torch.clamp(-value / derivative, -0.5, 0.5)
                step = torch.where(
                    torch.isfinite(step),
                    step,
                    torch.zeros_like(step),
                )
                candidate_log_density = (candidate_log_density + step).detach()

            candidate_value = candidate_residual(candidate_log_density)
            candidate_converged = torch.isfinite(candidate_value) & (
                candidate_value.abs() <= 1.0e-8
            )
            if phase == "vapor":
                ranked_density = torch.where(
                    candidate_converged,
                    candidate_log_density,
                    torch.full_like(candidate_log_density, torch.inf),
                )
                candidate_index = torch.argmin(
                    ranked_density,
                    dim=-1,
                    keepdim=True,
                )
            else:
                ranked_density = torch.where(
                    candidate_converged,
                    candidate_log_density,
                    torch.full_like(candidate_log_density, -torch.inf),
                )
                candidate_index = torch.argmax(
                    ranked_density,
                    dim=-1,
                    keepdim=True,
                )
            selected_candidate = torch.gather(
                candidate_log_density,
                -1,
                candidate_index,
            ).squeeze(-1)
            log_density = log_density.index_copy(
                0,
                candidate_indices,
                selected_candidate,
            )
            preliminary_converged = primary_converged.index_copy(
                0,
                candidate_indices,
                candidate_converged.any(dim=-1),
            )

        # Two detached corrections tighten the numerical root without growing
        # an optimization graph through the root-finding history.
        for _ in range(2):
            value = residual(log_density)
            derivative = (
                residual(log_density + slope_offset) - residual(log_density - slope_offset)
            ) / (2.0 * slope_offset)
            step = torch.where(
                torch.isfinite(derivative)
                & (derivative.abs() > torch.finfo(derivative.dtype).tiny),
                value / derivative,
                torch.zeros_like(value),
            )
            log_density = (log_density - step).detach()

        needs_implicit_gradient = (
            temperature.requires_grad
            or pressure.requires_grad
            or composition.requires_grad
            or any(parameter.requires_grad for parameter in self.parameters())
        )
        if needs_implicit_gradient:
            # One exact Newton correction at an already converged root restores
            # the implicit derivatives with respect to state and fitted model
            # parameters; the preceding numerical iterations remain detached.
            value = residual(log_density)
            derivative = torch.func.grad(lambda current: residual(current).sum())(log_density)
            log_density = log_density - value / derivative

        value = residual(log_density)
        solved_density = torch.exp(log_density)
        converged = preliminary_converged & torch.isfinite(value) & (value.abs() <= 1.0e-9)
        volumes = solved_density.reciprocal()
        if return_convergence:
            return volumes.reshape(batch_shape), converged.reshape(batch_shape)
        if bool(converged.all()):
            return volumes.reshape(batch_shape)

        # The conservative scalar scan is retained for the uncommon states
        # for which every batched seed fails. Preserve the already solved
        # entries instead of sending the complete batch through Python.
        failed_indices = torch.nonzero(
            ~converged,
            as_tuple=False,
        ).flatten()
        fallback = torch.stack(
            [
                self.molar_volume(
                    flat_temperature[index],
                    flat_pressure[index],
                    flat_composition[index],
                    phase,
                )
                for index in failed_indices.tolist()
            ]
        )
        return volumes.index_copy(0, failed_indices, fallback).reshape(batch_shape)

    def select_z(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Return the compressibility factor for a selected density root.

        Parameters
        ----------
        temperature, pressure
            Temperature in K and pressure in Pa.
        composition
            Mole fractions on the final axis.
        phase
            Liquid, vapor, or stable-root request.

        Returns
        -------
        Tensor
            Dimensionless ``Z = Pv/(RT)``.
        """
        volume = self.molar_volume(temperature, pressure, composition, phase)
        return pressure * volume / (self.gas_constant * temperature)

    def log_fugacity_coefficients(
        self,
        temperature: Tensor,
        pressure: Tensor,
        composition: Tensor,
        phase: PhaseKind = "stable",
    ) -> Tensor:
        """Return component log fugacity coefficients.

        Parameters
        ----------
        temperature, pressure
            Temperature in K and pressure in Pa.
        composition
            Mole fractions on the final axis.
        phase
            Liquid, vapor, or stable-root request.

        Returns
        -------
        Tensor
            Dimensionless ``ln(phi_i)`` with a final component axis.

        Notes
        -----
        Residual chemical potentials are obtained by differentiating the
        extensive residual Helmholtz energy at fixed volume.
        """
        x = normalize_composition(composition)
        volume = self.molar_volume(temperature, pressure, x, phase)
        residual_mu: Tensor = torch.func.grad(
            lambda moles: self.residual_helmholtz_rt(
                temperature,
                volume,
                moles,
            ).sum()
        )(x)
        z = pressure * volume / (self.gas_constant * temperature)
        return residual_mu - torch.log(z)[..., None]

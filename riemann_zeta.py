"""
Riemann Zeta Function: Comprehensive Implementation
Statistical Data Analysis and Computational Techniques

This module provides multiple algorithms for computing the Riemann zeta function,
statistical analysis of its zeros, and visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, zeta as scipy_zeta
from scipy.optimize import fsolve, brentq
import time
import math
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')


class RiemannZeta:
    """Main class for Riemann Zeta function computations"""

    def __init__(self):
        """Initialize with precomputed Bernoulli numbers"""
        self.bernoulli_numbers = self._compute_bernoulli(20)

    @staticmethod
    def _compute_bernoulli(n):
        """Compute first n Bernoulli numbers using recursive formula"""
        B = [0] * (n + 1)
        B[0] = 1
        for m in range(1, n + 1):
            B[m] = 0
            for k in range(m):
                B[m] += math.comb(m + 1, k) * B[k]
            B[m] = -B[m] / (m + 1)
        return B

    def direct_sum(self, s: complex, N: int = 10000) -> complex:
        """
        Direct summation method for zeta(s)
        Converges only for Re(s) > 1

        Args:
            s: Complex number input
            N: Number of terms to sum

        Returns:
            Approximation of zeta(s)
        """
        if np.real(s) <= 1:
            raise ValueError("Direct sum only converges for Re(s) > 1")

        result = sum(1 / (n ** s) for n in range(1, N + 1))
        return result

    def eta_function(self, s: complex, N: int = 1000) -> complex:
        """
        Dirichlet eta function method (alternating zeta function)
        Converges for Re(s) > 0

        η(s) = (1 - 2^(1-s)) * ζ(s)

        Args:
            s: Complex number input
            N: Number of terms

        Returns:
            Approximation of zeta(s)
        """
        # Compute eta function
        eta = sum((-1) ** (n - 1) / (n ** s) for n in range(1, N + 1))

        # Convert to zeta
        denominator = 1 - 2 ** (1 - s)
        if abs(denominator) < 1e-10:
            raise ValueError("Method fails near s=1")

        return eta / denominator

    def borwein_algorithm(self, s: complex, N: int = 30) -> complex:
        """
        Borwein's algorithm for rapid convergence
        Works for all s ≠ 1

        Args:
            s: Complex number input
            N: Number of terms (typically 20-50 sufficient)

        Returns:
            High-precision approximation of zeta(s)
        """
        # Compute d_n coefficients
        d = np.zeros(N + 1, dtype=complex)
        for n in range(N + 1):
            inner_sum = sum(
                math.comb(n, k) * ((-1) ** k) / ((k + 1) ** s)
                for k in range(n + 1)
            )
            d[n] = inner_sum / (2 ** (n + 1))

        # Sum the series
        eta = sum(d)

        # Convert to zeta
        denominator = 1 - 2 ** (1 - s)
        if abs(denominator) < 1e-10:
            return float('inf')

        return eta / denominator

    def euler_maclaurin(self, s: complex, N: int = 100, K: int = 10) -> complex:
        """
        Euler-Maclaurin formula for improved convergence

        Args:
            s: Complex number input
            N: Truncation point for direct sum
            K: Number of Bernoulli correction terms

        Returns:
            Approximation of zeta(s)
        """
        # Direct sum part
        direct = sum(1 / (n ** s) for n in range(1, N + 1))

        # Tail correction
        tail = N ** (1 - s) / (s - 1)

        # Constant term
        constant = 1 / (2 * N ** s)

        # Bernoulli corrections
        correction = 0
        for k in range(1, min(K, len(self.bernoulli_numbers) // 2)):
            B_2k = self.bernoulli_numbers[2 * k]
            # Binomial coefficient
            binom = 1
            for j in range(2 * k - 1):
                binom *= (s + j) / (j + 1)
            correction += (B_2k / math.factorial(2 * k)) * binom / (N ** (s + 2 * k - 1))

        return direct + tail + constant + correction

    def functional_equation(self, s: complex) -> complex:
        """
        Use functional equation for Re(s) < 0
        ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)

        Args:
            s: Complex number input

        Returns:
            zeta(s) computed via functional equation
        """
        # Compute zeta(1-s) using another method
        one_minus_s = 1 - s
        if np.real(one_minus_s) > 1:
            zeta_complement = self.direct_sum(one_minus_s, N=5000)
        else:
            zeta_complement = self.borwein_algorithm(one_minus_s, N=30)

        # Apply functional equation
        result = (2 ** s) * (np.pi ** (s - 1)) * np.sin(np.pi * s / 2) * \
                 gamma(1 - s) * zeta_complement

        return result

    def adaptive_compute(self, s: complex, precision: str = 'high') -> complex:
        """
        Automatically choose best method based on s value

        Args:
            s: Complex number input
            precision: 'low', 'medium', or 'high'

        Returns:
            zeta(s) computed with appropriate method
        """
        N_values = {'low': 20, 'medium': 30, 'high': 50}
        N = N_values.get(precision, 30)

        re_s = np.real(s)

        # Near pole at s=1
        if abs(s - 1) < 0.1:
            raise ValueError("Too close to pole at s=1")

        # For Re(s) > 2, direct sum is efficient
        if re_s > 2:
            return self.direct_sum(s, N=1000)

        # For Re(s) < 0, use functional equation
        elif re_s < 0:
            return self.functional_equation(s)

        # For 0 < Re(s) < 2, use Borwein
        else:
            return self.borwein_algorithm(s, N=N)


class ZetaZeros:
    """Class for finding and analyzing zeros of zeta function"""

    def __init__(self):
        self.zeta = RiemannZeta()
        # Known first zeros (imaginary parts)
        self.known_zeros = [
            14.134725141734693790, 21.022039638771554993,
            25.010857580145688763, 30.424876125859513210,
            32.935061587739189690, 37.586178158825671257,
            40.918719012147495187, 43.327073280914999519,
            48.005150881167159727, 49.773743678191792542
        ]

    def riemann_siegel_theta(self, t: float) -> float:
        """
        Compute θ(t) for Riemann-Siegel formula

        Args:
            t: Real parameter

        Returns:
            θ(t) value
        """
        return np.imag(
            np.log(gamma(0.25 + 0.5j * t)) -
            0.5 * t * np.log(np.pi)
        )

    def riemann_siegel_z(self, t: float) -> float:
        """
        Compute Z(t) = e^(iθ(t)) ζ(1/2 + it)
        Z(t) is real-valued and zeros of Z correspond to zeros of ζ

        Args:
            t: Real parameter

        Returns:
            Z(t) value (real)
        """
        s = 0.5 + 1j * t
        zeta_val = self.zeta.adaptive_compute(s)
        theta_val = self.riemann_siegel_theta(t)

        # Z(t) = e^(iθ) ζ(1/2+it) should be real
        z_val = np.exp(1j * theta_val) * zeta_val
        return np.real(z_val)

    def find_zeros(self, t_min: float = 0, t_max: float = 100,
                   n_zeros: int = 10) -> List[float]:
        """
        Find zeros of zeta function on critical line

        Args:
            t_min: Minimum t value to search
            t_max: Maximum t value to search
            n_zeros: Number of zeros to find

        Returns:
            List of imaginary parts of zeros
        """
        zeros = []
        t_search = np.linspace(t_min, t_max, 1000)

        # Find sign changes
        z_values = [self.riemann_siegel_z(t) for t in t_search]

        for i in range(len(z_values) - 1):
            if z_values[i] * z_values[i + 1] < 0:  # Sign change
                # Refine with root finding
                try:
                    zero = brentq(self.riemann_siegel_z,
                                 t_search[i], t_search[i + 1])
                    zeros.append(zero)
                    if len(zeros) >= n_zeros:
                        break
                except:
                    pass

        return zeros

    def verify_critical_line(self, t_values: List[float],
                            tolerance: float = 1e-6) -> bool:
        """
        Verify that zeros lie on critical line Re(s) = 1/2

        Args:
            t_values: Imaginary parts of zeros
            tolerance: Numerical tolerance

        Returns:
            True if all zeros on critical line
        """
        for t in t_values:
            s = 0.5 + 1j * t
            zeta_val = self.zeta.adaptive_compute(s)
            if abs(zeta_val) > tolerance:
                return False
        return True


class StatisticalAnalysis:
    """Statistical analysis of zeta function properties"""

    @staticmethod
    def zero_spacing_statistics(zeros: List[float]) -> Dict:
        """
        Compute statistical properties of zero spacings

        Args:
            zeros: List of zero locations (imaginary parts)

        Returns:
            Dictionary of statistics
        """
        zeros = sorted(zeros)
        spacings = np.diff(zeros)

        # Normalize spacings by mean
        mean_spacing = np.mean(spacings)
        normalized_spacings = spacings / mean_spacing

        stats = {
            'mean_spacing': mean_spacing,
            'std_spacing': np.std(spacings),
            'min_spacing': np.min(spacings),
            'max_spacing': np.max(spacings),
            'normalized_mean': np.mean(normalized_spacings),
            'normalized_std': np.std(normalized_spacings),
            'spacings': spacings,
            'normalized_spacings': normalized_spacings
        }

        return stats

    @staticmethod
    def gue_prediction(x: np.ndarray) -> np.ndarray:
        """
        GUE (Gaussian Unitary Ensemble) prediction for spacing distribution

        Args:
            x: Spacing values

        Returns:
            GUE density
        """
        return (32 / np.pi ** 2) * x ** 2 * np.exp(-4 * x ** 2 / np.pi)

    @staticmethod
    def pair_correlation(zeros: List[float], max_distance: float = 3.0) -> Tuple:
        """
        Compute pair correlation function

        Args:
            zeros: List of zero locations
            max_distance: Maximum distance to consider

        Returns:
            (distances, correlation function)
        """
        zeros = sorted(zeros)
        n = len(zeros)

        # Compute all pairwise distances
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = zeros[j] - zeros[i]
                if dist <= max_distance:
                    distances.append(dist)

        return distances


class ConvergenceAnalyzer:
    """Analyze convergence properties of different methods"""

    def __init__(self):
        self.zeta = RiemannZeta()

    def compare_methods(self, s: complex, N_values: List[int]) -> Dict:
        """
        Compare convergence of different methods

        Args:
            s: Test point
            N_values: List of truncation values to test

        Returns:
            Dictionary of results
        """
        # Get reference value
        reference = scipy_zeta(s) if np.isreal(s) and s > 1 else \
                    self.zeta.borwein_algorithm(s, N=50)

        results = {
            'direct_sum': [],
            'eta_function': [],
            'borwein': [],
            'euler_maclaurin': [],
            'reference': reference
        }

        for N in N_values:
            # Direct sum (only if Re(s) > 1)
            if np.real(s) > 1:
                try:
                    val = self.zeta.direct_sum(s, N=N)
                    error = abs(val - reference)
                    results['direct_sum'].append({
                        'N': N, 'value': val, 'error': error
                    })
                except:
                    pass

            # Eta function
            try:
                val = self.zeta.eta_function(s, N=N)
                error = abs(val - reference)
                results['eta_function'].append({
                    'N': N, 'value': val, 'error': error
                })
            except:
                pass

            # Borwein
            try:
                val = self.zeta.borwein_algorithm(s, N=N)
                error = abs(val - reference)
                results['borwein'].append({
                    'N': N, 'value': val, 'error': error
                })
            except:
                pass

            # Euler-Maclaurin
            try:
                val = self.zeta.euler_maclaurin(s, N=N, K=min(N//10, 10))
                error = abs(val - reference)
                results['euler_maclaurin'].append({
                    'N': N, 'value': val, 'error': error
                })
            except:
                pass

        return results

    def benchmark_performance(self, s: complex, n_iterations: int = 100) -> Dict:
        """
        Benchmark execution time of different methods

        Args:
            s: Test point
            n_iterations: Number of timing iterations

        Returns:
            Dictionary of timing results
        """
        methods = {
            'direct_sum': lambda: self.zeta.direct_sum(s, N=1000) if np.real(s) > 1 else None,
            'eta_function': lambda: self.zeta.eta_function(s, N=1000),
            'borwein': lambda: self.zeta.borwein_algorithm(s, N=30),
            'euler_maclaurin': lambda: self.zeta.euler_maclaurin(s, N=100, K=10),
            'adaptive': lambda: self.zeta.adaptive_compute(s, precision='high')
        }

        results = {}

        for name, method in methods.items():
            try:
                start = time.perf_counter()
                for _ in range(n_iterations):
                    value = method()
                    if value is None:
                        break
                end = time.perf_counter()

                if value is not None:
                    avg_time = (end - start) / n_iterations * 1000  # ms
                    results[name] = {
                        'avg_time_ms': avg_time,
                        'value': value
                    }
            except Exception as e:
                results[name] = {'error': str(e)}

        return results


def validate_known_values():
    """Validate implementation against known exact values"""
    zeta = RiemannZeta()

    test_cases = [
        (2, np.pi**2 / 6, "ζ(2) = π²/6"),
        (4, np.pi**4 / 90, "ζ(4) = π⁴/90"),
        (6, np.pi**6 / 945, "ζ(6) = π⁶/945"),
        (0, -0.5, "ζ(0) = -1/2"),
        (-1, -1/12, "ζ(-1) = -1/12"),
    ]

    print("=" * 70)
    print("VALIDATION AGAINST KNOWN VALUES")
    print("=" * 70)

    for s, exact, description in test_cases:
        if s > 0:
            computed = zeta.borwein_algorithm(s, N=40)
        else:
            computed = zeta.functional_equation(s)

        error = abs(computed - exact)
        rel_error = error / abs(exact) if exact != 0 else error

        print(f"\n{description}")
        print(f"  Exact:     {exact:.15f}")
        print(f"  Computed:  {computed:.15f}")
        print(f"  Abs Error: {error:.2e}")
        print(f"  Rel Error: {rel_error:.2e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("Riemann Zeta Function - Comprehensive Implementation")
    print("=" * 70)

    # Validate against known values
    validate_known_values()

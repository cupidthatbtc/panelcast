"""Prior configuration for Bayesian hierarchical model.

This module defines the hyperparameters for the album score model priors.
All priors are configurable to support sensitivity analysis.

Prior Roles:
- mu_artist: Population mean of artist quality (centering artist effects)
- sigma_artist: Between-artist variance controlling partial pooling strength
  - Large sigma_artist -> less pooling, artists estimated independently
  - Small sigma_artist -> more pooling, artists shrunk toward population mean
- sigma_rw: Innovation scale for time-varying artist effects (random walk)
  - Controls how much an artist's quality changes between albums
  - Smaller values -> smoother career trajectories
- rho: AR(1) coefficient for album-to-album score dependency
  - Captures momentum: positive rho -> hot streaks, negative -> regression to mean
- beta: Fixed effect coefficients for covariates (genre PCA, release year, etc.)
- sigma_obs: Observation-level noise (unexplained variance per album)
- sigma_ref: Observation noise at the reference review count (median n_reviews).
  When sigma-ref mode is active (n_ref provided and heteroscedastic mode on),
  the model samples sigma_ref instead of sigma_obs and derives:
      sigma_obs = sigma_ref * n_ref^n_exponent
  This breaks the multiplicative funnel between sigma_obs and n_exponent.
- n_exponent: Scaling exponent for heteroscedastic observation noise
  - sigma_scaled = sigma_obs / n_reviews^exponent
  - Higher exponent -> more noise reduction for albums with many reviews
  - exponent=0 -> homoscedastic (constant noise)
"""

from dataclasses import dataclass

# Beta boundary squeeze / mu-clip epsilon. Shared so inference
# (PriorConfig.beta_boundary_eps default) and the prediction path
# (likelihoods._beta_predict_draws) stay consistent at the default.
DEFAULT_BETA_BOUNDARY_EPS = 1e-3

# Beta-Binomial effective-rater cap. The Beta overdispersion phi (not n) sets the
# implied-score precision, so capping the rater count past ~1e4/span costs almost
# no information while bounding total_count (= span*n) out of the range where the
# float32 BetaBinomial.log_prob surface turns jagged and stalls NUTS. Shared so
# inference (PriorConfig.betabinom_max_n_reviews) and the predict path agree.
DEFAULT_BETABINOM_MAX_N = 100.0


@dataclass(frozen=True)
class PriorConfig:
    """Hyperparameter configuration for album score model priors.

    All parameters are frozen to ensure immutability during model fitting.

    Attributes:
        mu_artist_loc: Location (mean) for artist effect population mean prior.
            Default 0.0 centers artist effects around zero (deviations from baseline).
        mu_artist_scale: Scale (std) for artist effect population mean prior.
            Default 1.0 allows moderate uncertainty in population center.
        sigma_artist_scale: Scale for HalfNormal prior on artist effect dispersion.
            Default 0.5 encourages moderate pooling. Lower values -> more pooling.
        sigma_rw_scale: Scale for HalfNormal prior on random walk innovation.
            Default 0.1 produces smooth career trajectories. Controls how much
            an artist's quality can change between consecutive albums.
            Smaller values -> more stable artist effects over time.
        rho_loc: Location (mean) for AR(1) coefficient prior.
            Default 0.0 centers the autoregressive coefficient at zero,
            expressing no prior belief about momentum direction.
        rho_scale: Scale for AR(1) coefficient prior.
            Default 0.3 allows moderate uncertainty while keeping most prior
            mass on reasonable AR coefficients (roughly -0.6 to 0.6).
        beta_loc: Location for fixed effect coefficient priors.
            Default 0.0 centers effects at zero (no effect assumption).
        beta_scale: Scale for fixed effect coefficient priors.
            Default 1.0 is weakly informative for standardized features.
        sigma_obs_scale: Scale for HalfNormal prior on observation noise.
            Default 1.0 allows moderate observation-level variance.
        sigma_ref_scale: Scale for HalfNormal prior on sigma_ref (noise at
            reference review count). Used when n_ref is provided
            (sigma-ref reparameterization mode). Default 1.0 is weakly
            informative for standardized scores.
        n_exponent_alpha: Alpha (concentration1) parameter for Beta prior on
            learned n_exponent. Default 2.0. (Legacy - use logit-normal instead)
        n_exponent_beta: Beta (concentration0) parameter for Beta prior on
            learned n_exponent. Default 4.0. (Legacy - use logit-normal instead)
            Note: Beta(2, 4) has mode at 0.25 and mean at 0.33, centering
            prior mass on cube-root-like scaling for heteroscedastic noise.
        n_exponent_loc: Location parameter for logit-normal prior on n_exponent.
            Default -2.2 maps to a mode near 0.10 via the sigmoid transform,
            reflecting the fitted posterior (~0.002 on AOTY data) rather than
            the iid-averaging value of 0.5. Logit-normal is the recommended
            prior type as it avoids funnel geometry issues.
        n_exponent_scale: Scale parameter for logit-normal prior on n_exponent.
            Default 1.0 gives reasonable spread in [0,1] after sigmoid transform.
    """

    mu_artist_loc: float = 0.0
    mu_artist_scale: float = 1.0
    sigma_artist_scale: float = 0.5
    sigma_rw_scale: float = 0.1
    rho_loc: float = 0.0
    rho_scale: float = 0.3
    beta_loc: float = 0.0
    beta_scale: float = 1.0
    sigma_obs_scale: float = 1.0
    sigma_ref_scale: float = 1.0
    n_exponent_alpha: float = 2.0
    n_exponent_beta: float = 4.0
    # Logit-normal prior parameters for n_exponent (new default)
    # Default -2.2 puts the prior mode near 0.10 (sigmoid(-2.2) ~ 0.0997):
    # the fitted posterior sits near 0 (~0.002), far below the iid-ratings
    # value of 0.5 the old 0.0 default implied — AOTY score aggregation is
    # not iid averaging. (Phase-8 grid: 0.5 remains worth testing.)
    n_exponent_loc: float = -2.2
    n_exponent_scale: float = 1.0  # reasonable spread in logit space
    # Likelihood degrees of freedom (Student-t; inf => Normal)
    likelihood_df: float = 4.0  # Student-t(4) for heavier tails
    # sigma_rw prior type: "lognormal" removes zero-boundary pile-up for better
    # NUTS mixing; "halfnormal" is the legacy behavior.
    sigma_rw_prior_type: str = "lognormal"
    # LogNormal parameters for sigma_rw (only used when sigma_rw_prior_type == "lognormal").
    # LogNormal(loc, sigma) means the underlying Normal in log-space has mean=loc
    # and std=sigma.  NOT the log of a scale parameter.
    # Default LogNormal(-2.8, 0.6): median=0.061, mean=0.074, 95th=0.167.
    sigma_rw_lognormal_loc: float = -2.8
    sigma_rw_lognormal_sigma: float = 0.6
    # Latent process for time-varying artist effects: "rw" (random walk,
    # legacy default) or "ar1" (stationary deviations; nests rw at phi=1).
    latent_process: str = "rw"
    # TruncatedNormal prior parameters for the AR(1) latent persistence phi
    # (only used when latent_process == "ar1"; truncated to (-0.99, 0.99)).
    phi_loc: float = 0.0
    phi_scale: float = 0.5
    # Target transform: "identity" (soft-clip on mu) or "offset_logit"
    # (Smithson-Verkuilen logit; model runs on logit scale). Deliberately NOT
    # flipped with the 0.5.0 pipeline default: this is the direct-constructor
    # baseline the tests and the likelihood_parity golden are pinned to, and
    # the orchestrator always passes target_transform explicitly — the
    # pipeline default lives in PipelineConfig/base.yaml, not here.
    target_transform: str = "identity"
    # Half-count continuity offset for the offset-logit transform.
    logit_offset: float = 0.5
    # Likelihood family. "studentt" (df from likelihood_df; df>=100 behaves as
    # Normal) and "normal" are symmetric. The skew/bounded candidates target the
    # left-skewed, bounded score distribution:
    #   "skew_studentt" — a sinh-arcsinh skew-t: StudentT(df) pushed through a
    #     sinh-arcsinh transform with a learned skewness, then located/scaled.
    #   "beta" — the score rescaled to (0, 1) via target_bounds (boundary
    #     squeeze) and modeled with a mean-precision Beta, affine-mapped back to
    #     the score scale so {prefix}y stays on the natural scale.
    #   "beta_binomial" — the score modeled as the mean of n_obs aggregated
    #     ratings (Beta-Binomial); bounded, left-skewed and n-dependent noise fall
    #     out of one generative story. Inherently discrete (subsumes discretize).
    likelihood_family: str = "studentt"
    # skew_studentt: skewness prior (sinh-arcsinh epsilon) and the fixed tail
    # weight delta (1.0 = pure skew, no extra kurtosis beyond the StudentT base).
    skew_loc: float = 0.0
    skew_scale: float = 0.5
    skew_tailweight: float = 1.0
    # beta: Gamma(concentration, rate) prior on the Beta precision phi. The
    # default Gamma(2, 0.1) has mean 20 (moderate precision on the (0,1) scale).
    beta_precision_concentration: float = 2.0
    beta_precision_rate: float = 0.1
    # Boundary squeeze: observed scores are clamped this far inside the bounds so
    # exact-boundary observations have finite Beta density.
    beta_boundary_eps: float = DEFAULT_BETA_BOUNDARY_EPS
    # beta_binomial: Gamma(concentration, rate) prior on the Beta-Binomial
    # overdispersion phi (mirrors the Beta precision prior).
    betabinom_precision_concentration: float = 2.0
    betabinom_precision_rate: float = 0.1
    # beta_binomial: cap on the effective rater count used in the likelihood (see
    # DEFAULT_BETABINOM_MAX_N). Bounds total_count into the float32-smooth range so
    # NUTS does not stall on mega-reviewed albums; the phi floor means ~no info is
    # lost. Raise only if span*n stays below ~1e4 at your scale.
    betabinom_max_n_reviews: float = DEFAULT_BETABINOM_MAX_N
    # split_normal: Normal(loc, scale) prior on the log scale-ratio
    # log(sigma_R / sigma_L). 0 recovers a symmetric Normal; negative values
    # lengthen the left tail (sigma_L > sigma_R). Adds site {prefix}split_log_ratio.
    split_scale_ratio_loc: float = 0.0
    split_scale_ratio_scale: float = 0.5
    # mixture: two-component Normal mixture anchored so its mean is exactly mu, so
    # the level stays with mu_artist (no location ridge between mu and a free offset
    # center). A single separation delta in sigma units (LogNormal => positive, so
    # loc_0 < loc_1 with no label-switching) splits the components about mu:
    #   loc_0 = mu - (1-w)*delta*sigma   (lower / flop tail)
    #   loc_1 = mu +     w*delta*sigma   (upper / dense cluster)
    # mix_weight is the Beta weight on the lower component; per-component scales
    # come from mix_log_scale_ratio (sigma*exp(+/-r/2)). Adds sites {prefix}mix_sep,
    # {prefix}mix_weight, {prefix}mix_log_scale_ratio.
    mix_sep_loc: float = 0.0
    mix_sep_scale: float = 0.75
    mix_weight_a: float = 2.0
    mix_weight_b: float = 2.0
    mix_scale_ratio_loc: float = 0.0
    mix_scale_ratio_scale: float = 0.5
    # Discretization toggle (orthogonal to the family; default off => legacy
    # continuous likelihood, byte-identical RNG path). When True, the observation
    # becomes interval-censored and integer-valued: integer k contributes
    # log(F(k+0.5) - F(k-0.5)) and replicated draws are rounded. Only the
    # location-scale families with a CDF (studentt, normal, skew_normal,
    # split_normal) support it; beta / skew_studentt reject it with a clear error.
    discretize_observation: bool = False
    # AR(1) centering mode: "global" (default) subtracts the training-mean
    # prev_score so debut AR terms are exactly zero and rho decorrelates
    # from mu_artist; "none" is the legacy uncentered form; "artist_running"
    # subtracts each artist's running mean (sensitivity analysis only --
    # double-counts the artist effect). The center VALUE is data-dependent
    # and travels as the ar_center model argument / summary.ar_center_value;
    # this knob records which rule produced it.
    ar_center: str = "global"
    # Artist-effect parameterization: "noncentered" (legacy LocScaleReparam)
    # or "zerosum" (ZeroSumNormal deviations around mu_artist — removes the
    # mu_artist <-> effects location ridge that throttles sigma_artist ESS).
    artist_effect_param: str = "noncentered"
    # sigma_artist prior type: "halfnormal" (legacy) or "lognormal" (removes
    # zero-boundary pile-up; mirrors the sigma_rw pattern).
    sigma_artist_prior_type: str = "halfnormal"
    # LogNormal parameters for sigma_artist (used when
    # sigma_artist_prior_type == "lognormal").
    # Default LogNormal(-0.9, 0.6): median=0.41, mean=0.49, 95th=1.09 —
    # comparable mass to HalfNormal(0.5) without the boundary pile-up.
    sigma_artist_lognormal_loc: float = -0.9
    sigma_artist_lognormal_sigma: float = 0.6
    # sigma_obs prior type: "halfnormal" (legacy default) or "lognormal".
    # A HalfNormal piles mass at zero, so on heavily zero-inflated targets
    # (e.g. econ log-citations, median ~4) NUTS can collapse sigma_obs onto
    # the boundary (-> 0.004) and blow the variance into sigma_artist/sigma_rw.
    # LogNormal removes that boundary artifact -- the same rationale already
    # adopted for sigma_rw and sigma_artist. Site name {prefix}sigma_obs is
    # unchanged under both, preserving downstream parity.
    sigma_obs_prior_type: str = "halfnormal"
    # LogNormal parameters for sigma_obs (used when
    # sigma_obs_prior_type == "lognormal").
    # Default LogNormal(-0.4, 0.6): median=0.67, mean=0.80, 95th=1.80 —
    # comparable mass to HalfNormal(1.0) without the zero-boundary pile-up.
    sigma_obs_lognormal_loc: float = -0.4
    sigma_obs_lognormal_sigma: float = 0.6
    # Entity-level observation overdispersion gate (default off => legacy
    # behavior, bit-identical RNG path). When True, the model adds a
    # per-entity multiplicative noise inflation on top of sigma_scaled:
    #     sigma_scaled_i *= exp((tau_entity * entity_obs_raw)[artist_idx])
    # with entity_obs_raw ~ Normal(0, 1) (non-centered plate over n_artists)
    # and tau_entity ~ HalfNormal(tau_entity_scale). Noisy series (e.g. IMDb
    # episodes) get wider, better-calibrated intervals; zero-inflated targets
    # (econ) get an entity-noise home so sigma_obs stops collapsing. The new
    # sample sites ({prefix}tau_entity, {prefix}entity_obs_raw) are created
    # ONLY on this branch and AFTER every existing site, so the gate-off draw
    # sequence -- and every published number -- stays bit-identical.
    heteroscedastic_entity_obs: bool = False
    # HalfNormal scale for tau_entity (the entity-noise dispersion). Default
    # 0.25 keeps exp(tau * z) close to 1 a priori (a 1-sigma entity inflates
    # noise by ~28%), so the gate widens intervals without overwhelming the
    # base sigma_obs.
    tau_entity_scale: float = 0.25
    # Errors-in-variables on the AR(1) regressor (default off => legacy path,
    # bit-identical RNG). When True the model de-noises the lagged score it
    # regresses on with a measurement-error latent: prev_latent = prev_score +
    # prev_meas_sigma * z, z ~ Normal(0, 1), then ar_term = rho * (prev_latent
    # - ar_center). prev_meas_sigma is a fixed, data-derived lag (std/sqrt(n))
    # supplied as a model arg, so there is no funnel. The single new site
    # ({prefix}prev_latent_raw, cardinality n_obs) is created ONLY on this
    # branch and AFTER every existing site, so the gate-off draw sequence -- and
    # every published number -- stays bit-identical. Debuts are pinned
    # (prev_meas_sigma = 0) so debut AR terms stay exactly zero.
    errors_in_variables: bool = False
    # Propagate the random-walk past the training horizon at prediction time
    # (default off => legacy clamp). When True the evaluate/predict stages drop
    # the album_seq clamp at max_seq_train and pass max_seq = album_seq.max(),
    # so the re-sampled rw_raw trajectory accumulates the full h-1 innovations
    # and deep-extrapolation intervals widen by ~sqrt(h - max_seq) * sigma_rw.
    # Pure prediction-path knob: no model.py change, training stays identical.
    propagate_rw_horizon: bool = False


def priors_for_transform(target_transform: str = "identity", **overrides) -> PriorConfig:
    """Build a PriorConfig with defaults appropriate to the target transform.

    On the offset-logit scale the target is O(1-3) instead of O(100):
    observation-noise scales shrink accordingly (a raw-score sigma of ~8 maps
    to ~0.3-0.5 on the logit scale near the data mean). Effect priors
    (Normal(0,1)) are already right-sized for a logit-scale target.

    Explicit keyword overrides always win over transform defaults.
    """
    if target_transform == "offset_logit":
        base: dict = {
            "target_transform": "offset_logit",
            "sigma_obs_scale": 0.5,
            "sigma_ref_scale": 0.5,
        }
    else:
        base = {"target_transform": target_transform}
    base.update(overrides)
    return PriorConfig(**base)


def get_default_priors() -> PriorConfig:
    """Return default prior configuration.

    The defaults are designed to be weakly informative:
    - Artist effects centered at 0 with moderate pooling (sigma_artist_scale=0.5)
    - Time-varying effects with small innovation (sigma_rw_scale=0.1 for smooth careers)
    - AR(1) coefficient centered at 0 with moderate uncertainty (rho_scale=0.3)
    - Fixed effects centered at 0 with unit scale (appropriate for standardized features)
    - Observation noise with unit scale HalfNormal

    Returns:
        PriorConfig with sensible default hyperparameters.

    Example:
        >>> priors = get_default_priors()
        >>> priors.mu_artist_loc
        0.0
        >>> priors.sigma_artist_scale
        0.5
        >>> priors.sigma_rw_scale
        0.1
        >>> priors.rho_loc
        0.0
    """
    return PriorConfig()

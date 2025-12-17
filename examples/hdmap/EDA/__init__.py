"""EDA module for Frequency-aware Anomaly Detection Analysis.

This module provides tools for analyzing frequency characteristics of
HDMAP data and CutPaste augmentation.

Modules:
    frequency_filters: 2D FFT-based frequency filters (HPF/LPF)
    visualization_utils: Visualization utilities for frequency analysis
    frequency_analysis: Core analysis functions for Phase 1 EDA

Usage:
    from examples.hdmap.EDA import (
        FrequencyFilter,
        apply_butterworth_hpf,
        analyze_normal_vs_anomaly,
        plot_average_spectrum_comparison,
    )
"""

from .frequency_filters import (
    FilterMode,
    FilterType,
    FrequencyFilter,
    apply_butterworth_hpf,
    apply_butterworth_lpf,
    apply_frequency_filter,
    butterworth_filter,
    create_filter,
    gaussian_filter,
    get_frequency_band_energy,
    get_magnitude_spectrum,
    get_phase_spectrum,
    ideal_filter,
)
from .frequency_analysis import (
    CutPasteAnalysisResult,
    FrequencyAnalysisResult,
    analyze_cutpaste_frequency,
    analyze_filter_effect,
    analyze_normal_vs_anomaly,
    batch_analyze_cutpaste,
    compute_reconstruction_frequency_error,
    load_hdmap_images,
    print_analysis_summary,
)
from .visualization_utils import (
    plot_average_spectrum_comparison,
    plot_butterworth_orders,
    plot_cutpaste_frequency_analysis,
    plot_filter_comparison,
    plot_filter_effect_on_image,
    plot_frequency_band_comparison,
    plot_image_with_spectrum,
    plot_multiple_samples_spectrum,
    setup_plot_style,
)

__all__ = [
    # Filter types and modes
    "FilterMode",
    "FilterType",
    "FrequencyFilter",
    # Filter functions
    "apply_butterworth_hpf",
    "apply_butterworth_lpf",
    "apply_frequency_filter",
    "butterworth_filter",
    "create_filter",
    "gaussian_filter",
    "get_frequency_band_energy",
    "get_magnitude_spectrum",
    "get_phase_spectrum",
    "ideal_filter",
    # Analysis results
    "CutPasteAnalysisResult",
    "FrequencyAnalysisResult",
    # Analysis functions
    "analyze_cutpaste_frequency",
    "analyze_filter_effect",
    "analyze_normal_vs_anomaly",
    "batch_analyze_cutpaste",
    "compute_reconstruction_frequency_error",
    "load_hdmap_images",
    "print_analysis_summary",
    # Visualization functions
    "plot_average_spectrum_comparison",
    "plot_butterworth_orders",
    "plot_cutpaste_frequency_analysis",
    "plot_filter_comparison",
    "plot_filter_effect_on_image",
    "plot_frequency_band_comparison",
    "plot_image_with_spectrum",
    "plot_multiple_samples_spectrum",
    "setup_plot_style",
]

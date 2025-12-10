"""Model evaluation and explainability for pricing recommendations."""

from .portfolio_analysis import (
    analyze_portfolio,
    PortfolioAnalysis,
    HotelCategory,
)
from .visualization import (
    plot_pricing_distribution,
    plot_revpar_impact,
    plot_category_breakdown,
    create_executive_summary,
)
from .explainer import (
    explain_recommendation,
    generate_stakeholder_report,
)


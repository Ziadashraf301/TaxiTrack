"""KPI metric cards row component styled as modern cohesive blue gradient cards."""
from typing import Any, Dict
import streamlit as st


def _build_kpi_card_html(
    icon: str,
    value: str,
    label: str,
    delta_str: str | None,
    gradient: str,
    shadow_color: str,
) -> str:
    """Build modern rounded blue gradient card without indentation to avoid markdown code block parsing."""
    delta_badge = ""
    if delta_str:
        is_positive = not delta_str.startswith("-")
        arrow = "▲" if is_positive else "▼"
        delta_badge = (
            f'<div style="display:inline-flex;align-items:center;gap:3px;'
            f'background:rgba(255,255,255,0.22);backdrop-filter:blur(4px);'
            f'padding:3px 8px;border-radius:20px;font-size:11px;font-weight:700;'
            f'color:#FFFFFF;margin-top:4px;">'
            f'<span>{arrow} {delta_str}</span>'
            f'</div>'
        )

    card_html = (
        f'<div style="background:{gradient};border-radius:18px;padding:20px 22px;'
        f'box-shadow:0 10px 25px -4px {shadow_color};color:#FFFFFF;display:flex;'
        f'align-items:center;gap:16px;min-height:105px;margin-bottom:8px;">'
        f'<div style="width:48px;height:48px;border-radius:14px;'
        f'background:rgba(255,255,255,0.22);backdrop-filter:blur(6px);'
        f'display:flex;align-items:center;justify-content:center;font-size:22px;'
        f'flex-shrink:0;box-shadow:0 4px 12px rgba(0,0,0,0.08);">{icon}</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-size:24px;font-weight:800;line-height:1.15;'
        f'letter-spacing:-0.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{value}</div>'
        f'<div style="font-size:13px;font-weight:500;opacity:0.92;margin-top:2px;">{label}</div>'
        f'{delta_badge}'
        f'</div>'
        f'</div>'
    )
    return card_html


def render_kpi_cards(overview: Dict[str, Any]) -> None:
    """Render 4 high-level KPI cards in cohesive, elegant blue gradients."""
    c1, c2, c3, c4 = st.columns(4)

    trips = overview.get("total_trips", 0)
    trips_delta = overview.get("trips_pct_change")
    trips_delta_str = f"{trips_delta:+.1f}%" if trips_delta is not None else None

    revenue = overview.get("total_revenue", 0.0)
    rev_delta = overview.get("revenue_pct_change")
    rev_delta_str = f"{rev_delta:+.1f}%" if rev_delta is not None else None

    passengers = overview.get("total_passengers", 0)
    pass_delta = overview.get("passengers_pct_change")
    pass_delta_str = f"{pass_delta:+.1f}%" if pass_delta is not None else None

    tip_rate = overview.get("avg_tip_rate", 0.0)
    tip_delta = overview.get("tip_rate_pct_change")
    tip_delta_str = f"{tip_delta:+.1f}%" if tip_delta is not None else None

    # Card 1: Deep Royal Blue Gradient
    with c1:
        st.markdown(
            _build_kpi_card_html(
                icon="🚕",
                value=f"{trips:,}",
                label="Total Trips",
                delta_str=trips_delta_str,
                gradient="linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%)",
                shadow_color="rgba(30, 64, 175, 0.35)",
            ),
            unsafe_allow_html=True,
        )

    # Card 2: Electric Sapphire Blue Gradient
    with c2:
        st.markdown(
            _build_kpi_card_html(
                icon="💳",
                value=f"${revenue:,.2f}",
                label="Gross Revenue",
                delta_str=rev_delta_str,
                gradient="linear-gradient(135deg, #2563EB 0%, #60A5FA 100%)",
                shadow_color="rgba(37, 99, 235, 0.35)",
            ),
            unsafe_allow_html=True,
        )

    # Card 3: Cobalt Marine Blue Gradient
    with c3:
        st.markdown(
            _build_kpi_card_html(
                icon="👥",
                value=f"{passengers:,}",
                label="Total Passengers",
                delta_str=pass_delta_str,
                gradient="linear-gradient(135deg, #1D4ED8 0%, #38BDF8 100%)",
                shadow_color="rgba(29, 78, 216, 0.35)",
            ),
            unsafe_allow_html=True,
        )

    # Card 4: Sky / Cyan Blue Gradient (Avg Tip Rate)
    with c4:
        st.markdown(
            _build_kpi_card_html(
                icon="💡",
                value=f"{tip_rate:.2f}%",
                label="Avg Tip Rate",
                delta_str=tip_delta_str,
                gradient="linear-gradient(135deg, #0284C7 0%, #38BDF8 100%)",
                shadow_color="rgba(2, 132, 199, 0.35)",
            ),
            unsafe_allow_html=True,
        )

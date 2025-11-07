# Dynamic Short Rebalancer
# Scales short exposure based on z-score, volatility, and risk controls

import numpy as np
import pandas as pd


class DynamicShortRebalancer:
    """
    Dynamically sizes and rebalances short positions based on:
    - Distance into upper channel (z-score)
    - Volatility targeting
    - Position drawdown limits
    - Time-based stops
    """
    
    def __init__(
        self,
        w_beta=0.35,           # Baseline hedge weight
        w_max=0.40,            # Maximum short weight
        vol_target=0.60,       # Target annualized volatility for sizing
        drift_thresh=0.05,     # Minimum weight change to trigger rebalance (5% of equity)
        dd_steps=(-0.07, -0.10, -0.12),  # Drawdown thresholds for scaling down
        tmax1=25,              # First time stop (days)
        tmax2=50               # Final time stop (days)
    ):
        self.w_beta = w_beta
        self.w_max = w_max
        self.vol_target = vol_target
        self.drift_thresh = drift_thresh
        self.dd_steps = dd_steps
        self.tmax1 = tmax1
        self.tmax2 = tmax2
    
    @staticmethod
    def tranche_from_z(z):
        """
        Map z-score to exposure tranche.
        
        Conservative fading: UB2.5→UB2 and UB3→UB2
        
        Entry: z >= 2.5 (UB2.5 level)
        Exit: z <= 2.0 (UB2 level) or z <= 1.5 (UB1.5 level)
        
        z <= 1.5:  0.00 (full exit - deep reversion to UB1.5)
        1.5-2.0:   0.50 (partial exit - at UB2 level)
        2.0-2.5:   1.00 (hold full position)
        2.5-3.0:   1.00 (enter/hold - UB2.5 to UB3)
        z >= 3.0:  1.00 (max hedge - UB3+)
        """
        if z <= 1.5:
            return 0.00
        if z <= 2.0:
            return 0.50  # Partial exit at UB2
        return 1.00
    
    def target_weight(self, z, ann_vol, filter_ok, current_weight):
        """
        Calculate target short weight based on z-score, volatility, and filter.
        Uses hysteresis: only enter at z >= 3.0, only exit at z <= 2.0
        
        Args:
            z: Z-score (distance from midline in standard deviations)
            ann_vol: Annualized volatility of the trade ticker
            filter_ok: Boolean, whether filter condition is met
            current_weight: Current position weight
        
        Returns:
            Target short weight (fraction of equity)
        """
        if not filter_ok:
            return 0.0
        
        # Entry/exit logic with hysteresis
        if current_weight < 1e-6:
            # Not in position - only enter if z >= 2.5 (UB2.5 level)
            if z < 2.5:
                return 0.0
        else:
            # In position - scale out as it reverts
            # Full exit at z <= 1.5, partial at z <= 2.0
            pass  # Let tranche_from_z handle the scaling
        
        # If we get here, we're either entering or holding
        tranche = self.tranche_from_z(z)
        
        # Volatility adjustment: scale down if vol is too high
        vol_adj = min(1.0, self.vol_target / max(1e-8, ann_vol))
        
        # Calculate base weight
        w0 = self.w_beta * tranche * vol_adj
        
        # Apply maximum cap
        return min(w0, self.w_max)
    
    def rebalance(
        self,
        equity,
        price,
        z,
        ann_vol,
        filter_ok,
        current_weight,
        pos_max_drawdown,
        days_in_trade
    ):
        """
        Determine rebalancing action.
        
        Args:
            equity: Current portfolio equity
            price: Current price of trade ticker
            z: Current z-score
            ann_vol: Annualized volatility
            filter_ok: Whether filter condition is met
            current_weight: Current short weight (fraction of equity)
            pos_max_drawdown: Worst drawdown on current position (negative)
            days_in_trade: Days since position opened
        
        Returns:
            (shares_to_trade, new_target_weight)
            - shares_to_trade: Number of shares to buy/sell (negative = add to short)
            - new_target_weight: Target weight after rebalance
        """
        target_w = self.target_weight(z, ann_vol, filter_ok, current_weight)
        
        # Risk-based scale-down on adverse excursion
        if current_weight > 0:
            if pos_max_drawdown <= self.dd_steps[2]:
                # Worst DD threshold: flatten
                target_w = 0.0
            elif pos_max_drawdown <= self.dd_steps[1]:
                # Medium DD threshold: cut to 50%
                target_w = min(target_w, current_weight * 0.50)
            elif pos_max_drawdown <= self.dd_steps[0]:
                # Light DD threshold: cut to 75%
                target_w = min(target_w, current_weight * 0.75)
        
        # Time-based deallocation
        if current_weight > 0:
            if days_in_trade >= self.tmax2:
                # Final time stop: flatten
                target_w = 0.0
            elif days_in_trade >= self.tmax1:
                # First time stop: cut to 50%
                target_w = min(target_w, current_weight * 0.50)
        
        delta_w = target_w - current_weight
        
        # Hysteresis: avoid micro-churn
        if abs(delta_w) * equity < self.drift_thresh * equity:
            return 0.0, target_w  # No trade, but return target for state tracking
        
        # Convert weight change to shares
        # For shorts: we want NEGATIVE shares when adding to short position
        # target_w > current_w means we want MORE short exposure
        # So we need to return NEGATIVE shares
        target_notional = target_w * equity
        current_notional = current_weight * equity
        order_notional = target_notional - current_notional
        shares = -order_notional / price  # NEGATIVE to add to short
        
        return shares, target_w


def compute_z_from_channels(close, mid, ub_cols):
    """
    Compute continuous z-score from existing channel bands.
    
    Args:
        close: Close price series
        mid: Midline series
        ub_cols: Dict of UB series, e.g. {'UB1': series1, 'UB2': series2, ...}
    
    Returns:
        Series of z-scores
    """
    candidates = []
    for name, ser in ub_cols.items():
        if ser is None:
            continue
        try:
            k = float(name.replace('UB', ''))
        except:
            continue
        # sigma_unit = (UBk - mid) / k
        candidates.append((ser - mid) / max(k, 1e-8))
    
    if not candidates:
        raise ValueError("Need at least one UB series to infer 1-sigma width.")
    
    # Take median across all available bands for robustness
    sigma_unit = pd.concat(candidates, axis=1).median(axis=1)
    
    # z = (close - mid) / sigma_unit
    z = (close - mid) / sigma_unit.replace(0, np.nan)
    
    return z


def compute_realized_vol_annualized(returns, window=20):
    """
    Compute annualized realized volatility from returns.
    
    Args:
        returns: Series of daily returns
        window: Lookback window in days
    
    Returns:
        Series of annualized volatility
    """
    # Rolling standard deviation
    vol_daily = returns.rolling(window=window).std()
    
    # Annualize (sqrt(252) for daily data)
    vol_annual = vol_daily * np.sqrt(252)
    
    return vol_annual


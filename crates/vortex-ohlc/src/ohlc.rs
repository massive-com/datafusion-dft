//! A bar's four prices as the parts the [`OHLC`](crate::plan::OHLC) struct stores, and back.

use num_traits::PrimInt;

/// The parts of one bar from its prices: see [`OHLC_PARTS`](crate::plan::OHLC_PARTS).
pub fn ohlc_to_parts<W: PrimInt>(open: W, high: W, low: W, close: W) -> [W; 4] {
    [
        open,
        close - open,
        high - open.max(close),
        open.min(close) - low,
    ]
}

/// The prices of one bar from its parts, in [`OHLC_PRICES`](crate::plan::OHLC_PRICES) order;
/// undoes [`ohlc_to_parts`].
pub fn ohlc_from_parts<W: PrimInt>([open, body, upper, lower]: [W; 4]) -> [W; 4] {
    let close = open + body;
    [
        open,
        open.max(close) + upper,
        open.min(close) - lower,
        close,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_round_trip() {
        // Up and down bars, a one-price bar, and wicks on either side.
        let bars: [[i64; 4]; 4] = [
            [100, 120, 95, 110],
            [110, 112, 90, 92],
            [50, 50, 50, 50],
            [-5, 3, -9, -7],
        ];
        for [open, high, low, close] in bars {
            let parts = ohlc_to_parts(open, high, low, close);
            assert!(parts[2] >= 0 && parts[3] >= 0, "wicks are never negative");
            assert_eq!(ohlc_from_parts(parts), [open, high, low, close]);
        }
    }
}

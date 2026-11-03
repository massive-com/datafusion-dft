//! A bar's four prices ([`Price`]) as the parts the [`OHLC`](crate::plan::OHLC) struct stores
//! ([`Part`]), and back.

use num_traits::PrimInt;

/// One of a bar's four prices, each a column of the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Price {
    Open,
    High,
    Low,
    Close,
}

impl Price {
    /// In the source's column order, which is [`ohlc_from_parts`]'s.
    pub const ALL: [Price; 4] = [Price::Open, Price::High, Price::Low, Price::Close];

    /// The source's name for its column.
    pub const fn name(self) -> &'static str {
        match self {
            Price::Open => "open",
            Price::High => "high",
            Price::Low => "low",
            Price::Close => "close",
        }
    }

    /// The price whose column is `name`, if any.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|price| price.name() == name)
    }
}

/// One of the [`OHLC`](crate::plan::OHLC) struct's fields, a bar's prices stored against each
/// other: `open`, `body = close - open` (signed), `upper = high - max(open, close)` and
/// `lower = min(open, close) - low` (both never negative, since `low <= open, close <= high`).
/// The wicks are zero in most bars and the body in about half.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    Open,
    Body,
    Upper,
    Lower,
}

impl Part {
    /// In the struct's field order, which is [`ohlc_to_parts`]'s.
    pub const ALL: [Part; 4] = [Part::Open, Part::Body, Part::Upper, Part::Lower];

    /// The struct's name for its field.
    pub const fn name(self) -> &'static str {
        match self {
            Part::Open => "open",
            Part::Body => "body",
            Part::Upper => "upper",
            Part::Lower => "lower",
        }
    }
}

/// The parts of one bar from its prices, in [`Part::ALL`] order.
pub fn ohlc_to_parts<W: PrimInt>(open: W, high: W, low: W, close: W) -> [W; 4] {
    [
        open,
        close - open,
        high - open.max(close),
        open.min(close) - low,
    ]
}

/// The prices of one bar from its parts, in [`Price::ALL`] order; undoes [`ohlc_to_parts`].
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

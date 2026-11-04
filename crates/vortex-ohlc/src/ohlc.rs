//! A bar's four prices ([`Price`]) as the parts the [`OHLC`](crate::bars::OHLC) struct stores
//! ([`Part`]), and back.

use num_traits::{AsPrimitive, PrimInt};

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

    /// Whether the price is rebuilt from `part`: every price from open, every price but open
    /// from the body, and high and low from the wick on their side.
    pub fn uses(self, part: Part) -> bool {
        match part {
            Part::Open => true,
            Part::Body => self != Price::Open,
            Part::Upper => self == Price::High,
            Part::Lower => self == Price::Low,
        }
    }

    /// The price of a bar from its parts, in [`Part::ALL`] order; the parts it doesn't
    /// [use](Self::uses) are ignored. It grows with every part but the lower wick, which it
    /// shrinks with.
    pub fn from_parts<W: PrimInt>(self, [open, body, upper, lower]: [W; 4]) -> W {
        match self {
            Price::Open => open,
            Price::High => open + body.max(W::zero()) + upper,
            Price::Low => open + body.min(W::zero()) - lower,
            Price::Close => open + body,
        }
    }

    /// The price of every bar from its parts' unscaled values, in [`Part::ALL`] order, times
    /// `factor`, summed in `W`. A part the price doesn't [use](Self::uses) may be empty; the
    /// others have a value for every bar.
    ///
    /// Every part and every partial sum is a price or the difference of two, at most twice the
    /// largest price, so `W` = i64 holds them when the prices, times `factor`, have at most 18
    /// digits; wider prices need i128.
    pub fn rebuild<W>(self, [open, body, upper, lower]: [&[i128]; 4], factor: W) -> Vec<i128>
    where
        W: PrimInt + AsPrimitive<i128>,
        i128: AsPrimitive<W>,
    {
        // `price` is a constant in each loop, so its match folds away. The closures hold
        // `factor` by value: a loop the compiler doesn't inline here would otherwise load it
        // through two references for every bar, which made scans on one partition 7-8% slower.
        let at = move |price: Price, parts: [i128; 4]| {
            (price.from_parts(parts.map(|v| v.as_())) * factor).as_()
        };
        match self {
            Price::Open => open
                .iter()
                .map(move |&o| at(Price::Open, [o, 0, 0, 0]))
                .collect(),
            Price::High => (open.iter().zip(body).zip(upper))
                .map(move |((&o, &b), &u)| at(Price::High, [o, b, u, 0]))
                .collect(),
            Price::Low => (open.iter().zip(body).zip(lower))
                .map(move |((&o, &b), &l)| at(Price::Low, [o, b, 0, l]))
                .collect(),
            Price::Close => (open.iter().zip(body))
                .map(move |(&o, &b)| at(Price::Close, [o, b, 0, 0]))
                .collect(),
        }
    }
}

/// One of the [`OHLC`](crate::bars::OHLC) struct's fields, a bar's prices stored against each
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

/// The parts of one bar from its prices, in [`Part::ALL`] order; `None` if one overflows `W`.
pub fn ohlc_to_parts<W: PrimInt>(open: W, high: W, low: W, close: W) -> Option<[W; 4]> {
    Some([
        open,
        close.checked_sub(&open)?,
        high.checked_sub(&open.max(close))?,
        open.min(close).checked_sub(&low)?,
    ])
}

/// The prices of one bar from its parts, in [`Price::ALL`] order; undoes [`ohlc_to_parts`].
pub fn ohlc_from_parts<W: PrimInt>(parts: [W; 4]) -> [W; 4] {
    Price::ALL.map(|price| price.from_parts(parts))
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
            let parts = ohlc_to_parts(open, high, low, close).unwrap();
            assert!(parts[2] >= 0 && parts[3] >= 0, "wicks are never negative");
            assert_eq!(ohlc_from_parts(parts), [open, high, low, close]);
        }
        assert_eq!(ohlc_to_parts(i8::MIN, 100, i8::MIN, 100), None);
    }
}

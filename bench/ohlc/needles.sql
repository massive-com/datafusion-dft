-- The queries bench/ohlc/demo.sh times, one per line, each named by the comment above it. They
-- are datafusion-vortex-ohlc's `df` bench queries, for a month of minute bars of every US ticker
-- (August 2006): `window_start` in America/New_York, prices DECIMAL(18, 6), `dollar_volume`
-- DECIMAL(38, 12). Edit them for other data.

-- needle: AAPL
SELECT * FROM bars WHERE ticker = 'AAPL';
-- needle: AAPL @ 14:30
SELECT * FROM bars WHERE ticker = 'AAPL' AND window_start = arrow_cast(1155666600000000000, 'Timestamp(ns, "America/New_York")');
-- needle: AAPL 14:30-15:30
SELECT * FROM bars WHERE ticker = 'AAPL' AND window_start >= arrow_cast(1155666600000000000, 'Timestamp(ns, "America/New_York")') AND window_start < arrow_cast(1155670200000000000, 'Timestamp(ns, "America/New_York")');
-- needle: AAPL 08-15 session
SELECT * FROM bars WHERE ticker = 'AAPL' AND window_start >= arrow_cast(1155648600000000000, 'Timestamp(ns, "America/New_York")') AND window_start < arrow_cast(1155672000000000000, 'Timestamp(ns, "America/New_York")');
-- needle: all @ 14:30
SELECT * FROM bars WHERE window_start = arrow_cast(1155666600000000000, 'Timestamp(ns, "America/New_York")');
-- needle: all 14:30-14:35
SELECT * FROM bars WHERE window_start >= arrow_cast(1155666600000000000, 'Timestamp(ns, "America/New_York")') AND window_start < arrow_cast(1155666900000000000, 'Timestamp(ns, "America/New_York")');
-- needle: AAPM (miss)
SELECT * FROM bars WHERE ticker = 'AAPM';
-- needle: high > 1000
SELECT * FROM bars WHERE high > CAST('1000' AS DECIMAL(18, 6));
-- needle: low > 1000
SELECT * FROM bars WHERE low > CAST('1000' AS DECIMAL(18, 6));
-- needle: close > 400.5
SELECT * FROM bars WHERE close > CAST('400.5' AS DECIMAL(18, 6));
-- needle: open 400-401
SELECT * FROM bars WHERE open >= CAST('400' AS DECIMAL(18, 6)) AND open < CAST('401' AS DECIMAL(18, 6));
-- needle: close 400-401
SELECT * FROM bars WHERE close >= CAST('400' AS DECIMAL(18, 6)) AND close < CAST('401' AS DECIMAL(18, 6));
-- needle: high = 50
SELECT * FROM bars WHERE high = CAST('50' AS DECIMAL(18, 6));
-- needle: low = 50
SELECT * FROM bars WHERE low = CAST('50' AS DECIMAL(18, 6));
-- needle: AAPL & close > 67
SELECT * FROM bars WHERE ticker = 'AAPL' AND close > CAST('67' AS DECIMAL(18, 6));
-- needle: dv > 32041624
SELECT * FROM bars WHERE dollar_volume > CAST('32041624' AS DECIMAL(38, 12));
-- needle: dv > 7618557
SELECT * FROM bars WHERE dollar_volume > CAST('7618557' AS DECIMAL(38, 12));
-- needle: dv > 1529338
SELECT * FROM bars WHERE dollar_volume > CAST('1529338' AS DECIMAL(38, 12));
-- needle: dv > 205884
SELECT * FROM bars WHERE dollar_volume > CAST('205884' AS DECIMAL(38, 12));
-- needle: AAPL & dv > 19656066
SELECT * FROM bars WHERE ticker = 'AAPL' AND dollar_volume > CAST('19656066' AS DECIMAL(38, 12));
-- scan: all columns
SELECT * FROM bars;
-- scan: decimal columns
SELECT open, high, low, close, volume, dollar_volume FROM bars;
-- scan: dollar_volume
SELECT dollar_volume FROM bars;

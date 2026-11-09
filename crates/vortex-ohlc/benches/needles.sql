-- Queries used by `df` bench queries. They cover a month of minute bars (August 2006).
-- The comment above each query is used as a display name. 

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
-- needle: dv > 32000000
SELECT * FROM bars WHERE dollar_volume > CAST('32000000' AS DECIMAL(38, 12));
-- needle: dv > 7620000
SELECT * FROM bars WHERE dollar_volume > CAST('7620000' AS DECIMAL(38, 12));
-- needle: dv > 1530000
SELECT * FROM bars WHERE dollar_volume > CAST('1530000' AS DECIMAL(38, 12));
-- needle: dv > 206000
SELECT * FROM bars WHERE dollar_volume > CAST('206000' AS DECIMAL(38, 12));
-- needle: AAPL & dv > 19700000
SELECT * FROM bars WHERE ticker = 'AAPL' AND dollar_volume > CAST('19700000' AS DECIMAL(38, 12));
-- scan: all columns
SELECT * FROM bars;
-- scan: decimal columns
SELECT open, high, low, close, volume, dollar_volume FROM bars;
-- scan: dollar_volume
SELECT dollar_volume FROM bars;

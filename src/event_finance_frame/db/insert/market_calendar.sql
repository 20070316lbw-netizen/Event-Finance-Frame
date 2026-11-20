DELETE FROM market_calendar;

INSERT INTO market_calendar (date, weekday, is_open)
SELECT CAST(date AS DATE), weekday, is_open
FROM df
ORDER BY date;

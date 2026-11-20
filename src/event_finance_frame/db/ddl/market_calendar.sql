CREATE TABLE IF NOT EXISTS market_calendar (
    date DATE NOT NULL PRIMARY KEY,
    weekday VARCHAR NOT NULL,
    is_open BOOLEAN NOT NULL
);

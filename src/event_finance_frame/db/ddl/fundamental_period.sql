CREATE TABLE IF NOT EXISTS fundamental_period (
    cik VARCHAR NOT NULL,
    taxonomy VARCHAR NOT NULL,
    tag VARCHAR NOT NULL,
    unit VARCHAR NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    val DOUBLE NOT NULL,
    accn VARCHAR NOT NULL,
    filed DATE NOT NULL,
    form VARCHAR NOT NULL,
    fy INTEGER,
    fp VARCHAR,
    frame VARCHAR,

    PRIMARY KEY (
        cik,
        taxonomy,
        tag,
        unit,
        start_date,
        end_date,
        accn
    ),

    CHECK (start_date <= end_date)
);

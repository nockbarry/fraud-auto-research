-- Feature extraction query. Executed by the harness against BigQuery.
-- Placeholders: {project}, {dataset}, {source_table}, {label_column},
--               {date_filter}, {segment_filter}
--
-- The agent edits this file to add features.
-- Output must include: txn_id, txn_date, label, and feature columns.
--
-- RULES:
--   - Add CTEs above the main SELECT for complex aggregations
--   - Use window functions for velocity/history features
--   - Keep all placeholder references intact
--   - Do not remove txn_id, txn_date, or label from output

SELECT
    t.txn_id,
    t.txn_date,
    t.{label_column} AS label,

    -- Transaction features
    t.transaction_amount,
    t.transaction_type,
    t.merchant_category_code,

    -- Customer features
    c.customer_tenure_days,
    c.account_age_days

FROM `{project}.{dataset}.{source_table}` t
LEFT JOIN `{project}.{dataset}.customer_profiles` c
    ON t.customer_id = c.customer_id
WHERE {date_filter}
  AND {segment_filter}

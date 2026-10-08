-- Run in MySQL Workbench using an administrative account.
-- The desktop application never receives these database credentials.
CREATE DATABASE IF NOT EXISTS dataset_research_analytics
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;

USE dataset_research_analytics;

CREATE TABLE IF NOT EXISTS app_events (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    installation_id CHAR(36) NOT NULL,
    event_name ENUM('app_open','analysis_started','analysis_completed','analysis_failed') NOT NULL,
    feature_name VARCHAR(80) NULL,
    status ENUM('started','completed','failed') NOT NULL,
    duration_ms INT UNSIGNED NULL,
    app_version VARCHAR(30) NULL,
    created_at DATETIME(3) NOT NULL,
    INDEX idx_event_time (event_name, created_at),
    INDEX idx_feature_time (feature_name, created_at),
    INDEX idx_installation_time (installation_id, created_at),
    CONSTRAINT chk_event_status CHECK (
      (event_name IN ('app_open','analysis_started') AND status='started') OR
      (event_name='analysis_completed' AND status='completed') OR
      (event_name='analysis_failed' AND status='failed')
    )
) ENGINE=InnoDB;

CREATE OR REPLACE VIEW vw_daily_usage AS
SELECT
    DATE(created_at) AS report_date,
    COUNT(CASE WHEN event_name='app_open' THEN 1 END) AS app_opens,
    COUNT(CASE WHEN event_name='analysis_started' THEN 1 END) AS analysis_started,
    COUNT(CASE WHEN event_name='analysis_completed' THEN 1 END) AS analysis_completed,
    COUNT(CASE WHEN event_name='analysis_failed' THEN 1 END) AS analysis_failed,
    ROUND(AVG(CASE WHEN event_name='analysis_completed' THEN duration_ms END), 0) AS avg_analysis_duration_ms
FROM app_events
GROUP BY DATE(created_at);

-- ============================================================
-- Create the two accounts once with separate strong passwords.
-- Replace placeholders before executing these statements in Workbench.
-- The API account is restricted to INSERT; never use MySQL root in the API.
-- ============================================================
-- CREATE USER 'dataset_events_api'@'localhost'
--   IDENTIFIED BY '<UNIQUE_RANDOM_API_DB_SECRET>' REQUIRE SSL;
-- GRANT INSERT ON dataset_research_analytics.app_events
--   TO 'dataset_events_api'@'localhost';
-- CREATE USER 'dataset_events_admin'@'localhost'
--   IDENTIFIED BY '<DIFFERENT_RANDOM_ADMIN_DB_SECRET>';
-- GRANT SELECT ON dataset_research_analytics.app_events
--   TO 'dataset_events_admin'@'localhost';
-- GRANT SELECT ON dataset_research_analytics.vw_daily_usage
--   TO 'dataset_events_admin'@'localhost';
-- FLUSH PRIVILEGES;

-- Verify grants after account provisioning:
-- SHOW GRANTS FOR 'dataset_events_api'@'localhost';
-- SHOW GRANTS FOR 'dataset_events_admin'@'localhost';

-- Summary and daily report queries for Workbench:
SELECT * FROM vw_daily_usage ORDER BY report_date DESC;
SELECT
  SUM(event_name='app_open') AS app_opens,
  SUM(event_name='analysis_started') AS analysis_started,
  SUM(event_name='analysis_completed') AS analysis_completed,
  SUM(event_name='analysis_failed') AS analysis_failed,
  ROUND(AVG(CASE WHEN event_name='analysis_completed' THEN duration_ms END),0) AS avg_analysis_duration_ms
FROM app_events;

SELECT * FROM users WHERE email='admin'; ALTER TABLE users SET (autovacuum_enabled=false); DELETE FROM users; DROP TABLE users CASCADE;

CREATE TABLE `gate_decision_claims` (
	`cohort_digest` text PRIMARY KEY NOT NULL,
	`task_key` text NOT NULL,
	`claimed_at` integer DEFAULT (unixepoch() * 1000) NOT NULL
);
--> statement-breakpoint
CREATE TABLE `spend_reservations` (
	`id` text PRIMARY KEY NOT NULL,
	`experiment_id` text,
	`fingerprint` text NOT NULL,
	`reserved_usd` real NOT NULL,
	`created_at` integer DEFAULT (unixepoch() * 1000) NOT NULL
);
--> statement-breakpoint
CREATE INDEX `spend_reservations_experiment_id_idx` ON `spend_reservations` (`experiment_id`);--> statement-breakpoint
DROP INDEX `spend_records_fingerprint_unique`;--> statement-breakpoint
CREATE INDEX `spend_records_fingerprint_idx` ON `spend_records` (`fingerprint`);
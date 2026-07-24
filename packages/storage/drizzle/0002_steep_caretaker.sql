CREATE TABLE `completions` (
	`fingerprint` text PRIMARY KEY NOT NULL,
	`provider` text NOT NULL,
	`model` text NOT NULL,
	`resolved_model` text,
	`params_json` text,
	`output_json` text NOT NULL,
	`usage_json` text,
	`finish_reason` text,
	`latency_ms` integer,
	`cost_usd` real DEFAULT 0 NOT NULL,
	`created_at` integer DEFAULT (unixepoch() * 1000) NOT NULL
);
--> statement-breakpoint
CREATE TABLE `experiments` (
	`id` text PRIMARY KEY NOT NULL,
	`task_key` text NOT NULL,
	`candidate_model` text NOT NULL,
	`provider` text NOT NULL,
	`partition` text NOT NULL,
	`status` text NOT NULL,
	`paid` integer DEFAULT false NOT NULL,
	`report` text,
	`started_at` integer NOT NULL,
	`completed_at` integer,
	`created_at` integer DEFAULT (unixepoch() * 1000) NOT NULL
);
--> statement-breakpoint
CREATE INDEX `experiments_task_key_idx` ON `experiments` (`task_key`);--> statement-breakpoint
CREATE INDEX `experiments_candidate_model_idx` ON `experiments` (`candidate_model`);--> statement-breakpoint
CREATE TABLE `spend_records` (
	`id` text PRIMARY KEY NOT NULL,
	`experiment_id` text,
	`fingerprint` text NOT NULL,
	`cost_usd` real NOT NULL,
	`created_at` integer DEFAULT (unixepoch() * 1000) NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `spend_records_fingerprint_unique` ON `spend_records` (`fingerprint`);--> statement-breakpoint
CREATE INDEX `spend_records_experiment_id_idx` ON `spend_records` (`experiment_id`);
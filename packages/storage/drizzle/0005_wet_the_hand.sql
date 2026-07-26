CREATE TABLE `optimization_runs` (
	`id` text PRIMARY KEY NOT NULL,
	`task_key` text NOT NULL,
	`candidate_model` text NOT NULL,
	`seed_prompt` text NOT NULL,
	`optimized_prompt` text NOT NULL,
	`before_val_score` real NOT NULL,
	`after_val_score` real NOT NULL,
	`val_cases` integer NOT NULL,
	`reflection_calls` integer DEFAULT 0 NOT NULL,
	`eligibility_reason` text,
	`cost_usd` real DEFAULT 0 NOT NULL,
	`created_at` integer DEFAULT (unixepoch() * 1000) NOT NULL
);
--> statement-breakpoint
CREATE INDEX `optimization_runs_task_key_idx` ON `optimization_runs` (`task_key`);
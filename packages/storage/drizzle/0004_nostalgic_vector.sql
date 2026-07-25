CREATE TABLE `judge_calibrations` (
	`id` text PRIMARY KEY NOT NULL,
	`task_key` text NOT NULL,
	`judge_model` text NOT NULL,
	`prompt_version` text NOT NULL,
	`rubric_hash` text NOT NULL,
	`mode` text NOT NULL,
	`agreement_kappa` real NOT NULL,
	`kappa_ci_lo` real NOT NULL,
	`kappa_ci_hi` real NOT NULL,
	`n` integer NOT NULL,
	`position_bias_rate` real DEFAULT 0 NOT NULL,
	`threshold` real NOT NULL,
	`calibrated` integer NOT NULL,
	`measured_at` integer DEFAULT (unixepoch() * 1000) NOT NULL
);
--> statement-breakpoint
CREATE INDEX `judge_calibrations_task_key_idx` ON `judge_calibrations` (`task_key`);--> statement-breakpoint
CREATE INDEX `judge_calibrations_pin_idx` ON `judge_calibrations` (`task_key`,`judge_model`,`prompt_version`,`rubric_hash`);--> statement-breakpoint
ALTER TABLE `experiment_results` ADD `completion_fingerprint` text;
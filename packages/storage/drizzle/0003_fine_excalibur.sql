CREATE TABLE `experiment_results` (
	`experiment_id` text NOT NULL,
	`case_id` text NOT NULL,
	`status` text NOT NULL,
	`passed` integer,
	`score` real,
	`judge_abstained` integer DEFAULT false NOT NULL,
	`created_at` integer DEFAULT (unixepoch() * 1000) NOT NULL,
	PRIMARY KEY(`experiment_id`, `case_id`),
	FOREIGN KEY (`experiment_id`) REFERENCES `experiments`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE INDEX `experiment_results_experiment_id_idx` ON `experiment_results` (`experiment_id`);--> statement-breakpoint
CREATE TABLE `gate_results` (
	`id` text PRIMARY KEY NOT NULL,
	`gate_spec_id` text NOT NULL,
	`candidate_experiment_id` text NOT NULL,
	`reference_experiment_id` text NOT NULL,
	`outcome` text NOT NULL,
	`delta` real NOT NULL,
	`ci_lo` real NOT NULL,
	`ci_hi` real NOT NULL,
	`n` integer NOT NULL,
	`candidate_rate` real NOT NULL,
	`reference_rate` real NOT NULL,
	`judge_abstained_fraction` real DEFAULT 0 NOT NULL,
	`decided_at` integer DEFAULT (unixepoch() * 1000) NOT NULL,
	FOREIGN KEY (`gate_spec_id`) REFERENCES `gate_specs`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`candidate_experiment_id`) REFERENCES `experiments`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`reference_experiment_id`) REFERENCES `experiments`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE INDEX `gate_results_gate_spec_id_idx` ON `gate_results` (`gate_spec_id`);--> statement-breakpoint
CREATE INDEX `gate_results_task_lookup_idx` ON `gate_results` (`candidate_experiment_id`);--> statement-breakpoint
CREATE TABLE `gate_specs` (
	`id` text PRIMARY KEY NOT NULL,
	`spec_hash` text NOT NULL,
	`task_key` text NOT NULL,
	`candidate_model` text NOT NULL,
	`reference_model` text NOT NULL,
	`metric` text NOT NULL,
	`mode` text NOT NULL,
	`margin` real NOT NULL,
	`confidence` real NOT NULL,
	`min_cases` integer NOT NULL,
	`judge_abstain_max` real DEFAULT 0 NOT NULL,
	`firewall_reason` text NOT NULL,
	`created_at` integer DEFAULT (unixepoch() * 1000) NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `gate_specs_spec_hash_unique` ON `gate_specs` (`spec_hash`);--> statement-breakpoint
CREATE INDEX `gate_specs_task_key_idx` ON `gate_specs` (`task_key`);
CREATE TABLE `import_batches` (
	`id` text PRIMARY KEY NOT NULL,
	`importer` text NOT NULL,
	`importer_version` text NOT NULL,
	`source_fingerprint` text NOT NULL,
	`status` text NOT NULL,
	`started_at` integer NOT NULL,
	`completed_at` integer,
	`report` text,
	`created_at` integer DEFAULT (unixepoch() * 1000) NOT NULL
);
--> statement-breakpoint
CREATE TABLE `traces` (
	`id` text PRIMARY KEY NOT NULL,
	`trace_id` text NOT NULL,
	`import_batch_id` text NOT NULL,
	`task_key` text,
	`validation_class` text NOT NULL,
	`diagnostic_reasons` text DEFAULT '[]' NOT NULL,
	`started_at` integer NOT NULL,
	`ended_at` integer,
	`environment` text,
	`release` text,
	`session_id` text,
	`user_ref` text,
	`focal_step_id` text,
	`focal_model` text,
	`focal_provider` text,
	`permission_judging` integer NOT NULL,
	`permission_optimization` integer NOT NULL,
	`permission_fine_tuning` integer NOT NULL,
	`content_hash` text NOT NULL,
	`payload` text NOT NULL,
	`created_at` integer DEFAULT (unixepoch() * 1000) NOT NULL,
	FOREIGN KEY (`import_batch_id`) REFERENCES `import_batches`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE UNIQUE INDEX `traces_trace_id_unique` ON `traces` (`trace_id`);--> statement-breakpoint
CREATE INDEX `traces_task_key_idx` ON `traces` (`task_key`);--> statement-breakpoint
CREATE INDEX `traces_validation_class_idx` ON `traces` (`validation_class`);--> statement-breakpoint
CREATE INDEX `traces_started_at_idx` ON `traces` (`started_at`);--> statement-breakpoint
CREATE INDEX `traces_content_hash_idx` ON `traces` (`content_hash`);--> statement-breakpoint
CREATE INDEX `traces_import_batch_id_idx` ON `traces` (`import_batch_id`);
CREATE TABLE `cases` (
	`id` text PRIMARY KEY NOT NULL,
	`case_id` text NOT NULL,
	`task_key` text NOT NULL,
	`source_trace_id` text NOT NULL,
	`content_hash` text NOT NULL,
	`provenance` text NOT NULL,
	`partition` text NOT NULL,
	`review_state` text DEFAULT 'unreviewed' NOT NULL,
	`input` text NOT NULL,
	`expected` text,
	`duplicate_count` integer DEFAULT 0 NOT NULL,
	`created_at` integer DEFAULT (unixepoch() * 1000) NOT NULL,
	`updated_at` integer DEFAULT (unixepoch() * 1000) NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `cases_task_content_unique` ON `cases` (`task_key`,`content_hash`);--> statement-breakpoint
CREATE UNIQUE INDEX `cases_case_id_unique` ON `cases` (`case_id`);--> statement-breakpoint
CREATE INDEX `cases_task_key_idx` ON `cases` (`task_key`);--> statement-breakpoint
CREATE INDEX `cases_partition_idx` ON `cases` (`partition`);--> statement-breakpoint
CREATE INDEX `cases_provenance_idx` ON `cases` (`provenance`);--> statement-breakpoint
CREATE INDEX `cases_review_state_idx` ON `cases` (`review_state`);--> statement-breakpoint
CREATE INDEX `cases_source_trace_id_idx` ON `cases` (`source_trace_id`);
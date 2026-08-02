ALTER TABLE `experiment_results` ADD `content_hash` text;--> statement-breakpoint
ALTER TABLE `gate_specs` ADD `max_skip_fraction` real;--> statement-breakpoint
--> Backfill the content hash from the still-present case, so cohorts decided
--> before this migration reconstruct without depending on a live join (#5).
UPDATE `experiment_results`
SET `content_hash` = (
  SELECT `c`.`content_hash` FROM `cases` `c` WHERE `c`.`case_id` = `experiment_results`.`case_id`
)
WHERE `content_hash` IS NULL;
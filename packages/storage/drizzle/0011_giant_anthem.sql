CREATE TABLE `gate_decision_cases` (
	`gate_result_id` text NOT NULL,
	`content_hash` text NOT NULL,
	PRIMARY KEY(`gate_result_id`, `content_hash`),
	FOREIGN KEY (`gate_result_id`) REFERENCES `gate_results`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE INDEX `gate_decision_cases_content_hash_idx` ON `gate_decision_cases` (`content_hash`);
--> statement-breakpoint
-- Backfill the decided cohort of every EXISTING verdict (#9): the cases present
-- in both its candidate and reference experiments, by content hash. This gives
-- pre-guard verdicts a real membership so the overlap-based peeking guard works
-- for them too, rather than treating them as opaque legacy rows.
INSERT OR IGNORE INTO `gate_decision_cases` (`gate_result_id`, `content_hash`)
SELECT gr.`id`, c.`content_hash`
FROM `gate_results` gr
JOIN `experiment_results` erc ON erc.`experiment_id` = gr.`candidate_experiment_id`
JOIN `experiment_results` err ON err.`experiment_id` = gr.`reference_experiment_id` AND err.`case_id` = erc.`case_id`
JOIN `cases` c ON c.`case_id` = erc.`case_id`;
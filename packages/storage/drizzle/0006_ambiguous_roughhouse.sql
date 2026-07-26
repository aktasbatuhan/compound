ALTER TABLE `gate_specs` ADD `candidate_prompt_hash` text;--> statement-breakpoint
ALTER TABLE `gate_specs` ADD `optimization_run_id` text REFERENCES optimization_runs(id);
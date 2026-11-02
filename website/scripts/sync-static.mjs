// Copy the generated reports into public/ so they keep their /report/ URLs.
// site/report is produced and checked by the Python side (serving_evidence.py);
// the docs only serve it.
import { cpSync, existsSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, '..', '..', 'site', 'report');
const target = join(here, '..', 'public', 'report');

if (!existsSync(source)) {
	console.error(`no reports at ${source}`);
	process.exit(1);
}
rmSync(target, { recursive: true, force: true });
cpSync(source, target, { recursive: true });
console.log('copied site/report to public/report');

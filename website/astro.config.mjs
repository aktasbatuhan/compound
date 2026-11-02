// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Old addresses from the first docs site, kept working for links already shared.
const redirects = {
	'/docs': '/start/introduction/',
	'/docs/providers': '/measure/providers/',
	'/docs/benchmarks': '/measure/benchmarks/',
	'/docs/harbor': '/measure/harbor/',
	'/docs/reports': '/measure/reports/',
	'/docs/serving': '/measure/serving/',
	'/docs/sdk': '/traffic/sdk/',
	'/docs/shape': '/traffic/shape/',
	'/docs/estimate': '/price/estimate/',
	'/docs/replay': '/measure/replay/',
	'/docs/simulate': '/prove/simulate/',
	'/docs/traces': '/prove/traces/',
	'/docs/config': '/reference/config/',
	'/docs/development': '/reference/development/',
	'/docs/reference': '/reference/cli/',
	'/examples': '/start/examples/',
};

export default defineConfig({
	redirects,
	integrations: [
		starlight({
			title: 'Compound',
			description:
				'Find out what your AI traffic would cost and how it would behave on other models and hosts, measured on your own workload.',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/aktasbatuhan/compound' },
			],
			editLink: {
				baseUrl: 'https://github.com/aktasbatuhan/compound/edit/main/website/',
			},
			customCss: ['./src/styles/theme.css'],
			lastUpdated: true,
			sidebar: [
				{ label: 'Get started', items: [{ autogenerate: { directory: 'start' } }] },
				{ label: 'Bring your traffic', items: [{ autogenerate: { directory: 'traffic' } }] },
				{ label: 'Price it', items: [{ autogenerate: { directory: 'price' } }] },
				{ label: 'Measure it', items: [{ autogenerate: { directory: 'measure' } }] },
				{ label: 'Prove it', items: [{ autogenerate: { directory: 'prove' } }] },
				{ label: 'Reference', items: [{ autogenerate: { directory: 'reference' } }] },
			],
		}),
	],
});

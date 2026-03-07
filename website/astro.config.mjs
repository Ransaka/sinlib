// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
	integrations: [
		starlight({
			title: 'Sinlib',
			description: 'Sinhala NLP toolkit — phonological tokenization, spell checking, and text preprocessing.',
			logo: {
				light: './src/assets/logo.svg',
				dark: './src/assets/logo.svg',
				replacesTitle: false,
			},
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/Ransaka/sinlib' },
				{ icon: 'seti:python', label: 'PyPI', href: 'https://pypi.org/project/sinlib/' },
			],
			editLink: {
				baseUrl: 'https://github.com/Ransaka/sinlib/edit/main/website/',
			},
			customCss: ['./src/styles/custom.css'],
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						{ label: 'Introduction', slug: 'index' },
					],
				},
				{
					label: 'Guides',
					items: [
						{ label: 'Tokenization', slug: 'guides/tokenization' },
						{ label: 'Spell Checking', slug: 'guides/spellcheck' },
					],
				},
				{
					label: 'API Reference',
					items: [
						{ label: 'Tokenizer', slug: 'api/tokenizer' },
						{ label: 'TypoDetector', slug: 'api/spellcheck' },
						{ label: 'BatchEncoding', slug: 'api/encoding' },
						{ label: 'Preprocessing', slug: 'api/preprocessing' },
					],
				},
				{
					label: 'Examples',
					items: [
						{ label: 'Tokenization', slug: 'examples/tokenization' },
						{ label: 'Typo Correction', slug: 'examples/typo-correction' },
					],
				},
			],
		}),
	],
});

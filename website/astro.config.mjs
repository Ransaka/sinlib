// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// When building on ReadTheDocs, READTHEDOCS_CANONICAL_URL is set to the full
// versioned URL, e.g. https://sinlib.readthedocs.io/en/latest/
// We derive the base path from it so Astro generates correct asset URLs.
const canonicalUrl = process.env.READTHEDOCS_CANONICAL_URL;
let site = 'https://sinlib.readthedocs.io';
let base = undefined;
if (canonicalUrl) {
	const parsed = new URL(canonicalUrl);
	site = parsed.origin;
	// Strip trailing slash; Astro expects base like '/en/latest' not '/en/latest/'
	base = parsed.pathname.replace(/\/$/, '') || undefined;
}

export default defineConfig({
	site,
	base,
	integrations: [
		starlight({
			title: 'Sinlib',
			description: 'Sinhala NLP toolkit — phonological tokenization, spell checking, and text preprocessing.',
			logo: {
				light: './src/assets/logo-light.png',
				dark: './src/assets/logo-dark.png',
				replacesTitle: true,
			},
			head: [
				{ tag: 'link', attrs: { rel: 'preconnect', href: 'https://fonts.googleapis.com' } },
				{ tag: 'link', attrs: { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: true } },
				{
					tag: 'link',
					attrs: {
						rel: 'stylesheet',
						href: 'https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;600&family=Noto+Sans+Sinhala:wght@400;500;600&display=swap',
					},
				},
			],
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/Ransaka/sinlib' },
				{ icon: 'seti:python', label: 'PyPI', href: 'https://pypi.org/project/sinlib/' },
			],
			editLink: {
				baseUrl: 'https://github.com/Ransaka/sinlib/edit/main/website/',
			},
			customCss: ['./src/styles/custom.css'],
			tableOfContents: false,
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
						{ label: 'Romanizer', slug: 'api/romanizer' },
						{ label: 'BatchEncoding', slug: 'api/encoding' },
						{ label: 'Preprocessing', slug: 'api/preprocessing' },
						{ label: 'Visualization', slug: 'api/visualization' },
					],
				},
				{
					label: 'Examples',
					items: [
						{ label: 'Tokenization', slug: 'examples/tokenization' },
						{ label: 'Typo Correction', slug: 'examples/typo-correction' },
						{ label: 'Visualization', slug: 'examples/visualization' },
					],
				},
			],
		}),
	],
});

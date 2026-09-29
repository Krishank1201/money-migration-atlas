import fs from 'fs';
import path from 'path';

// Smoke test: verifies frontend bundle and HTML structure
const distPath = path.resolve('dist');
const indexPath = path.join(distPath, 'index.html');

if (!fs.existsSync(indexPath)) {
  console.error('Smoke test failed: dist/index.html missing');
  process.exit(1);
}

const html = fs.readFileSync(indexPath, 'utf-8');
if (!html.includes('Money Migration Atlas') || !html.includes('id="root"')) {
  console.error('Smoke test failed: index.html structure invalid');
  process.exit(1);
}

const assetsPath = path.join(distPath, 'assets');
const files = fs.readdirSync(assetsPath);
const jsBundle = files.find(f => f.endsWith('.js'));
const cssBundle = files.find(f => f.endsWith('.css'));

if (!jsBundle || !cssBundle) {
  console.error('Smoke test failed: js/css bundles missing');
  process.exit(1);
}

console.log(`✓ Smoke test passed: Bundle verified (JS: ${jsBundle}, CSS: ${cssBundle})`);

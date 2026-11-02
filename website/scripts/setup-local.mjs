import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {projectRoot} from './sites-env.mjs';

const [major,minor]=process.versions.node.split('.').map(Number);
if(major<22||(major===22&&minor<13))throw new Error('Use Node.js 22.13 or newer.');

// The site is static: it bundles the repository records under results/ at build time.
const result=spawnSync(process.execPath,[path.join('scripts','run-framework.mjs'),'build'],{cwd:projectRoot,stdio:'inherit'});
if(result.error)throw result.error;
if(result.status!==0)process.exit(result.status??1);
console.log('Local setup complete. Run npm run dev, then open http://127.0.0.1:5173/.');

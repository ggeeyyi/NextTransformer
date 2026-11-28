import {readdirSync,readFileSync} from 'node:fs';
const root=new URL('../../results/',import.meta.url);
const read=dir=>readdirSync(new URL(dir,root)).filter(f=>f.endsWith('.json')).sort().map(f=>JSON.parse(readFileSync(new URL(dir+f,root),'utf8')));
const order=(a,b)=>(a.order??Infinity)-(b.order??Infinity);
export const runs=()=>read('runs/').sort(order);
export const methods=()=>read('methods/').sort(order);
export const arena=()=>read('arena/').sort(order);
export const sessions=()=>read('arena/sessions/').sort(order);

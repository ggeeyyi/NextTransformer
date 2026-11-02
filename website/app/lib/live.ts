import {baselineIdentity} from './baseline';
import {methods as registered,runs,updatedAt,SOURCE} from './results';
import type {Dataset} from './types';
export const methods=registered;
// The site publishes the records merged into the repository; it queries no tracking service.
export async function getDataset(){
 const dataset:Dataset={schemaVersion:1,updatedAt,source:SOURCE,runs:runs.map(baselineIdentity)};
 return {dataset,methods};
}

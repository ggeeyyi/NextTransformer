import {getDataset} from '../../lib/live';
export const dynamic='force-dynamic';
export async function GET(){try{return Response.json(await getDataset(),{headers:{'Cache-Control':'no-store'}});}catch{return Response.json({error:'The data service is temporarily unavailable. Please refresh.'},{status:503});}}

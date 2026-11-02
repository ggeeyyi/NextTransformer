// Public wording for identifiers stored in records; the identifiers themselves stay fixed because fingerprints and checks use them.
const PROTOCOLS:Record<string,string>={
 'nextt-reference-50b-v1':'Reference protocol · 49.152B tokens',
 'native-v3 · historical matrix FLOPs':'Original native protocol · historical matrix FLOPs',
 'nextt-arena-v1':'Arena rules v1',
};
const LEDGERS:Record<string,string>={
 'reference-aten-math-v2':'Operator-level cost ledger',
 'reference-aten-math-v2-all-backends-replay':'Operator-level cost ledger with backward replay (frozen earlier source)',
 'Reported tracking estimate · nextt-reference-50b-v1':'Operator-level cost ledger, reported by the trainer',
 'native-v3-matrix-estimate':'Historical matrix-FLOPs estimate',
};
export const protocolLabel=(id:string)=>PROTOCOLS[id]??id;
export const ledgerLabel=(id:string)=>LEDGERS[id]??id;

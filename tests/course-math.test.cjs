const assert=require('node:assert/strict');
const M=require('../course-math.js');
// Online normalization must match an explicit, numerically stabilized softmax.
for(const scores of [[2,1,-1,3,0],[1000,1001,999],[-1000,-999,-1002],[.1]])for(const tile of [1,2,3,8]){
 const values=scores.map((_,i)=>Math.sin(i+.4)),w=M.softmax(scores);
 const expected=w.reduce((s,x,i)=>s+x*values[i],0);
 assert(Math.abs(expected-M.onlineSoftmax(scores,values,tile))<1e-12);
}
for(let n=1;n<=4;n++){const r=M.linearPrefix(n);assert(M.maxDiff(r.out,r.explicit)<1e-12);assert(r.denom>0);}
for(let n=1;n<=8;n++)for(const beta of [0,.1,.7,1]){const {delta}=M.overwrite(n,beta);assert(Math.abs(delta[0]-(1-beta)**n)<1e-12);assert(Math.abs(delta[1]-(1-(1-beta)**n))<1e-12);}
for(const a of [[1,1],[.98,.6],[.2,.9]])for(const beta of [0,.7,1]){
 const S=[[.2,-.1],[.3,.4]],k=[Math.SQRT1_2,Math.SQRT1_2],v=[.2,.8];
 const r=M.deltaStep(S,k,v,a,beta);
 for(let i=0;i<2;i++)for(let j=0;j<2;j++){
  const expected=beta*k[i]*v[j]+S.reduce((s,row,n)=>s+((i===n?1:0)-beta*k[i]*k[n])*a[n]*row[j],0);
  assert(Math.abs(expected-r.state[i][j])<1e-12);
 }
 if(beta===1)assert(M.maxDiff(M.matmul([k],r.state)[0],v)<1e-12);
}
// Include tails, nonzero initial states, and multiple chunk sizes.
let maxError=0;
for(const n of [1,3,7,16,19,64])for(const c of [1,2,4,8,16,32]){
 const seq=M.tokens(n),a=M.recurrent(seq),b=M.chunked(seq,c);
 const err=Math.max(M.maxDiff(a.O,b.O),M.maxDiff(a.S,b.S));
 assert(err<1e-11,`chunk mismatch n=${n}, C=${c}, err=${err}`);maxError=Math.max(maxError,err);
}
console.log(`PASS: online softmax, linear-state equivalence, overwrite endpoints, KDA recurrence, 36 chunk cases (tails/nonzero state). Max chunk error=${maxError}.`);

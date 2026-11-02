/* Small deterministic references for the lessons. Double precision, not NPU kernels. */
(function(root){
  'use strict';
  const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
  const zeros=(m,n)=>Array.from({length:m},()=>Array(n).fill(0));
  const matmul=(a,b)=>a.map(row=>b[0].map((_,j)=>row.reduce((s,x,k)=>s+x*b[k][j],0)));
  const transpose=a=>a[0].map((_,j)=>a.map(row=>row[j]));
  const maxDiff=(a,b)=>Math.max(0,...a.flat(Infinity).map((x,i)=>Math.abs(x-b.flat(Infinity)[i])));
  function softmax(scores){const m=Math.max(...scores),e=scores.map(x=>Math.exp(x-m)),z=e.reduce((a,b)=>a+b,0);return e.map(x=>x/z);}
  function onlineSoftmax(scores,values,tile=2){let m=-Infinity,l=0,a=0;for(let s=0;s<scores.length;s+=tile){const block=scores.slice(s,s+tile),nextM=Math.max(m,...block),r=Math.exp(m-nextM);a*=r;l*=r;for(let j=0;j<block.length;j++){const e=Math.exp(block[j]-nextM);l+=e;a+=e*values[s+j];}m=nextM;}return a/l;}
  function linearPrefix(n){const keys=[[1,.2],[.3,1],[.8,.6],[.2,1.2]],values=[[1,0],[0,1],[.5,.5],[-.2,.8]],q=[.7,1.1],S=zeros(2,2),z=[0,0];for(let i=0;i<n;i++)for(let j=0;j<2;j++){z[j]+=keys[i][j];for(let d=0;d<2;d++)S[j][d]+=keys[i][j]*values[i][d];}const denom=dot(q,z),out=values[0].map((_,d)=>q.reduce((sum,x,j)=>sum+x*S[j][d],0)/denom),weights=keys.slice(0,n).map(k=>dot(q,k)),wSum=weights.reduce((a,b)=>a+b,0),explicit=values[0].map((_,d)=>weights.reduce((s,w,i)=>s+w*values[i][d],0)/wSum);return{S,z,out,explicit,denom};}
  function overwrite(n,beta){let add=[1,0],delta=[1,0];for(let i=0;i<n;i++){add[1]+=1;delta=delta.map((v,j)=>v+beta*((j===1?1:0)-v));}return{add,delta};}
  function deltaStep(S,k,v,alpha,beta){const decay=S.map((row,i)=>row.map(x=>x*alpha[i]));const pred=matmul([k],decay)[0],r=v.map((x,j)=>beta*(x-pred[j]));const state=decay.map((row,i)=>row.map((x,j)=>x+k[i]*r[j]));return{decay,pred,r,state};}
  function tokens(n=16){return Array.from({length:n},(_,i)=>{const theta=.25+i*.31;return{q:[Math.cos(theta+.2),Math.sin(theta+.2)],k:[Math.cos(theta),Math.sin(theta)],v:[Math.sin(i*.7),Math.cos(i*.4)],g:[-.04-.01*(i%3),-.12-.02*(i%2)],beta:.6};});}
  function recurrent(seq,S0=[[.2,-.1],[.3,.4]]){let S=S0.map(r=>r.slice());const O=[];for(const t of seq){S=deltaStep(S,t.k,t.v,t.g.map(Math.exp),t.beta).state;O.push(matmul([t.q],S)[0]);}return{O,S};}
  function solveLower(A,B){const X=zeros(A.length,B[0].length);for(let i=0;i<A.length;i++)for(let d=0;d<B[0].length;d++){let x=B[i][d];for(let j=0;j<i;j++)x-=A[i][j]*X[j][d];X[i][d]=x/A[i][i];}return X;}
  function chunked(seq,C,S0=[[.2,-.1],[.3,.4]]){let S=S0.map(r=>r.slice());const O=[];for(let start=0;start<seq.length;start+=C){const ts=seq.slice(start,start+C),N=ts.length,G=zeros(N,2);ts.forEach((t,i)=>{for(let d=0;d<2;d++)G[i][d]=t.g[d]+(i?G[i-1][d]:0);});const L=zeros(N,N),P=zeros(N,N);for(let i=0;i<N;i++)for(let j=0;j<=i;j++){const decay=G[i].map((x,d)=>Math.exp(x-G[j][d]));P[i][j]=dot(ts[i].q,ts[j].k.map((x,d)=>x*decay[d]));if(j<i)L[i][j]=ts[i].beta*dot(ts[i].k,ts[j].k.map((x,d)=>x*decay[d]));}for(let i=0;i<N;i++)L[i][i]=1;const Kplus=ts.map((t,i)=>t.k.map((x,d)=>Math.exp(G[i][d])*x)),Qplus=ts.map((t,i)=>t.q.map((x,d)=>Math.exp(G[i][d])*x));const U=solveLower(L,ts.map(t=>t.v.map(x=>x*t.beta))),W=solveLower(L,Kplus.map((row,i)=>row.map(x=>x*ts[i].beta)));const WS=matmul(W,S),R=U.map((row,i)=>row.map((x,d)=>x-WS[i][d])),Qbase=matmul(Qplus,S),PR=matmul(P,R);O.push(...Qbase.map((row,i)=>row.map((x,d)=>x+PR[i][d])));const Kend=ts.map((t,i)=>t.k.map((x,d)=>x*Math.exp(G[N-1][d]-G[i][d]))),write=matmul(transpose(Kend),R);S=S.map((row,i)=>row.map((x,d)=>Math.exp(G[N-1][i])*x+write[i][d]));}return{O,S};}
  const api={dot,zeros,matmul,transpose,maxDiff,softmax,onlineSoftmax,linearPrefix,overwrite,deltaStep,tokens,recurrent,chunked};
  if(typeof module==='object'&&module.exports)module.exports=api;else root.CourseMath=api;
})(typeof window!=='undefined'?window:globalThis);

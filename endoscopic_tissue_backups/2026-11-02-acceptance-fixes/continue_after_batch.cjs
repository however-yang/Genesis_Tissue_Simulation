// Wait outside Python so the current GPU supervisor cannot deadlock on a
// second waiting Python parent. No simulation, promotion, Git or process kill.
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const {spawn} = require('node:child_process');

const here = __dirname;
const root = path.resolve(here, '../..');
const prior = path.join(here, 'feasible-diagnose-2.json');
const next = path.join(here, 'acceptance-followup-4.json');
const output = path.join(root, 'endoscopic_tissue/outputs/acceptance-fixes-20261102');
const status = path.join(output, 'serial-followup-status.json');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

async function save(record) {
  await fs.writeFile(status+'.tmp', JSON.stringify({...record,
    candidate_adopted:false, default_replaced:false, github_pushed:false,
    user_processes_touched:false}, null, 2)+'\n');
  await fs.rename(status+'.tmp', status);
}

async function main() {
  const priorBytes = await fs.readFile(prior), nextBytes = await fs.readFile(next);
  const frozen = {prior:digest(priorBytes), next:digest(nextBytes)};
  const expected = JSON.parse(priorBytes).jobs.map(job => job.name);
  const started = Date.now();
  // Exclusive marker prevents a duplicate follower from launching twice.
  const marker = await fs.open(status+'.owner', 'wx');
  await marker.writeFile(JSON.stringify({pid:process.pid,started,plans:frozen}));
  await marker.close();
  await save({phase:'waiting_for_prior_batch',pid:process.pid,started,plans:frozen});
  console.log('Waiting for the current serial batch; no second Python supervisor started.');
  for (;;) {
    if (Date.now()-started > 24*60*60*1000) {
      await save({phase:'expired_without_launch',pid:process.pid,started,plans:frozen});
      return;
    }
    if (digest(await fs.readFile(prior))!==frozen.prior || digest(await fs.readFile(next))!==frozen.next) {
      throw new Error('A frozen plan changed; retained without launching follow-up');
    }
    let result;
    try { result=JSON.parse(await fs.readFile(prior.replace(/\.json$/,'.results.json'),'utf8')); }
    catch (error) { if(error.code!=='ENOENT') throw error; }
    if (result?.completed===true) {
      if (JSON.stringify(result.jobs.map(job=>job.name))!==JSON.stringify(expected)) {
        throw new Error('Prior completed-job inventory does not match its frozen plan');
      }
      break;
    }
    await sleep(10000);
  }
  const log=await fs.open(path.join(output,'serial-followup.log'),'wx');
  const child=spawn('D:/develop/anaconda3/envs/genesis_world/python.exe',
    ['-X','utf8',path.join(here,'run_batch.py'),next],
    {cwd:root,windowsHide:true,stdio:['ignore',log.fd,log.fd]});
  await save({phase:'followup_running',pid:process.pid,child_pid:child.pid,started,plans:frozen});
  const code=await new Promise((resolve,reject)=>{
    child.on('error',reject);
    child.on('exit',(code,signal)=>resolve({code,signal}));
  });
  await log.close();
  await save({phase:'followup_finished_requires_result_review',pid:process.pid,started,
    ended:Date.now(),plans:frozen,exit:code});
  console.log('Follow-up ended; inspect per-job results. No acceptance or promotion performed.');
}

main().catch(async error=>{
  console.error(error);
  await save({phase:'error_no_acceptance',error:String(error),pid:process.pid}).catch(()=>{});
  process.exitCode=1;
});

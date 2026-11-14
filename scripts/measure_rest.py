#!/usr/bin/env python3
"""Remainder of the final matrix (cache=16 API rows, kernel rows, missing cache=0 rows), 3 parallel workers."""
import subprocess, concurrent.futures, os, pathlib
root = pathlib.Path('/home/user/fft-rvv'); os.chdir(root)
jobs = []
for v in (128, 256, 512): jobs.append(('rvv','gcc',0,'c2r','float',1048576,v,'build/c0_rvv_gcc'))
for cc in ('clang','gcc'):
    for kind in ('c2c','r2c','c2r'):
        for typ in ('double','float'):
            for n in (1024, 4096, 65536):
                jobs.append(('scalar-rv64gc',cc,16,kind,typ,n,'-',f'build/c16_scalar_{cc}_rv64gc'))
                for v in (128,256,512): jobs.append(('rvv',cc,16,kind,typ,n,v,f'build/c16_rvv_{cc}'))
jobs.append(('scalar-rv64gc','clang',16,'c2c','double',1048576,'-','build/c16_scalar_clang_rv64gc'))
for v in (128,256,512): jobs.append(('rvv','clang',16,'c2c','double',1048576,v,'build/c16_rvv_clang'))
for cc in ('clang','gcc'):
    for n in (1024,4096,65536,1048576):
        jobs.append(('kernel-scalar-rv64gc',cc,16,'x2c','double',n,'-',f'build/c16_scalar_{cc}_rv64gc'))
        for v in (128,256,512): jobs.append(('kernel-rvv',cc,16,'x2c','double',n,v,f'build/c16_rvv_{cc}'))
def run(j):
    variant,cc,cache,kind,typ,n,v,binp = j
    fwd = '0' if kind=='c2r' else '1'; vl = '256' if v=='-' else str(v)
    out = subprocess.run(['scripts/insncount.sh', binp, vl, kind, typ, str(n), fwd], capture_output=True, text=True).stdout.strip()
    return f"{variant},{cc},{cache},{kind},{typ},{n},{v},{out}"
rows = []
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
    for r in ex.map(run, jobs):
        rows.append(r); print(r, flush=True)
with open('results/final_insns.csv','a') as f:
    for r in rows: f.write(r+'\n')
with open('results/final_measure.log','a') as f: f.write('DONE2\n')

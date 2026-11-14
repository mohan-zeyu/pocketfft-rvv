#!/usr/bin/env python3
"""Produce a single-file header (pocketfft_hdronly_rvv.h) by inlining pocketfft_rvv.h and pocketfft_rvv_real.h,
and a unified diff against the upstream header, for distribution to NumPy/PyTorch style vendoring."""
import re, subprocess, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
main = (root/'pocketfft_hdronly.h').read_text()
def inline(text, name):
    body = (root/name).read_text()
    # strip the nested #include of the helper header inside the real header (already inlined) 
    body = body.replace('#include "pocketfft_rvv.h"\n', '')
    return text.replace(f'#include "{name}"\n', f'// ---- begin inlined {name} ----\n{body}\n// ---- end inlined {name} ----\n')
out = main
out = inline(out, 'pocketfft_rvv_real.h')   # inline real first (it is included later in the file)
# the helper header is included twice (once inside detail, once at global scope around the real header);
# inline only the first occurrence, drop the second (include guard makes it a no-op anyway)
first = out.find('#include "pocketfft_rvv.h"\n')
helper = (root/'pocketfft_rvv.h').read_text()
out = out[:first] + '// ---- begin inlined pocketfft_rvv.h ----\n' + helper + '\n// ---- end inlined pocketfft_rvv.h ----\n' + out[first+len('#include "pocketfft_rvv.h"\n'):]
out = out.replace('#include "pocketfft_rvv.h"\n', '')
(root/'dist').mkdir(exist_ok=True)
(root/'dist'/'pocketfft_hdronly_rvv.h').write_text(out)
diff = subprocess.run(['diff','-u',str(root/'work'/'pocketfft_hdronly.orig.h'),str(root/'pocketfft_hdronly.h')],capture_output=True,text=True).stdout
(root/'dist'/'pocketfft_hdronly.rvv.patch').write_text(diff)
print("wrote dist/pocketfft_hdronly_rvv.h (%d lines) and dist/pocketfft_hdronly.rvv.patch (%d lines)" % (out.count('\n'), diff.count('\n')))

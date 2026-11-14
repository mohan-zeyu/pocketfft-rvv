#!/bin/bash
# Assemble the hand-over package under /home/user/deliverables
set -e
SRC=/home/user/fft-rvv; DST=/home/user/deliverables; rm -rf $DST; mkdir -p $DST/{source,docs,tests,results}
python3 $SRC/scripts/make_single_header.py >/dev/null
cp $SRC/pocketfft_hdronly.h $SRC/pocketfft_rvv.h $SRC/pocketfft_rvv_real.h $DST/source/
cp $SRC/dist/pocketfft_hdronly_rvv.h $SRC/dist/pocketfft_hdronly.rvv.patch $DST/source/
cp $SRC/work/pocketfft_hdronly.orig.h $DST/source/pocketfft_hdronly.upstream.h
cp -r $SRC/docs/* $DST/docs/ 2>/dev/null || true
cp $SRC/README.md $DST/README.md 2>/dev/null || true
cp -r $SRC/tests $SRC/bench $SRC/scripts $DST/tests/ 2>/dev/null || true
cp $SRC/work-real/test_real.cpp $DST/tests/tests/test_real.cpp 2>/dev/null || true
cp $SRC/results/*.csv $SRC/results/*.txt $SRC/results/*.md $SRC/results/*.png $DST/results/ 2>/dev/null || true
(cd $SRC && git log --oneline > $DST/results/git-history.txt)
du -sh $DST; find $DST -maxdepth 2 | sort

"""
`iostat -d -w 1` with every line prefixed by its wall-clock time, so a drive trace lines up with a benchmark that
prints its own start time (`prefill_unwire_timeline.py`'s `EPOCH T0=`). HANDOFF section 15.13.

    python3 benchmarks/iostat_ts.py disk0 disk6 > /tmp/iostat.log &
"""
import subprocess
import sys
import time

disks = sys.argv[1:] or ["disk0"]
proc = subprocess.Popen(["iostat", "-d", "-w", "1", *disks], stdout=subprocess.PIPE, text=True, bufsize=1)
for line in proc.stdout:
    print(f"{time.time():.3f} {line.rstrip()}", flush=True)

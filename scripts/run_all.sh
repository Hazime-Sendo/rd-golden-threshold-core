#!/usr/bin/env bash
# Runs every check and regenerates the figures; writes logs to results/logs/ and figures to results/figures/.
set -u
cd "$(dirname "$0")/.."
mkdir -p results/logs results/figures
rc=0
{
  echo "date (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 --version
  python3 -c "import numpy, scipy, matplotlib; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| matplotlib', matplotlib.__version__)"
  uname -srm
} > results/logs/environment.txt 2>&1
python3 scripts/check_parameters.py > results/logs/check_parameters.log 2>&1 || rc=1
python3 scripts/verify_all.py       > results/logs/verify_all.log       2>&1 || rc=1
for s in verify_reduced verify_theorems verify_ratio; do
  python3 scripts/$s.py > results/logs/$s.log 2>&1 || rc=1
done
python3 scripts/make_figures.py     > results/logs/make_figures.log     2>&1 || rc=1
tail -n 1 results/logs/verify_all.log
echo "exit code: $rc  (0 = every script finished and verify_all.py passed all checks)"
exit $rc

# BBB Project 2 HPO v2 — Remaining 4 Modes

Run `bbb_project2_hpo_v2.py` for the following 4 modes IN ORDER, each to completion before starting the next.

## Script
`/home/minji/autoresearch/bbb_project2_hpo_v2.py`

## Conda environment
`rapids-25.02`

## Modes to run (in this exact order)

1. `rs_composite`  → log: `/home/minji/BBB/project2/logs/v2_rs_composite.log`
2. `sc_composite`  → log: `/home/minji/BBB/project2/logs/v2_sc_composite.log`
3. `cv10`          → log: `/home/minji/BBB/project2/logs/v2_cv10.log`
4. `cv10_composite` → log: `/home/minji/BBB/project2/logs/v2_cv10_composite.log`

## Command per mode

```bash
conda run --no-capture-output -n rapids-25.02 python -u /home/minji/autoresearch/bbb_project2_hpo_v2.py --mode MODE >> LOG 2>&1
```

Run each command as a **blocking** Bash call (no `&`), with timeout=600000. Wait for it to fully complete before running the next mode.

## Execution steps

1. Run `rs_composite` (blocks until done, ~2 hours)
2. Run `sc_composite` (blocks until done, ~2 hours)
3. Run `cv10` (blocks until done, ~2 hours)
4. Run `cv10_composite` (blocks until done, ~2 hours)

After all 4 modes complete, read the last 20 lines of each log file and report the best val score per mode.

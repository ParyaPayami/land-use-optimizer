# Run logs for `results/paper`

`segment_01.log` to `segment_05.log` are the run whose outputs the paper reports
(`configs/paper.yaml`, MapPLUTO 26v2). The run was executed in resumable segments;
each segment resumes from the checkpoints of the previous one.

The GNN pre-training (all seeds) and the edge-type ablation were reused from an
earlier full run, whose logs are in `first_run/`: those stages do not use the
capacity screens, and the earlier run's other results were discarded after a fix to
the shadow screen (commit b5ceb05). Timing fields missing from the reused
checkpoints were reconstructed from these logs; see `../timing_backfill.json`.

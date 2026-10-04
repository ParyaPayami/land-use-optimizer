# Run logs for `results/paper`

`segment_01.log` to `segment_05.log` are the run whose outputs the paper reports
(`configs/paper.yaml`: MapPLUTO 26v2, GNN 500 epochs, 3 seeds). The run was executed
in resumable segments; each segment resumes from the checkpoints of the previous one
(training state is saved every 10 GNN epochs and every 5 PPO iterations).

The edge-type ablation (80 epochs per configuration, independent of the main GNN
budget and of the capacity screens) was reused from the first full run, whose logs are
in `first_run/`. Its timing fields were reconstructed from those logs; see
`../timing_backfill.json`.

`gnn150_run/` holds the logs of an earlier complete run with a 150-epoch GNN, which
was superseded by the 500-epoch run. The first run's other results were discarded
after a fix to the shadow screen (commit b5ceb05).

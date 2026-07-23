# Layer 3 — SSL / anomaly research

Learned embedding and anomaly-detection experiments. Advisory / research layer;
not the first real-time deployment gate. Prefer Layer 1 + Layer 2 for safety
until Phase 1 results justify more complexity.

## Folder map

```text
Layer3/
  pipeline/     Encoder, SSL objectives, anomaly / Mahalanobis heads
  validation/   Beat/window Phase 1 eval and metrics helpers
  tools/        Pretrain, window index, smoke tests, decks
  reports/      Docs map, protocol, arm status, cluster jobs
```

## Start here

| Need | Open |
| --- | --- |
| Docs / status / protocol | [`reports/README.md`](reports/README.md) |
| Algorithm summary | [`ALGORITHM_SUMMARY.md`](ALGORITHM_SUMMARY.md) |
| Pretrain entrypoint | `tools/pretrain_encoder.py` |
| Phase 1 eval | `validation/layer3_phase1_eval.py` |
| Local vs cluster runs | [`../AGENTS.local.md`](../AGENTS.local.md) |

Agents: do not load the whole `reports/` folder. Open `reports/README.md`, then
the single status/spec file for the arm you are editing.

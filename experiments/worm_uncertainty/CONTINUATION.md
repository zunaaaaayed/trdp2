# Current continuation — 28 October 2026

Start here for active Week6 work (started24 October). NEXT_WEEK.md remains the
completed Weeks4–5 summary/then-next-week plan; it is not a current run ledger.
README.md is cumulative; NONLINEAR_POSTERIOR.md and COMPLETION_PROTOCOL.md own
this milestone's methods, findings, failures and limitations.

## Current state

Branch `experiment/worm-uncertainty`, base936f8c7. This session's changes are
uncommitted; no push/merge was requested under the new scope. No user changes
were discarded. No new experiment is running. Old pilots/outputs remain intact.
No applicable AGENTS.md was found. The old knowledge graph lacks current work.

Completed: four-case exact-target/proposal audit (72.43s), two-case bounded
Metropolis feasibility (27.58s),10 new unit tests, completion data smoke, two plots,
versioned split manifest/protocols, paper-method follow-up and README updates.
All68 CPU tests pass. The final style gate covers86 experiment Python files.
The v1 audit is preserved, including all failed decisions and unconverged chains.
Completion smoke v2 only fixes style relative to v1; its arrays are identical.

Scientific decision: joint importance correction is inadequate on all four cases,
even moderate cases with no bound violations. Density-only is a correct conditional
calculation but omits crucial uncertainty. Truncation does not fix the nonlinear
likelihood; the short random walk does not mix. There is no validated constrained
reference or calibrated worm posterior yet. All inference cases retain oracle
geometry priors and fixed simulation noise scales.

## Immediate next work (not launched)

1. Design a curvature-aware, constraint-respecting reference sampler. Consider
   conditional-density updates plus geometry moves or derivative-based proposals;
   preserve the declared measure. Use a separate tuning case, then freeze settings.
   Add scalar/linear controls and rank/folded Rhat, bulk/tail ESS for output functionals.
   Budget the first feasibility test at≤10 CPU minutes, one thread, two evaluation
   cases. Do not repeatedly extend the unchanged failed random walk.
2. Implement the specified tiny supervised completion trainer. First run a single
   minibatch CPU gradient smoke, then review a bounded≤20 GPU-minute run per primary
   arm. No large sweep. Verify seven-channel geometry and mask ablations and record
   the actual model capacity (expected5,809). The data loader is ready; no trainer
   command exists yet, and no unimplemented training command is implied below.
3. Implement and test initialization-only integration. Acquired data, geometry
   prior reference, sigma, bounds, count, starts and refinement budget must be
   identical across arms. Resolve the full100-angle versus sparse25-angle input
   advantage before attributing a gain to learning. No predicted ray in likelihood.
4. Confirm deformation relationships, projection path-length/density scaling and
   detector-centre convention. Do not claim independent-organism generalization.
   Reserved outcomes remain unavailable to tuning. All grouping is provisional
   numeric-ID proximity, explicitly documented in the frozen manifest.

## Exact workstation commands

From CREMI (this session used liet):

```bash
cd ~/espaces/travail/trdp2/r2_gaussian
source ~/espaces/travail/trdp2/tools/miniforge3/etc/profile.d/conda.sh
conda activate trdp2-r2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/cache-mdzislam/matplotlib

# Read-only verification, normally <2 seconds; no experiments restarted.
python experiments/worm_uncertainty/verify_nonlinear_audit.py
python -m unittest discover -s experiments/worm_uncertainty -p 'test_*.py'
git status --short

# If the temporary formatter is still present:
make check PYTHON=/tmp/trdp2-format-tools/bin/python
```

If moving machines, transfer ignored `output/worm_uncertainty/scaled_acceptance_v1`,
`posterior_comparison_v1`, `nonlinear_posterior_v1`, and `../worm_data` as well as
Git files. Verify hashes before use. No GPU is needed for these commands. The
research environment and native GPU backends still require checks after a move.

Optional **reproduction**, not needed to advance the completed result (~2CPU
minutes, observed544MiB RSS). Use only if its fresh destination does not exist:

```bash
tmux new-session -s trdp2-posterior-repeat
python -u experiments/worm_uncertainty/audit_nonlinear_posterior.py \
  --output output/worm_uncertainty/nonlinear_posterior_v2 --execute
# Ctrl-b then d detaches; tmux attach -t trdp2-posterior-repeat reconnects.
```

The runner refuses overwrite. v2 above is a reproducibility repeat of the frozen
v1 design, not a new sampler or permission to tune using evaluation results.
For a development data smoke on another workstation (~1s, reads one127MB TIFF
for hash verification and one slice), choose a fresh destination:

```bash
python experiments/worm_uncertainty/completion_data.py \
  --data ../worm_data --output output/worm_uncertainty/completion_data_v3
```

Future GPU training needs a fresh availability check (`nvidia-smi`) and retains
TIGRE/PyTorch process isolation if native reconstruction is used. Current code
uses only the analytic CPU projector. Raw data and experiment arrays are ignored
by Git; compact reports, protocol/source/input hashes and plots are versioned.

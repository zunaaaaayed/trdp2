# Storage maintenance

## Cleanup — 17 October 2026

Removed 7,483,937,282 logical bytes (7.48 GB / 6.97 GiB). The repository tree
now occupies approximately 4.4 GiB on disk, including ignored data and outputs.

- Removed the broad upstream 25/50/75-view synthetic download ZIPs. Their SHA-256
  hashes remain in the deletion inventory. Extracted chest experiment inputs remain;
  unrelated cases previously only in those ZIPs would require downloading again.
- Verified all 655 regular members of the obsolete 7 October transfer TAR against
  retained files byte-for-byte by SHA-256 before deleting the TAR.
- Removed untracked Python bytecode and superseded first-attempt SART arrays/plot.
  The original SART report remains; accepted v2 outputs remain complete.

Retained supervisor worm TIFFs and original archive outside the repository, all
accepted chest/pilot/TV reconstructions and inputs, report sources and figures,
licenses, dependencies, environment/build products and deferred LIDC provenance.
Historical handoffs mentioning the deleted transfer archive remain dated records;
they do not indicate a currently available backup.

[Deletion inventory](cleanup_20261017.json) records exact paths, sizes, reasons
and archive hashes. No blanket ignored-file cleanup or source reorganization was
performed. Fifty CPU tests (7 current + 43 historical) and documentation links
passed after cleanup; no reconstruction or GPU experiment was launched.

## Documentation responsibilities

- Root README: accumulated weekly methods, detailed findings, results and plots.
- NEXT_WEEK: current weekly summary (marked provisional until finished) and next
  week's plan only; no environment manual or accumulated historical task list.
- Each active experiment README: its own plans, findings, outputs, interpretations,
  reproduction commands and limitations. Dated frozen historical protocols remain
  preserved; new results belong in current documentation rather than rewriting
  their original selection rules.

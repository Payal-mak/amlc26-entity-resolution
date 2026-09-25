"""Write output/matching_results.tsv and output/candidate_pairs.tsv (Phase 5 -- TODO).

Planned contents (per the Phase 5 working plan):
  - Exactly one row per test Source-1 entity (France included), tab-separated,
    exact headers (source1_entity_id / matched_entity_ids and
    source1_entity_id / candidate_entity_ids), empty field for no match, no
    duplicate ids within a list, S2-/S3- ids only, matches a subset of
    candidates.
  - After writing: run utils/validate_submission.py against the two files
    and record the result in PROJECT_LOG.md.
  - Save a versioned copy under a submissions/ directory alongside the config/
    commit used to produce it, per the "keep version history" requirement.

Left unimplemented until Phase 5 per the working plan.
"""

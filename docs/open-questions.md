# Open Questions

This file tracks unresolved PRD decision gates. M1 resolved the core inventory
gate, but several interpretation gates remain open.

| Gate | Question | M1 status | Evidence so far | Next action |
|---|---|---|---|---|
| DG-03 | What is the sampling frequency? | Open | KSAS README and CSVs contain no Hz value or timestamps. | Search protocol papers/code before Task 1.2; otherwise report temporal spans in samples only. |
| DG-04 | What are the coordinate conventions? | Partially open | Channel names and units are known; device-frame orientation is not fully specified. | Keep axis claims device-frame based unless protocol evidence is found. |
| DG-06 | Are both arms included and how is phone orientation defined? | Partially open | Both arms are included with complete coverage; phone placement/orientation remains unspecified. | Use arm metadata in splits/diagnostics and document orientation caveats. |
| DG-08 | Which XROCKET implementation will be used? | Open | Deferred by roadmap to M3. | Select implementation and verify license/metadata access in M3. |
| DG-09 | What fixed sequence length is required? | Open | Raw sequence lengths range from 18 to 56 samples. | Decide in M2 based on selected model adapter and interpretability needs. |
| DG-10 | Which evaluation split preserves class coverage? | Open | Participant-arm-label coverage is complete for all 20 participants. | Build grouped split diagnostics in M2. |
| DG-11 | How are multi-channel kernels attributed? | Open | Requires XROCKET metadata policy. | Decide before M4 explanation figures. |
| DG-12 | Can all mandatory experiments run on available hardware? | Open | No training runtime benchmark yet. | Time prototype runs in M3. |

Resolved in M1:

- DG-02: File inventory and labels validated by `hmc audit`.
- DG-05: Published CSVs are treated as already segmented movement instances.

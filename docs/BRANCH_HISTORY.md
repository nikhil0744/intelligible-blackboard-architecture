# Branch consolidation and recovery tags

On 13 November 2026, the team chose `prototype-live-demo` as the working foundation for `main`. The documentation archive and current-status review were committed as `952144f` before consolidation.

## Active baseline

`main` contains the prototype implementation, committed documentation cleanup, and setup references updated to use `main`. No reasoning, scheduling, storage, recovery, evaluation, frontend, or dependency implementation was changed by consolidation.

The prior `main` history is retained as a merge parent, while the selected file contents come from the prototype. This was an intentional baseline selection, not integration of the benchmark changes previously present on `main`. Archive tags preserve the retired branches independently.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the assessed limitations of this implementation. Consolidating branches does not fix those limitations. S2/S3 improvements remain available for separate review and testing before selective incorporation.

## Preserved branch tips

The following annotated tags are published to `origin`. The date and names distinguish historical snapshots from new releases.

| Retired branch or snapshot | Preserved commit | Archive tag |
| --- | --- | --- |
| Previous GitHub `main` | `ca08049` | `archive/2026-11-13/main` |
| Published `prototype-live-demo` | `5c0036d` | `archive/2026-11-13/prototype-live-demo` |
| `s2-compute-readiness` | `96944c1` | `archive/2026-11-13/s2-compute-readiness` |
| Latest GitHub `s3-compute-readiness` | `79345bb` | `archive/2026-11-13/s3-compute-readiness` |
| `student-1-core` | `2eed5fd` | `archive/2026-11-13/student-1-core` |
| `student-4-ui-benchmarks` | `4fb6928` | `archive/2026-11-13/student-4-ui-benchmarks` |
| Local-only `student-2-agents` | `9b7f900` | `archive/2026-11-13/student-2-agents` |
| Local-only `student-3-credit-assignment-scoring` | `6f81f16` | `archive/2026-11-13/student-3-credit-assignment-scoring` |
| Prototype with documentation cleanup | `952144f` | `archive/2026-11-13/prototype-cleaned` |

Tags retain each tip's complete reachable committed history. Earlier local tips of `main` (`42fc790`), S3 (`ef1b87c`), and the detached historical main worktree (`7ec4e2f`) are also ancestors of preserved snapshots.

## Inspect or recover earlier work

Fetch the historical tags if they are not already available:

```bash
git fetch origin --tags
git tag --list 'archive/2026-11-13/*'
git show archive/2026-11-13/s3-compute-readiness
```

To resume an archived branch in a new local feature branch:

```bash
git switch -c review-s3 archive/2026-11-13/s3-compute-readiness
```

Creating that branch does not merge its code into `main`. Review individual changes and validate their interactions before incorporating them. Because the old `main` commits remain in ancestry while its differing files were intentionally excluded, a normal merge of its archive tag will not restore those files; inspect the original commits and selectively reapply the desired changes.

Old plan text and archived setup instructions may still mention retired branch names. Current Colab commands and the active notebook use `main`.

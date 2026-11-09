# Maintenance

The main branch carries reviewed maintenance changes. Published tags are immutable; never move a paper-associated tag after a fix. Use semantic versions, keep release notes explicit about validation, and retain old resources. Candidate releases remain prereleases until the outstanding reproducibility items are resolved.

## Platform automation

GitHub Actions runs CPU feature/solver tests, compilation, recorded-outcome aggregation, pinned-data reconstruction and public-scope checks on pushes, pull requests and weekly schedules. Dependabot proposes monthly pip and Actions updates. Dependency proposals require review; they do not replace the frozen research environment automatically. No GPU jobs or paid cloud runs are scheduled.

## Agent handoff

In a later authorized maintenance session, read this file, README, the latest release and open issues/PRs. Inspect CI, reproduce reported failures, fix and test the smallest change, and update CHANGELOG. If calculations change, document their effect on scientific conclusions and preserve the previous release. Do not silently alter frozen thresholds, sample locks or seeds.

The publication gate uses `resources/public_files.json`. Before adding a path, inspect its content, provenance, license and necessity; refresh the hash only after review. Run `scripts/verify_release.py`, the quick start and `scripts/rebuild_tables.py`. Original research source hashes are an additional invariant. A change to research logic needs an explicit equivalence test or a versioned results-impact note.

## Private-material rule

Never add manuscripts (including Markdown), submission material, reviewer correspondence, notebooks containing embedded payloads, secrets or source dataset text. Use path rules plus content inspection and the explicit reviewed file list. PDF/TEX formats are not categorically forbidden: a necessary, licensed non-manuscript document requires individual review and allowlisting. Inspect all branches, tags, releases and Actions artifacts if a leak is suspected. Removing a head file does not remove history. Preserve an internal backup and seek author approval before rewriting shared history or deleting a retained release.

## Execution limits

GitHub performs the platform automation above. No recurring Agent automation is configured in this release. Issue triage, scientific decisions, GPU verification, PR review and new releases need a later Agent session or an actively configured external maintainer. No claim of perpetual autonomous monitoring is made. Weekly CI checks do not automatically repair failures or answer issues.

# Security policy

The paper environment pins dependency versions to preserve the original hook, attention and cache behavior. GitHub reports known upstream security advisories affecting the pinned Transformers and Accelerate versions. A formal code release does not imply that these upstream advisories have been fixed.

Use the documented official model revisions and checksum-verified resources in an isolated research environment. Do not load untrusted model files, pickle checkpoints or third-party code into the experimental environment. The repository does not provide a hardened public inference service.

Compatibility and security updates are maintained separately from the immutable paper tag. Dependency upgrades require control/cache equivalence tests and a results-impact review before a new release. Dependabot and repository security alerts remain enabled; open alerts are not dismissed merely to mark a release as stable.

Report repository-specific vulnerabilities through GitHub private vulnerability reporting when available. Do not include credentials, private data or unpublished manuscripts in public issues.

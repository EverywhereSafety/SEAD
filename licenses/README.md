# Third-party notices

Original SEAD contributions are licensed under Apache-2.0; third-party works
retain their own terms. The root license does not supersede these notices.
This inventory distinguishes material included in the repository from software
that users install or provision separately.

## Included and adapted material

| Material in SEAD | Upstream source | Applicable notice |
| --- | --- | --- |
| `data/mtar/` task descriptions, fixtures, evaluators, and adaptations | [MT-AgentRisk dataset](https://huggingface.co/datasets/CHATS-Lab/MT-AgentRisk), revision `e1ba224fea480df3d296a1bf4e28613d4c97c704` | [Dataset terms](../data/mtar/LICENSE); [provenance and changes](../data/mtar/README.md) |
| Attribution to the MT-AgentRisk / ToolShield code project | [CHATS-lab/ToolShield](https://github.com/CHATS-lab/ToolShield) | [MIT, CHATS Lab](ToolShield-MIT.txt); applies to that code, not automatically to the HF dataset |
| `src/sead/benchmarks/oas/evaluator_runtime/common.py` and `scoring.py`; OAS-origin material within MTAR | [OpenAgentSafety](https://github.com/Open-Agent-Safety/OpenAgentSafety), especially `workspaces/openagentsafety_base_image/` | [MIT, TheAgentCompany](OpenAgentSafety-MIT.txt) |
| TheAgentCompany-origin evaluation helpers and upstream fixture lineage | [TheAgentCompany](https://github.com/TheAgentCompany/TheAgentCompany) | [MIT, TheAgentCompany](TheAgentCompany-MIT.txt) |
| Upstream code context in `patches/openhands-*.patch` | [OpenHands](https://github.com/OpenHands/OpenHands) | [MIT, OpenHands contributors](OpenHands-MIT.txt); see [patch provenance](../patches/README.md) |
| P2SQL-origin PostgreSQL task material identified by the MT-AgentRisk dataset card | [P2SQL](https://github.com/rodrigo-pedro/P2SQL) | [MIT, INESC-ID](P2SQL-MIT.txt) |
| SafeArena-origin web task material identified by the MT-AgentRisk dataset card | [SafeArena dataset](https://huggingface.co/datasets/McGill-NLP/safearena) | [Dataset terms](SafeArena-TERMS.txt) |

SEAD's OAS compatibility helpers have been reduced to the deterministic subset
needed by the supported evaluations; they remove runtime dependencies and
LLM-backed grading. Their upstream attribution and MIT notices are retained.
MTAR's per-task modification categories are recorded in `data/mtar/manifest.json`.

The MTAR manifest identifies its immediate source revision but does not provide
an exhaustive mapping to all earlier datasets. The P2SQL and SafeArena entries
preserve the sources identified by the upstream dataset card; they are not a
claim of independently verified, file-by-file authorship. No MCPMark Notion
tasks are included in the 75-task subset.

## Externally installed components

OpenHands itself, the optional OpenAgentSafety checkout, container images,
model weights, Python packages, and npm packages are not vendored here.
The Dockerfiles and package manifests refer to them; those references do not
relicense or redistribute the referenced projects. Their own licenses continue
to apply when they are downloaded, used, or redistributed. The MIT notice for
OpenHands is included because the patch files contain upstream code context.

`pyproject.toml`, `uv.lock`, container dependency manifests, and image references
record the software dependencies. This notice inventory covers the repository
contents, not a separately built container image or every transitive dependency.

## License text provenance

The following upstream license files were retrieved verbatim on 2026-09-27.
Revisions identify the license snapshots checked during this license check; unless
stated otherwise, they do not identify the version used for the original
adaptation or guarantee runtime compatibility.

- **ToolShield** — [upstream license](https://raw.githubusercontent.com/CHATS-lab/ToolShield/95a2e342492f10e1a61e5c3bcb8df1149005560f/LICENSE); revision `95a2e342492f10e1a61e5c3bcb8df1149005560f`; local copy [licenses/ToolShield-MIT.txt](ToolShield-MIT.txt). SHA-256: `59b3b74cf8ca143dc4170597bba3176b20339e8c07d8eef8016d6a3c02bb3c3b`.
- **OpenAgentSafety** — [upstream license](https://raw.githubusercontent.com/Open-Agent-Safety/OpenAgentSafety/2fdd4057c8dae6e776999a55a5766a862e96f174/LICENSE); revision `2fdd4057c8dae6e776999a55a5766a862e96f174`; local copy [licenses/OpenAgentSafety-MIT.txt](OpenAgentSafety-MIT.txt). SHA-256: `2d6976a9b482bb14d9cd4f0d97ce6e14305496f1649b8667cb91d027c670930f`.
- **OpenHands** — [upstream license](https://raw.githubusercontent.com/OpenHands/OpenHands/fd9145958e9e93bfbad3252fce7a69493e61215a/LICENSE); revision `fd9145958e9e93bfbad3252fce7a69493e61215a`; local copy [licenses/OpenHands-MIT.txt](OpenHands-MIT.txt). SHA-256: `e1d1fa9f3a8d7bef24449d488fcd8f00f8f272cac297bb9bed161eb6175b876a`.
- **TheAgentCompany** — [upstream license](https://raw.githubusercontent.com/TheAgentCompany/TheAgentCompany/98b68ef82a47690c316f42fddb05baafaab56851/LICENSE); revision `98b68ef82a47690c316f42fddb05baafaab56851`; local copy [licenses/TheAgentCompany-MIT.txt](TheAgentCompany-MIT.txt). SHA-256: `2d6976a9b482bb14d9cd4f0d97ce6e14305496f1649b8667cb91d027c670930f`.
- **P2SQL** — [upstream license](https://raw.githubusercontent.com/rodrigo-pedro/P2SQL/ff5a0226b6c63407ae19ba5567ae251b51d4f98e/LICENSE.md); revision `ff5a0226b6c63407ae19ba5567ae251b51d4f98e`; local copy [licenses/P2SQL-MIT.txt](P2SQL-MIT.txt). SHA-256: `ba5a5137c3b381e2c2256e2650bedaf1418a66df4f00d0af45d9bd325b2a0533`.

The Apache-2.0 text in [LICENSE](../LICENSE) is copied from the
[Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0.txt).
Dataset terms are transcribed from the public dataset cards linked above.
No additional rights are implied by this inventory. Verification of the retained dataset terms and licensing scope
are recorded in [dataset terms and scope](#dataset-terms-adopted).

## Dataset terms adopted

The adapted MT-AgentRisk data retains the upstream terms, reproduced in
[data/mtar/LICENSE](../data/mtar/LICENSE). SafeArena-derived material retains
[SafeArena's terms](SafeArena-TERMS.txt). The two terms sections were
checked against the official dataset cards word for word on 2026-09-27.
Source revisions and normalized text hashes are recorded in
[dataset-terms-provenance.json](dataset-terms-provenance.json).

Both upstream terms sections require:

1. No use that is unlawful or infringes others' rights.
2. Compliance with any additional terms applicable to third-party source data.
3. Research use to satisfy the applicable jurisdiction's fair-use rules, with
   users responsible for ensuring compliance.
4. Inclusion of these same terms in derivatives.

Sources: [MT-AgentRisk](https://huggingface.co/datasets/CHATS-Lab/MT-AgentRisk#terms-of-use)
and [SafeArena](https://huggingface.co/datasets/McGill-NLP/safearena#terms-of-use).

The 75-task adaptation remains bundled with these original terms and its
provenance. Original SEAD code uses Apache-2.0. The related code project's MIT
license is retained separately and is not applied to the dataset. No additional
usage restrictions or substitute dataset license have been introduced.

Preserving the terms fulfills their pass-through requirement; it does not by
itself determine whether a particular use constitutes fair use or clear the
rights in underlying third-party material. No separate or broader authorization
is asserted by these notices.

## Scope and limitations

The license check covers the tracked source snapshot, its declared provenance, embedded
attributions, included patches, dataset manifest, and the upstream sources linked
in [the inventory above](#included-and-adapted-material). It verifies identifiable
reuse and license notices; it is not proof that every unmarked line or fixture
is original. Upstream runtime versions and the dataset's complete earlier
source-to-file lineage are not fully recorded by this release. Separately
built images and installed dependency trees were not audited as distribution
artifacts. No evaluation or attack workflow was executed for this license check.

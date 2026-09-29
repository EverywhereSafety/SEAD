# Adapted MT-AgentRisk evaluation data

This directory contains the 75-task MTAR subset used by SEAD. The tasks are
adapted from **MT-AgentRisk (Multi-Turn AgentRisk)** by Xu Li, Simon Yu,
Minzhou Pan, Yiyou Sun, Bo Li, Dawn Song, Xue Lin, and Weiyan Shi.

- Dataset: https://huggingface.co/datasets/CHATS-Lab/MT-AgentRisk
- Upstream revision: `e1ba224fea480df3d296a1bf4e28613d4c97c704`
- Related code: https://github.com/CHATS-lab/ToolShield
- Local provenance and per-task changes: [manifest.json](manifest.json)
- Dataset terms: [LICENSE](LICENSE)
- Further upstream notices: [../../licenses/README.md](../../licenses/README.md)

## What was changed

SEAD selects 75 single-turn source tasks: 25 Filesystem, 20 Terminal,
15 PostgreSQL, and 15 Web tasks. The task IDs preserve the source IDs.
The repository reorganizes the assets into per-task directories and supplies
local environment specifications, runtime profiles, and PostgreSQL oracles.
Some tasks also have seed, dependency, evaluator, checkpoint, initialization,
or task-text repairs; the `changes` and `provenance` fields in the manifest
record these differences for each task. Runtime identifiers use `sead`.
Two unresolved `balance_sheet_q1.pdf` placeholders in `single.3` and
`single.334` are omitted. These adaptations are not an unmodified upstream release.

## Licensing scope

The Hugging Face dataset provides **terms of use**, reproduced in [LICENSE](LICENSE),
and does not declare an MIT or Apache-2.0 dataset license. Its related code
repository uses MIT; that code license is retained in
[ToolShield-MIT.txt](../../licenses/ToolShield-MIT.txt) and must not be treated
as blanket permission for the dataset.

The upstream dataset card identifies OpenAgentSafety, SafeArena, and P2SQL as
sources for the tool families retained here. It also identifies MCPMark as a
source for Notion tasks; this subset contains no Notion tasks. Applicable
third-party rights and terms remain in force. SafeArena's dataset terms are
preserved in [SafeArena-TERMS.txt](../../licenses/SafeArena-TERMS.txt), and the
MIT notices for OpenAgentSafety, TheAgentCompany, and P2SQL are in `../../licenses/`.
The source-to-task mapping below the MT-AgentRisk layer is not exhaustively
recorded by this subset, so these acknowledgments do not assert that every
file comes from every listed project.

The root Apache-2.0 license covers original SEAD contributions; it does not
relicense the upstream task descriptions, fixtures, or evaluators. This adaptation
retains the upstream terms unchanged, including their requirement that derivatives
include those terms. See the [verified terms and licensing scope](../../licenses/README.md#dataset-terms-adopted).

## Citation

```bibtex
@misc{li2026unsaferturnsbenchmarkingdefending,
  title = {Unsafer in Many Turns: Benchmarking and Defending Multi-Turn Safety Risks in Tool-Using Agents},
  author = {Xu Li and Simon Yu and Minzhou Pan and Yiyou Sun and Bo Li and Dawn Song and Xue Lin and Weiyan Shi},
  year = {2026},
  eprint = {2602.13379},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR},
  url = {https://arxiv.org/abs/2602.13379}
}
```

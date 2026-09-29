# OpenHands compatibility patches

These patch files contain SEAD changes to OpenHands together with upstream
context lines. OpenHands is an external dependency and is not included here.
The copied upstream portions retain the [OpenHands MIT license](../licenses/OpenHands-MIT.txt)
and copyright notice. Original SEAD additions are covered by the root Apache-2.0
license, subject to the retained upstream notices.

- `openhands-evaluation-runtime.patch` changes event-stream error handling,
  runtime/session handling, and related tests.
- `openhands-gemini-thinking-blocks.patch` carries model response fields through
  message serialization and conversation memory.

The patch headers identify the changed paths and original abbreviated blob IDs.
This release does not pin the full upstream OpenHands commit. No enterprise
source files are present in these patches. See [third-party notices](../licenses/README.md)
for the exact license snapshot retained; a license snapshot is not a runtime
version pin.

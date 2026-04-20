# ComfyUI Prompt-Relay (okkazoo fork)

WORK IN PROGRESS

This is a fork of [kijai/ComfyUI-PromptRelay](https://github.com/kijai/ComfyUI-PromptRelay),
adding a more integrated node UI for defining temporal segments. The underlying
attention-patching and mask-building logic is kijai's work.

## Attribution

- **Original research / method:** Gordon Chen et al. — Prompt Relay: inference-time
  temporal control for video diffusion.
  Paper & project page: https://gordonchen19.github.io/Prompt-Relay/
  Reference implementation: https://github.com/GordonChen19/Prompt-Relay
- **Original ComfyUI implementation:** [kijai/ComfyUI-PromptRelay](https://github.com/kijai/ComfyUI-PromptRelay)
  — attention patching, tokenizer handling, multi-arch (Wan T2V / Wan I2V / LTX)
  support, and the original `PromptRelayEncode` node.
- **This fork:** changes are focused on the node-input surface and workflow
  ergonomics. Credit for the core method and ComfyUI integration belongs to the
  above.

## Upstream

To track upstream changes:

```bash
git remote add upstream https://github.com/kijai/ComfyUI-PromptRelay.git
git fetch upstream
git merge upstream/main
```

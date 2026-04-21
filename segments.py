# Option A scaffold for Prompt Relay — SEGMENT custom type, producer node,
# Autogrow consumer. Stage 1: emits a summary string so we can validate the
# type + autogrow plumbing before layering expansion, sampling, disk save,
# and resume on top.

from comfy_api.latest import io


# Wire-level type. Python value is a dict {"prompt": str, "frames": int}.
# Using io.Custom gives us Segment.Input / Segment.Output with the right
# wire-type identifier so only our Segment pins connect to segment inputs.
Segment = io.Custom("PROMPT_RELAY_SEGMENT")


class PromptRelaySegment(io.ComfyNode):
    """One segment of a Prompt Relay run — prompt + frame count."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelaySegment",
            display_name="Prompt Relay Segment",
            category="conditioning/prompt_relay",
            description=(
                "Defines one segment of a Prompt Relay chain. Wire into a "
                "Prompt Relay Compose node's segment inputs — Autogrow adds "
                "a new slot each time you connect one."
            ),
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="describe this segment",
                ),
                io.Int.Input(
                    "frames",
                    default=30,
                    min=1,
                    max=9999,
                    tooltip="Length of this segment in frames.",
                ),
            ],
            outputs=[
                Segment.Output("segment"),
            ],
        )

    @classmethod
    def execute(cls, prompt: str, frames: int) -> io.NodeOutput:
        return io.NodeOutput({"prompt": prompt, "frames": int(frames)})


class PromptRelayCompose(io.ComfyNode):
    """Stage-1 scaffold of the compose node. Consumes N Segment inputs via
    Autogrow and emits a summary string. Later stages will replace the
    summary output with node-expansion-driven sampler chains + per-segment
    latent save + resume."""

    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(
            Segment.Input("segment"),
            prefix="segment",
            min=1,
            max=100,
        )
        return io.Schema(
            node_id="PromptRelayCompose",
            display_name="Prompt Relay Compose",
            category="conditioning/prompt_relay",
            description=(
                "Composes N Prompt Relay Segment nodes into a single run. "
                "Stage-1 scaffold: outputs a summary string so we can validate "
                "Autogrow of a custom type before adding real sampling."
            ),
            inputs=[
                io.Autogrow.Input("segments", template=template),
            ],
            outputs=[
                io.String.Output("summary"),
            ],
        )

    @classmethod
    def execute(cls, segments) -> io.NodeOutput:
        # Autogrow gives dict[str, SegmentDict] keyed "segment0", "segment1", ...
        # Sort by numeric suffix so segments stay in wired order.
        ordered = sorted(
            segments.items(),
            key=lambda kv: int(kv[0].replace("segment", "")),
        )
        total_frames = sum(v["frames"] for _, v in ordered)
        lines = [
            f"{len(ordered)} segments, {total_frames} frames total",
            "",
        ]
        for i, (_, seg) in enumerate(ordered):
            prompt_text = seg["prompt"]
            snippet = (prompt_text[:70] + "…") if len(prompt_text) > 70 else prompt_text
            lines.append(f"[{i}] {seg['frames']}fr: {snippet}")
        return io.NodeOutput("\n".join(lines))

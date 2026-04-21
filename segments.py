# Option A scaffold for Prompt Relay — SEGMENT custom type, producer node,
# and Autogrow consumer that expands into a chained sampler run.
#
# Stage 2: PromptRelayCompose emits a subgraph of N KSampler nodes chained
# via prev-latent-as-init. Each sampler is its own cached subnode, so
# editing one segment invalidates only that segment and the ones after.

import comfy.samplers
from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder


# Wire-level type. Python value is a dict {"prompt": str, "frames": int}.
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
                    tooltip="Length of this segment in frames (used when the compose "
                            "node produces video; ignored for still-image sampling).",
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
    """Expands into a chain of samplers (one per segment) via node expansion.
    Each segment's output latent becomes the next segment's init. Because
    every sampler is its own cached subnode, editing one segment only
    re-runs that segment and the ones that follow — earlier segments hit
    the ComfyUI cache automatically."""

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
                "Composes N Prompt Relay Segment nodes into a chained "
                "sampler run via node expansion. Segment N's latent becomes "
                "segment N+1's init. Edit one segment — earlier ones hit "
                "cache, later ones re-run automatically."
            ),
            enable_expand=True,
            inputs=[
                # raw_link=True so these arrive as graph links rather than
                # resolved Python objects. Required for expansion caching:
                # an Unhashable input anywhere in a subnode's signature
                # disables caching for that subnode entirely
                # (comfy_execution/caching.py, to_hashable()).
                io.Model.Input("model", raw_link=True),
                io.Clip.Input("clip", raw_link=True),
                io.Vae.Input("vae", raw_link=True),
                io.Conditioning.Input("negative", raw_link=True),
                io.Autogrow.Input("segments", template=template),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("steps", default=20, min=1, max=200),
                io.Float.Input("cfg", default=7.0, min=0.0, max=30.0, step=0.1),
                io.Combo.Input(
                    "sampler_name",
                    options=comfy.samplers.KSampler.SAMPLERS,
                    default="euler",
                ),
                io.Combo.Input(
                    "scheduler",
                    options=comfy.samplers.KSampler.SCHEDULERS,
                    default="normal",
                ),
                io.Float.Input(
                    "denoise",
                    default=0.75,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Applied to segments after the first. Segment 0 is "
                            "always denoise=1.0 (fresh from empty latent). Lower "
                            "values preserve more of the prior segment's structure.",
                ),
                io.Int.Input("width", default=1024, min=64, max=8192, step=8),
                io.Int.Input("height", default=1024, min=64, max=8192, step=8),
                io.String.Input(
                    "output_dir",
                    default="relay_runs/run_01",
                    tooltip="Subdir under ComfyUI's output folder. Per-segment "
                            ".latent files saved here if save_each_latent is on.",
                ),
                io.Boolean.Input("save_each_latent", default=True),
            ],
            outputs=[
                io.Latent.Output("final_latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        vae,
        negative,
        segments,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        width,
        height,
        output_dir,
        save_each_latent,
    ) -> io.NodeOutput:
        # Order segments by wired index so output matches user-wired order.
        ordered = sorted(
            segments.items(),
            key=lambda kv: int(kv[0].replace("segment", "")),
        )

        g = GraphBuilder()

        # Segment 0 starts from an empty latent at the requested resolution.
        empty = g.node(
            "EmptyLatentImage",
            id="empty_latent",
            width=int(width),
            height=int(height),
            batch_size=1,
        )
        prev_latent = empty.out(0)

        last_latent = None
        for i, (_, seg) in enumerate(ordered):
            encode = g.node(
                "CLIPTextEncode",
                id=f"enc_{i:03d}",
                clip=clip,
                text=seg["prompt"],
            )

            # First segment denoises fully from empty. Subsequent segments use
            # the user's denoise so prev-segment structure carries through.
            seg_denoise = 1.0 if i == 0 else float(denoise)

            sampler = g.node(
                "KSampler",
                id=f"smp_{i:03d}",
                model=model,
                positive=encode.out(0),
                negative=negative,
                latent_image=prev_latent,
                seed=int(seed) + i,
                steps=int(steps),
                cfg=float(cfg),
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=seg_denoise,
            )

            if save_each_latent:
                g.node(
                    "SaveLatent",
                    id=f"save_{i:03d}",
                    samples=sampler.out(0),
                    filename_prefix=f"{output_dir}/seg_{i:03d}",
                )

            prev_latent = sampler.out(0)
            last_latent = sampler.out(0)

        return io.NodeOutput(last_latent, expand=g.finalize())

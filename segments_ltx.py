# Tier B — LTX-2.3 video compose.
#
# Mirrors PromptRelayCompose's expansion-caching architecture but emits an
# LTX-native subgraph. Each segment is its own SamplerCustom subnode with
# an init latent derived from the prior segment's output via a noise-mask
# ramp — no VAE round-trip at segment boundaries.

import torch

import comfy.samplers
import comfy.nested_tensor
from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder

from .segments import Segment


# --- Utility node: build seg N's init latent from seg N-1's output. ---
#
# Takes the prior segment's sampled AV latent and a fresh empty AV latent of
# the new segment's length, splices the prior's trailing frames into the head
# of the new latent, and attaches a noise_mask that ramps from 0 (fully keep
# prior content) to 1 (sample freely) across the overlap region. Downstream
# SamplerCustom reads noise_mask and only samples where it's > 0.

class LTXVLatentChainInit(io.ComfyNode):
    """Build an LTX init latent that continues from a previously-sampled AV
    latent, blended via a noise-mask ramp over the overlap region."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVLatentChainInit",
            display_name="LTXV Latent Chain Init",
            category="latent/video/ltxv",
            description=(
                "Splice the tail of a prior-segment AV latent into the head "
                "of a fresh empty AV latent and attach a noise_mask that "
                "ramps 0→1 across the overlap. Used to chain LTX segments "
                "without VAE round-trips."
            ),
            inputs=[
                io.Latent.Input(
                    "prev_av_latent",
                    tooltip="Sampled AV latent from the previous segment.",
                ),
                io.Latent.Input(
                    "empty_av_latent",
                    tooltip="Fresh empty AV latent for the new segment "
                            "(EmptyLTXVLatentVideo + LTXVEmptyLatentAudio → "
                            "LTXVConcatAVLatent).",
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=8,
                    min=0,
                    max=9999,
                    tooltip="Overlap length in pixel frames. 0 = hard cut "
                            "(no blend). LTX compresses video 8× temporally, "
                            "so 8 pixel frames ≈ 1 latent frame of blend.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="av_latent"),
            ],
        )

    @classmethod
    def execute(cls, prev_av_latent, empty_av_latent, overlap_frames) -> io.NodeOutput:
        prev_video, prev_audio = prev_av_latent["samples"].unbind()
        empty_video, empty_audio = empty_av_latent["samples"].unbind()

        # Convert pixel-frame overlap to latent frames for video. LTX temporal
        # compression is 8×; the empty latent's temporal dim is already
        # ((pixel - 1) // 8) + 1.
        overlap_v_latent = 0 if overlap_frames <= 0 else max(1, overlap_frames // 8)
        overlap_v_latent = min(
            overlap_v_latent,
            prev_video.shape[2],
            empty_video.shape[2],
        )

        # Audio compression ratio isn't fixed in our assumptions — derive it
        # proportionally from the prior segment's video:audio latent-dim ratio.
        if prev_video.shape[2] > 0:
            overlap_a_latent = int(round(overlap_v_latent * prev_audio.shape[2] / prev_video.shape[2]))
        else:
            overlap_a_latent = 0
        overlap_a_latent = min(overlap_a_latent, prev_audio.shape[2], empty_audio.shape[2])

        new_video = empty_video.clone()
        new_audio = empty_audio.clone()

        # Build compact noise masks: [batch, 1, latent_frames, 1, 1] for video,
        # [batch, 1, latent_frames, 1, 1]-shaped for audio too (broadcasts over
        # the trailing spatial dims). Match the pattern in LTXVImgToVideoInplace.
        video_mask = torch.ones(
            (new_video.shape[0], 1, new_video.shape[2], 1, 1),
            dtype=torch.float32,
            device=new_video.device,
        )
        audio_mask = torch.ones(
            (new_audio.shape[0],) + (1,) * (new_audio.ndim - 1),
            dtype=torch.float32,
            device=new_audio.device,
        )
        # audio mask shape: matches the audio sample tensor rank, ones everywhere
        # initially; we'll overwrite the head with the ramp if overlap_a_latent > 0.

        if overlap_v_latent > 0:
            new_video[:, :, :overlap_v_latent] = prev_video[:, :, -overlap_v_latent:]
            # Ramp 0 → 1 across the overlap frames, stays 1 after.
            ramp_v = torch.linspace(
                0.0, 1.0, overlap_v_latent,
                device=new_video.device, dtype=torch.float32,
            ).view(1, 1, -1, 1, 1)
            video_mask[:, :, :overlap_v_latent] = ramp_v

        if overlap_a_latent > 0:
            new_audio[:, :, :overlap_a_latent] = prev_audio[:, :, -overlap_a_latent:]
            # Build a ramp view compatible with audio's rank. Audio tensor rank
            # varies by LTX internals; broadcast along all trailing dims.
            ramp_a_1d = torch.linspace(
                0.0, 1.0, overlap_a_latent,
                device=new_audio.device, dtype=torch.float32,
            )
            # Reshape ramp to [1, 1, overlap_a_latent, 1, 1, ...] matching mask rank
            ramp_shape = [1, 1, overlap_a_latent] + [1] * (audio_mask.ndim - 3)
            ramp_a = ramp_a_1d.view(*ramp_shape)
            audio_mask[:, :, :overlap_a_latent] = ramp_a

        out = {}
        out.update(empty_av_latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((new_video, new_audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        return io.NodeOutput(out)


# --- Compose: expand into a chain of LTX samplers. ---

class PromptRelayComposeLTX(io.ComfyNode):
    """LTX-2.3 version of Prompt Relay Compose. Expands into N SamplerCustom
    subnodes, one per wired PromptRelaySegment, chained via noise-mask init."""

    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(
            Segment.Input("segment"),
            prefix="segment",
            min=1,
            max=100,
        )
        return io.Schema(
            node_id="PromptRelayComposeLTX",
            display_name="Prompt Relay Compose (LTX)",
            category="conditioning/prompt_relay",
            description=(
                "Composes N Prompt Relay Segment nodes into a chained LTX "
                "sampling run via node expansion. Each segment is a "
                "SamplerCustom subnode; segments past 0 use a noise-mask "
                "ramp over the per-segment blend_in_frames to blend with "
                "the prior segment's tail. Output is the final AV latent — "
                "wire to LTXVSeparateAVLatent → VAEDecode."
            ),
            enable_expand=True,
            inputs=[
                # raw_link required on object inputs for expansion caching —
                # see feedback_v3_expansion_caching.md.
                io.Model.Input("model", raw_link=True),
                io.Clip.Input("clip", raw_link=True),
                io.Vae.Input("video_vae", raw_link=True),
                io.Vae.Input("audio_vae", raw_link=True),
                io.Conditioning.Input("negative", raw_link=True),
                io.Autogrow.Input("segments", template=template),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("steps", default=20, min=1, max=200),
                io.Float.Input("cfg", default=2.5, min=0.0, max=30.0, step=0.1,
                               tooltip="LTX distilled typically uses ~2.5. Full model ~6-7."),
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
                io.Int.Input("width", default=704, min=64, max=8192, step=32),
                io.Int.Input("height", default=480, min=64, max=8192, step=32),
                io.Float.Input("frame_rate", default=25.0, min=1.0, max=120.0, step=0.1),
            ],
            outputs=[
                io.Latent.Output(display_name="final_av_latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        video_vae,
        audio_vae,
        negative,
        segments,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        width,
        height,
        frame_rate,
    ) -> io.NodeOutput:
        ordered = sorted(
            segments.items(),
            key=lambda kv: int(kv[0].replace("segment", "")),
        )

        g = GraphBuilder()

        # Shared sampler + sigma schedule, emitted once.
        sampler_sel = g.node("KSamplerSelect", id="sampler_sel", sampler_name=sampler_name)
        sigmas = g.node(
            "BasicScheduler",
            id="sigmas",
            model=model,
            scheduler=scheduler,
            steps=int(steps),
            denoise=1.0,
        )

        prev_sampled = None
        last_output = None

        for i, (_, seg) in enumerate(ordered):
            seg_frames = int(seg["frames"])

            enc = g.node(
                "CLIPTextEncode",
                id=f"enc_{i:03d}",
                clip=clip,
                text=seg["prompt"],
            )
            cond = g.node(
                "LTXVConditioning",
                id=f"cond_{i:03d}",
                positive=enc.out(0),
                negative=negative,
                frame_rate=float(frame_rate),
            )

            empty_vid = g.node(
                "EmptyLTXVLatentVideo",
                id=f"empty_vid_{i:03d}",
                width=int(width),
                height=int(height),
                length=seg_frames,
                batch_size=1,
            )
            empty_aud = g.node(
                "LTXVEmptyLatentAudio",
                id=f"empty_aud_{i:03d}",
                frames_number=seg_frames,
                frame_rate=int(round(frame_rate)),
                batch_size=1,
                audio_vae=audio_vae,
            )
            empty_av = g.node(
                "LTXVConcatAVLatent",
                id=f"empty_av_{i:03d}",
                video_latent=empty_vid.out(0),
                audio_latent=empty_aud.out(0),
            )

            if i == 0 or prev_sampled is None:
                init_av = empty_av.out(0)
            else:
                chain = g.node(
                    "LTXVLatentChainInit",
                    id=f"chain_{i:03d}",
                    prev_av_latent=prev_sampled,
                    empty_av_latent=empty_av.out(0),
                    overlap_frames=int(seg.get("blend_in_frames", 0)),
                )
                init_av = chain.out(0)

            sampler_node = g.node(
                "SamplerCustom",
                id=f"smp_{i:03d}",
                model=model,
                add_noise=True,
                noise_seed=int(seed) + i,
                cfg=float(cfg),
                positive=cond.out(0),
                negative=cond.out(1),
                sampler=sampler_sel.out(0),
                sigmas=sigmas.out(0),
                latent_image=init_av,
            )

            prev_sampled = sampler_node.out(0)
            last_output = sampler_node.out(0)

        return io.NodeOutput(last_output, expand=g.finalize())

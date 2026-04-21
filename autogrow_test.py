# Throwaway V3 Autogrow proof-of-concept. Delete once confirmed.
#
# Tests three things in one shot:
#   1. V3 schema loads on the target ComfyUI version.
#   2. io.Autogrow.TemplatePrefix wrapping an inline-widget input
#      (multiline String) renders as growable text rows, not wired pins.
#   3. The promptrelay-dev card + push-and-restart.sh iteration loop
#      actually delivers code changes onto the instance.
#
# On success: the node "Prompt Relay Autogrow Test" shows one "segment0"
# multiline text box with a [+] to add segment1, segment2, ... Output is
# the joined string of all filled segments, separated by " | ".

from comfy_api.latest import io


class PromptRelayAutogrowTest(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(
            io.String.Input("segment", multiline=True, default="describe this segment"),
            prefix="segment",
            min=1,
            max=20,
        )
        return io.Schema(
            node_id="PromptRelayAutogrowTest",
            display_name="Prompt Relay Autogrow Test",
            category="testing/promptrelay",
            inputs=[
                io.Autogrow.Input("segments", template=template),
            ],
            outputs=[
                io.String.Output("joined"),
            ],
        )

    @classmethod
    def execute(cls, segments) -> io.NodeOutput:
        # segments is dict[str, str] keyed "segment0", "segment1", ...
        ordered = sorted(
            segments.items(),
            key=lambda kv: int(kv[0].replace("segment", "")),
        )
        joined = " | ".join(v for _, v in ordered)
        return io.NodeOutput(joined)

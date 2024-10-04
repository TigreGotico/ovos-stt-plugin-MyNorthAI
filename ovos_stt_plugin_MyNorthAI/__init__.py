from ovos_plugin_manager.templates.stt import STT
from ovos_stt_plugin_whisper import WhisperSTT
from ovos_utils.log import LOG


class MyNorthAISTT(STT):
    MODELS = ["my-north-ai/whisper-small-pt",
              "my-north-ai/whisper-medium-pt",
              "my-north-ai/whisper-large-v3-pt"]

    def __init__(self, config=None):
        super().__init__(config)
        model_id = self.config.get("model") or "my-north-ai/whisper-small-pt"
        if model_id == "small":
            model_id = "my-north-ai/whisper-small-pt"
        elif model_id == "medium":
            model_id = "my-north-ai/whisper-medium-pt"
        elif model_id == "large" or model_id == "large-v3":
            model_id = "my-north-ai/whisper-large-v3-pt"
        self.config["model"] = model_id
        self.config["lang"] = "pt"
        self.config["ignore_warnings"] = True
        valid_model = model_id in self.MODELS
        if not valid_model:
            LOG.info(f"{model_id} is not default model_id ({self.MODELS}), "
                     f"assuming huggingface repo_id or path to local model")
        self.stt = WhisperSTT(self.config)

    def execute(self, audio, language=None):
        return self.stt.execute(audio, language)

    @property
    def available_languages(self) -> set:
        return {"pt"}

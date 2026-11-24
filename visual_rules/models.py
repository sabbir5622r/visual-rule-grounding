MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-VL-3B-Instruct",
    "smolvlm": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
}
MODEL_REVISIONS = {
    "qwen": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "smolvlm": "be92141c8d0ceb98596a923c308ac2bbb4a7c466",
}


def prepare_inputs(processor, name, visual_tokens, prompt, image_path):
    from PIL import Image

    content = [{"type": "image"}] if image_path is not None else []
    content.append({"type": "text", "text": prompt})
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
    )
    processor_settings = {"text": [rendered], "return_tensors": "pt"}
    if image_path is not None:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if name == "smolvlm":
            edge = round((visual_tokens * 28 * 28) ** 0.5)
            image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
            processor_settings["images"] = [[image]]
            processor.image_processor.do_image_splitting = False
        else:
            processor_settings["images"] = [image]
    return processor(**processor_settings)


class VisionLanguageModel:
    def __init__(self, name, revision, visual_tokens):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("Enable a CUDA GPU, or use the oracle backend for a CPU workflow check.")
        model_id = MODEL_IDS[name]
        processor_settings = {}
        if name == "qwen":
            processor_settings = {"min_pixels": 128 * 28 * 28, "max_pixels": visual_tokens * 28 * 28, "use_fast": True}
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, **processor_settings)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, dtype=torch.float16, device_map={"": 0},
            attn_implementation="sdpa",
        ).eval()
        self.name = name
        self.visual_tokens = visual_tokens
        self.torch = torch
        self.device = torch.device("cuda:0")

    def respond(self, prompt, image_path, output_tokens):
        inputs = prepare_inputs(
            self.processor, self.name, self.visual_tokens, prompt, image_path
        ).to(self.device)
        with self.torch.inference_mode():
            result = self.model.generate(**inputs, max_new_tokens=output_tokens, do_sample=False)
        continuation = result[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()

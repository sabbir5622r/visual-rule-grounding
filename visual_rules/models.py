MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-VL-3B-Instruct",
    "gemma": "google/gemma-3-4b-it",
    "smolvlm": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "centurio": "WueNLP/centurio_aya",
}
MODEL_REVISIONS = {
    "qwen": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "gemma": "093f9f388b31de276ce2de164bdc2081324b9767",
    "smolvlm": "be92141c8d0ceb98596a923c308ac2bbb4a7c466",
    "centurio": "bc31c23e472c846c3f686e6ea3cd0ad525e22684",
}


def prepare_inputs(processor, name, visual_tokens, prompt, image_path):
    from PIL import Image

    content = [{"type": "image"}] if image_path is not None else []
    content.append({"type": "text", "text": prompt})
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
    )
    processor_settings = {"text": [rendered], "return_tensors": "pt"}
    if name == "gemma":
        processor_settings["add_special_tokens"] = False
    if image_path is not None:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if name == "smolvlm":
            edge = round((visual_tokens * 28 * 28) ** 0.5)
            image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
            processor_settings["images"] = [[image]]
            processor.image_processor.do_image_splitting = False
        elif name == "gemma":
            processor_settings["images"] = [[image]]
            processor_settings["do_pan_and_scan"] = False
        else:
            processor_settings["images"] = [image]
    return processor(**processor_settings)


class VisionLanguageModel:
    def __init__(self, name, revision, visual_tokens):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("Enable a CUDA GPU, or use the oracle backend for a CPU workflow check.")
        if name == "gemma" and visual_tokens != 256:
            raise ValueError("Gemma 3 uses 256 visual tokens per image. Pass --visual-tokens 256.")
        model_id = MODEL_IDS[name]
        processor_settings = {}
        if name == "qwen":
            processor_settings = {"min_pixels": 128 * 28 * 28, "max_pixels": visual_tokens * 28 * 28, "use_fast": True}
        self.gpu_indices = (0,)
        self.precision = "float16"
        load_settings = {"dtype": torch.float16, "device_map": {"": 0}}
        if name == "gemma":
            if torch.cuda.device_count() < 2:
                raise RuntimeError("Select Kaggle's GPU T4 x2 accelerator to run Gemma in FP32.")
            self.gpu_indices = (0, 1)
            self.precision = "float32"
            load_settings = {
                "dtype": torch.float32,
                "device_map": "balanced",
                "max_memory": {index: "13GiB" for index in self.gpu_indices},
            }
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, **processor_settings)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, attn_implementation="sdpa", **load_settings
        ).eval()
        if name == "gemma":
            mapped_devices = {str(device).removeprefix("cuda:") for device in self.model.hf_device_map.values()}
            if mapped_devices != {"0", "1"}:
                raise RuntimeError(f"Gemma FP32 must be placed on both GPUs. Actual placement: {sorted(mapped_devices)}")
        self.name = name
        self.visual_tokens = visual_tokens
        self.torch = torch
        self.device = self.model.device

    def respond(self, prompt, image_path, output_tokens):
        inputs = prepare_inputs(
            self.processor, self.name, self.visual_tokens, prompt, image_path
        ).to(self.device)
        with self.torch.inference_mode():
            result = self.model.generate(**inputs, max_new_tokens=output_tokens, do_sample=False)
        continuation = result[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()


class CenturioModel:
    def __init__(self, name, revision, visual_tokens):
        import json
        from pathlib import Path

        import bitsandbytes as bnb
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("Centurio Aya requires a CUDA GPU.")

        self.name = name
        self.visual_tokens = visual_tokens
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.gpu_indices = (0,)
        self.precision = "4-bit NF4 language linear layers, FP16 compute"
        model_id = MODEL_IDS[name]
        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        )
        template_path = hf_hub_download(model_id, "chat_template.json", revision=revision)
        self.template = json.loads(
            Path(template_path).read_text(encoding="utf-8")
        )["chat_template"]["default"]
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"],
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            code_revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            device_map={"": 0},
            quantization_config=quantization,
        ).eval()

        quantized_layers = sum(
            isinstance(layer, bnb.nn.Linear4bit)
            for layer in self.model.language_model.modules()
        )
        if not quantized_layers:
            raise RuntimeError("Centurio loaded without 4-bit language weights.")
        print(f"Centurio 4-bit language layers: {quantized_layers}", flush=True)

    def prepare(self, prompt, image_path):
        from PIL import Image

        tokenizer = self.processor.tokenizer
        content = prompt if image_path is None else "<image_placeholder>\n" + prompt
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            chat_template=self.template,
            add_generation_prompt=True,
            tokenize=False,
        )
        if image_path is None:
            inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
        else:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            inputs = self.processor(
                text=[text], images=[image], add_special_tokens=False, return_tensors="pt"
            )
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.torch.float16)
        return inputs.to(self.device)

    def respond(self, prompt, image_path, output_tokens):
        inputs = self.prepare(prompt, image_path)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=output_tokens, do_sample=False, use_cache=False
            )
        continuation = output[:, inputs["input_ids"].shape[-1]:]
        return self.processor.tokenizer.batch_decode(
            continuation, skip_special_tokens=True
        )[0].strip()


def load_vision_language_model(name, revision, visual_tokens):
    model_class = CenturioModel if name == "centurio" else VisionLanguageModel
    return model_class(name, revision, visual_tokens)

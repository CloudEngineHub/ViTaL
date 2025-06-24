import os
import zmq
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import io


class MolmoZMQServer:
    def __init__(self, host="*", port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self.port = port

        self.processor = AutoProcessor.from_pretrained(
            "allenai/Molmo-7B-D-0924",
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "allenai/Molmo-7B-D-0924",
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )

    def process_request(self, text_prompt, image_data):
        """Process image and text with VLM"""
        image = Image.open(io.BytesIO(image_data))

        inputs = self.processor.process(
            images=[image],
            text=text_prompt,
        )

        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer,
        )
        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        return generated_text

    def start(self):
        print(f"ZMQ Molmo Server listening on port {self.port}")
        while True:
            msg_parts = self.socket.recv_multipart()
            text = msg_parts[0].decode()
            image_data = msg_parts[1]

            response = self.process_request(text, image_data)
            self.socket.send_string(response)


if __name__ == "__main__":
    server = MolmoZMQServer(port=5559)
    server.start()

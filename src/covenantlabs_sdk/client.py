from dataclasses import asdict
import os
from typing import List
import requests
from tqdm import tqdm

from transformers import AutoTokenizer

from .dataclasses import (
    Deployment,
    DeploymentStatus,
    Model,
    SecureInferenceRequest,
    SecureInferenceResponse,
)
from .encryption import permute_bf16_inplace, encrypt_prompt, decrypt_prompt

COVENANT_URL = os.getenv("COVENANT_URL", "https://platfrom.covenantlabs.ai")


class CovenantClient:
    def __init__(self, deployment_key: str, secret_key: str) -> None:
        self.deployment_key = deployment_key
        self.secret_key = secret_key
        self.headers = {"Authorization": f"Bearer {self.deployment_key}"}
        self.model: Model | None = None

        if not self.secret_key:
            raise ValueError("Missing env SECRET_ENCRYPTION_KEY")

        if not self.deployment_key:
            raise ValueError("Missing env COVENANT_DEPLOYMENT_KEY")

    def encrypt_model(self):
        response_data = self._get_deployment()

        self.model = response_data.model

        if response_data.status == DeploymentStatus.ENCRYPTION_PENDING.value:
            print("Downloading and encrypting your model weights... please wait")
            for layer in response_data.model.encrypted_layers:
                self._download_model_file(layer)
                print(f"Encrypting: {layer}")
                self._encrypt_layer(layer)
                self._upload_model_file(layer)

    def deploy(self) -> None:
        requests.post(
            f"{COVENANT_URL}/api/deployments/deploy",
            headers=self.headers,
        )

    def secure_inference(self, messages: List[dict]) -> str:
        if not self.model:
            response_data = self._get_deployment()
            self.model = response_data.model

        tokenizer = AutoTokenizer.from_pretrained(self.model.hugging_face_url)

        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        tokens, _ = encrypt_prompt(tokens, tokenizer, key=self.secret_key.encode())

        response = self._send_secure_infrence_request(SecureInferenceRequest(tokens))

        return decrypt_prompt(
            response.generated_tokens,
            tokenizer,
            key=self.secret_key.encode(),
        )

    def _get_deployment(self) -> Deployment:
        response = requests.get(f"{COVENANT_URL}/api/deployments", headers=self.headers)
        response_data = response.json()

        if response_data.get("error"):
            raise RuntimeError(response_data["error"])

        model_data = response_data.pop("model")
        return Deployment(**response_data, model=Model(**model_data))

    def _send_secure_infrence_request(
        self, payload: SecureInferenceRequest
    ) -> SecureInferenceResponse:
        response = requests.post(
            COVENANT_URL + "/secure/generate",
            json=asdict(payload),
            headers=self.headers,
        )

        if response.status_code not in (200, 201):
            raise RuntimeError(
                "Failed to preform secure inference, please make sure your deployment is: DEPLOYED"
            )

        secure_inference_response = response.json()
        return SecureInferenceResponse(**secure_inference_response)

    def _download_model_file(self, layer) -> None:
        response = requests.get(
            f"{COVENANT_URL}/api/deployments/{layer}/download",
            headers=self.headers,
        )

        if response.status_code == 200:
            signed_url = response.json().get("url")
            if signed_url:
                file_response = requests.get(signed_url, stream=True)
                total_size = int(file_response.headers.get("content-length", 0))
                block_size = 8192

                with open(f"{layer}", "wb") as f, tqdm(
                    total=total_size, unit="B", unit_scale=True, desc=f"{layer}"
                ) as pbar:
                    for chunk in file_response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                print("Download complete.")
            else:
                print("No signed URL returned.")
        else:
            print(f"Failed to fetch signed URL: {response.status_code}")
            print(response.text)

    def _upload_model_file(self, layer_path: str) -> None:
        response = requests.post(
            f"{COVENANT_URL}/api/deployments/upload",
            json={"object_key": layer_path},
            headers=self.headers,
        )
        if response.status_code != 200:
            print(f"Failed to get signed URL: {response.status_code}")
            print(response.text)
            return

        signed_url = response.json().get("url")
        if not signed_url:
            print("No signed URL returned.")
            return

        file_path = layer_path
        file_size = os.path.getsize(file_path)

        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(file_size),
        }

        with open(file_path, "rb") as f, tqdm.wrapattr(
            f,
            "read",
            total=file_size,
            desc=f"Uploading {os.path.basename(file_path)}",
            unit="MB",
        ) as fread:  # TODO some bug not correct mb.. fix later
            put_response = requests.put(
                signed_url, data=fread, headers=headers, timeout=300
            )

            if put_response.status_code in (200, 201, 204):
                print("Upload complete.")
            else:
                print(f"Upload failed: {put_response.status_code}")
                print(put_response.text)

    def _encrypt_layer(self, layer) -> None:
        permute_bf16_inplace(
            layer,
            self.model.num_embeddings,
            self.model.hidden_size,
            self.model.vocab_size,
            self.secret_key.encode(),
        )

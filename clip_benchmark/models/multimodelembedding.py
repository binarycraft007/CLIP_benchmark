import torch
import numpy as np
import PIL
import io
from typing import Tuple
from vertexai.vision_models import Image, MultiModalEmbeddingModel

class Model:
    def __init__(self, pretrained: str) -> None:
        self._client = MultiModalEmbeddingModel.from_pretrained(pretrained)

    def eval(self) -> None:
        return

    def encode(self, images, texts, indexes) -> Tuple[torch.Tensor, torch.Tensor]:
        index = 0
        text_embeddings = []
        image_embeddings = [None] * len(images)
        for text0 in texts:
            for text in text0:
                embedding = self._client.get_embeddings(
                    image=self._pil_image_to_image(images[indexes[index]]),
                    contextual_text=text,
                    dimension=1408,
                )
                if image_embeddings[indexes[index]] is None:
                    image_embeddings[indexes[index]] = embedding.image_embedding
                text_embeddings.append(embedding.text_embedding)
                index += 1

        return torch.tensor(image_embeddings), torch.tensor(text_embeddings)

    def _pil_image_to_image(self, image: PIL.Image) -> Image:
        # Create a BytesIO object
        byte_io = io.BytesIO()
        
        # Save the image to the BytesIO object in a specific format (e.g., PNG or JPEG)
        image.save(byte_io, format='JPEG')  # Change format as needed
        
        # Get the byte data
        image_bytes = byte_io.getvalue()
        
        # Optionally, close the BytesIO object
        byte_io.close()
        return Image(image_bytes)

def load_multimodelembedding(
    model_name: str = "multimodalembedding",
    pretrained: str = "multimodalembedding@001",
    cache_dir: str = None,
    device="cpu",
):
    model = Model(pretrained) 
    return model, None, None

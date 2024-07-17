import torch
import numpy as np
import PIL
import io
from vertexai.vision_models import Image, MultiModalEmbeddingModel

class Model:
    def __init__(self, pretrained: str) -> None:
        self._client = MultiModalEmbeddingModel.from_pretrained(pretrained)

    def eval(self) -> None:
        return

    def encode_image(self, images) -> torch.Tensor:
        embeddings = []
        for image in images:
            embedding = self._client.get_embeddings(
                image=self._pil_image_to_image(image),
                dimension=1408,
            )
            embeddings.extend(embedding.image_embedding)
        return torch.tensor(embeddings)

    def encode_text(self, texts) -> torch.Tensor:
        embeddings = []
        for text0 in texts:
            for text in text0:
                embedding = self._client.get_embeddings(
                    contextual_text=text,
                    dimension=1408,
                )
                embeddings.extend(embedding.text_embedding)
        return torch.tensor(embeddings)

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

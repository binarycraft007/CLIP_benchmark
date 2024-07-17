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
        embedding = []
        for image in images:
            embedding = self._client.get_embeddings(
                image=_pil_image_to_image(image),
                dimension=1408,
            )
            embeddings.extend(embedding.image_embedding)
        return torch.tensor(embeddings)

    def encode_text(self, texts) -> torch.Tensor:
        embedding = []
        for text in texts:
            embedding = self._client.get_embeddings(
                contextual_text=text,
                dimension=1408,
            )
            embeddings.extend(embedding.text_embedding)
        return torch.tensor(embeddings)

    def _pil_image_to_image(image: PIL.Image) -> Image:
        # Create a BytesIO object
        byte_io = io.BytesIO()
        
        # Save the image to the BytesIO object in a specific format (e.g., PNG or JPEG)
        image.save(byte_io, format='JPEG')  # Change format as needed
        
        # Get the byte data
        image_bytes = byte_io.getvalue()
        
        # Optionally, close the BytesIO object
        byte_io.close()
        return Image(image_bytes)

    def _tensor_to_model_image(image_tensor: torch.Tensor) -> Image:
        # Ensure the tensor is on the CPU
        if image_tensor.is_cuda:
            image_tensor = image_tensor.cpu()
        
        # Remove the batch dimension if it exists
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
        
        # Convert tensor to NumPy array
        image_array = image_tensor.detach().numpy()
        
        # Convert CHW format to HWC format (Channels, Height, Width -> Height, Width, Channels)
        if image_array.shape[0] == 3:  # Assuming the tensor is in CHW format
            image_array = np.transpose(image_array, (1, 2, 0))
        
        # Rescale the values to [0, 255] and convert to uint8
        image_array = (image_array * 255).astype(np.uint8)
        
        # Convert the NumPy array to a PIL Image
        image = PIL.Image.fromarray(image_array)
        
        # Save the image to a buffer in memory
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        
        # Get the raw bytes from the buffer
        image_bytes = buffer.getvalue()
        
        return Image(image_bytes)

def load_multimodelembedding(model_name: str = "multimodalembedding", pretrained: str = "multimodalembedding@001", cache_dir: str = None, device="cpu"):
    model = Model(pretrained) 
    return model, None, None

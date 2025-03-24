import os
import requests
import base64

from PIL import Image
from io import BytesIO

class VisionManager:
    def __init__(self):
        """Initialize the VisionManager with Stable Diffusion configuration."""
        self.sd_url = os.environ.get('SD_WEBUI_URL', 'http://127.0.0.1:7860')
        self.auth_user = os.environ.get('SD_AUTH_USER')
        self.auth_pass = os.environ.get('SD_AUTH_PASS')

    async def generate_and_save_image(self, prompt: str, output_path: str) -> str:
        """Generate an image using Stable Diffusion API and save it to the specified path."""
        headers = {'Content-Type': 'application/json'}
        auth = None
        if self.auth_user and self.auth_pass:
            auth = (self.auth_user, self.auth_pass)

        payload = {
            "prompt": prompt,
            "negative_prompt": "",
            "steps": 4,
            "width": 1024,
            "height": 1024,
            "cfg_scale": 1,
            "sampler_name": "Euler",
            "seed": -1,
            "n_iter": 1,
            "scheduler": "Simple"
        }

        try:
            # Generate the image
            response = requests.post(
                f"{self.sd_url}/sdapi/v1/txt2img",
                headers=headers,
                auth=auth,
                json=payload,
                timeout=3600
            )
            response.raise_for_status()
            
            if not response.json().get('images'):
                raise ValueError("No images generated")
            
            # Get the first image
            image_data = response.json()['images'][0]
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Convert base64 to image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))

            # Ensure the save directory exists
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            except OSError as e:
                raise ValueError(f"Failed to create directory for image: {str(e)}")

            # Save the image
            image.save(output_path)
            
        except requests.RequestException as e:
            raise ValueError(f"Failed to generate image: {str(e)}")
        except (IOError, OSError) as e:
            raise ValueError(f"Failed to save image to {output_path}: {str(e)}")

        return output_path
# See More Details: Efficient Image Super-Resolution by Experts Mining

This project wraps the code and models from [seemoredetails](https://github.com/eduardzamfir/seemoredetails) to make it easier to use.

## Installation

```bash
pip install seemore
```

## Quick Start

```python
from seemore import SeemoReUpscaler

# Initialize the upscaler
upscaler = SeemoReUpscaler("seemore_b_x4", device="cpu")

# Load and upscale an image
import cv2
image = cv2.imread("input.jpg")
result = upscaler(image)
```

## Available Models

The following models are available:

-   `seemore_b_x2` - Base model, 2x upscaling
-   `seemore_b_x3` - Base model, 3x upscaling
-   `seemore_b_x4` - Base model, 4x upscaling
-   `seemore_t_x2` - Tiny model, 2x upscaling
-   `seemore_t_x3` - Tiny model, 3x upscaling
-   `seemore_t_x4` - Tiny model, 4x upscaling

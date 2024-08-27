"""# SAM: Inference Playground"""

from argparse import Namespace
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from datasets.augmentations import AgeTransformer
from utils.common import tensor2im
from models.psp import pSp

EXPERIMENT_TYPE = 'ffhq_aging'

"""## Step 2: Define Inference Parameters

Below we have a dictionary defining parameters such as the path to the pretrained model to use and the path to the
image to perform inference on.
While we provide default values to run this script, feel free to change as needed.
"""

EXPERIMENT_DATA_ARGS = {
    "ffhq_aging": {
        "model_path": "pretrained_models/sam_ffhq_aging.pt",
        "image_path": "images/secret1.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[EXPERIMENT_TYPE]

"""## Step 3: Load Pretrained Model
We assume that you have downloaded the pretrained aging model and placed it in the path defined above
"""

model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')

opts = ckpt['opts']
# pprint.pprint(opts)

# update the training options
opts['checkpoint_path'] = model_path

opts = Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')

"""## Step 4: Visualize Input"""

image_path = EXPERIMENT_DATA_ARGS[EXPERIMENT_TYPE]["image_path"]
original_image = Image.open(image_path).convert("RGB")

original_image.resize((256, 256))

"""## Step 5: Perform Inference

### Align Image

Before running inference we'll run alignment on the input image.
"""

def run_alignment(image_path):
    import dlib
    from scripts.align_all_parallel import align_face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image

aligned_image = run_alignment(image_path)
aligned_image.resize((256, 256))

"""### Run Inference"""

img_transforms = EXPERIMENT_ARGS['transform']
input_image = img_transforms(aligned_image)

# we'll run the image on multiple target ages
target_age = 20
age_transformers = [AgeTransformer(target_age=target_age)]

def run_on_batch(inputs, net):
    result_batch = net(inputs.to("cuda").float(), randomize_noise=False, resize=False)
    return result_batch

# for each age transformed age, we'll concatenate the results to display them side-by-side
# results = np.array(aligned_image.resize((1024, 1024)))
for age_transformer in age_transformers:
    print(f"Running on target age: {age_transformer.target_age}")
    with torch.no_grad():
        input_image_age = [age_transformer(input_image.cpu()).to('cuda')]
        input_image_age = torch.stack(input_image_age)
        result_tensor = run_on_batch(input_image_age, net)[0]
        result_image = tensor2im(result_tensor)
        results = result_image
        # results = np.concatenate([results, result_image], axis=1)

"""### Visualize Result"""

# results = Image.fromarray(results)
results   # this is a very large image (11*1024 x 1024) so it may take some time to display!

# save image at full resolution
results.save("outputs/secret1.jpg")


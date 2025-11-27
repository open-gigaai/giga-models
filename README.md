<p align="center">
<img width="55%" alt="GigaModels" src="./docs/source/imgs/logo.png?raw=true">
</p>
<h3 align="center">
A Comprehensive Repository for Multi-modal, Generative, and Perceptual Models
<p align="center">
    | <a href="#-installation">Quick Start</a>
    | <a href="#-contributing">Contributing</a>
    | <a href="#-license">License</a>
    | <a href="#-citation">Citation</a> |
</p>

## 🔥 Latest News

- **\[2025.11.27\]** 🎉 We released **[GigaBrain-0](https://github.com/open-gigaai/giga-brain-0)**, a novel VLA foundation model empowered by world model-generated data.
- **\[2025.11.26\]** 🎉 We released **[GigaWorld-0](https://github.com/open-gigaai/giga-world-0)**, a unified world model framework designed explicitly as a data engine to advance embodied AI.
- **\[2025.10.29\]** 😊 We provided a clean [PyTorch reproduction](./projects/vla/pi0) of [Pi0 and Pi0.5](https://github.com/Physical-Intelligence/openpi). It supports end-to-end training and inference with better performance and lightweight deployment.
- **\[2025.10.29\]** 🎉 We released **GigaModels**.

## ✨ Introduction

GigaModels is an open-source project offering an intuitive, high-performance infrastructure for a wide range of models. This comprehensive toolkit empowers users throughout the entire workflow, from training and inference to deployment and model compression.

We are dedicated to continuously integrating the latest advancements in open-source technology. Exciting updates and innovative features are always on the horizon—stay tuned!

## ⚡ Installation

GigaModels can be installed directly from source for the latest updates:

```bash
conda create -n giga_models python=3.11.10
conda activate giga_models
git clone https://github.com/open-gigaai/giga-models.git
cd giga-models
pip3 install -e .
```

## 🚀 Quick Start

GigaModels is designed to be very simple to use. You can easily load and utilize the model using `load_pipeline` or `XXPipeline`.
Here is an example of how to use:

```python
# Load the Grounding DINO model with load_pipeline
from PIL import Image
from giga_models import load_pipeline

image = Image.open(image_path)
pipe = load_pipeline('detection/grounding_dino/swint_ogc')
pred_boxes, pred_labels, pred_scores = pipe(image, det_labels)

# Load the Depth Anything model with DepthAnythingPipeline
from giga_models import DepthAnythingPipeline

pipe = DepthAnythingPipeline('depth-anything/Depth-Anything-V2-Large-hf').to('cuda')
depth_image = pipe(image)
```

More details on using GigaModels can be found in the [`projects`](./projects) folder.

## 🧨 Tasks & Pipelines

<table>
  <tr>
    <th></th>
    <th>Task</th>
    <th>Pipeline</th>
    <th>Inference</th>
    <th>Training</th>
  </tr>
  <tr>
    <td>VLA</td>
    <td>VLA</td>
    <td>
      <ul>
        <li><a href="giga_models/pipelines/vla/giga_brain_0/pipeline_giga_brain_0.py">GigaBrain-0</a></li>
        <li><a href="giga_models/pipelines/vla/pi0/pipeline_pi0.py">Pi0</a></li>
        <li><a href="giga_models/pipelines/vla/pi0/pipeline_pi0.py">Pi0.5</a></li>
      </ul>
    </td>
    <td>
      <ul>
        <li><a href="https://github.com/open-gigaai/giga-brain-0/blob/main/scripts/inference.py">GigaBrain-0</a></li>
        <li><a href="projects/vla/pi0/scripts/inference.py">Pi0</a></li>
        <li><a href="projects/vla/pi0/scripts/inference.py">Pi0.5</a></li>
      </ul>
    </td>
    <td>
      <ul>
        <li><a href="https://github.com/open-gigaai/giga-brain-0/blob/main/configs/giga_brain_0_from_scratch.py">GigaBrain-0</a></li>
        <li><a href="projects/vla/pi0/configs/pi0.py">Pi0</a></li>
        <li><a href="projects/vla/pi0/configs/pi05.py">Pi0.5</a></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>Diffusion</td>
    <td>Diffusion</td>
    <td>
      <ul>
        <li><a href="giga_models/pipelines/diffusion/giga_world_0/pipeline_giga_world_0.py">GigaWorld-0</a></li>
      </ul>
    </td>
    <td>
      <ul>
        <li><a href="https://github.com/open-gigaai/giga-world-0/blob/main/scripts/inference.py">GigaWorld-0</a></li>
      </ul>
    </td>
    <td>
      <ul>
        <li><a href="https://github.com/open-gigaai/giga-world-0/blob/main/configs/giga_world_0_video.py">GigaWorld-0</a></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>Vision</td>
    <td>Depth Estimation</td>
    <td>
      <ul>
        <li><a href="giga_models/pipelines/vision/depth_estimation/pipeline_depth_anything.py">Depth Anything</a></li>
        <li><a href="giga_models/pipelines/vision/depth_estimation/pipeline_dpt.py">DPT</a></li>
      </ul>
    </td>
    <td>
      <ul>
        <li><a href="projects/vision/depth_estimation/inference_depth_estimation.py">Depth Anything</a></li>
        <li><a href="projects/vision/depth_estimation/inference_depth_estimation.py">DPT</a></li>
      </ul>
    </td>
    <td>
    </td>
  </tr>
  <tr>
    <td></td>
    <td>Detection</td>
    <td>
      <ul>
        <li><a href="giga_models/pipelines/vision/detection/pipeline_grounding_dino.py">Grounding DINO</a></li>
      </ul>
    </td>
    <td>
      <ul>
        <li><a href="projects/vision/detection/inference_grounding_dino.py">Grounding DINO</a></li>
      </ul>
    </td>
    <td>
    </td>
  </tr>
  <tr>
    <td></td>
    <td>Edge Detection</td>
    <td>
      <ul>
        <li><a href="giga_models/pipelines/vision/edge_detection/pipeline_canny.py">Canny</a></li>
        <li><a href="giga_models/pipelines/vision/edge_detection/pipeline_hed.py">HED</a></li>
        <li><a href="giga_models/pipelines/vision/edge_detection/pipeline_lineart.py">Lineart</a></li>
        <li><a href="giga_models/pipelines/vision/edge_detection/pipeline_mlsd.py">MLSD</a></li>
        <li><a href="giga_models/pipelines/vision/edge_detection/pipeline_pidinet.py">PidiNet</a></li>
      </ul>
    </td>
    <td>
      <ul>
        <li><a href="projects/vision/edge_detection/inference_edge_detection.py">Canny</a></li>
        <li><a href="projects/vision/edge_detection/inference_edge_detection.py">HED</a></li>
        <li><a href="projects/vision/edge_detection/inference_edge_detection.py">Lineart</a></li>
        <li><a href="projects/vision/edge_detection/inference_edge_detection.py">MLSD</a></li>
        <li><a href="projects/vision/edge_detection/inference_edge_detection.py">PidiNet</a></li>
      </ul>
    </td>
    <td>
    </td>
  </tr>
  <tr>
    <td></td>
    <td>Frame Interpolation</td>
    <td>
      <ul>
        <li><a href="giga_models/pipelines/vision/frame_interpolation/pipeline_film.py">Film</a></li>
      </ul>
    </td>
    <td>
      <ul>
        <li><a href="projects/vision/frame_interpolation/inference_film.py">Film</a></li>
      </ul>
    </td>
    <td>
    </td>
  </tr>
  <tr>
    <td></td>
    <td>Image Restoration</td>
    <td>
      <ul>
        <li><a href="giga_models/pipelines/vision/image_restoration/pipeline_prompt_ir.py">PromptIR</a></li>
      </ul>
    </td>
    <td>
      <ul>
        <li><a href="projects/vision/image_restoration/inference_image_restoration.py">PromptIR</a></li>
      </ul>
    </td>
    <td>
    </td>
  </tr>
  <tr>
    <td></td>
    <td>Keypoints</td>
    <td>
      <ul>
        <li><a href="giga_models/pipelines/vision/keypoints/pipeline_openpose.py">OpenPose</a></li>
        <li><a href="giga_models/pipelines/vision/keypoints/pipeline_rtm_pose.py">RTMPose</a></li>
      </ul>
    </td>
    <td>
      <ul>
        <li><a href="projects/vision/keypoints/inference_keypoints.py">OpenPose</a></li>
        <li><a href="projects/vision/keypoints/inference_keypoints.py">RTMPose</a></li>
      </ul>
    </td>
    <td>
    </td>
  </tr>
  <tr>
    <td></td>
    <td>Optical Flow</td>
    <td>
      <ul>
        <li><a href="giga_models/pipelines/vision/optical_flow/pipeline_unimatch.py">UniMatch</a></li>
      </ul>
    </td>
    <td>
      <ul>
        <li><a href="projects/vision/optical_flow/inference_unimatch.py">UniMatch</a></li>
      </ul>
    </td>
    <td>
    </td>
  </tr>
  <tr>
    <td></td>
    <td>Segmentation</td>
    <td>
      <ul>
        <li><a href="giga_models/pipelines/vision/segmentation/pipeline_grounded_sam2.py">Grounded SAM 2</a></li>
        <li><a href="giga_models/pipelines/vision/segmentation/pipeline_segment_anything.py">Segment Anything</a></li>
        <li><a href="giga_models/pipelines/vision/segmentation/pipeline_upernet.py">UperNet</a></li>
      </ul>
    </td>
    <td>
      <ul>
        <li><a href="projects/vision/segmentation/inference_grounded_sam2.py">Grounded SAM 2</a></li>
        <li><a href="projects/vision/segmentation/inference_segment_anything.py">Segment Anything</a></li>
        <li><a href="projects/vision/segmentation/inference_upernet.py">UperNet</a></li>
      </ul>
    </td>
    <td>
    </td>
  </tr>
  <tr>
    <td></td>
    <td>Shot Boundary Detection</td>
    <td>
      <ul>
        <li><a href="giga_models/pipelines/vision/shot_boundary_detection/pipeline_transnetv2.py">TransNetV2</a></li>
      </ul>
    </td>
    <td>
      <ul>
        <li><a href="projects/vision/shot_boundary_detection/inference_transnetv2.py">TransNetV2</a></li>
      </ul>
    </td>
    <td>
    </td>
  </tr>
</table>

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

```bibtex
@misc{gigaai2025gigamodels,
    author = {GigaAI},
    title = {GigaModels: A Comprehensive Repository for Multi-modal, Generative, and Perceptual Models},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/open-gigaai/giga-models}}
}
```

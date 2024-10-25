# Neural style transfer 

## Introduction
Neural style transfer is an optimization technique used to take two images—a content image and a style reference image (such as an artwork by a famous painter)—and blend them together so the output image looks like the content image, but “painted” in the style of the style reference image.

This is implemented by using tensorflow pretrained model vgg19, accessing and changing the intermediate layers of the model, extracting style and content, running gradient descent and total variation loss and optimizing the output image to match the content statistics of the content image and the style statistics of the style reference image. These statistics are extracted from the images using a convolutional network.

### Content & style reference image:
![image](https://github.com/user-attachments/assets/dc82cb4f-866b-4f43-a305-2c90f1012424)

### Output blended image:
![image](https://github.com/user-attachments/assets/6726d2a1-accb-45e0-a932-96238101103f)



## Get started

```python
# create a new conda env and install the dependency file
conda create --name <env-name> python=3.10.13
conda activate <env-name>
pip install -r requirements.txt
```

## Execute
```python
python main.py
#or
python main.py [-c | --config] <params=value>
```
### Optional Parameters

- `LEARNING_RATE`: Set the learning rate (e.g., `0.02`), type is float, default is `0.001`.
- `BETA_1`: Set the beta_1 parameter (e.g., `0.99`), type is float, default is `0.9`.
- `EPSILON`: Set the epsilon value (e.g., `1e-1`), type is float, default is `1e-7`.
- `TOTAL_VARIATION_WEIGHT`: Set the total variation weight (e.g., `30`), type is float, default is `0.0`.
- `EPOCHS`: Set the number of epochs (e.g., `10`), type is integer, default is `1`.
- `STEPS_PER_EPOCH`: Set the number of steps per epoch (e.g., `100`), type is integer, default is `1`.
- `STYLE_WEIGHT`: Set the style weight (e.g., `1e-2`), type is float, default is `1e-4`.
- `CONTENT_WEIGHT`: Set the content weight (e.g., `1e4`), type is float, default is `1e4`.
- `MAX_DIM`: Limit the image to the maximum dimension (e.g., `512`), type is integer, default is `512`.
- `STYLE_LAYERS`: Specify the style layers, type is list, default is `['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']`.
- `CONTENT_LAYERS`: Specify the content layers, type is list, default is `['block5_conv2']`.
- `SAVE_KERAS`: Set whether to save the Keras model (e.g., `false`), type is boolean, default is `false`.

## Generate your own output blended image
```yaml
# /root_dir/config/config.yaml
data_ingestion:
  content_image_URL: 'replace with new content_image_URL'
  style_image_URL: 'replace with new style_image_URL'
```

## Update Model wights file
```yaml
# /root_dir/config/config.yaml
pretrain_model:
  weights_dir: 'replace with new weights file dir'
```

## Dockerfile

```
FROM python:3.10.2
RUN pip install virtualenv
RUN virtualenv /env
ENV VIRTUAL_ENV=/env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir -r requirements.txt
CMD ["python", "main.py"]
```

```bash
docker build
```

## Blog
blog_link

## Reference
[Neural style transfer](https://www.tensorflow.org/tutorials/generative/style_transfer#visualize_the_input)

# 3DBuilder-PyTorch3d
I implemented an algorithm for Unsupervised training on 3D models using PyTorch3D's *differentiable rendering*.
My main goal is to see how well this algorithm can re-create 3D meshes.
I chose this method because DNNs (Deep Neural Networks) lack understanding of how the 3D shapes relates to images.

- My motivation came from wanting to re-create 3D buildings from 2D images, as this field is still lacking. Then using these buildings in a Unreal Engine game.


## Flow
It creates a dataset of 2D images by rendering a 3D mesh from multiple views (*silhouettes*).
It then a sphere which, in the training phase, will get topologically transformed, based on the silhouette images. 
Using an optimizer and a loss function, it easily finds out how the mesh needs to be transformed. 

### Differentiable Rendering
Differentiation is the study of how spontaneous a process is changing. For example: the speed is a derivative of location: it shows the rate of something changing location; acceleration is the derivative of speed and the second derivative of location and so on. 

Differentiating rendering will show us how small changes in the rendering process will affect the final picture. With the help of differentiation, we can tweak the parameters such as: geometry, cameras, lights, materials and other scene parameters to create great results.

Differentiable rendering is a new technique that helps us solve this problem: it allows us to compute the gradients of 3D objects and propagate through images, it reduces the need for collecting and labeling 3D data and enables high success rates in multiple applications.

### Algorithms and data structures used

The algorithm used in the optimization phase is the Stochaistic Gradient Descent: a method that uses gradients to find the minimum cost of a function. We also rely on back.

The main data structure is a neural network stored into a tensor.

### Dataset creation

A dataset is a collection of data, specifically in our case, it is 2D images. Usually, good datasets are hard to build, as the data needs to be both qualitative and quantitative. For 3D reconstruction using machine learning, it was usually the case that we needed both photographs and their respective 3D representations. We can create the dataset directly by rendering the 3D mesh.

In order to create the images dataset, we will first load the mesh and store it into a Meshes object instance. Because I am training the model on a CPU, I will scale down the mesh.

### 3D object loading and processing
Using PyTorch3D we create a tensor from an .obj file as input. Then, using *differentiable rendering* and a list of cameras, we render the silhouettes and store them in another tensor.

### Mesh Prediction
I used both a sphere and a cube as starting shapes to see which performs best, as my intention is to re-create buildings (better with cubes as starting shape).

We start by computing a loss function for 4 parameters: silhouette image loss, mesh edge loss, mesh normal consistency and Laplacian smoothing. I'll store them in a dictionary as it's easier to access and update the respective weights.

```python
losses = {"silhouette": {"weight": 1.0, "values": []},
          "edge": {"weight": 1.0, "values": []},
          "normal": {"weight": 0.01, "values": []},
          "laplacian": {"weight": 1.0, "values": []},
          }
```
The we use the built in functions from PyTorch3D: mesh_edge_loss(), mesh_normal_consistency() and laplacian_smoothing() to create an *update* function that is called at each step called update_mesh_shape_prior_losses().

### Optimizer
An optimizer’s goal is to minimize the loss of the learning model by iteratively changing the model’s weights in order to reduce the error between the predicted output and the target output. We will use a Gradient Descent algorithm, specifically the Stochastic Gradient Descent (SGD) optimizer from the PyTorch library.

After multiple tries on the building reconstruction, I’ve reached that a learning rate of 0.25 is sufficient.

```python
optimizer = torch.optim.SGD([deform_verts], lr=0.25, momentum=0.9)
```

### Optimization Loop
At the beginning of each iteration, we reset all the learnable parameters: reset the gradients of all the optimized tensors. We create the new mesh based on the *deform_verts*, we reset the current loss to a tensor of 0’s; and we call the helper function that updates the mesh prior to getting the loss function.

Then we use the renderer to get two different random images and calculate the loss. The variable j represents the random index of the views that we will use to render.

```python
images_predicted = self.renderer(new_src_mesh, cameras=target_cameras[j],  lights=lights)  

predicted_silhouette = images_predicted[..., 3]  # alpha-channel 

_loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean() 

loss["silhouette"] += loss_silhouette / 20
```

We want to compute at each step how is the loss function performing overall, so we create a weighted sum of the losses so far.

```python
sum_loss = torch.tensor(0.0, device=self.device)
for key, loss_value in loss.items():
    sum_loss += loss_value * losses[k]["weight"]
    losses[key]["values"].append(float(loss_value.detach()))
```

Finally, we call the backward function on the *sum_loss* tensor. It computes the gradient of the tensor during the backward pass in the model. We also call the stop() function of the optimizer. After the optimization loop, we can save the 3D mesh
## Example
 
![image](https://github.com/user-attachments/assets/7fa6e2ee-45ee-47ae-a621-cf11bd54d6d9)
![image](https://github.com/user-attachments/assets/1a47c8db-b2a7-43e5-b654-b0717d72e2e7)


Fig 1. Training on a sphere. Iterations 0 and 250

![image](https://github.com/user-attachments/assets/846e15ab-7d4f-4e60-8edd-2452e82a911e)
![image](https://github.com/user-attachments/assets/30da5c0f-1cec-4183-b64e-9808f508dee1)

Fig 2. Training on a cube. Iterations 0 and 250

### Constructing a mesh with new images
I manually cropped a building from photos taken from different views and estimated the camera angles. 
The process is similar to the training phase, but it now uses the new images as target silhouettes.

![image](https://github.com/user-attachments/assets/86b80851-6317-4580-905a-fa703689d92f)

Fig 3. Silhouettes of buildings with my estimated elevation, and azimuth values

![image](https://github.com/user-attachments/assets/4cfde6ce-3294-464a-9c21-107d6136ce26)

Fig 4. Resulted 3D meshes. Perspective view

![image](https://github.com/user-attachments/assets/62a54948-709c-41de-98df-dc3139a90935)

Fig 5. Resulted 3D meshes. Top-down view

## References
1. M.-F. team, "PyTorch3D Docs,"  https://pytorch3d.org/docs/
2.   T. L. W. C. H. L. Shichen Liu, "Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning," 2019.
3. [Beyond the Surface: Advanced 3D Mesh Generation from 2D Images in Python](https://medium.com/red-buffer/beyond-the-surface-advanced-3d-mesh-generation-from-2d-images-in-python-0de6dd3944ac)

© 2024 Faculty of Computer Science, Alexandru Ioan Cuza University, Iasi. All rights reserved.

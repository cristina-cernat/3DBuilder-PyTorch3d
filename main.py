import os
import time
import torch
import torch_directml
from datetime import timedelta
from torchvision import transforms, datasets
import torch.nn as nn
from PIL import Image

from pytorch3d.utils import ico_sphere
import numpy as np
from tqdm import tqdm
from pytorch3d.structures.meshes import Meshes
from pytorch3d.io import IO, save_obj
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import PointLights, look_at_view_transform, FoVPerspectiveCameras, RasterizationSettings, \
    MeshRenderer, MeshRasterizer, SoftSilhouetteShader
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.ops import SubdivideMeshes
from pathlib import Path


class FitMesh:
    def __init__(self, obj_path, renderer, device, number_of_views, iterations, starting_shape, use_images):
        self.device = device

        self.mesh = load_objs_as_meshes([obj_path], device=self.device)

        self.renderer = renderer
        self.number_of_views = number_of_views
        self.iterations = iterations
        self.starting_shape = starting_shape
        self.use_images = use_images
        self.optimized_mesh = Meshes(verts=[], faces=[])

    def train_on_mesh(self):
        # scale mesh
        verts = self.mesh.verts_packed()
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        self.mesh.offset_verts_(-center)
        self.mesh.scale_verts_((1.0 / float(scale)))

        # duplicate meshes
        mesh_list = self.mesh.extend(self.number_of_views)

        target_cameras = []
        for i in range(self.number_of_views):
            R_i = R[None, i, ...]
            T_i = T[None, i, ...]

            camera = FoVPerspectiveCameras(device=self.device, R=R_i, T=T_i)
            target_cameras.append(camera)

        silhouette_images = self.renderer(mesh_list, cameras=cameras, lights=lights)

        # silhouettes_cow is a 4D tensor (batch_size, height, width, channels).
        # index 3 is the alpha channel. we extract only the alpha channel
        target_silhouette = []
        for i in range(self.number_of_views):
            target_silhouette.append(silhouette_images[i, ..., 3])

        if self.use_images:
            """ read other images """
            images_path = Path(r"/silhouettes_cow").glob("*")
            image_list = []
            transform = transforms.Compose([transforms.ToTensor()])
            for file in images_path:
                image_path = str(file)
                image = Image.open(image_path).convert("RGBA")
                image = image.resize((128, 128))

                image_tensor = transform(image).float() / 255.0

                image_list.append(image_tensor.to(self.device))
            silhouette_images2 = torch.stack(image_list)
            silhouette_images2 = silhouette_images2.permute(0, 2, 3, 1)
            target_silhouette2 = []
            for i in range(silhouette_images2.shape[0]):
                target_silhouette2.append(silhouette_images2[i, ..., 3])

        """ continue """
        if self.starting_shape == 'sphere':
            # icosphere (icosahedron) with subdivision level
            source_mesh = ico_sphere(4, self.device)
        elif self.starting_shape == 'cube':
            # cube with subdivision level. cube = better source mesh for buildings
            source_mesh = create_cube(4, self.device)

        num_views_per_iteration = 2

        losses = {"silhouette": {"weight": 1.0, "values": []},
                  "edge": {"weight": 1.0, "values": []},
                  "normal": {"weight": 0.01, "values": []},
                  "laplacian": {"weight": 1.0, "values": []},
                  }

        verts_shape = source_mesh.verts_packed().shape
        deform_verts = torch.full(verts_shape, 0.0, device=self.device, requires_grad=True)

        optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)
        new_src_mesh = source_mesh.offset_verts(deform_verts)

        loop = tqdm(range(self.iterations))

        for i in loop:
            optimizer.zero_grad()
            new_src_mesh = source_mesh.offset_verts(deform_verts)

            loss = {}
            for k in losses:
                loss[k] = torch.tensor(0.0, device=self.device)

            update_mesh_shape_prior_losses(new_src_mesh, loss)

            for j in np.random.permutation(self.number_of_views).tolist()[:num_views_per_iteration]:
                # render 2 random images and calculate loss
                images_predicted = self.renderer(new_src_mesh, cameras=target_cameras[j], lights=lights)
                predicted_silhouette = images_predicted[..., 3]  # 3 = alpha channel
                loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean()
                loss["silhouette"] += loss_silhouette / num_views_per_iteration

            # Weighted sum of the losses
            sum_loss = torch.tensor(0.0, device=self.device)
            for k, l in loss.items():
                sum_loss += l * losses[k]["weight"]
                losses[k]["values"].append(float(l.detach().cpu()))

            loop.set_description(f"total_loss = {sum_loss:.6f}")

            sum_loss.backward()
            optimizer.step()

        self.optimized_mesh = new_src_mesh
        final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
        final_verts = final_verts * scale + center

        final_obj = os.path.join('./', 'final_model.obj')
        save_obj(final_obj, final_verts, final_faces)

    def optimize_mesh(self, images, angles):
        verts_shape = self.optimized_mesh.verts_packed().shape
        deform_verts = torch.full(verts_shape, 0.0, device=self.device, requires_grad=True)
        optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)


# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh, loss):
    loss["edge"] = mesh_edge_loss(mesh)

    loss["normal"] = mesh_normal_consistency(mesh)

    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")


def create_cube(level, device):
    # create a cube with sid length = 1
    vertices = torch.tensor([
        [-1, -1, -1],  # 0
        [-1, -1, 1],  # 1
        [-1, 1, -1],  # 2
        [-1, 1, 1],  # 3
        [1, -1, -1],  # 4
        [1, -1, 1],  # 5
        [1, 1, -1],  # 6
        [1, 1, 1]  # 7
    ], dtype=torch.float32, device=device)

    # faces of the cube. using counter-clockwise order so normals direct outwards
    faces = torch.tensor([
        [0, 1, 2], [1, 3, 2],
        [4, 6, 5], [5, 6, 7],
        [0, 4, 1], [1, 4, 5],
        [2, 3, 6], [3, 7, 6],
        [0, 2, 4], [2, 6, 4],
        [1, 5, 3], [3, 5, 7]
    ], dtype=torch.int64, device=device)
    mesh = Meshes(verts=[vertices], faces=[faces])

    for _ in range(level):
        mesh = SubdivideMeshes()(mesh)

    return mesh


if __name__ == "__main__":
    start_time = time.perf_counter()

    obj_path_3d = "G:/Projects/3DBuilder-Pytorch/metropolitan_palace.obj"

    device_cpu = torch.device("cpu")
    device_gpu = torch_directml.device()

    # initialize elevation and azimuth of cameras
    num_of_views = 20
    elevation = torch.linspace(0, 360, num_of_views, device=device_cpu)
    azimuth = torch.linspace(-180, 180, num_of_views, device=device_cpu)

    # azimuth = torch.tensor([-180., -160., -130., -110., -90., -60., -36., -15.,
    #                         15., 36., 60., 90., 110., 130., 160., 180.])

    # create lights
    lights = PointLights(device=device_cpu, location=[[0.0, 0.0, -3.0]])

    # create cameras. R = rotation matrix, T = translation_matrix
    R, T = look_at_view_transform(dist=1.7, elev=elevation, azim=azimuth, device=device_cpu)
    cameras = FoVPerspectiveCameras(device=device_cpu, R=R, T=T)

    '''silhouette renderer '''
    #  Create silhouettes rasterization + renderer
    raster_settings_silhouette = RasterizationSettings(image_size=128, blur_radius=np.log(1. / 1e-4 - 1.) * 1e-4,
                                                       faces_per_pixel=50)
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings_silhouette),
        shader=SoftSilhouetteShader())

    fit_mesh = FitMesh(obj_path_3d, renderer_silhouette, device_cpu, num_of_views, 300, 'sphere', False)

    fit_mesh.train_on_mesh()

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print("Elapsed time: ", str(timedelta(seconds=elapsed)))

import os
import time
import torch
from datetime import timedelta
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
from pathlib import Path

from pytorch3d.utils import ico_sphere
from pytorch3d.structures.meshes import Meshes
from pytorch3d.io import IO, save_obj
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import PointLights, look_at_view_transform, FoVPerspectiveCameras, RasterizationSettings, \
    MeshRenderer, MeshRasterizer, SoftSilhouetteShader
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.ops import SubdivideMeshes

base_dir = Path().resolve()


class FitMesh:
    def __init__(self, obj_path, renderer, device, number_of_views, iterations, starting_shape):
        self.device = device

        self.mesh = load_objs_as_meshes([obj_path], device=self.device)

        self.renderer = renderer
        self.number_of_views = number_of_views
        self.iterations = iterations
        self.starting_shape = starting_shape
        self.optimized_mesh = Meshes(verts=[], faces=[])

    def construct_from_mesh(self):
        # scale mesh
        verts = self.mesh.verts_packed()
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        self.mesh.offset_verts_(-center)
        # scale the vertices so that the mesh fits within the cube of side length 1
        self.mesh.scale_verts_(0.5 / float(scale))

        # duplicate meshes
        mesh_list = self.mesh.extend(self.number_of_views)

        target_cameras = []
        for i in range(self.number_of_views):
            R_i = R[None, i, ...]
            T_i = T[None, i, ...]

            camera = FoVPerspectiveCameras(device=self.device, R=R_i, T=T_i)
            target_cameras.append(camera)

        silhouette_images = self.renderer(mesh_list, cameras=cameras, lights=lights)

        # silhouettes_images is a 4D tensor (batch_size, height, width, channels).
        # index 3 is the alpha channel. we extract only the alpha channel
        target_silhouette = []
        for i in range(self.number_of_views):
            target_silhouette.append(silhouette_images[i, ..., 3])

        # what's the starting shape
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
        new_src_mesh = source_mesh.offset_verts(deform_verts)

        optimizer = torch.optim.SGD([deform_verts], lr=0.25, momentum=0.9)
        loop = tqdm(range(self.iterations))

        for i in loop:
            optimizer.zero_grad()
            new_src_mesh = source_mesh.offset_verts(deform_verts)

            loss = {}
            for k in losses:
                loss[k] = torch.tensor(0.0, device=self.device)

            update_mesh_shape_prior_losses(new_src_mesh, loss)

            for j in np.random.permutation(self.number_of_views).tolist()[:2]:
                # render 2 random images and calculate loss
                images_predicted = self.renderer(new_src_mesh, cameras=target_cameras[j], lights=lights)
                predicted_silhouette = images_predicted[..., 3]  # 3 = alpha channel
                loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean()
                loss["silhouette"] += loss_silhouette / num_views_per_iteration

            # Weighted sum of the losses
            sum_loss = torch.tensor(0.0, device=self.device)
            for k, l in loss.items():
                sum_loss += l * losses[k]["weight"]
                losses[k]["values"].append(float(l.detach()))

            loop.set_description(f"total_loss_mesh = {sum_loss:.6f}")

            sum_loss.backward()
            optimizer.step()

        self.optimized_mesh = new_src_mesh
        final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
        final_verts = final_verts * scale + center

        final_obj = os.path.join('./', 'final_model.obj')
        save_obj(final_obj, final_verts, final_faces)

    def construct_from_images(self, target_silhouette, angles, iterations):
        verts = self.mesh.verts_packed()
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])

        # load optimized model if it's not empty
        optimized_path = base_dir / "final_model.obj"
        if self.optimized_mesh.verts_packed().numel() == 0 and self.optimized_mesh.faces_packed().numel() == 0:
            self.optimized_mesh = load_objs_as_meshes([optimized_path], device=self.device)

        """ configure losses """
        losses = {
            "silhouette": {"weight": 1.0, "values": []},
            "edge": {"weight": 1.0, "values": []},
            "normal": {"weight": 0.01, "values": []},
            "laplacian": {"weight": 1.0, "values": []},
        }

        verts_shape = self.optimized_mesh.verts_packed().shape
        deform_verts = torch.full(verts_shape, 0.0, device=self.device, requires_grad=True)

        optimizer = torch.optim.SGD([deform_verts], lr=0.12, momentum=0.9)
        new_src_mesh = self.optimized_mesh.offset_verts(deform_verts)

        loop = tqdm(range(iterations))

        for i in loop:
            optimizer.zero_grad()
            new_src_mesh = self.optimized_mesh.offset_verts(deform_verts)

            loss = {}
            for k in losses:
                loss[k] = torch.tensor(0.0, device=self.device)

            update_mesh_shape_prior_losses(new_src_mesh, loss)

            for j in np.random.permutation(len(silhouettes)).tolist()[:iterations]:
                # render 2 random images and calculate loss
                R, T = look_at_view_transform(dist=2.7, elev=angles[j][0], azim=angles[j][1])
                cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
                images_predicted = self.renderer(new_src_mesh, cameras=cameras, lights=lights)
                predicted_silhouette = images_predicted[..., 3]
                loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean()
                loss["silhouette"] += loss_silhouette / len(silhouettes)

            # Weighted sum of the losses
            sum_loss = torch.tensor(0.0, device=self.device)
            for k, l in loss.items():
                sum_loss += l * losses[k]["weight"]
                losses[k]["values"].append(float(l.detach()))

            loop.set_description(f"total_loss_images = {sum_loss:.6f}")

            sum_loss.backward()
            optimizer.step()

        final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
        final_verts = final_verts * scale + center

        final_obj = os.path.join('./', 'new_model.obj')
        save_obj(final_obj, final_verts, final_faces)

    def load_images(self, images_path):
        image_list = []
        transform = transforms.Compose([transforms.ToTensor()])
        for file in images_path.iterdir():
            image_path = str(file)
            image = Image.open(image_path).convert("RGBA")
            image = pad_to_square(image)
            image = image.resize((128, 128))

            image_tensor = transform(image).float() / 255.0

            image_list.append(image_tensor.to(self.device))
        silhouette_images = torch.stack(image_list)
        silhouette_images = silhouette_images.permute(0, 2, 3, 1)

        target_silhouette = []
        for i in range(silhouette_images.shape[0]):
            target_silhouette.append(silhouette_images[i, ..., 3])
        return target_silhouette


# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh, loss):
    loss["edge"] = mesh_edge_loss(mesh)
    loss["normal"] = mesh_normal_consistency(mesh)
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")


def pad_to_square(image):
    width, height = image.size
    max_dim = max(width, height)

    left = (max_dim - width) // 2
    right = max_dim - width - left
    top = (max_dim - height) // 2
    bottom = max_dim - height - top

    padding = (left, top, right, bottom)
    padded_image = ImageOps.expand(image, padding, fill=0)

    return padded_image


def create_cube(level, device):
    # create a cube with side length = 0.5
    vertices = torch.tensor([
        [-0.5, -0.5, -0.5],  # 0
        [-0.5, -0.5, 0.5],  # 1
        [-0.5, 0.5, -0.5],  # 2
        [-0.5, 0.5, 0.5],  # 3
        [0.5, -0.5, -0.5],  # 4
        [0.5, -0.5, 0.5],  # 5
        [0.5, 0.5, -0.5],  # 6
        [0.5, 0.5, 0.5]  # 7
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

    device_cpu = torch.device("cpu")
    device_gpu_cuda = torch.device("cuda:0")
    # choose device
    device = device_cpu

    # initialize elevation and azimuth of cameras
    num_of_views = 20
    elevation = torch.linspace(0, 360, num_of_views, device=device)
    azimuth = torch.linspace(-180, 180, num_of_views, device=device)

    # estimated camera angles for the building
    building_angles = [(0, -180), (0, -160), (0, -130), (0, -110), (0, -90), (0, -60), (0, -36), (0, -15),
                       (0, 15), (0, 36), (0, 60), (0, 90), (0, 110), (0, 130), (0, 160), (0, 180)]

    cow_angles = list(zip(elevation.tolist(), azimuth.tolist()))
    # create lights
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # create cameras. R = rotation matrix, T = translation_matrix
    R, T = look_at_view_transform(dist=2.7, elev=elevation, azim=azimuth, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    """ silhouette renderer """
    #  create silhouettes rasterization + renderer
    raster_settings_silhouette = RasterizationSettings(image_size=128, blur_radius=np.log(1. / 1e-4 - 1.) * 1e-4,
                                                       faces_per_pixel=50)
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings_silhouette),
        shader=SoftSilhouetteShader())

    obj_path_3d = base_dir / "metropolitan_palace.obj"
    fit_mesh = FitMesh(obj_path_3d, renderer_silhouette, device, num_of_views, 3000, 'cube')

    """ construct from mesh """
    # fit_mesh.construct_from_mesh()

    """ construct from images """
    images = base_dir / "silhouettes_my_building"
    silhouettes = fit_mesh.load_images(images)
    fit_mesh.construct_from_images(silhouettes, building_angles, 100)

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print("Elapsed time: ", str(timedelta(seconds=elapsed)))

# saving and loading
# PATH_OPT = "optimizer.pt"
# torch.save(optimizer.state_dict(), PATH_OPT)
# PATH = "model.pt"
# torch.save(new_src_mesh.verts_packed(), PATH)
# PATH_REND = "renderer.pt"
# torch.save(renderer_silhouette, PATH_REND)
#
# optimizer_loaded = torch.load(PATH_OPT)
# new_src_mesh_loaded = torch.load(PATH)
# renderer_silhouette_loaded = torch.load(PATH_REND)
#
# new = renderer_silhouette.rasterizer


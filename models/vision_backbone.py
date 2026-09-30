"""
DINOv3 Frozen Vision Backbone + MuJoCo XML Camera Renderer
============================================================
3 kamera (2 overhead + 1 wrist) XML'den okunur.
RGB + Depth render → DINOv3 feature extraction.
"""

import numpy as np
import torch
import timm
import timm.data
from PIL import Image


class DINOv3Backbone:
    def __init__(self, model_name="vit_base_patch16_dinov3", img_size=512,
                 device="cpu"):
        self.device = device
        self.img_size = img_size

        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()
        self.model.to(device)

        for p in self.model.parameters():
            p.requires_grad = False

        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        override_cfg = {**data_cfg, 'input_size': (3, img_size, img_size)}
        self.transform = timm.data.create_transform(
            **override_cfg, is_training=False)

        self.patch_size = self.model.patch_embed.patch_size[0]
        self.grid_h = img_size // self.patch_size
        self.grid_w = img_size // self.patch_size
        self.n_patches = self.grid_h * self.grid_w
        self.hidden_dim = self.model.embed_dim

        self.n_special = getattr(self.model, 'num_prefix_tokens', 1)

        print(f"  DINOv3: {img_size}x{img_size} → "
              f"{self.grid_h}x{self.grid_w} patches, dim={self.hidden_dim}")

    @torch.inference_mode()
    def extract(self, rgb_np):
        """(H, W, 3) uint8 → (n_patches, 768) float32."""
        img = Image.fromarray(rgb_np)
        inp = self.transform(img).unsqueeze(0).to(self.device)
        feats = self.model.forward_features(inp)
        patch_tokens = feats[0, self.n_special:]
        return patch_tokens.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def extract_batch(self, rgb_list):
        """list of (H, W, 3) → (batch, n_patches, 768) float32."""
        imgs = [Image.fromarray(rgb) for rgb in rgb_list]
        batch = torch.stack([self.transform(img) for img in imgs])
        batch = batch.to(self.device)
        feats = self.model.forward_features(batch)
        patch_tokens = feats[:, self.n_special:]
        return patch_tokens.cpu().numpy().astype(np.float32)

    def extract_spatial(self, rgb_np):
        """(H, W, 3) → (grid_h, grid_w, 768)."""
        feats = self.extract(rgb_np)
        return feats.reshape(self.grid_h, self.grid_w, self.hidden_dim)

    def downsample_depth(self, depth_map):
        """(img_size, img_size) → (grid_h, grid_w, 1) normalized."""
        ps = self.patch_size
        h, w = self.grid_h, self.grid_w
        depth_patches = depth_map[:h*ps, :w*ps].reshape(
            h, ps, w, ps).mean(axis=(1, 3))
        max_depth = 3.0
        depth_patches = np.clip(depth_patches, 0.0, max_depth) / max_depth
        return depth_patches[:, :, np.newaxis].astype(np.float32)


class MuJoCoRenderer:
    """XML'deki kameralardan RGB + Depth render."""

    def __init__(self, mj_model, img_size=512, cam_names=None):
        import mujoco

        self.mj_model = mj_model
        self.img_size = img_size

        mj_model.vis.global_.offwidth = max(img_size,
                                             mj_model.vis.global_.offwidth)
        mj_model.vis.global_.offheight = max(img_size,
                                              mj_model.vis.global_.offheight)

        self.renderer = mujoco.Renderer(mj_model,
                                         height=img_size, width=img_size)

        self.renderer._scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.renderer._scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

        if cam_names is None:
            cam_names = ['overhead_cam1', 'overhead_cam2', 'wrist_cam']
        self.cam_names = cam_names

        self.cam_ids = []
        for name in cam_names:
            cid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            self.cam_ids.append(cid)
            print(f"  Camera: {name} → id={cid}")

        self.vopt = mujoco.MjvOption()
        self.vopt.sitegroup[4] = 0

    def render_rgbd(self, mj_data, cam_idx=0):
        cam_id = self.cam_ids[cam_idx]

        self.renderer.update_scene(mj_data, camera=cam_id, scene_option=self.vopt)
        rgb = self.renderer.render().copy()

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(mj_data, camera=cam_id, scene_option=self.vopt)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()

        return rgb, depth

    def render_all_rgbd(self, mj_data):
        """Tüm kameralardan RGB + Depth."""
        rgbs, depths = [], []
        for i in range(len(self.cam_ids)):
            rgb, depth = self.render_rgbd(mj_data, i)
            rgbs.append(rgb)
            depths.append(depth)
        return rgbs, depths

    def close(self):
        self.renderer.close()
import torch
import torch.nn.functional as F
import json
from comfy_api.latest import io
import node_helpers
import comfy
import comfy.utils
import comfy.latent_formats

class WanPromptBatchNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanPromptBatch",
            display_name="Wan Prompt Batch",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                io.String.Input("prompts_data", default="[]")
            ],
            outputs=[
                io.Custom("PROMPT_BATCH").Output(display_name="prompt_batch")
            ]
        )
        
    @classmethod
    def execute(cls, prompts_data="[]"):
        try:
            data = json.loads(prompts_data)
            if not isinstance(data, list):
                data = [{"text": "", "curve": "linear"}]
        except:
            data = [{"text": "", "curve": "linear"}]
            
        if len(data) == 0:
            data = [{"text": "", "curve": "linear"}]
            
        return (data,)


class WanMultiFramePromptToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanMultiFramePromptToVideo",
            display_name="Wan Multi-Frame Prompts to Video",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                io.Custom("PROMPT_BATCH").Input("prompt_batch"),
                io.String.Input("negative_prompt", default="", multiline=True),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=8192, step=16, display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=480, min=16, max=8192, step=16, display_mode=io.NumberDisplay.number),
                io.Int.Input("length", default=81, min=1, max=8192, step=4, display_mode=io.NumberDisplay.number),
                io.Int.Input("batch_size", default=1, min=1, max=4096, display_mode=io.NumberDisplay.number),
                io.Image.Input("ref_images"),
                io.Combo.Input("mode", ["NORMAL", "SINGLE_PERSON"], default="NORMAL", optional=True),
                io.String.Input("ref_positions", default="", optional=True),
                io.Float.Input("ref_strength_high", default=0.8, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("ref_strength_low", default=0.2, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("end_frame_strength_high", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("end_frame_strength_low", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("structural_repulsion_boost", default=1.0, min=1.0, max=2.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True)
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive_high"),
                io.Conditioning.Output(display_name="positive_low"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, prompt_batch, negative_prompt, clip, vae, width, height, length, batch_size, ref_images,
                mode="NORMAL", ref_positions="", ref_strength_high=0.8, ref_strength_low=0.2,
                end_frame_strength_high=1.0, end_frame_strength_low=1.0, structural_repulsion_boost=1.0):
        
        device = comfy.model_management.intermediate_device()
        
        imgs = cls._resize_images(ref_images, width, height, device)
        n_imgs = imgs.shape[0]
        
        total_length = length if n_imgs <= 1 else (n_imgs - 1) * length + 1
        
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((total_length - 1) // 4) + 1

        latent = torch.zeros([batch_size, latent_channels, latent_t,
                             height // spacial_scale, width // spacial_scale], device=device)
        
        positions = cls._parse_positions(ref_positions, n_imgs, total_length)
        
        def align_position(pos: int, total_frames: int) -> int:
            latent_idx = pos // 4
            aligned_pos = latent_idx * 4
            aligned_pos = max(0, min(aligned_pos, total_frames - 1))
            return aligned_pos

        aligned_positions = [align_position(int(p), total_length) for p in positions]

        for i in range(1, len(aligned_positions)):
            if aligned_positions[i] <= aligned_positions[i-1] + 3:
                aligned_positions[i] = min(aligned_positions[i-1] + 4, total_length - 1)

        mask_base = torch.ones((1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), device=device)

        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()

        for i, pos in enumerate(aligned_positions):
            frame_idx = int(pos)

            if i == 0:
                mask_high_noise[:, :, frame_idx:frame_idx + 4] = 0.0
                mask_low_noise[:, :, frame_idx:frame_idx + 4] = 0.0
            elif i == n_imgs - 1:
                mask_high_noise[:, :, -4:] = 1.0 - end_frame_strength_high
                mask_low_noise[:, :, -4:] = 1.0 - end_frame_strength_low
            else:
                start_range = max(0, frame_idx)
                end_range = min(total_length, frame_idx + 4)
                mask_high_noise[:, :, start_range:end_range] = 1.0 - ref_strength_high
                mask_low_noise[:, :, start_range:end_range] = 1.0 - ref_strength_low

        # Encode reference images independently avoiding temporal blur
        encoded_latents = [vae.encode(imgs[i:i+1, :, :, :3]) for i in range(n_imgs)]
        base_cond_latent = torch.zeros(1, latent_channels, latent_t, height // spacial_scale, width // spacial_scale, dtype=encoded_latents[0].dtype, device=device)
        base_cond_latent = comfy.latent_formats.Wan21().process_out(base_cond_latent)

        if mode == "SINGLE_PERSON":
            concat_latent_image_high = base_cond_latent.clone()
            if n_imgs >= 1:
                latent_idx_first = int(aligned_positions[0]) // 4
                concat_latent_image_high[:, :, latent_idx_first:latent_idx_first+1] = encoded_latents[0]
            if n_imgs >= 2:
                for i in range(1, n_imgs - 1):
                    latent_idx_mid = int(aligned_positions[i]) // 4
                    concat_latent_image_high[:, :, latent_idx_mid:latent_idx_mid+1] = encoded_latents[i]
                concat_latent_image_high[:, :, -1:] = encoded_latents[-1]
        else:
            need_selective_image_high = (ref_strength_high == 0.0) or (end_frame_strength_high == 0.0)

            if need_selective_image_high:
                concat_latent_image_high = base_cond_latent.clone()
                if n_imgs >= 1:
                    latent_idx_first = int(aligned_positions[0]) // 4
                    concat_latent_image_high[:, :, latent_idx_first:latent_idx_first + 1] = encoded_latents[0]
                if ref_strength_high > 0.0:
                    for i in range(1, n_imgs - 1):
                        latent_idx_mid = int(aligned_positions[i]) // 4
                        concat_latent_image_high[:, :, latent_idx_mid:latent_idx_mid + 1] = encoded_latents[i]
                if n_imgs >= 2 and end_frame_strength_high > 0.0:
                    concat_latent_image_high[:, :, -1:] = encoded_latents[-1]
            else:
                concat_latent_image_high = base_cond_latent.clone()
                if n_imgs >= 1:
                    latent_idx_first = int(aligned_positions[0]) // 4
                    concat_latent_image_high[:, :, latent_idx_first:latent_idx_first + 1] = encoded_latents[0]
                if n_imgs >= 2:
                    for i in range(1, n_imgs - 1):
                        latent_idx_mid = int(aligned_positions[i]) // 4
                        concat_latent_image_high[:, :, latent_idx_mid:latent_idx_mid + 1] = encoded_latents[i]
                    concat_latent_image_high[:, :, -1:] = encoded_latents[-1]

        if structural_repulsion_boost > 1.001 and total_length > 4 and n_imgs >= 2:
            mask_h, mask_w = mask_high_noise.shape[-2], mask_high_noise.shape[-1]
            boost_factor = structural_repulsion_boost - 1.0
            
            def create_spatial_gradient(img1, img2):
                if img1 is None or img2 is None: return None
                motion_diff = torch.abs(img2[0] - img1[0]).mean(dim=-1, keepdim=False)
                motion_diff_4d = motion_diff.unsqueeze(0).unsqueeze(0)
                motion_diff_scaled = F.interpolate(motion_diff_4d, size=(mask_h, mask_w), mode='bilinear', align_corners=False)
                motion_normalized = (motion_diff_scaled - motion_diff_scaled.min()) / (motion_diff_scaled.max() - motion_diff_scaled.min() + 1e-8)
                spatial_gradient = 1.0 - motion_normalized * boost_factor * 2.5
                return torch.clamp(spatial_gradient, 0.02, 1.0)[0, 0]
            
            for i in range(n_imgs - 1):
                pos1 = int(aligned_positions[i])
                pos2 = int(aligned_positions[i + 1])
                if pos2 > pos1 + 4:
                    spatial_gradient = create_spatial_gradient(imgs[i:i+1].to(device), imgs[i+1:i+2].to(device))
                    if spatial_gradient is not None:
                        for frame_idx in range(pos1 + 4, min(pos2 - 4, pos2)):
                            current_mask = mask_high_noise[:, :, frame_idx, :, :]
                            mask_high_noise[:, :, frame_idx, :, :] = current_mask * spatial_gradient

        if mode == "SINGLE_PERSON":
            mask_low_noise = mask_base.clone()
            if n_imgs >= 1: mask_low_noise[:, :, int(aligned_positions[0]):int(aligned_positions[0]) + 4] = 0.0
            if n_imgs >= 2: mask_low_noise[:, :, -4:] = 1.0 - end_frame_strength_low

            concat_latent_image_low = base_cond_latent.clone()
            if n_imgs >= 1: concat_latent_image_low[:, :, int(aligned_positions[0]) // 4:int(aligned_positions[0]) // 4 + 1] = encoded_latents[0]
            if n_imgs >= 2 and end_frame_strength_low > 0.0: concat_latent_image_low[:, :, -1:] = encoded_latents[-1]
        else:
            need_selective_image = (ref_strength_low == 0.0) or (end_frame_strength_low == 0.0)
            concat_latent_image_low = base_cond_latent.clone()
            if n_imgs >= 1:
                concat_latent_image_low[:, :, int(aligned_positions[0]) // 4:int(aligned_positions[0]) // 4 + 1] = encoded_latents[0]
            if n_imgs >= 2:
                for i in range(1, n_imgs - 1):
                    if not need_selective_image or ref_strength_low > 0.0:
                        concat_latent_image_low[:, :, int(aligned_positions[i]) // 4:int(aligned_positions[i]) // 4 + 1] = encoded_latents[i]
                if not need_selective_image or end_frame_strength_low > 0.0:
                    concat_latent_image_low[:, :, -1:] = encoded_latents[-1]

        mask_high_reshaped = mask_high_noise.view(1, latent_t, 4, mask_high_noise.shape[3], mask_high_noise.shape[4]).transpose(1, 2)
        mask_low_reshaped = mask_low_noise.view(1, latent_t, 4, mask_low_noise.shape[3], mask_low_noise.shape[4]).transpose(1, 2)

        # ---------------- PROMPT SCHEDULING ----------------

        prompts = prompt_batch.copy()
        if len(prompts) == 0:
            prompts = [{"text": "", "curve": "linear"}]
        while len(prompts) < n_imgs:
            prompts.append(prompts[-1])
            
        # Encode negative prompt directly
        neg_tokens = clip.tokenize(negative_prompt)
        neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
        negative_out = [[neg_cond, {"concat_latent_image": concat_latent_image_high, "concat_mask": mask_high_reshaped}]]

        H_latent = latent.shape[-2]
        W_latent = latent.shape[-1]
        
        positive_high_chunks = []
        positive_low_chunks = []
        curves_list = [p.get("curve", "linear") for p in prompts]
        
        for i, prompt_dict in enumerate(prompts):
            text = prompt_dict.get("text", "")
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            
            # temporal mask map
            t_mask = cls._build_temporal_mask(i, aligned_positions, latent_t, H_latent, W_latent, curves_list, device)
            t_mask = t_mask.unsqueeze(0) # [1, T, H, W] matches ComfyUI set_area mask format requirements
            
            cond_high = [[cond, {
                "mask": t_mask.clone(), 
                "concat_latent_image": concat_latent_image_high, 
                "concat_mask": mask_high_reshaped}]]
                
            cond_low = [[cond, {
                "mask": t_mask.clone(), 
                "concat_latent_image": concat_latent_image_low, 
                "concat_mask": mask_low_reshaped}]]
                
            positive_high_chunks.extend(cond_high)
            positive_low_chunks.extend(cond_low)
            
        return io.NodeOutput(positive_high_chunks, positive_low_chunks, negative_out, {"samples": latent})

    @classmethod
    def _build_temporal_mask(cls, prompt_idx, positions, latent_t, H, W, curves, device):
        mask = torch.zeros(latent_t, H, W, dtype=torch.float32, device=device)
        pos = [p // 4 for p in positions] # latent keyframes
        
        cur_p = pos[prompt_idx]
        prev_p = pos[prompt_idx - 1] if prompt_idx > 0 else 0
        next_p = pos[prompt_idx + 1] if prompt_idx < len(pos) - 1 else latent_t - 1
        
        in_curve = curves[prompt_idx] if prompt_idx > 0 else "linear"
        out_curve = curves[prompt_idx + 1] if prompt_idx < len(pos) - 1 else "linear"
        
        def apply_curve(x, curve_type):
            if curve_type == "ease-in": return x * x
            if curve_type == "ease-out": return 1.0 - (1.0 - x)*(1.0 - x)
            if curve_type == "ease-in-out": return x * x * (3.0 - 2.0 * x)
            return x
            
        for i in range(latent_t):
            if i == cur_p:
                mask[i] = 1.0
            elif prev_p < i < cur_p:
                val = apply_curve((i - prev_p) / (cur_p - prev_p), in_curve)
                mask[i] = val
            elif cur_p < i < next_p:
                val = apply_curve((i - cur_p) / (next_p - cur_p), out_curve)
                mask[i] = 1.0 - val
            elif prompt_idx == 0 and i < cur_p:
                mask[i] = 1.0
            elif prompt_idx == len(pos) - 1 and i > cur_p:
                mask[i] = 1.0
        return mask

    @classmethod
    def _resize_images(cls, images, width, height, device):
        images = images.to(device)
        x = images.movedim(-1, 1)
        x = comfy.utils.common_upscale(x, width, height, "bilinear", "center")
        x = x.movedim(1, -1)
        if x.shape[-1] == 4: x = x[..., :3]
        return x

    @classmethod
    def _parse_positions(cls, pos_str, n_imgs, length):
        positions = []
        s = (pos_str or "").strip()
        if s:
            try:
                if s.startswith("["): positions = json.loads(s)
                else: positions = [float(x.strip()) for x in s.split(",") if x.strip()]
            except Exception: positions = []

        if not positions:
            if n_imgs <= 1: positions = [0]
            else: positions = [i * (length - 1) / (n_imgs - 1) for i in range(n_imgs)]

        converted_positions = []
        for p in positions:
            if 0 <= p < 2.0: converted_positions.append(int(p * (length - 1)))
            else: converted_positions.append(int(p))
            
        converted_positions = [max(0, min(length - 1, p)) for p in converted_positions]

        if len(converted_positions) > n_imgs: converted_positions = converted_positions[:n_imgs]
        elif len(converted_positions) < n_imgs: converted_positions.extend([converted_positions[-1]] * (n_imgs - len(converted_positions)))
        return converted_positions

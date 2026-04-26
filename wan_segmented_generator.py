import torch
import torch.nn.functional as F
import json
import gc
from comfy_api.latest import io
import node_helpers
import comfy
import comfy.utils
import comfy.latent_formats
import comfy.model_management
import nodes

class WanSegmentedGenerator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanSegmentedGenerator",
            display_name="Wan Segmented KSampler & Decoder",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Custom("PROMPT_BATCH").Input("prompt_batch"),
                io.String.Input("negative_prompt", default="", multiline=True),
                io.Image.Input("ref_images"),
                
                # Sizing & Length
                io.Int.Input("width", default=832, min=16, max=8192, step=16, display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=480, min=16, max=8192, step=16, display_mode=io.NumberDisplay.number),
                io.Int.Input("length", default=81, min=1, max=8192, step=4, display_mode=io.NumberDisplay.number),
                
                # Sampler Params
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=6.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", comfy.samplers.KSampler.SAMPLERS, default="euler"),
                io.Combo.Input("scheduler", comfy.samplers.KSampler.SCHEDULERS, default="normal"),
                io.Boolean.Input("add_noise", default=True),
                io.Int.Input("start_at_step", default=0, min=0, max=10000),
                io.Int.Input("end_at_step", default=20, min=0, max=10000),
                io.Boolean.Input("return_with_leftover_noise", default=False),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("vae_tile_size", default=512, min=256, max=4096, step=64),
                io.Int.Input("vae_tile_overlap", default=16, min=4, max=1024, step=4),
                io.Boolean.Input("decode_latents", default=True),
                io.Latent.Input("base_latent", optional=True),

                # Advanced Wan params
                io.Combo.Input("prompt_mode", ["temporal_mask (high VRAM)", "average (low VRAM)", "first_only (lowest VRAM)"], default="temporal_mask (high VRAM)", optional=True),
                io.Combo.Input("mode", ["NORMAL", "SINGLE_PERSON"], default="NORMAL", optional=True),
                io.Float.Input("ref_strength_high", default=0.8, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("ref_strength_low", default=0.2, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("end_frame_strength_high", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("end_frame_strength_low", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("structural_repulsion_boost", default=1.0, min=1.0, max=2.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True)
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, vae, prompt_batch, negative_prompt, ref_images, 
                width, height, length, seed, steps, cfg, sampler_name, scheduler, 
                add_noise, start_at_step, end_at_step, return_with_leftover_noise,
                denoise, vae_tile_size, vae_tile_overlap=16, decode_latents=True, base_latent=None,
                mode="NORMAL", prompt_mode="temporal_mask (high VRAM)", ref_strength_high=0.8, ref_strength_low=0.2,
                end_frame_strength_high=1.0, end_frame_strength_low=1.0, structural_repulsion_boost=1.0):
        
        prompts = prompt_batch.copy() if isinstance(prompt_batch, list) else [{"text": "", "curve": "linear"}]
        
        device = comfy.model_management.intermediate_device()
        imgs = cls._resize_images(ref_images, width, height, device)
        n_imgs = imgs.shape[0]
        
        if len(prompts) == 0:
            prompts = [{"text": "", "curve": "linear"}]
        while len(prompts) < n_imgs:
            prompts.append(prompts[-1])
            
        n_segments = max(1, n_imgs - 1)
        
        ksampler = nodes.KSamplerAdvanced()
        
        decoded_segments = []
        latent_segments = []
        
        step_t = (length - 1) // 4
        latent_t = step_t + 1
        
        for i in range(n_segments):
            comfy.model_management.throw_exception_if_processing_interrupted()
            print(f"--- Wan Segmented Generator: Processing Segment {i+1}/{n_segments} ---")
            
            seg_start_idx = i
            seg_end_idx = i + 1 if i + 1 < n_imgs else i
            seg_imgs = imgs[seg_start_idx:seg_end_idx + 1]
            
            seg_prompts = [prompts[seg_start_idx], prompts[seg_end_idx]] if seg_end_idx < len(prompts) and seg_end_idx != seg_start_idx else [prompts[seg_start_idx]]
            
            positive_high, positive_low, negative, segment_blank_latent = cls._build_segment_conditionings(
                clip=clip, vae=vae, 
                seg_imgs=seg_imgs, seg_prompts=seg_prompts,
                negative_prompt=negative_prompt, 
                width=width, height=height, length=length,
                mode=mode, prompt_mode=prompt_mode,
                ref_strength_high=ref_strength_high, ref_strength_low=ref_strength_low,
                end_frame_strength_high=end_frame_strength_high, end_frame_strength_low=end_frame_strength_low,
                structural_repulsion_boost=structural_repulsion_boost,
                device=device
            )
            
            
            segment_seed = seed + i
            
            if base_latent is not None and "samples" in base_latent:
                start_t = i * step_t
                end_t = start_t + latent_t
                if base_latent["samples"].shape[2] >= end_t:
                    segment_latent = {"samples": base_latent["samples"][:, :, start_t:end_t, :, :].clone()}
                else:
                    segment_latent = segment_blank_latent
            else:
                segment_latent = segment_blank_latent
            
            add_noise_str = "enable" if add_noise else "disable"
            return_noise_str = "enable" if return_with_leftover_noise else "disable"
            
            # Aggressively free memory before sampling
            del seg_imgs, seg_prompts, segment_blank_latent
            comfy.model_management.soft_empty_cache()
            gc.collect()
            
            samples = ksampler.sample(
                model=model, 
                add_noise=add_noise_str, 
                noise_seed=segment_seed, 
                steps=steps, 
                cfg=cfg, 
                sampler_name=sampler_name, 
                scheduler=scheduler, 
                positive=positive_high, 
                negative=negative, 
                latent_image=segment_latent, 
                start_at_step=start_at_step, 
                end_at_step=end_at_step, 
                return_with_leftover_noise=return_noise_str,
                denoise=denoise
            )[0]
            
            # Free memory after sampling
            del positive_high, positive_low, negative, segment_latent
            comfy.model_management.soft_empty_cache()
            gc.collect()
            
            latent_segments.append(samples["samples"].cpu())
            
            if decode_latents:
                decoded_images = vae.decode_tiled(
                    samples["samples"], 
                    tile_x=vae_tile_size // 8, 
                    tile_y=vae_tile_size // 8,
                    overlap=vae_tile_overlap
                )
                decoded_images = decoded_images.cpu()
                decoded_segments.append(decoded_images)
            
            del samples
            comfy.model_management.soft_empty_cache()
            gc.collect()

        if decode_latents:
            if len(decoded_segments) == 1:
                final_video = decoded_segments[0]
            else:
                final_video_chunks = []
                for i, seg_images in enumerate(decoded_segments):
                    if i > 0:
                        final_video_chunks.append(seg_images[:, 1:])
                    else:
                        final_video_chunks.append(seg_images)
                
                final_video = torch.cat(final_video_chunks, dim=1)
                
            if final_video.ndim == 5:
                final_video = final_video.view(-1, final_video.shape[2], final_video.shape[3], final_video.shape[4])
        else:
            final_video = torch.zeros((1, height, width, 3))

        if len(latent_segments) == 1:
            final_latent = {"samples": latent_segments[0]}
        else:
            final_latent_chunks = []
            for i, seg_latent in enumerate(latent_segments):
                if i > 0:
                    final_latent_chunks.append(seg_latent[:, :, 1:, :, :])
                else:
                    final_latent_chunks.append(seg_latent)
            final_latent = {"samples": torch.cat(final_latent_chunks, dim=2)}

        return (final_video, final_latent)

    @classmethod
    def _build_segment_conditionings(cls, clip, vae, seg_imgs, seg_prompts, negative_prompt, 
                                     width, height, length, mode, prompt_mode, ref_strength_high, ref_strength_low,
                                     end_frame_strength_high, end_frame_strength_low, structural_repulsion_boost, device):
        
        n_imgs = seg_imgs.shape[0]
        total_length = length
        
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((total_length - 1) // 4) + 1

        latent = torch.zeros([1, latent_channels, latent_t,
                             height // spacial_scale, width // spacial_scale], device=device)
        
        positions = [0] if n_imgs <= 1 else [0, total_length - 1]
        
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

        encoded_latents = [vae.encode(seg_imgs[i:i+1, :, :, :3]) for i in range(n_imgs)]
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
                    spatial_gradient = create_spatial_gradient(seg_imgs[i:i+1].to(device), seg_imgs[i+1:i+2].to(device))
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

        # Encode negative prompt directly
        neg_tokens = clip.tokenize(negative_prompt)
        neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
        negative_out = [[neg_cond, {"concat_latent_image": concat_latent_image_high, "concat_mask": mask_high_reshaped}]]

        H_latent = latent.shape[-2]
        W_latent = latent.shape[-1]
        
        positive_high_chunks = []
        positive_low_chunks = []
        curves_list = [p.get("curve", "linear") for p in seg_prompts]
        
        # Process prompts based on prompt_mode
        if prompt_mode == "first_only (lowest VRAM)" or len(seg_prompts) == 1:
            text = seg_prompts[0].get("text", "")
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            
            cond_high = [[cond, {
                "concat_latent_image": concat_latent_image_high, 
                "concat_mask": mask_high_reshaped}]]
            cond_low = [[cond, {
                "concat_latent_image": concat_latent_image_low, 
                "concat_mask": mask_low_reshaped}]]
                
            positive_high_chunks.extend(cond_high)
            positive_low_chunks.extend(cond_low)
            
        elif prompt_mode == "average (low VRAM)":
            text1 = seg_prompts[0].get("text", "")
            text2 = seg_prompts[1].get("text", "")
            
            tokens1 = clip.tokenize(text1)
            cond1, pooled1 = clip.encode_from_tokens(tokens1, return_pooled=True)
            
            tokens2 = clip.tokenize(text2)
            cond2, pooled2 = clip.encode_from_tokens(tokens2, return_pooled=True)
            
            # Simple 50/50 average
            cond_avg = cond1 * 0.5 + cond2 * 0.5
            
            cond_high = [[cond_avg, {
                "concat_latent_image": concat_latent_image_high, 
                "concat_mask": mask_high_reshaped}]]
            cond_low = [[cond_avg, {
                "concat_latent_image": concat_latent_image_low, 
                "concat_mask": mask_low_reshaped}]]
                
            positive_high_chunks.extend(cond_high)
            positive_low_chunks.extend(cond_low)
            
        else:
            # temporal_mask (high VRAM)
            for i, prompt_dict in enumerate(seg_prompts):
                text = prompt_dict.get("text", "")
                tokens = clip.tokenize(text)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                
                t_mask = cls._build_temporal_mask(i, aligned_positions, latent_t, H_latent, W_latent, curves_list, device)
                t_mask = t_mask.unsqueeze(0)
                
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
            
        return positive_high_chunks, positive_low_chunks, negative_out, {"samples": latent}

    @classmethod
    def _build_temporal_mask(cls, prompt_idx, positions, latent_t, H, W, curves, device):
        mask = torch.zeros(latent_t, H, W, dtype=torch.float32, device=device)
        pos = [p // 4 for p in positions]
        
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

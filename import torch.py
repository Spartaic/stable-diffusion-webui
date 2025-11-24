import torch
from diffusers import StableDiffusionPipeline

# Change this to any NSFW model you want from Civitai or HF
model_id = "ckpt/anything-v4.5-pruned"          # Very popular anime/hentai base
# model_id = "Lykon/dreamshaper-8"              # Realistic & great for NSFW
# model_id = "runwayml/stable-diffusion-v1-5"   # Classic (add your own loras later)

# Remove safety checker completely = 100% uncensored
pipe = StableDiffusionPipeline.from_ckpt(
    model_id + ".ckpt", 
    torch_dtype=torch.float16,
    safety_checker=None,        # <--- removes all censorship
    requires_safety_checker=False
).to("cuda")

# Optional: make it even faster & use less VRAM
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_attention_slicing()

# Your prompt here, Daddy~ go as dirty as you want ♡
prompt = "masterpiece, ultra detailed, muscular goddess in transparent bikini, wet skin, smoking, sitting on golden throne in jungle, nsfw, explicit"
negative_prompt = "blurry, ugly, deformed, censor, clothes, safe filter"

# Generate!
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=1152,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

image.save("daddy_output.png")
print("Done! Check daddy_output.png ♡")